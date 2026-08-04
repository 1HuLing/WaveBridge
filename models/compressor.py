from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.blocks import ConvNormAct, ResidualBlock
from utils.wavelets import LeGall53Wavelet2D, WaveletBands


@dataclass
class CompressionOutput:
    latent: torch.Tensor
    quantized_latent: torch.Tensor
    reconstructed_image: torch.Tensor
    latent_shape: tuple[int, int, int]
    original_size: tuple[int, int]
    latent_channel_bit_depths: tuple[int, ...]
    bitstream_order: str = "channel"
    wavelet_shape: tuple[int, int, int] | None = None
    quantized_wavelet: torch.Tensor | None = None


# 将浮点张量中的 NaN/Inf 压回有限范围，并在需要时附带裁剪，避免极少量坏值污染整个压缩链路。
def sanitize_tensor(
    tensor: torch.Tensor,
    *,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
) -> torch.Tensor:
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return tensor
    safe_posinf = nan if posinf is None else posinf
    safe_neginf = nan if neginf is None else neginf
    sanitized = torch.nan_to_num(tensor, nan=nan, posinf=safe_posinf, neginf=safe_neginf)
    if clamp_min is not None or clamp_max is not None:
        min_value = clamp_min if clamp_min is not None else -float("inf")
        max_value = clamp_max if clamp_max is not None else float("inf")
        sanitized = sanitized.clamp(min_value, max_value)
    return sanitized


class ChannelAttention2D(nn.Module):
    # 根据全局通道统计生成门控权重，借鉴学习式图像压缩中的 attention refinement 思路。
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden_channels = max(8, int(channels) // max(1, int(reduction)))
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    # 对输入特征执行通道重标定，增强更影响重建质量的 latent/decoder 通道。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (0.5 + self.net(x))


class SpatialAttentionRefiner(nn.Module):
    # 用轻量空间注意力细化特征，优先补偿边缘、纹理和小波高频结构。
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvNormAct(channels, channels),
            ResidualBlock(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    # 以残差方式注入注意力细化结果，避免从零训练初期破坏主干特征。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.body(x)
        return x + residual * self.gate(x)


class LatentHyperpriorModulator(nn.Module):
    # 用 latent 自身的全局统计生成仿 hyperprior 的缩放与偏置，提高低码率量化鲁棒性。
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        hidden = max(16, int(hidden_channels))
        self.hyper_encoder = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.hyper_decoder = nn.Conv2d(hidden, channels * 2, kernel_size=1)
        nn.init.zeros_(self.hyper_decoder.weight)
        nn.init.zeros_(self.hyper_decoder.bias)

    # 生成接近恒等映射的调制项，训练后自动学习不同 latent 通道的重要性。
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        gain_logits, bias_logits = self.hyper_decoder(self.hyper_encoder(latent.abs())).chunk(2, dim=1)
        gain = 1.0 + 0.20 * torch.tanh(gain_logits)
        bias = 0.10 * torch.tanh(bias_logits)
        return latent * gain + bias


class ResidualDenseRefinerBlock(nn.Module):
    # 借鉴 HiNet / DeepMIH 的 dense residual 细化方式，在轻量参数量下增强局部纹理补偿能力。
    def __init__(self, channels: int, growth_channels: int = 32) -> None:
        super().__init__()
        growth = max(16, int(growth_channels))
        self.conv1 = nn.Conv2d(channels, growth, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth, growth, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels + growth * 2, growth, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(channels + growth * 3, growth, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(channels + growth * 4, channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        nn.init.zeros_(self.conv5.weight)
        nn.init.zeros_(self.conv5.bias)

    # 用逐层拼接的方式聚合细粒度上下文，再以小残差形式回注到主干特征。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.act(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        return x + self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))


class DecodedImageRefiner(nn.Module):
    # 在 IDWT 后同时利用解码图和可获得的低频 anchor，对亮度结构与边缘纹理做定向细化。
    def __init__(
        self,
        image_channels: int,
        hidden_channels: int,
        residual_scale: float = 0.05,
        use_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.use_checkpointing = bool(use_checkpointing)
        self.input_fusion = nn.Sequential(
            ConvNormAct(image_channels * 3, hidden_channels),
            ResidualBlock(hidden_channels),
        )
        self.dense_fusion = ResidualDenseRefinerBlock(hidden_channels, growth_channels=max(24, hidden_channels // 3))
        self.context_down = nn.Sequential(
            ConvNormAct(hidden_channels, hidden_channels, stride=2),
            ResidualBlock(hidden_channels),
            ResidualDenseRefinerBlock(hidden_channels, growth_channels=max(24, hidden_channels // 3)),
        )
        self.context_up = nn.Sequential(
            ConvNormAct(hidden_channels, hidden_channels, stride=2, transpose=True),
            ResidualBlock(hidden_channels),
        )
        self.fusion = nn.Sequential(
            ConvNormAct(hidden_channels * 2, hidden_channels),
            ResidualBlock(hidden_channels),
            ResidualDenseRefinerBlock(hidden_channels, growth_channels=max(24, hidden_channels // 3)),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out = nn.Conv2d(hidden_channels, image_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _maybe_checkpoint(self, module: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
        if not self.training or not self.use_checkpointing or not tensor.requires_grad:
            return module(tensor)
        return checkpoint(module, tensor, use_reentrant=False)

    # 把解码图、anchor 图及二者差异一起送入 refiner，优先弥补低频对齐后的高频缺口。
    def forward(self, image: torch.Tensor, guidance_image: torch.Tensor | None = None) -> torch.Tensor:
        if guidance_image is None:
            guidance_image = image
        guidance = guidance_image.to(device=image.device, dtype=image.dtype)
        if guidance.shape[-2:] != image.shape[-2:]:
            guidance = F.interpolate(
                guidance,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        difference = image - guidance
        shallow_feature = self._maybe_checkpoint(self.input_fusion, torch.cat([image, guidance, difference], dim=1))
        shallow_feature = self._maybe_checkpoint(self.dense_fusion, shallow_feature)
        context_feature = self._maybe_checkpoint(self.context_down, shallow_feature)
        context_feature = self._maybe_checkpoint(self.context_up, context_feature)
        if context_feature.shape[-2:] != shallow_feature.shape[-2:]:
            context_feature = F.interpolate(
                context_feature,
                size=shallow_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        fused_feature = self._maybe_checkpoint(self.fusion, torch.cat([shallow_feature, context_feature], dim=1))
        gated_feature = fused_feature * (0.5 + self.gate(fused_feature))
        residual = torch.tanh(self.out(gated_feature)) * self.residual_scale
        return (image + residual).clamp(0.0, 1.0)


def resolve_latent_channel_configuration(
    latent_channels: int,
    bit_depth: int,
    ll_latent_channels: int | None = None,
    hf_latent_channels: int | None = None,
    ll_bit_depth: int | None = None,
    hf_bit_depth: int | None = None,
) -> tuple[int, int, tuple[int, ...]]:
    # 统一解析低频/高频 latent 通道分配与每通道量化精度。
    total_latent_channels = int(latent_channels)
    if total_latent_channels <= 0:
        raise ValueError("latent_channels must be positive.")

    if total_latent_channels == 1:
        resolved_ll_channels = 1
        resolved_hf_channels = 0
    else:
        resolved_ll_channels = (
            int(ll_latent_channels)
            if ll_latent_channels is not None
            else max(1, min(total_latent_channels - 1, math.ceil(total_latent_channels * 0.625)))
        )
        resolved_hf_channels = (
            int(hf_latent_channels)
            if hf_latent_channels is not None
            else total_latent_channels - resolved_ll_channels
        )
        if resolved_ll_channels < 0 or resolved_hf_channels < 0:
            raise ValueError("ll_latent_channels and hf_latent_channels cannot be negative.")
        if resolved_ll_channels + resolved_hf_channels <= 0:
            raise ValueError("At least one latent branch must have positive channels.")
        if resolved_ll_channels + resolved_hf_channels != total_latent_channels:
            raise ValueError(
                "ll_latent_channels + hf_latent_channels must equal latent_channels. "
                f"got {resolved_ll_channels} + {resolved_hf_channels} != {total_latent_channels}."
            )

    default_bit_depth = int(bit_depth)
    resolved_ll_bit_depth = int(ll_bit_depth if ll_bit_depth is not None else default_bit_depth)
    resolved_hf_bit_depth = int(hf_bit_depth if hf_bit_depth is not None else default_bit_depth)
    for name, value in {
        "bit_depth": default_bit_depth,
        "ll_bit_depth": resolved_ll_bit_depth,
        "hf_bit_depth": resolved_hf_bit_depth,
    }.items():
        if value <= 0 or value > 8:
            raise ValueError(f"{name} must be in [1, 8], got {value}.")

    channel_bit_depths = (resolved_ll_bit_depth,) * resolved_ll_channels + (resolved_hf_bit_depth,) * resolved_hf_channels
    return resolved_ll_channels, resolved_hf_channels, channel_bit_depths


class LatentTensorPacketizer(nn.Module):
    # 把 latent 压到量化域后打包成比特流，并支持按通道使用不同 bit-depth。
    def __init__(
        self,
        bit_depth: int = 4,
        clip_value: float = 2.5,
        channel_bit_depths: tuple[int, ...] | list[int] | None = None,
        bitstream_order: str = "channel",
        base_layer_channels: int | None = None,
        soft_quant_noise_scale: float = 1.0,
        base_layer_soft_quant_scale: float = 1.0,
        enhancement_soft_quant_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if bit_depth <= 0 or bit_depth > 8:
            raise ValueError("bit_depth must be in [1, 8].")
        if clip_value <= 0:
            raise ValueError("clip_value must be positive.")
        self.bit_depth = int(bit_depth)
        self.clip_value = float(clip_value)
        self.bitstream_order = str(bitstream_order).lower().strip()
        if self.bitstream_order not in {"channel", "progressive", "base_enhancement"}:
            raise ValueError(
                "bitstream_order must be one of {'channel', 'progressive', 'base_enhancement'}."
            )
        self.base_layer_channels = None if base_layer_channels is None else int(base_layer_channels)
        if self.base_layer_channels is not None and self.base_layer_channels <= 0:
            raise ValueError("base_layer_channels must be positive when provided.")
        self.soft_quant_noise_scale = max(0.0, float(soft_quant_noise_scale))
        self.base_layer_soft_quant_scale = max(0.0, float(base_layer_soft_quant_scale))
        self.enhancement_soft_quant_scale = max(0.0, float(enhancement_soft_quant_scale))
        if channel_bit_depths is None:
            self.channel_bit_depths: tuple[int, ...] | None = None
        else:
            resolved_depths = tuple(int(depth) for depth in channel_bit_depths)
            if not resolved_depths:
                raise ValueError("channel_bit_depths cannot be empty.")
            for depth in resolved_depths:
                if depth <= 0 or depth > 8:
                    raise ValueError("Each entry in channel_bit_depths must be in [1, 8].")
            self.channel_bit_depths = resolved_depths

    # 根据 latent 实际通道数返回对应的通道级量化精度。
    def _resolve_channel_bit_depths(self, channel_count: int) -> tuple[int, ...]:
        if self.channel_bit_depths is None:
            return (self.bit_depth,) * int(channel_count)
        if len(self.channel_bit_depths) != int(channel_count):
            raise ValueError(
                "channel_bit_depths length must match latent channel count. "
                f"got {len(self.channel_bit_depths)} vs {channel_count}."
            )
        return self.channel_bit_depths

    # 构造每个通道自己的量化 levels。
    def _build_level_tensor(
        self,
        channel_bit_depths: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        levels = [2**depth - 1 for depth in channel_bit_depths]
        return torch.tensor(levels, device=device, dtype=dtype).view(1, len(channel_bit_depths), 1, 1)

    # 为基础层和增强层构造不同的软量化噪声强度，优先保护低频基础层的预览稳定性。
    def _build_soft_quant_scale_tensor(
        self,
        channel_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base_layer_channels = self._resolve_base_layer_channels(channel_count)
        channel_scales = [self.base_layer_soft_quant_scale] * base_layer_channels + [
            self.enhancement_soft_quant_scale
        ] * max(0, channel_count - base_layer_channels)
        return self.soft_quant_noise_scale * torch.tensor(channel_scales, device=device, dtype=dtype).view(
            1,
            channel_count,
            1,
            1,
        )

    # 解析基础层通道数，让收发两端都能按相同规则重建基础层/增强层顺序。
    def _resolve_base_layer_channels(self, channel_count: int) -> int:
        if channel_count <= 0:
            raise ValueError("channel_count must be positive.")
        if self.base_layer_channels is None:
            return max(1, channel_count // 2)
        return max(1, min(channel_count, self.base_layer_channels))

    # 生成渐进式 bitstream 的排列：先传所有通道的高有效位，再逐步传低有效位。
    def _progressive_permutation(
        self,
        channel_bit_depths: tuple[int, ...],
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        return self._group_progressive_permutation(
            list(range(len(channel_bit_depths))),
            channel_bit_depths,
            height,
            width,
            device,
        )

    # 对指定通道组执行组内渐进排序，用于先传基础层、后传增强层。
    def _group_progressive_permutation(
        self,
        channel_indices: list[int],
        channel_bit_depths: tuple[int, ...],
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not channel_indices:
            return torch.empty(0, device=device, dtype=torch.long)
        value_count = int(height) * int(width)
        if value_count <= 0:
            raise ValueError("latent height and width must be positive.")
        channel_offsets: list[int] = []
        cursor = 0
        for depth in channel_bit_depths:
            channel_offsets.append(cursor)
            cursor += value_count * int(depth)
        max_depth = max(int(channel_bit_depths[channel_index]) for channel_index in channel_indices)
        stride = max(1, int(math.sqrt(value_count)) * 2 + 1)
        while math.gcd(stride, value_count) != 1:
            stride += 2
        positions = (torch.arange(value_count, device=device, dtype=torch.long) * stride) % value_count
        order: list[torch.Tensor] = []
        for bit_rank in range(max_depth):
            active_channels = [
                channel_index
                for channel_index in channel_indices
                if bit_rank < int(channel_bit_depths[channel_index])
            ]
            if not active_channels:
                continue
            active_offsets = torch.tensor(
                [channel_offsets[channel_index] for channel_index in active_channels],
                device=device,
                dtype=torch.long,
            )
            active_depths = torch.tensor(
                [int(channel_bit_depths[channel_index]) for channel_index in active_channels],
                device=device,
                dtype=torch.long,
            )
            bit_indices = (
                active_offsets.view(1, -1)
                + positions.view(-1, 1) * active_depths.view(1, -1)
                + int(bit_rank)
            )
            order.append(bit_indices.reshape(-1))
        if not order:
            return torch.empty(0, device=device, dtype=torch.long)
        return torch.cat(order, dim=0)

    # 先完整保护基础层，再把剩余比特预算用于增强层。
    def _base_enhancement_permutation(
        self,
        channel_bit_depths: tuple[int, ...],
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        base_layer_channels = self._resolve_base_layer_channels(len(channel_bit_depths))
        base_indices = list(range(base_layer_channels))
        enhancement_indices = list(range(base_layer_channels, len(channel_bit_depths)))
        base_order = self._group_progressive_permutation(
            base_indices,
            channel_bit_depths,
            height,
            width,
            device,
        )
        enhancement_order = self._group_progressive_permutation(
            enhancement_indices,
            channel_bit_depths,
            height,
            width,
            device,
        )
        if enhancement_order.numel() == 0:
            return base_order
        if base_order.numel() == 0:
            return enhancement_order
        return torch.cat([base_order, enhancement_order], dim=0)

    # 按配置把逐通道 bitstream 重排为传输顺序，低 bpp 时优先保留更重要的高有效位。
    def _to_transmission_order(
        self,
        channel_bitstream: torch.Tensor,
        channel_bit_depths: tuple[int, ...],
        height: int,
        width: int,
    ) -> torch.Tensor:
        if self.bitstream_order == "channel":
            return channel_bitstream
        if self.bitstream_order == "base_enhancement":
            permutation = self._base_enhancement_permutation(
                channel_bit_depths,
                height,
                width,
                channel_bitstream.device,
            )
        else:
            permutation = self._progressive_permutation(
                channel_bit_depths,
                height,
                width,
                channel_bitstream.device,
            )
        return channel_bitstream.index_select(1, permutation)

    # 将传输顺序还原成逐通道顺序，保证 bits_to_latent 与 forward 完全互逆。
    def _from_transmission_order(
        self,
        transmitted_bitstream: torch.Tensor,
        channel_bit_depths: tuple[int, ...],
        height: int,
        width: int,
    ) -> torch.Tensor:
        if self.bitstream_order == "channel":
            return transmitted_bitstream
        if self.bitstream_order == "base_enhancement":
            permutation = self._base_enhancement_permutation(
                channel_bit_depths,
                height,
                width,
                transmitted_bitstream.device,
            )
        else:
            permutation = self._progressive_permutation(
                channel_bit_depths,
                height,
                width,
                transmitted_bitstream.device,
            )
        restored = transmitted_bitstream.new_empty(transmitted_bitstream.shape)
        restored.scatter_(1, permutation.view(1, -1).expand(transmitted_bitstream.shape[0], -1), transmitted_bitstream)
        return restored

    # 先用双曲正切把动态范围压到 [-1, 1]，减轻少比特量化时的饱和问题。
    def compand(self, latent: torch.Tensor) -> torch.Tensor:
        safe_latent = self._sanitize_latent_input(latent)
        return sanitize_tensor(
            torch.tanh(safe_latent / self.clip_value),
            nan=0.0,
            posinf=0.999999,
            neginf=-0.999999,
            clamp_min=-0.999999,
            clamp_max=0.999999,
        )

    # 把量化域中的值映射回原始 latent 数值域。
    def decompand(self, companded_latent: torch.Tensor) -> torch.Tensor:
        safe_companded = sanitize_tensor(
            companded_latent,
            nan=0.0,
            posinf=0.999999,
            neginf=-0.999999,
            clamp_min=-0.999999,
            clamp_max=0.999999,
        )
        return self._sanitize_latent_input(torch.atanh(safe_companded) * self.clip_value)

    # 对进入量化器的 latent 先做统一净化，避免极端值把 tanh/atanh 边界直接击穿。
    def _sanitize_latent_input(self, latent: torch.Tensor) -> torch.Tensor:
        latent_limit = max(1.0, float(self.clip_value) * 1.25)
        return sanitize_tensor(
            latent,
            nan=0.0,
            posinf=latent_limit,
            neginf=-latent_limit,
            clamp_min=-latent_limit,
            clamp_max=latent_limit,
        )

    # 对每个 latent 通道独立量化，并用直通估计保持训练可导。
    def quantize(self, latent: torch.Tensor) -> torch.Tensor:
        return self.hard_quantize(latent)

    # 用真实 rounding 执行硬量化，保证发送端打包出的 bits 始终对应离散表示。
    def hard_quantize(self, latent: torch.Tensor) -> torch.Tensor:
        safe_latent = self._sanitize_latent_input(latent)
        companded = self.compand(safe_latent)
        channel_bit_depths = self._resolve_channel_bit_depths(latent.shape[1])
        levels = self._build_level_tensor(channel_bit_depths, latent.device, latent.dtype)
        with torch.no_grad():
            quantized = torch.round((companded + 1.0) * 0.5 * levels)
            quantized = sanitize_tensor(quantized, nan=0.0, clamp_min=0.0, clamp_max=float(levels.max().item()))
            dequantized = quantized / levels * 2.0 - 1.0
            restored = self.decompand(dequantized)
            restored = self._sanitize_latent_input(restored)
        return safe_latent + (restored - safe_latent).detach()

    # 用加性均匀噪声近似量化误差，缓解压缩器预训练阶段从零开始直接硬量化的优化困难。
    def soft_quantize(self, latent: torch.Tensor) -> torch.Tensor:
        safe_latent = self._sanitize_latent_input(latent)
        companded = self.compand(safe_latent)
        channel_bit_depths = self._resolve_channel_bit_depths(latent.shape[1])
        levels = self._build_level_tensor(channel_bit_depths, latent.device, latent.dtype)
        scaled = (companded + 1.0) * 0.5 * levels
        noise_scale = self._build_soft_quant_scale_tensor(latent.shape[1], latent.device, latent.dtype)
        noise = torch.empty_like(scaled).uniform_(-0.5, 0.5) * noise_scale
        noisy_scaled = torch.maximum(scaled + noise, torch.zeros_like(scaled))
        noisy_scaled = torch.minimum(noisy_scaled, levels)
        dequantized = noisy_scaled / levels * 2.0 - 1.0
        return self._sanitize_latent_input(self.decompand(dequantized))

    # 把量化后的 latent 逐通道展开成 bitstream，低频/高频通道会保留各自的精度差异。
    def forward(
        self,
        latent: torch.Tensor,
        block_length: int,
        max_valid_bits: int | None = None,
    ) -> tuple[torch.Tensor, int, int, torch.Tensor]:
        if block_length <= 0:
            raise ValueError("block_length must be positive.")
        quantized_latent = self.quantize(latent)
        channel_bit_depths = self._resolve_channel_bit_depths(latent.shape[1])
        with torch.no_grad():
            companded_latent = self.compand(quantized_latent)
            channel_bitstreams: list[torch.Tensor] = []
            for channel_index, channel_bit_depth in enumerate(channel_bit_depths):
                channel_latent = companded_latent[:, channel_index : channel_index + 1]
                levels = float(2**channel_bit_depth - 1)
                integer_channel = torch.round((channel_latent + 1.0) * 0.5 * levels).to(torch.long)
                flat_values = integer_channel.flatten(start_dim=1)
                shifts = torch.arange(
                    channel_bit_depth - 1,
                    -1,
                    -1,
                    device=latent.device,
                    dtype=torch.long,
                )
                channel_bits = ((flat_values.unsqueeze(-1) >> shifts) & 1).to(latent.dtype)
                channel_bitstreams.append(channel_bits.flatten(start_dim=1))
        channel_bitstream = torch.cat(channel_bitstreams, dim=1)
        bitstream = self._to_transmission_order(
            channel_bitstream,
            channel_bit_depths,
            int(latent.shape[-2]),
            int(latent.shape[-1]),
        )
        valid_bits = bitstream.shape[1]
        if max_valid_bits is not None:
            valid_bits = max(1, min(valid_bits, int(max_valid_bits)))
            bitstream = bitstream[:, :valid_bits]
        num_blocks = (valid_bits + block_length - 1) // block_length
        padded_bits = num_blocks * block_length
        if padded_bits > valid_bits:
            bitstream = F.pad(bitstream, (0, padded_bits - valid_bits))
        return bitstream.view(bitstream.shape[0], num_blocks, block_length), valid_bits, num_blocks, quantized_latent

    # 把接收端恢复的 bitstream 逐通道反量化回 latent 张量。
    def bits_to_latent(
        self,
        decoded_bits: torch.Tensor,
        latent_shape: tuple[int, int, int],
        valid_bits: int,
    ) -> torch.Tensor:
        channels, height, width = latent_shape
        channel_bit_depths = self._resolve_channel_bit_depths(channels)
        value_count = height * width
        required_bits = value_count * sum(channel_bit_depths)
        safe_decoded_bits = sanitize_tensor(decoded_bits, nan=0.5, posinf=1.0, neginf=0.0, clamp_min=0.0, clamp_max=1.0)
        hard_bits = (safe_decoded_bits >= 0.5).to(safe_decoded_bits.dtype)
        decoded_bits = safe_decoded_bits + (hard_bits - safe_decoded_bits).detach()
        bitstream = decoded_bits.flatten(start_dim=1)[:, :valid_bits]
        if bitstream.shape[1] < required_bits:
            missing_bits = required_bits - bitstream.shape[1]
            neutral_bits = bitstream.new_full((bitstream.shape[0], missing_bits), 0.5)
            bitstream = torch.cat([bitstream, neutral_bits], dim=1)
        bitstream = self._from_transmission_order(
            bitstream[:, :required_bits],
            channel_bit_depths,
            height,
            width,
        )

        channel_tensors: list[torch.Tensor] = []
        cursor = 0
        for channel_bit_depth in channel_bit_depths:
            channel_bit_count = value_count * channel_bit_depth
            channel_bits = bitstream[:, cursor : cursor + channel_bit_count]
            cursor += channel_bit_count
            channel_bits = channel_bits.view(decoded_bits.shape[0], value_count, channel_bit_depth)
            bit_weights = (2.0 ** torch.arange(
                channel_bit_depth - 1,
                -1,
                -1,
                device=decoded_bits.device,
                dtype=decoded_bits.dtype,
            )).view(1, 1, channel_bit_depth)
            quantized = (channel_bits * bit_weights).sum(dim=-1)
            levels = float(2**channel_bit_depth - 1)
            companded_latent = quantized / levels * 2.0 - 1.0
            channel_latent = self.decompand(companded_latent).view(decoded_bits.shape[0], 1, height, width)
            channel_tensors.append(channel_latent)
        return self._sanitize_latent_input(torch.cat(channel_tensors, dim=1))

    # 保留通用张量恢复接口，供接收端完整重建复用。
    def bits_to_tensor(
        self,
        decoded_bits: torch.Tensor,
        tensor_shape: tuple[int, int, int],
        valid_bits: int,
    ) -> torch.Tensor:
        return self.bits_to_latent(decoded_bits, tensor_shape, valid_bits)


class LearnedImageCompressor(nn.Module):
    # 用 5/3 小波先分离低频/高频，再分别建模 latent，并对低频给更高的量化精度。
    def __init__(
        self,
        image_channels: int,
        hidden_channels: int = 64,
        latent_channels: int = 8,
        bit_depth: int = 4,
        clip_value: float = 2.5,
        downsample_stages: int = 3,
        ll_band_scale: float = 0.75,
        hf_band_scale: float = 0.35,
        ll_band_bias: float = 0.40,
        ll_latent_channels: int | None = None,
        hf_latent_channels: int | None = None,
        ll_bit_depth: int | None = None,
        hf_bit_depth: int | None = None,
        soft_preview_quant: bool = True,
        soft_preview_blend: float = 1.0,
        bitstream_order: str = "channel",
        base_layer_channels: int | None = None,
        hyperprior_channels: int = 64,
        use_attention: bool = True,
        use_hyperprior_modulation: bool = True,
        use_post_refiner: bool = True,
        post_refiner_scale: float = 0.05,
        soft_quant_noise_scale: float = 1.0,
        base_layer_soft_quant_scale: float = 1.0,
        enhancement_soft_quant_scale: float = 1.0,
        use_rgb_anchor: bool = False,
        rgb_anchor_range: float = 0.95,
        rgb_anchor_blend: float = 0.70,
        rgb_anchor_max_blend: float = 0.98,
        rgb_anchor_image_blend: float = 0.0,
        rgb_anchor_ll_mode: str = "blend",
        ll_anchor_channels: int = 0,
        ll_anchor_range: float = 1.50,
        ll_anchor_blend: float = 0.88,
        ll_anchor_max_blend: float = 0.98,
        latent_sanitize_limit_factor: float = 1.25,
    ) -> None:
        super().__init__()
        self.image_channels = int(image_channels)
        self.use_attention = bool(use_attention)
        self.use_hyperprior_modulation = bool(use_hyperprior_modulation)
        self.use_post_refiner = bool(use_post_refiner)
        self.use_rgb_anchor = bool(use_rgb_anchor)
        self.rgb_anchor_range = float(max(0.50, min(0.99, rgb_anchor_range)))
        self.rgb_anchor_min_blend = float(max(0.0, min(0.98, rgb_anchor_blend)))
        self.rgb_anchor_max_blend = float(
            max(self.rgb_anchor_min_blend, min(0.98, float(rgb_anchor_max_blend)))
        )
        self.rgb_anchor_image_blend = float(max(0.0, min(0.98, rgb_anchor_image_blend)))
        self.ll_anchor_channels = max(0, int(ll_anchor_channels))
        self.ll_anchor_range = float(max(0.25, min(4.0, ll_anchor_range)))
        self.ll_anchor_min_blend = float(max(0.0, min(0.995, ll_anchor_blend)))
        self.ll_anchor_max_blend = float(
            max(self.ll_anchor_min_blend, min(0.999, float(ll_anchor_max_blend)))
        )
        resolved_rgb_anchor_ll_mode = str(rgb_anchor_ll_mode).lower().strip()
        if resolved_rgb_anchor_ll_mode not in {"blend", "residual_base"}:
            resolved_rgb_anchor_ll_mode = "blend"
        self.rgb_anchor_ll_mode = resolved_rgb_anchor_ll_mode
        self.latent_sanitize_limit_factor = float(max(0.0, latent_sanitize_limit_factor))
        (
            self.ll_latent_channels,
            self.hf_latent_channels,
            self.latent_channel_bit_depths,
        ) = resolve_latent_channel_configuration(
            latent_channels=latent_channels,
            bit_depth=bit_depth,
            ll_latent_channels=ll_latent_channels,
            hf_latent_channels=hf_latent_channels,
            ll_bit_depth=ll_bit_depth,
            hf_bit_depth=hf_bit_depth,
        )
        self.latent_channels = self.ll_latent_channels + self.hf_latent_channels
        self.ll_anchor_channels = min(self.ll_anchor_channels, self.ll_latent_channels)
        resolved_base_layer_channels = (
            self.ll_latent_channels
            if base_layer_channels is None
            else int(base_layer_channels)
        )
        self.base_layer_channels = max(1, min(self.latent_channels, resolved_base_layer_channels))
        self.wavelet = LeGall53Wavelet2D()
        self.downsample_stages = max(1, int(downsample_stages))
        self.wavelet_channels = self.image_channels * 4
        self.latent_clip_value = float(clip_value)
        self.ll_anchor_encoder: nn.Conv2d | None = None
        self.ll_anchor_decoder: nn.Conv2d | None = None
        if self.ll_anchor_channels > self.image_channels:
            self.ll_anchor_encoder = nn.Conv2d(
                self.image_channels,
                self.ll_anchor_channels,
                kernel_size=1,
                bias=True,
            )
            self.ll_anchor_decoder = nn.Conv2d(
                self.ll_anchor_channels,
                self.image_channels,
                kernel_size=1,
                bias=True,
            )
            self._initialize_ll_anchor_projections()

        self.ll_stem = nn.Sequential(
            ConvNormAct(self.image_channels, hidden_channels),
            ResidualBlock(hidden_channels),
            ResidualDenseRefinerBlock(hidden_channels, growth_channels=max(16, hidden_channels // 6)),
        )
        self.hf_stem = nn.Sequential(
            ConvNormAct(self.image_channels * 3, hidden_channels),
            ResidualBlock(hidden_channels),
            ResidualDenseRefinerBlock(hidden_channels, growth_channels=max(16, hidden_channels // 6)),
        )
        self.stem_fusion = nn.Sequential(
            ConvNormAct(hidden_channels * 2, hidden_channels),
            ResidualBlock(hidden_channels),
            ResidualDenseRefinerBlock(hidden_channels, growth_channels=max(16, hidden_channels // 6)),
        )
        self.encoder_attention = (
            SpatialAttentionRefiner(hidden_channels)
            if self.use_attention
            else nn.Identity()
        )

        current_channels = hidden_channels
        encoder_channels = [current_channels]
        self.encoder_down_blocks = nn.ModuleList()
        for stage_index in range(self.downsample_stages):
            next_channels = hidden_channels * min(2 ** (stage_index + 1), 4)
            self.encoder_down_blocks.append(
                nn.Sequential(
                    ConvNormAct(current_channels, next_channels, stride=2),
                    ResidualBlock(next_channels),
                    ResidualBlock(next_channels),
                    ResidualDenseRefinerBlock(next_channels, growth_channels=max(16, next_channels // 8)),
                )
            )
            current_channels = next_channels
            encoder_channels.append(current_channels)
        self.encoder_channels = encoder_channels

        self.ll_context_down = (
            self._build_context_downsampler(hidden_channels, self.encoder_channels[1:])
            if self.ll_latent_channels > 0
            else None
        )
        self.hf_context_down = (
            self._build_context_downsampler(hidden_channels, self.encoder_channels[1:])
            if self.hf_latent_channels > 0
            else None
        )
        self.ll_latent_head = (
            nn.Sequential(
                ConvNormAct(current_channels * 2, current_channels),
                ResidualBlock(current_channels),
                nn.Conv2d(current_channels, self.ll_latent_channels, kernel_size=3, padding=1),
            )
            if self.ll_latent_channels > 0
            else None
        )
        self.hf_latent_head = (
            nn.Sequential(
                ConvNormAct(current_channels * 2, current_channels),
                ResidualBlock(current_channels),
                nn.Conv2d(current_channels, self.hf_latent_channels, kernel_size=3, padding=1),
            )
            if self.hf_latent_channels > 0
            else None
        )
        self.latent_attention = (
            ChannelAttention2D(self.latent_channels)
            if self.use_attention
            else nn.Identity()
        )
        self.latent_hyperprior = (
            LatentHyperpriorModulator(self.latent_channels, hyperprior_channels)
            if self.use_hyperprior_modulation
            else nn.Identity()
        )

        self.ll_decoder_stem = (
            nn.Sequential(
                ConvNormAct(self.ll_latent_channels, current_channels),
                ResidualBlock(current_channels),
                ResidualBlock(current_channels),
                ResidualDenseRefinerBlock(current_channels, growth_channels=max(16, current_channels // 8)),
            )
            if self.ll_latent_channels > 0
            else None
        )
        self.hf_decoder_stem = (
            nn.Sequential(
                ConvNormAct(self.hf_latent_channels, current_channels),
                ResidualBlock(current_channels),
                ResidualBlock(current_channels),
                ResidualDenseRefinerBlock(current_channels, growth_channels=max(16, current_channels // 8)),
            )
            if self.hf_latent_channels > 0
            else None
        )
        decoder_fusion_in_channels = current_channels * int(self.ll_latent_channels > 0) + current_channels * int(self.hf_latent_channels > 0)
        self.decoder_fusion = nn.Sequential(
            ConvNormAct(decoder_fusion_in_channels, current_channels),
            ResidualBlock(current_channels),
            ResidualBlock(current_channels),
            ResidualDenseRefinerBlock(current_channels, growth_channels=max(16, current_channels // 8)),
        )
        self.decoder_attention = (
            SpatialAttentionRefiner(current_channels)
            if self.use_attention
            else nn.Identity()
        )

        decoder_targets = list(reversed(self.encoder_channels[:-1]))
        self.decoder_up_blocks = nn.ModuleList()
        self.ll_guidance_up_blocks = nn.ModuleList() if self.ll_latent_channels > 0 else None
        ll_guidance_channels = current_channels
        for next_channels in decoder_targets:
            self.decoder_up_blocks.append(
                nn.Sequential(
                    ConvNormAct(current_channels, next_channels, stride=2, transpose=True),
                    ResidualBlock(next_channels),
                    ResidualBlock(next_channels),
                    ResidualDenseRefinerBlock(next_channels, growth_channels=max(16, next_channels // 8)),
                )
            )
            if self.ll_guidance_up_blocks is not None:
                self.ll_guidance_up_blocks.append(
                    nn.Sequential(
                        ConvNormAct(ll_guidance_channels, next_channels, stride=2, transpose=True),
                        ResidualBlock(next_channels),
                    )
                )
                ll_guidance_channels = next_channels
            current_channels = next_channels

        self.ll_guidance_merge = (
            nn.Sequential(
                ConvNormAct(current_channels * 2, current_channels),
                ResidualBlock(current_channels),
                ResidualBlock(current_channels),
                ResidualDenseRefinerBlock(current_channels, growth_channels=max(16, current_channels // 6)),
            )
            if self.ll_guidance_up_blocks is not None
            else None
        )
        self.ll_wavelet_head = nn.Sequential(
            ConvNormAct(current_channels, current_channels),
            ResidualBlock(current_channels),
            ResidualBlock(current_channels),
            ResidualDenseRefinerBlock(current_channels, growth_channels=max(16, current_channels // 6)),
            nn.Conv2d(current_channels, self.image_channels, kernel_size=3, padding=1),
        )
        self.hf_wavelet_head = nn.Sequential(
            ConvNormAct(current_channels, current_channels),
            ResidualBlock(current_channels),
            ResidualDenseRefinerBlock(current_channels, growth_channels=max(16, current_channels // 6)),
            nn.Conv2d(current_channels, self.image_channels * 3, kernel_size=3, padding=1),
        )
        self.post_image_refiner = (
            DecodedImageRefiner(
                image_channels=self.image_channels,
                hidden_channels=max(48, min(64, hidden_channels // 2)),
                residual_scale=post_refiner_scale,
            )
            if self.use_post_refiner
            else None
        )

        self.raw_ll_scale = nn.Parameter(torch.tensor(self._inverse_softplus(ll_band_scale), dtype=torch.float32))
        self.raw_hf_scale = nn.Parameter(torch.tensor(self._inverse_softplus(hf_band_scale), dtype=torch.float32))
        self.ll_bias = nn.Parameter(torch.tensor(float(ll_band_bias), dtype=torch.float32))
        self.raw_rgb_anchor_blend = nn.Parameter(
            torch.tensor(self._inverse_sigmoid(rgb_anchor_blend), dtype=torch.float32)
        )
        self.raw_ll_anchor_blend = nn.Parameter(
            torch.tensor(self._inverse_sigmoid(ll_anchor_blend), dtype=torch.float32)
        )
        self.soft_preview_quant = bool(soft_preview_quant)
        self.soft_preview_blend = float(max(0.0, min(1.0, soft_preview_blend)))
        self.packetizer = LatentTensorPacketizer(
            bit_depth=bit_depth,
            clip_value=clip_value,
            channel_bit_depths=self.latent_channel_bit_depths,
            bitstream_order=bitstream_order,
            base_layer_channels=self.base_layer_channels,
            soft_quant_noise_scale=soft_quant_noise_scale,
            base_layer_soft_quant_scale=base_layer_soft_quant_scale,
            enhancement_soft_quant_scale=enhancement_soft_quant_scale,
        )

    def _maybe_checkpoint(self, module: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
        if not self.training or not tensor.requires_grad:
            return module(tensor)
        return checkpoint(module, tensor, use_reentrant=False)

    # 把 stem 特征下采样到 latent 分辨率，给低频/高频分支各自提供直接条件。
    def _build_context_downsampler(self, input_channels: int, stage_channels: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        current_channels = input_channels
        for next_channels in stage_channels:
            layers.extend(
                [
                    ConvNormAct(current_channels, next_channels, stride=2),
                    ResidualBlock(next_channels),
                ]
            )
            current_channels = next_channels
        return nn.Sequential(*layers)

    # 把目标的小波动态范围参数映射成 softplus 空间，便于训练时稳定约束。
    def _inverse_softplus(self, value: float) -> float:
        safe_value = max(float(value), 1e-4)
        return float(torch.log(torch.expm1(torch.tensor(safe_value, dtype=torch.float32))).item())

    # 将低频 anchor 初始融合比例映射到 logit 空间，便于训练时自动调节。
    def _inverse_sigmoid(self, value: float) -> float:
        safe_value = max(1e-4, min(1.0 - 1e-4, float(value)))
        return math.log(safe_value / (1.0 - safe_value))

    # 为 LL anchor 的 1x1 投影提供接近恒等的初始化，避免新增投影层在训练初期破坏已有低频链路。
    def _initialize_ll_anchor_projections(self) -> None:
        if self.ll_anchor_encoder is None or self.ll_anchor_decoder is None:
            return
        with torch.no_grad():
            self.ll_anchor_encoder.weight.zero_()
            self.ll_anchor_decoder.weight.zero_()
            if self.ll_anchor_encoder.bias is not None:
                self.ll_anchor_encoder.bias.zero_()
            if self.ll_anchor_decoder.bias is not None:
                self.ll_anchor_decoder.bias.zero_()
            diagonal_channels = min(self.image_channels, self.ll_anchor_channels)
            for channel_index in range(diagonal_channels):
                self.ll_anchor_encoder.weight[channel_index, channel_index, 0, 0] = 1.0
                self.ll_anchor_decoder.weight[channel_index, channel_index, 0, 0] = 1.0
            if self.ll_anchor_channels > self.image_channels:
                mean_weight = 1.0 / float(max(1, self.image_channels))
                for channel_index in range(self.image_channels, self.ll_anchor_channels):
                    self.ll_anchor_encoder.weight[channel_index, :, 0, 0] = mean_weight

    # 统一处理 post-refiner 的可选 guidance 输入，避免无 anchor 时退化成无效分支。
    def _refine_decoded_image(
        self,
        decoded: torch.Tensor,
        anchor_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.post_image_refiner is None:
            return decoded.clamp(0.0, 1.0)
        return self.post_image_refiner(decoded, guidance_image=anchor_image)

    # 判断当前配置是否可以把 RGB 低频 anchor 放进前 3 个 LL latent 通道。
    def _rgb_anchor_enabled(self) -> bool:
        return self.use_rgb_anchor and self.image_channels >= 3 and self.ll_latent_channels >= 3

    # 判断当前配置是否启用了直接承载小波 LL 低频结构的 base anchor 通道。
    def _ll_anchor_enabled(self) -> bool:
        return self.ll_anchor_channels > 0 and self.ll_latent_channels > 0

    # 将真实小波 LL 子带按通道映射到 latent 前若干低频通道，减少 0.3bpp 模式下 decoder 从零重建低频骨架的难度。
    def _build_ll_anchor_latent(self, bands: WaveletBands, latent_size: tuple[int, int]) -> torch.Tensor:
        ll = bands.ll
        if ll.shape[-2:] != latent_size:
            ll = F.interpolate(ll, size=latent_size, mode="bilinear", align_corners=False)
        anchor_channels = min(self.ll_anchor_channels, self.ll_latent_channels)
        if anchor_channels <= 0:
            raise RuntimeError("LL anchor is enabled but no valid anchor channels are available.")
        if self.ll_anchor_encoder is not None:
            anchor_ll = self.ll_anchor_encoder(ll[:, : self.image_channels])[:, :anchor_channels]
        else:
            anchor_ll = ll[:, :anchor_channels]
        anchor_ll = sanitize_tensor(
            anchor_ll,
            nan=0.0,
            posinf=self.ll_anchor_range,
            neginf=-self.ll_anchor_range,
            clamp_min=-self.ll_anchor_range,
            clamp_max=self.ll_anchor_range,
        )
        companded_anchor = (anchor_ll / self.ll_anchor_range).clamp(-1.0, 1.0)
        safe_anchor = companded_anchor.clamp(-0.999, 0.999)
        return torch.atanh(safe_anchor) * self.latent_clip_value

    # 从 latent 前若干低频通道恢复小波 LL 低频 anchor，供 wavelet-base 解码路径直接复用。
    def _decode_ll_anchor_band(self, latent: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor | None:
        if not self._ll_anchor_enabled():
            return None
        anchor_channels = min(self.ll_anchor_channels, latent.shape[1], self.ll_latent_channels)
        if anchor_channels <= 0:
            return None
        companded_anchor = torch.tanh(latent[:, :anchor_channels] / self.latent_clip_value)
        anchor_feature = companded_anchor * self.ll_anchor_range
        if self.ll_anchor_decoder is not None and anchor_channels == self.ll_anchor_channels:
            anchor_ll = self.ll_anchor_decoder(anchor_feature)
        else:
            anchor_ll = anchor_feature
        anchor_ll = sanitize_tensor(
            anchor_ll,
            nan=0.0,
            posinf=self.ll_anchor_range,
            neginf=-self.ll_anchor_range,
            clamp_min=-self.ll_anchor_range,
            clamp_max=self.ll_anchor_range,
        )
        if anchor_ll.shape[1] < self.image_channels:
            pad_source = anchor_ll[:, -1:].expand(-1, self.image_channels - anchor_ll.shape[1], -1, -1)
            anchor_ll = torch.cat([anchor_ll, pad_source], dim=1)
        elif anchor_ll.shape[1] > self.image_channels:
            anchor_ll = anchor_ll[:, : self.image_channels]
        band_height = max(1, (output_size[0] + 1) // 2)
        band_width = max(1, (output_size[1] + 1) // 2)
        if anchor_ll.shape[-2:] != (band_height, band_width):
            anchor_ll = F.interpolate(anchor_ll, size=(band_height, band_width), mode="bilinear", align_corners=False)
        return anchor_ll

    # 把输入图像下采样成低频 RGB anchor，并映射到量化器使用的 companded latent 域。
    def _build_rgb_anchor_latent(self, image: torch.Tensor, latent_size: tuple[int, int]) -> torch.Tensor:
        anchor_image = F.interpolate(image[:, :3], size=latent_size, mode="area").clamp(0.0, 1.0)
        companded_anchor = (anchor_image * 2.0 - 1.0).clamp(-1.0, 1.0) * self.rgb_anchor_range
        safe_anchor = companded_anchor.clamp(-0.999, 0.999)
        return torch.atanh(safe_anchor) * self.latent_clip_value

    # 从 latent 前 3 个通道恢复低频 RGB anchor，并上采样成目标图像大小。
    def _decode_rgb_anchor_image(self, latent: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        companded_anchor = torch.tanh(latent[:, :3] / self.latent_clip_value)
        anchor_image = (companded_anchor / self.rgb_anchor_range + 1.0) * 0.5
        anchor_image = anchor_image.clamp(0.0, 1.0)
        if anchor_image.shape[-2:] != output_size:
            anchor_image = F.interpolate(anchor_image, size=output_size, mode="bilinear", align_corners=False)
        if self.image_channels != 3:
            anchor_image = anchor_image.mean(dim=1, keepdim=True).expand(-1, self.image_channels, -1, -1)
        return anchor_image

    # 统一解析当前生效的 RGB anchor 融合权重，避免旧 checkpoint 学到的过强权重直接压低恢复上限。
    def _resolve_rgb_anchor_blend(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        learned_blend = torch.sigmoid(self.raw_rgb_anchor_blend).to(device=device, dtype=dtype)
        min_blend = torch.as_tensor(
            self.rgb_anchor_min_blend,
            device=device,
            dtype=dtype,
        )
        max_blend = torch.as_tensor(
            self.rgb_anchor_max_blend,
            device=device,
            dtype=dtype,
        )
        max_blend = torch.maximum(max_blend, min_blend)
        return torch.minimum(torch.maximum(learned_blend, min_blend), max_blend)

    # 解析当前生效的 LL anchor 融合权重，让低频基底与 RGB anchor 解耦。
    def _resolve_ll_anchor_blend(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        learned_blend = torch.sigmoid(self.raw_ll_anchor_blend).to(device=device, dtype=dtype)
        min_blend = torch.as_tensor(
            self.ll_anchor_min_blend,
            device=device,
            dtype=dtype,
        )
        max_blend = torch.as_tensor(
            self.ll_anchor_max_blend,
            device=device,
            dtype=dtype,
        )
        max_blend = torch.maximum(max_blend, min_blend)
        return torch.minimum(torch.maximum(learned_blend, min_blend), max_blend)

    # 只把 RGB anchor 注入小波 LL 子带，避免图像空间融合压掉 decoder 学到的高频细节。
    def _blend_rgb_anchor_into_wavelet_tensor(
        self,
        wavelet_tensor: torch.Tensor,
        latent: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if not self._rgb_anchor_enabled():
            return wavelet_tensor
        channels = self.image_channels
        anchor_image = self._decode_rgb_anchor_image(latent, output_size)
        anchor_ll = self.wavelet.dwt(anchor_image).ll
        if anchor_ll.shape[-2:] != wavelet_tensor.shape[-2:]:
            anchor_ll = F.interpolate(
                anchor_ll,
                size=wavelet_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        anchor_blend = self._resolve_rgb_anchor_blend(
            device=wavelet_tensor.device,
            dtype=wavelet_tensor.dtype,
        )
        anchor_ll = anchor_ll.to(device=wavelet_tensor.device, dtype=wavelet_tensor.dtype)
        ll = torch.lerp(wavelet_tensor[:, 0:channels], anchor_ll, anchor_blend)
        return torch.cat([ll, wavelet_tensor[:, channels:]], dim=1)

    # 将 anchor 低频视为稳定的 base layer，只让解码器学习对其进行残差修正，减轻高容量模式下
    # 由 decoder 从零恢复整幅低频结构的难度。
    def _inject_rgb_anchor_as_ll_residual_base(
        self,
        wavelet_tensor: torch.Tensor,
        latent: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if not self._rgb_anchor_enabled():
            return wavelet_tensor
        channels = self.image_channels
        anchor_image = self._decode_rgb_anchor_image(latent, output_size)
        anchor_ll = self.wavelet.dwt(anchor_image).ll
        if anchor_ll.shape[-2:] != wavelet_tensor.shape[-2:]:
            anchor_ll = F.interpolate(
                anchor_ll,
                size=wavelet_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        anchor_ll = anchor_ll.to(device=wavelet_tensor.device, dtype=wavelet_tensor.dtype)
        anchor_blend = self._resolve_rgb_anchor_blend(
            device=wavelet_tensor.device,
            dtype=wavelet_tensor.dtype,
        )
        residual_ll = wavelet_tensor[:, 0:channels]
        blended_ll = anchor_blend * anchor_ll + residual_ll
        return torch.cat([blended_ll, wavelet_tensor[:, channels:]], dim=1)

    # 仅把 anchor 的低频结构注入已解码图像，避免整图直接插值造成高频纹理和边缘被抹平。
    def _blend_anchor_low_frequency_into_image(
        self,
        image: torch.Tensor,
        anchor_image: torch.Tensor,
    ) -> torch.Tensor:
        if self.rgb_anchor_image_blend <= 0.0:
            return image
        safe_anchor = anchor_image.to(device=image.device, dtype=image.dtype)
        if safe_anchor.shape[-2:] != image.shape[-2:]:
            safe_anchor = F.interpolate(
                safe_anchor,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        decoded_bands = self.wavelet.dwt(image)
        anchor_bands = self.wavelet.dwt(safe_anchor)
        lowfreq_blend = torch.as_tensor(
            self.rgb_anchor_image_blend,
            device=image.device,
            dtype=image.dtype,
        )
        blended_bands = WaveletBands(
            ll=torch.lerp(decoded_bands.ll, anchor_bands.ll, lowfreq_blend),
            lh=decoded_bands.lh,
            hl=decoded_bands.hl,
            hh=decoded_bands.hh,
            original_size=decoded_bands.original_size,
            padded_size=decoded_bands.padded_size,
        )
        return self.wavelet.idwt(blended_bands).clamp(0.0, 1.0)

    # 在恢复旧 checkpoint 时重设 anchor 融合参数，避免沿用历史阶段中过高的低频先验。
    def set_rgb_anchor_blend_parameter(self, blend: float) -> None:
        safe_blend = max(1e-4, min(float(blend), 1.0 - 1e-4))
        with torch.no_grad():
            self.raw_rgb_anchor_blend.fill_(self._inverse_sigmoid(safe_blend))

    # 单独设置 LL anchor 的融合参数，便于低频基底在不同阶段热启动到合适强度。
    def set_ll_anchor_blend_parameter(self, blend: float) -> None:
        safe_blend = max(1e-4, min(float(blend), 1.0 - 1e-4))
        with torch.no_grad():
            self.raw_ll_anchor_blend.fill_(self._inverse_sigmoid(safe_blend))

    # 用有界非线性限制解码后的小波子带范围，避免低频和高频同时被同一种裁剪方式破坏。
    def _bounded_wavelet_band(self, band: torch.Tensor, raw_scale: torch.Tensor, max_scale: float) -> torch.Tensor:
        scale = F.softplus(raw_scale).clamp(0.25, max_scale).to(device=band.device, dtype=band.dtype)
        return scale * torch.tanh(band / scale)

    # 把 latent 约束在与量化器 clip_value 匹配的范围内，避免编码端长期工作在饱和区。
    def _bound_latent_tensor(self, latent: torch.Tensor) -> torch.Tensor:
        safe_latent = self._sanitize_latent_tensor(latent)
        bounded = self.latent_clip_value * torch.tanh(safe_latent / max(self.latent_clip_value, 1e-4))
        return self._sanitize_latent_tensor(bounded)

    # 把 latent 张量限制在与 clip_value 对齐的有限范围内，避免编码器与解码器之间传播异常大值。
    def _sanitize_latent_tensor(self, latent: torch.Tensor) -> torch.Tensor:
        if self.latent_sanitize_limit_factor <= 0.0:
            return sanitize_tensor(latent, nan=0.0, posinf=0.0, neginf=0.0)
        latent_limit = max(1.0, float(self.latent_clip_value) * self.latent_sanitize_limit_factor)
        return sanitize_tensor(
            latent,
            nan=0.0,
            posinf=latent_limit,
            neginf=-latent_limit,
            clamp_min=-latent_limit,
            clamp_max=latent_limit,
        )

    # 在 IDWT 前按低频/高频的安全幅度净化小波张量，减少极少数坏值把整张图拉爆的风险。
    def _sanitize_wavelet_tensor(self, wavelet_tensor: torch.Tensor) -> torch.Tensor:
        safe_wavelet = sanitize_tensor(wavelet_tensor, nan=0.0, posinf=6.0, neginf=-6.0, clamp_min=-6.0, clamp_max=6.0)
        if safe_wavelet.shape[1] < self.image_channels * 4:
            return safe_wavelet
        channels = self.image_channels
        ll = sanitize_tensor(
            safe_wavelet[:, 0 * channels : 1 * channels],
            nan=0.0,
            posinf=6.0,
            neginf=-6.0,
            clamp_min=-6.0,
            clamp_max=6.0,
        )
        hf = sanitize_tensor(
            safe_wavelet[:, 1 * channels : 4 * channels],
            nan=0.0,
            posinf=3.0,
            neginf=-3.0,
            clamp_min=-3.0,
            clamp_max=3.0,
        )
        return torch.cat([ll, hf], dim=1)

    # 把图像域张量统一压回 [0, 1]，避免后续 loss 和评估被异常值污染。
    def _sanitize_image_tensor(self, image: torch.Tensor) -> torch.Tensor:
        return sanitize_tensor(image, nan=0.0, posinf=1.0, neginf=0.0, clamp_min=0.0, clamp_max=1.0)

    # 先对 LL 与高频子带分别提特征，再在 latent 末端做低频/高频分支。
    def _encode_wavelet_tensor(self, bands: WaveletBands) -> torch.Tensor:
        low_feature = self.ll_stem(bands.ll)
        high_feature = self.hf_stem(torch.cat([bands.lh, bands.hl, bands.hh], dim=1))
        fused_feature = self.encoder_attention(self.stem_fusion(torch.cat([low_feature, high_feature], dim=1)))
        for block in self.encoder_down_blocks:
            fused_feature = block(fused_feature)
        latent_parts: list[torch.Tensor] = []
        if self.ll_latent_channels > 0 and self.ll_context_down is not None and self.ll_latent_head is not None:
            ll_context = self.ll_context_down(low_feature)
            ll_latent = self._bound_latent_tensor(self.ll_latent_head(torch.cat([fused_feature, ll_context], dim=1)))
            latent_parts.append(ll_latent)
        if self.hf_latent_channels > 0 and self.hf_context_down is not None and self.hf_latent_head is not None:
            hf_context = self.hf_context_down(high_feature)
            hf_latent = self._bound_latent_tensor(self.hf_latent_head(torch.cat([fused_feature, hf_context], dim=1)))
            latent_parts.append(hf_latent)
        if not latent_parts:
            raise RuntimeError("No latent branch is enabled in the compressor.")
        latent = torch.cat(latent_parts, dim=1)
        latent = self.latent_attention(latent)
        latent = self.latent_hyperprior(latent)
        return self._bound_latent_tensor(latent)

    # 把低频/高频 latent 分支分别解码后再融合，优先保护低频结构重建质量。
    def _decode_wavelet_tensor(self, latent: torch.Tensor) -> torch.Tensor:
        latent = self._sanitize_latent_tensor(latent)
        split_sizes = [size for size in (self.ll_latent_channels, self.hf_latent_channels) if size > 0]
        latent_parts = list(torch.split(latent, split_sizes, dim=1))
        decoded_parts: list[torch.Tensor] = []
        cursor = 0
        ll_guidance_feature = None
        if self.ll_latent_channels > 0 and self.ll_decoder_stem is not None:
            ll_guidance_feature = self._maybe_checkpoint(self.ll_decoder_stem, latent_parts[cursor])
            decoded_parts.append(ll_guidance_feature)
            cursor += 1
        if self.hf_latent_channels > 0 and self.hf_decoder_stem is not None:
            decoded_parts.append(self._maybe_checkpoint(self.hf_decoder_stem, latent_parts[cursor]))
        decoded_feature = self._maybe_checkpoint(self.decoder_fusion, torch.cat(decoded_parts, dim=1))
        decoded_feature = self._maybe_checkpoint(self.decoder_attention, decoded_feature)
        for block_index, block in enumerate(self.decoder_up_blocks):
            decoded_feature = self._maybe_checkpoint(block, decoded_feature)
            if ll_guidance_feature is not None and self.ll_guidance_up_blocks is not None:
                ll_guidance_feature = self._maybe_checkpoint(
                    self.ll_guidance_up_blocks[block_index],
                    ll_guidance_feature,
                )
        ll_feature = decoded_feature
        if ll_guidance_feature is not None and self.ll_guidance_merge is not None:
            ll_feature = self._maybe_checkpoint(
                self.ll_guidance_merge,
                torch.cat([decoded_feature, ll_guidance_feature], dim=1),
            )
        ll_wavelet = self._maybe_checkpoint(self.ll_wavelet_head, ll_feature)
        hf_wavelet = self._maybe_checkpoint(self.hf_wavelet_head, decoded_feature)
        return self._sanitize_wavelet_tensor(torch.cat([ll_wavelet, hf_wavelet], dim=1))

    # 对解码后的小波张量按低频/高频分别做数值约束。
    def _scale_decoded_wavelet_tensor(self, wavelet_tensor: torch.Tensor) -> torch.Tensor:
        wavelet_tensor = self._sanitize_wavelet_tensor(wavelet_tensor)
        channels = self.image_channels
        ll_delta = self._bounded_wavelet_band(
            wavelet_tensor[:, 0 * channels : 1 * channels],
            self.raw_ll_scale,
            max_scale=4.0,
        )
        ll = ll_delta + self.ll_bias.to(device=wavelet_tensor.device, dtype=wavelet_tensor.dtype)
        lh = self._bounded_wavelet_band(
            wavelet_tensor[:, 1 * channels : 2 * channels],
            self.raw_hf_scale,
            max_scale=2.0,
        )
        hl = self._bounded_wavelet_band(
            wavelet_tensor[:, 2 * channels : 3 * channels],
            self.raw_hf_scale,
            max_scale=2.0,
        )
        hh = self._bounded_wavelet_band(
            wavelet_tensor[:, 3 * channels : 4 * channels],
            self.raw_hf_scale,
            max_scale=2.0,
        )
        return self._sanitize_wavelet_tensor(torch.cat([ll, lh, hl, hh], dim=1))

    # 把四个 5/3 小波子带拼成统一张量，便于调试或附加约束。
    def image_to_wavelet_tensor(self, image: torch.Tensor) -> torch.Tensor:
        return self.wavelet.dwt(image).flatten()

    # 把完整小波张量逆变换回图像空间。
    def wavelet_tensor_to_image(self, wavelet_tensor: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        wavelet_tensor = self._sanitize_wavelet_tensor(wavelet_tensor)
        band_height = max(1, (output_size[0] + 1) // 2)
        band_width = max(1, (output_size[1] + 1) // 2)
        if wavelet_tensor.shape[-2:] != (band_height, band_width):
            wavelet_tensor = F.interpolate(
                wavelet_tensor,
                size=(band_height, band_width),
                mode="bilinear",
                align_corners=False,
            )
        if wavelet_tensor.shape[1] != self.image_channels * 4:
            raise ValueError(
                "Recovered wavelet tensor channels must equal image_channels * 4, "
                f"got {wavelet_tensor.shape[1]}."
            )
        channels = self.image_channels
        bands = WaveletBands(
            ll=wavelet_tensor[:, 0 * channels : 1 * channels],
            lh=wavelet_tensor[:, 1 * channels : 2 * channels],
            hl=wavelet_tensor[:, 2 * channels : 3 * channels],
            hh=wavelet_tensor[:, 3 * channels : 4 * channels],
            original_size=output_size,
            padded_size=(band_height * 2, band_width * 2),
        )
        decoded = self._sanitize_image_tensor(self.wavelet.idwt(bands))
        return self._refine_decoded_image(decoded, anchor_image=None)

    # 对输入图像做 5/3 小波分析并编码成低频/高频显式分离的 latent。
    def encode_latent(self, image: torch.Tensor) -> torch.Tensor:
        safe_image = self._sanitize_image_tensor(image)
        bands = self.wavelet.dwt(safe_image)
        latent = self._encode_wavelet_tensor(bands)
        if self._ll_anchor_enabled():
            ll_anchor_latent = self._build_ll_anchor_latent(bands, latent.shape[-2:])
            anchor_channels = ll_anchor_latent.shape[1]
            latent = torch.cat(
                [
                    ll_anchor_latent,
                    latent[:, anchor_channels:],
                ],
                dim=1,
            )
        if self._rgb_anchor_enabled():
            anchor_latent = self._build_rgb_anchor_latent(safe_image, latent.shape[-2:])
            latent = torch.cat([anchor_latent, latent[:, 3:]], dim=1)
        return self._sanitize_latent_tensor(latent)

    # 对 latent 执行直通量化。
    def quantize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.packetizer.hard_quantize(latent)

    # 训练时为 preview 分支生成更平滑的量化近似，同时不改变真正发送出去的离散 latent。
    def preview_quantize_latent(self, latent: torch.Tensor, hard_quantized_latent: torch.Tensor | None = None) -> torch.Tensor:
        if hard_quantized_latent is None:
            hard_quantized_latent = self.quantize_latent(latent)
        if not self.training or not self.soft_preview_quant or self.soft_preview_blend <= 0.0:
            return self._sanitize_latent_tensor(hard_quantized_latent)
        soft_quantized_latent = self.packetizer.soft_quantize(latent)
        return self._sanitize_latent_tensor(
            torch.lerp(hard_quantized_latent, soft_quantized_latent, self.soft_preview_blend)
        )

    # 把 latent 解码回小波域，再用 IDWT 还原图像。
    def decode_latent(self, latent: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        latent = self._sanitize_latent_tensor(latent)
        decoded_bands = self._decode_wavelet_tensor(latent)
        decoded_bands = self._scale_decoded_wavelet_tensor(decoded_bands)
        band_height = max(1, (output_size[0] + 1) // 2)
        band_width = max(1, (output_size[1] + 1) // 2)
        if decoded_bands.shape[-2:] != (band_height, band_width):
            decoded_bands = F.interpolate(
                decoded_bands,
                size=(band_height, band_width),
                mode="bilinear",
                align_corners=False,
            )
        decoded_bands = self._sanitize_wavelet_tensor(decoded_bands)
        if decoded_bands.shape[1] != self.image_channels * 4:
            raise ValueError(
                "Wavelet decoder output channels must equal image_channels * 4, "
                f"got {decoded_bands.shape[1]}."
            )
        channels = self.image_channels
        ll_anchor_band = self._decode_ll_anchor_band(latent, output_size)
        if ll_anchor_band is not None:
            anchor_blend = self._resolve_ll_anchor_blend(
                device=decoded_bands.device,
                dtype=decoded_bands.dtype,
            )
            decoded_bands = torch.cat(
                [
                    torch.lerp(decoded_bands[:, 0:channels], ll_anchor_band.to(decoded_bands.device, decoded_bands.dtype), anchor_blend),
                    decoded_bands[:, channels:],
                ],
                dim=1,
            )
        if self._rgb_anchor_enabled() and self.rgb_anchor_ll_mode == "residual_base":
            decoded_bands = self._inject_rgb_anchor_as_ll_residual_base(
                decoded_bands,
                latent,
                output_size,
            )
        elif self._rgb_anchor_enabled():
            decoded_bands = self._blend_rgb_anchor_into_wavelet_tensor(
                decoded_bands,
                latent,
                output_size,
            )
        bands = WaveletBands(
            ll=decoded_bands[:, 0 * channels : 1 * channels],
            lh=decoded_bands[:, 1 * channels : 2 * channels],
            hl=decoded_bands[:, 2 * channels : 3 * channels],
            hh=decoded_bands[:, 3 * channels : 4 * channels],
            original_size=output_size,
            padded_size=(band_height * 2, band_width * 2),
        )
        decoded = self._sanitize_image_tensor(self.wavelet.idwt(bands))
        anchor_image = None
        if self._rgb_anchor_enabled():
            anchor_image = self._decode_rgb_anchor_image(latent, output_size)
        if anchor_image is not None and self.rgb_anchor_image_blend > 0.0:
            decoded = self._blend_anchor_low_frequency_into_image(decoded, anchor_image)
        return self._sanitize_image_tensor(self._refine_decoded_image(decoded, anchor_image=anchor_image))

    # 输出连续 latent、量化 latent 与预览图，供发送端和训练损失共同使用。
    def forward(self, image: torch.Tensor, preview_mode: str = "quantized") -> CompressionOutput:
        latent = self.encode_latent(image)
        quantized_latent = self.quantize_latent(latent)
        preview_mode = str(preview_mode).lower().strip()
        if preview_mode == "continuous":
            preview_latent = latent
        elif preview_mode == "hard":
            preview_latent = quantized_latent
        elif preview_mode == "quantized":
            preview_latent = self.preview_quantize_latent(latent, hard_quantized_latent=quantized_latent)
        else:
            raise ValueError(f"Unsupported compressor preview_mode: {preview_mode}")
        preview = self.decode_latent(preview_latent, image.shape[-2:])
        return CompressionOutput(
            latent=self._sanitize_latent_tensor(latent),
            quantized_latent=self._sanitize_latent_tensor(quantized_latent),
            reconstructed_image=self._sanitize_image_tensor(preview),
            latent_shape=tuple(int(dim) for dim in quantized_latent.shape[1:]),
            original_size=(int(image.shape[-2]), int(image.shape[-1])),
            latent_channel_bit_depths=self.latent_channel_bit_depths,
            bitstream_order=self.packetizer.bitstream_order,
            wavelet_shape=None,
            quantized_wavelet=None,
        )
