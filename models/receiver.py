from __future__ import annotations

from contextlib import nullcontext
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.blocks import ConvNormAct, MLP, ResidualBlock, build_spatial_norm
from models.comm import build_grouped_chip_codes, build_low_correlation_chip_codes, validate_index_tensor
from models.compressor import LatentTensorPacketizer
from models.polar_codec import PolarCRCCodec
from utils.wavelets import LeGall53Wavelet2D


@dataclass
class ReceiverOutput:
    recovered_latent: torch.Tensor
    decoded_image: torch.Tensor
    received_chips: torch.Tensor
    block_symbol_hints: torch.Tensor | None
    despread_signal: torch.Tensor
    llr_signal: torch.Tensor
    decoded_code_logits: torch.Tensor
    decoded_bit_logits: torch.Tensor
    decoded_code_bits: torch.Tensor
    decoded_bits: torch.Tensor
    hard_decoded_code_bits: torch.Tensor | None
    hard_decoded_info_bits: torch.Tensor | None
    hard_decoded_payload_bits: torch.Tensor | None
    crc_pass_mask: torch.Tensor | None
    metric_code_bits: torch.Tensor | None
    metric_payload_bits: torch.Tensor | None
    reconstruction_payload_bits: torch.Tensor | None
    restored_image: torch.Tensor
    decoded_block_indices: torch.Tensor | None
    decoded_group_indices: torch.Tensor | None
    info_indices: torch.Tensor | None
    raw_external_llr_signal: torch.Tensor | None = None
    decoder_llr_signal: torch.Tensor | None = None
    used_external_llr: bool = False
    bridge_external_llr_for_metrics_only: bool = False


# 生成固定长度的位置编码，让接收端能区分不同极化码块的顺序。
def build_sinusoidal_position_embedding(
    length: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if length <= 0:
        raise ValueError("length must be positive.")
    if dim <= 0:
        raise ValueError("dim must be positive.")
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / max(dim, 1))
    )
    embedding = torch.zeros(length, dim, device=device, dtype=dtype)
    embedding[:, 0::2] = torch.sin(position * div_term)
    if dim > 1:
        embedding[:, 1::2] = torch.cos(position * div_term)[:, : embedding[:, 1::2].shape[1]]
    return embedding.unsqueeze(0)


class GroupedNeuralCDMADespreader(nn.Module):
    # 从载密图像特征中恢复按组排列的接收码片矩阵，并为每个组单独生成对应的码片结构。
    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        code_length: int,
        chips_per_symbol: int,
        group_size: int = 4,
        chip_seed: int = 20260520,
        max_groups: int = 128,
    ) -> None:
        super().__init__()
        self.code_length = code_length
        self.chips_per_symbol = chips_per_symbol
        self.hidden_dim = hidden_dim
        self.group_size = max(1, int(group_size))
        self.chip_seed = int(chip_seed)
        self.max_groups = max(1, int(max_groups))
        code_grid_size = int(math.sqrt(code_length))
        if code_grid_size * code_grid_size != code_length:
            raise ValueError("GroupedNeuralCDMADespreader expects code_length to be a square number.")
        self.code_grid_size = code_grid_size
        fixed_codes = build_low_correlation_chip_codes(code_length, chips_per_symbol, seed=self.chip_seed)
        self.register_buffer("chip_codes", fixed_codes)
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            ResidualBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            ResidualBlock(hidden_dim),
        )
        self.group_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.group_embedding = nn.Embedding(self.max_groups, hidden_dim)
        self.group_token_norm = nn.LayerNorm(hidden_dim)
        self.group_token_mixer = nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.group_token_fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.group_scale = nn.Linear(hidden_dim, hidden_dim)
        self.group_bias = nn.Linear(hidden_dim, hidden_dim)
        self.group_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            ResidualBlock(hidden_dim),
            nn.Conv2d(hidden_dim, self.group_size, kernel_size=3, padding=1),
        )

    # 为当前参与恢复的组构造与发送端一致的局部 block 码片模板。
    def _build_group_block_chip_templates(
        self,
        resolved_group_indices: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        total_group_count: int | None = None,
    ) -> torch.Tensor:
        if total_group_count is None:
            max_group_index = int(resolved_group_indices.max().item()) if resolved_group_indices.numel() > 0 else 0
            total_blocks = max(1, (max_group_index + 1) * self.group_size)
        else:
            total_blocks = max(1, int(total_group_count) * self.group_size)
        block_codebook = build_grouped_chip_codes(
            total_blocks,
            self.chips_per_symbol,
            group_size=self.group_size,
            seed=self.chip_seed,
            device=device,
            dtype=dtype,
        )
        local_offsets = torch.arange(self.group_size, device=device, dtype=torch.long).view(1, self.group_size)
        block_indices = resolved_group_indices.view(-1, 1) * self.group_size + local_offsets
        validate_index_tensor(block_indices.flatten(), block_codebook.shape[0], "despreader_block_indices")
        block_codes = block_codebook.index_select(0, block_indices.flatten()).view(
            resolved_group_indices.numel(),
            self.group_size,
            self.chips_per_symbol,
        )
        chip_templates = F.normalize(
            self.chip_codes.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
            * block_codes.unsqueeze(2),
            p=2,
            dim=-1,
        )
        return chip_templates

    # 将当前解码阶段的组索引映射回发送端的全局组编号，避免局部组号和真实组号错位。
    def _resolve_group_indices(
        self,
        num_groups: int,
        device: torch.device,
        group_indices: torch.Tensor | None = None,
        total_group_count: int | None = None,
    ) -> torch.Tensor:
        resolved_total_group_count = int(total_group_count or num_groups)
        if resolved_total_group_count <= 0:
            raise ValueError("total_group_count must be positive.")
        if resolved_total_group_count > self.max_groups:
            raise ValueError(
                f"total_group_count={resolved_total_group_count} exceeds configured max_groups={self.max_groups}."
            )
        if group_indices is None:
            if num_groups > resolved_total_group_count:
                raise ValueError(
                    f"num_groups={num_groups} cannot exceed total_group_count={resolved_total_group_count}."
                )
            if num_groups > self.max_groups:
                raise ValueError(
                    f"num_groups={num_groups} exceeds configured max_groups={self.max_groups}."
                )
            return torch.arange(num_groups, device=device, dtype=torch.long)
        checked_indices = group_indices.to(device=device, dtype=torch.long).flatten()
        if checked_indices.numel() != num_groups:
            raise ValueError(
                "group_indices length must match num_groups. "
                f"got indices={checked_indices.numel()}, num_groups={num_groups}."
            )
        validate_index_tensor(checked_indices, resolved_total_group_count, "receiver_group_indices")
        return checked_indices

    # 构建与发送端 tile 路由一致的 group 空间先验，让每个 group 优先关注自己的承载区域。
    def _build_group_spatial_masks(
        self,
        resolved_group_indices: torch.Tensor,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        total_group_count: int | None = None,
    ) -> torch.Tensor:
        if total_group_count is None:
            total_groups = max(1, int(resolved_group_indices.max().item()) + 1) if resolved_group_indices.numel() > 0 else 1
        else:
            total_groups = max(1, int(total_group_count))
        groups_per_band = max(1, int(total_groups))
        groups_per_row = max(1, math.ceil(math.sqrt(groups_per_band)))
        groups_per_col = max(1, math.ceil(groups_per_band / groups_per_row))
        tile_height = max(1, math.ceil(height / groups_per_col))
        tile_width = max(1, math.ceil(width / groups_per_row))
        masks = torch.zeros(
            1,
            resolved_group_indices.numel(),
            1,
            height,
            width,
            device=device,
            dtype=dtype,
        )
        for local_index, group_index in enumerate(resolved_group_indices.detach().cpu().tolist()):
            slot_index = int(group_index)
            row_index = slot_index // groups_per_row
            col_index = slot_index % groups_per_row
            row_start = row_index * tile_height
            col_start = col_index * tile_width
            row_end = min(row_start + tile_height, height)
            col_end = min(col_start + tile_width, width)
            masks[:, local_index, :, row_start:row_end, col_start:col_end] = 1.0
        return masks

    # 输出每个 CDMA 组对应的码片张量，并返回组内局部 block 的符号提示。
    def forward(
        self,
        x: torch.Tensor,
        num_groups: int,
        group_indices: torch.Tensor | None = None,
        total_group_count: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_groups = max(1, int(num_groups))
        resolved_total_group_count = int(total_group_count or num_groups)
        resolved_group_indices = self._resolve_group_indices(
            num_groups,
            x.device,
            group_indices,
            total_group_count=resolved_total_group_count,
        )
        base_features = self.feature_adapter(x)
        if base_features.shape[-2:] != (self.code_grid_size, self.code_grid_size):
            base_features = F.adaptive_avg_pool2d(base_features, (self.code_grid_size, self.code_grid_size))
        global_context = self.group_context(base_features).unsqueeze(1)
        group_tokens = self.group_embedding(resolved_group_indices).to(dtype=base_features.dtype)
        group_tokens = group_tokens.unsqueeze(0).expand(base_features.shape[0], -1, -1)
        full_position_embedding = build_sinusoidal_position_embedding(
            resolved_total_group_count,
            self.hidden_dim,
            base_features.device,
            base_features.dtype,
        )
        position_embedding = full_position_embedding.index_select(1, resolved_group_indices)
        group_tokens = self.group_token_norm(group_tokens + global_context + position_embedding)
        mixed_tokens, _ = self.group_token_mixer(group_tokens)
        fused_tokens = self.group_token_fuse(mixed_tokens)
        scale = torch.tanh(self.group_scale(fused_tokens)).unsqueeze(-1).unsqueeze(-1)
        bias = self.group_bias(fused_tokens).unsqueeze(-1).unsqueeze(-1)
        group_features = base_features.unsqueeze(1) * (1.0 + scale) + bias
        group_spatial_masks = self._build_group_spatial_masks(
            resolved_group_indices=resolved_group_indices,
            height=self.code_grid_size,
            width=self.code_grid_size,
            device=base_features.device,
            dtype=base_features.dtype,
            total_group_count=resolved_total_group_count,
        )
        group_features = group_features * (0.25 + 0.75 * group_spatial_masks)
        group_features = group_features.reshape(-1, self.hidden_dim, self.code_grid_size, self.code_grid_size)
        local_symbol_maps = torch.tanh(self.group_head(group_features))
        local_symbol_maps = local_symbol_maps.view(
            base_features.shape[0],
            num_groups,
            self.group_size,
            self.code_grid_size,
            self.code_grid_size,
        )
        local_symbol_maps = local_symbol_maps.view(
            base_features.shape[0],
            num_groups,
            self.code_length,
            self.group_size,
        ).permute(0, 1, 3, 2).contiguous()
        chip_templates = self._build_group_block_chip_templates(
            resolved_group_indices=resolved_group_indices,
            device=base_features.device,
            dtype=base_features.dtype,
            total_group_count=resolved_total_group_count,
        )
        synthesized_chips = torch.einsum(
            "bgsk,gskc->bgkc",
            local_symbol_maps,
            chip_templates,
        ) / math.sqrt(float(self.group_size))
        return synthesized_chips.contiguous(), local_symbol_maps.contiguous()


class GroupAwareChipCorrelator(nn.Module):
    # 使用组索引选择对应的接收码片，再与发送端一致的扰码做相关解扩。
    def __init__(
        self,
        code_length: int,
        chips_per_symbol: int,
        chip_seed: int = 20260520,
        group_size: int = 4,
    ) -> None:
        super().__init__()
        self.chips_per_symbol = chips_per_symbol
        self.chip_seed = chip_seed
        self.group_size = max(1, int(group_size))
        fixed_codes = build_low_correlation_chip_codes(code_length, chips_per_symbol, seed=chip_seed)
        self.register_buffer("chip_codes", fixed_codes)

    # 构建与发送端一致的 block 级低相关扰码。
    def _build_block_chip_codes(
        self,
        num_blocks: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return build_grouped_chip_codes(
            num_blocks,
            self.chips_per_symbol,
            group_size=self.group_size,
            seed=self.chip_seed,
            device=device,
            dtype=dtype,
        )

    # 将每个 block 映射到自己的组码片后执行相关解扩。
    def forward(
        self,
        chips: torch.Tensor,
        block_indices: torch.Tensor | None = None,
        total_blocks: int | None = None,
        available_group_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if chips.dim() != 4:
            raise ValueError("GroupAwareChipCorrelator expects chips with shape [B, G, code_length, chips_per_symbol].")
        total_blocks = int(total_blocks or chips.shape[1] * self.group_size)
        if block_indices is None:
            block_indices = torch.arange(total_blocks, device=chips.device, dtype=torch.long)
        else:
            block_indices = block_indices.to(device=chips.device, dtype=torch.long)
        validate_index_tensor(block_indices, total_blocks, "decoded_block_indices")
        global_group_indices = torch.div(block_indices, self.group_size, rounding_mode="floor")
        if available_group_indices is None:
            group_indices = global_group_indices
        else:
            available_group_indices = available_group_indices.to(device=chips.device, dtype=torch.long)
            total_groups = max(1, math.ceil(total_blocks / self.group_size))
            validate_index_tensor(available_group_indices, total_groups, "available_group_indices")
            local_group_indices = torch.searchsorted(available_group_indices, global_group_indices)
            validate_index_tensor(local_group_indices, available_group_indices.numel(), "local_group_indices")
            recovered_global_indices = available_group_indices.index_select(0, local_group_indices)
            if not torch.equal(recovered_global_indices, global_group_indices):
                raise ValueError("available_group_indices must contain every decoded block group.")
            group_indices = local_group_indices
        validate_index_tensor(group_indices, chips.shape[1], "decoded_group_indices")
        group_chips = chips.index_select(1, group_indices)
        group_chips = group_chips - group_chips.mean(dim=-1, keepdim=True)
        group_norm = group_chips.float().norm(p=2, dim=-1, keepdim=True).clamp_min(1e-4).to(group_chips.dtype)
        group_chips = group_chips / group_norm
        block_codebook = self._build_block_chip_codes(total_blocks, chips.device, chips.dtype)
        validate_index_tensor(block_indices, block_codebook.shape[0], "block_code_indices")
        block_codes = block_codebook.index_select(0, block_indices)
        chip_template = self.chip_codes.to(device=chips.device, dtype=chips.dtype)
        block_chip_codes = F.normalize(
            chip_template.unsqueeze(0) * block_codes.unsqueeze(1),
            p=2,
            dim=-1,
        )
        return torch.nan_to_num((group_chips * block_chip_codes.unsqueeze(0)).sum(dim=-1), nan=0.0)


class LearnableLLRScaler(nn.Module):
    # 用可学习尺度把相关输出标定成更适合极化码解码的 LLR。
    def __init__(
        self,
        init_scale: float = 6.0,
        max_scale: float = 24.0,
        clamp_value: float = 20.0,
    ) -> None:
        super().__init__()
        self.max_scale = max_scale
        self.clamp_value = clamp_value
        raw_init = math.log(math.exp(max(init_scale, 1e-3)) - 1.0)
        self.raw_scale = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

    # 输出数值稳定的 LLR，避免极端值破坏训练。
    def forward(self, soft_symbols: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.raw_scale).clamp(0.1, self.max_scale)
        soft_symbols = torch.nan_to_num(soft_symbols, nan=0.0, posinf=1.0, neginf=-1.0)
        llr = soft_symbols * scale
        return llr.clamp(-self.clamp_value, self.clamp_value)


class ExternalLLRAdapter(nn.Module):
    # 对 QIM 提取出来的外部 LLR 做轻量幅度和偏置补偿，减少训练分布与最终评测分布的偏移。
    def __init__(
        self,
        code_length: int,
        clamp_value: float = 20.0,
        init_scale: float = 1.0,
        global_scale_min: float = 0.25,
        global_scale_max: float = 4.0,
        adaptive_gain_limit: float = 0.25,
        adaptive_bias_limit: float = 0.10,
        residual_mix: float = 0.20,
        delta_scale_limit: float = 0.0,
    ) -> None:
        super().__init__()
        if code_length <= 0:
            raise ValueError("code_length must be positive.")
        safe_init_scale = max(float(init_scale), 1e-3)
        self.code_length = int(code_length)
        self.clamp_value = float(clamp_value)
        self.global_scale_min = float(global_scale_min)
        self.global_scale_max = float(max(global_scale_min, global_scale_max))
        self.adaptive_gain_limit = float(max(0.0, adaptive_gain_limit))
        self.adaptive_bias_limit = float(max(0.0, adaptive_bias_limit))
        self.residual_mix = float(max(0.0, residual_mix))
        self.delta_scale_limit = float(max(0.0, delta_scale_limit))
        self.raw_global_scale = nn.Parameter(torch.tensor(math.log(math.exp(safe_init_scale) - 1.0), dtype=torch.float32))
        self.raw_delta_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.block_norm = nn.LayerNorm(self.code_length)
        self.stat_head = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 2),
        )
        self.residual_head = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=9, padding=4),
        )
        nn.init.zeros_(self.stat_head[-1].weight)
        nn.init.zeros_(self.stat_head[-1].bias)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    # 在接近恒等映射的前提下吸收外部 LLR 的局部格点噪声，并做每个码包的自适应校准。
    def forward(self, llr_signal: torch.Tensor) -> torch.Tensor:
        safe_llr = torch.nan_to_num(
            llr_signal,
            nan=0.0,
            posinf=self.clamp_value,
            neginf=-self.clamp_value,
        ).clamp(-self.clamp_value, self.clamp_value)
        block_mean = safe_llr.mean(dim=-1)
        block_abs_mean = safe_llr.abs().mean(dim=-1)
        block_std = safe_llr.float().std(dim=-1, unbiased=False).to(dtype=safe_llr.dtype)
        block_stats = torch.stack([block_mean, block_abs_mean, block_std], dim=-1)
        stat_gain_bias = self.stat_head(block_stats)
        adaptive_gain = 1.0 + self.adaptive_gain_limit * torch.tanh(stat_gain_bias[..., :1])
        adaptive_bias = self.adaptive_bias_limit * torch.tanh(stat_gain_bias[..., 1:2])
        normalized_llr = self.block_norm(safe_llr)
        residual = self.residual_head(normalized_llr.reshape(-1, 1, self.code_length)).reshape_as(safe_llr)
        base_global_scale = F.softplus(self.raw_global_scale).clamp(self.global_scale_min, self.global_scale_max)
        if self.delta_scale_limit > 0.0:
            delta_factor = 1.0 + self.delta_scale_limit * torch.tanh(self.raw_delta_scale).to(dtype=safe_llr.dtype)
        else:
            delta_factor = safe_llr.new_tensor(1.0)
        global_scale = (base_global_scale.to(dtype=safe_llr.dtype) * delta_factor).clamp(
            self.global_scale_min,
            self.global_scale_max,
        )
        adapted_llr = global_scale * adaptive_gain * safe_llr + adaptive_bias + self.residual_mix * residual
        return adapted_llr.clamp(-self.clamp_value, self.clamp_value)


class FrozenPilotLLRCalibrator(nn.Module):
    # 利用极化码冻结位作为零额外容量 pilot，校准整包 LLR 的正负号和幅度。
    def __init__(
        self,
        code_length: int,
        info_indices: torch.Tensor,
        enabled: bool = True,
        target_abs: float = 1.6,
        min_gain: float = 0.50,
        max_gain: float = 3.00,
        sample_count: int = 512,
        clamp_value: float = 20.0,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.target_abs = float(target_abs)
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)
        self.sample_count = max(1, int(sample_count))
        self.clamp_value = float(clamp_value)
        frozen_mask = torch.ones(int(code_length), dtype=torch.bool)
        frozen_mask[info_indices.detach().cpu().long()] = False
        frozen_indices = torch.nonzero(frozen_mask, as_tuple=False).flatten()
        if frozen_indices.numel() > self.sample_count:
            sample_positions = torch.linspace(0, frozen_indices.numel() - 1, steps=self.sample_count).round().long()
            frozen_indices = frozen_indices.index_select(0, sample_positions)
        self.register_buffer("frozen_indices", frozen_indices.long())

    # 对 LLR 做 per-image/per-block 的方向校准，避免整包符号翻转导致 BER 接近 0.5。
    def forward(self, llr_signal: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.frozen_indices.numel() == 0:
            return llr_signal
        safe_llr = torch.nan_to_num(llr_signal, nan=0.0, posinf=self.clamp_value, neginf=-self.clamp_value)
        frozen_llr = safe_llr.index_select(-1, self.frozen_indices.to(device=safe_llr.device))
        frozen_mean = frozen_llr.mean(dim=-1, keepdim=True)
        direction = torch.where(
            frozen_mean.detach() >= 0.0,
            torch.ones_like(frozen_mean),
            -torch.ones_like(frozen_mean),
        )
        corrected = safe_llr * direction
        corrected_frozen_abs = corrected.index_select(-1, self.frozen_indices.to(device=safe_llr.device)).abs()
        gain = (safe_llr.new_tensor(self.target_abs) / corrected_frozen_abs.mean(dim=-1, keepdim=True).clamp_min(1e-4))
        gain = gain.clamp(self.min_gain, self.max_gain).detach()
        return torch.nan_to_num(corrected * gain, nan=0.0, posinf=self.clamp_value, neginf=-self.clamp_value).clamp(
            -self.clamp_value,
            self.clamp_value,
        )


class SymbolDirectionCalibrator(nn.Module):
    # 学习接收端解扩符号的全局方向和局部幅度，降低整条链路因正负号翻转导致 BER 接近 0.5 的风险。
    def __init__(self, hidden_dim: int = 24) -> None:
        super().__init__()
        self.global_direction = nn.Parameter(torch.tensor(3.0, dtype=torch.float32))
        raw_scale = math.log(math.exp(1.0) - 1.0)
        self.global_raw_scale = nn.Parameter(torch.tensor(raw_scale, dtype=torch.float32))
        self.stats_head = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.stats_head[-1].weight)
        nn.init.zeros_(self.stats_head[-1].bias)

    # 根据每个码包的符号统计量输出校准后的软符号。
    def forward(self, symbols: torch.Tensor) -> torch.Tensor:
        safe_symbols = torch.nan_to_num(symbols, nan=0.0, posinf=4.0, neginf=-4.0)
        mean = safe_symbols.mean(dim=-1, keepdim=True)
        abs_mean = safe_symbols.abs().mean(dim=-1, keepdim=True)
        std = safe_symbols.float().std(dim=-1, keepdim=True, unbiased=False).to(safe_symbols.dtype)
        signed_balance = torch.tanh(mean)
        stats = torch.cat([mean, abs_mean, std, signed_balance], dim=-1)
        delta = self.stats_head(stats)
        raw_direction = self.global_direction.to(dtype=safe_symbols.dtype) + 0.25 * delta[..., :1]
        direction = torch.tanh(raw_direction)
        direction = torch.where(
            direction >= 0,
            direction.clamp_min(0.15),
            direction.clamp_max(-0.15),
        )
        raw_scale = self.global_raw_scale.to(dtype=safe_symbols.dtype) + 0.20 * delta[..., 1:2]
        scale = F.softplus(raw_scale).clamp(0.50, 4.0)
        bias = 0.05 * torch.tanh(delta[..., 2:3])
        calibrated = (safe_symbols + bias) * direction * scale
        return torch.nan_to_num(calibrated, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)


class LatentBitstreamRestorer(nn.Module):
    # 将完整解码出的 bitstream 反量化回 latent 张量，供共享压缩器 decoder 重建图像。
    def __init__(
        self,
        bit_depth: int = 4,
        latent_clip_value: float = 2.5,
        channel_bit_depths: tuple[int, ...] | list[int] | None = None,
        bitstream_order: str = "channel",
        base_layer_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.packetizer = LatentTensorPacketizer(
            bit_depth=bit_depth,
            clip_value=latent_clip_value,
            channel_bit_depths=channel_bit_depths,
            bitstream_order=bitstream_order,
            base_layer_channels=base_layer_channels,
        )

    # 把 bitstream 恢复成指定形状的 latent。
    def forward(
        self,
        decoded_bits: torch.Tensor,
        latent_shape: tuple[int, int, int],
        valid_info_bits: int,
    ) -> torch.Tensor:
        decoded_bit_count = decoded_bits.flatten(start_dim=1).shape[1]
        if decoded_bit_count < valid_info_bits:
            raise ValueError(
                "Latent restoration requires all valid bits to be decoded. "
                "Use force_full_decode=True for the final evaluation path."
            )
        return self.packetizer.bits_to_tensor(decoded_bits, latent_shape, valid_info_bits)


class MambaLikeBlock(nn.Module):
    # 构建轻量 Mamba 风格状态空间块，用于建模极化码块之间的顺序依赖。
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.state_proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.mlp = MLP(dim, dim * 2)

    # 对输入序列做局部状态更新和门控融合。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        local_state = self.depthwise(x.transpose(1, 2)).transpose(1, 2)
        state = torch.tanh(self.state_proj(x + local_state))
        gate = torch.sigmoid(self.gate(x))
        x = residual + gate * state
        return x + self.mlp(x)


class MambaDecoder(nn.Module):
    # 堆叠多个状态空间块，逐步增强块级别的序列表示。
    def __init__(self, dim: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MambaLikeBlock(dim) for _ in range(num_layers)])

    # 顺序通过所有状态空间层。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class BidirectionalMambaDecoder(nn.Module):
    # 同时做正向和反向扫描，把双向上下文融合进 block 表示。
    def __init__(self, dim: int, num_layers: int) -> None:
        super().__init__()
        self.forward_decoder = MambaDecoder(dim, num_layers=num_layers)
        self.backward_decoder = MambaDecoder(dim, num_layers=num_layers)
        self.forward_norm = nn.LayerNorm(dim)
        self.backward_norm = nn.LayerNorm(dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

    # 融合双向扫描结果，得到更稳定的块级上下文表示。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forward_state = self.forward_norm(self.forward_decoder(x))
        backward_state = self.backward_norm(self.backward_decoder(torch.flip(x, dims=[1])))
        backward_state = torch.flip(backward_state, dims=[1])
        merged = torch.cat([forward_state, backward_state], dim=-1)
        gate = self.gate(merged)
        return x + gate * self.fusion(merged)


class NeuralBPDecoder(nn.Module):
    # 用 logits 形式输出码字和信息位预测，让 Mamba 更像 LLR 细化器而不是纯分类器。
    def __init__(
        self,
        dim: int,
        code_length: int,
        info_length: int,
        info_indices: torch.Tensor,
        bp_iterations: int = 2,
    ) -> None:
        super().__init__()
        if info_indices.numel() != info_length:
            raise ValueError(
                f"NeuralBPDecoder expected {info_length} info indices, got {info_indices.numel()}."
            )
        self.block_code_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, code_length),
        )
        self.block_channel_gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )
        self.block_bp_updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * 2),
                    nn.GELU(),
                    nn.Linear(dim * 2, code_length),
                )
                for _ in range(bp_iterations)
            ]
        )
        self.register_buffer("info_indices", info_indices.detach().clone().long())
        info_hidden_dim = max(info_length * 2, dim * 4)
        self.info_proj = nn.Sequential(
            nn.LayerNorm(code_length * 2 + info_length),
            nn.Linear(code_length * 2 + info_length, info_hidden_dim),
            nn.GELU(),
            nn.Linear(info_hidden_dim, info_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(info_hidden_dim // 2, info_length),
        )
        self.info_structural_gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, info_length),
            nn.Sigmoid(),
        )
        self.code_feedback_gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )
        self.code_feedback_gain = 0.20
        self.logit_clamp = 12.0

    # 把 bit=1 logits 转成极化逆变换可用的符号均值，数值范围稳定在 [-1, 1]。
    def _bit1_logits_to_sign_mean(self, bit1_logits: torch.Tensor) -> torch.Tensor:
        return -torch.tanh(bit1_logits * 0.5)

    # 把 logits 拉回有限范围，避免结构反馈和低精度联合作用把极化头推到无穷大。
    def _sanitize_logits(self, logits: torch.Tensor) -> torch.Tensor:
        safe_logits = torch.nan_to_num(
            logits.float(),
            nan=0.0,
            posinf=self.logit_clamp,
            neginf=-self.logit_clamp,
        )
        return safe_logits.clamp(-self.logit_clamp, self.logit_clamp).to(dtype=logits.dtype)

    # 在符号均值域执行与发送端一致的极化变换，利用 G_N 自反性质近似恢复 u 域软值。
    def _soft_inverse_polar_transform(self, code_sign_mean: torch.Tensor) -> torch.Tensor:
        state = code_sign_mean
        stage = 1
        while stage < state.shape[-1]:
            state = state.view(*state.shape[:-1], -1, stage * 2)
            left = state[..., :stage]
            right = state[..., stage:]
            state = torch.cat([left * right, right], dim=-1)
            state = state.view(*code_sign_mean.shape[:-1], code_sign_mean.shape[-1])
            stage *= 2
        return state

    # 把恢复出的符号均值重新映射回 bit=1 logits，供 BCE/BER 损失直接使用。
    def _sign_mean_to_bit1_logits(self, sign_mean: torch.Tensor) -> torch.Tensor:
        safe_mean = torch.nan_to_num(sign_mean.float(), nan=0.0, posinf=0.999, neginf=-0.999)
        safe_mean = safe_mean.clamp(-0.999, 0.999)
        safe_logits = -2.0 * torch.atanh(safe_mean)
        safe_logits = torch.nan_to_num(
            safe_logits,
            nan=0.0,
            posinf=self.logit_clamp,
            neginf=-self.logit_clamp,
        )
        return safe_logits.clamp(-self.logit_clamp, self.logit_clamp).to(dtype=sign_mean.dtype)

    # 把软信息位重新映射回完整极化码字的结构先验，用来反向校正码字分支。
    def _build_code_logits_from_info_logits(self, info_logits: torch.Tensor) -> torch.Tensor:
        info_sign_mean = self._bit1_logits_to_sign_mean(info_logits)
        code_length = self.block_code_head[-1].out_features
        full_u_sign_mean = torch.ones(
            *info_logits.shape[:-1],
            code_length,
            device=info_logits.device,
            dtype=info_logits.dtype,
        )
        scatter_indices = self.info_indices.to(device=info_logits.device, dtype=torch.long)
        expand_shape = [1] * info_logits.dim()
        expand_shape[-1] = scatter_indices.numel()
        expanded_indices = scatter_indices.view(*expand_shape).expand(*info_logits.shape[:-1], -1)
        full_u_sign_mean.scatter_(-1, expanded_indices, info_sign_mean)
        code_sign_mean = self._soft_inverse_polar_transform(full_u_sign_mean)
        return self._sign_mean_to_bit1_logits(code_sign_mean)

    # 输出 block 级码字 logits 和信息位 logits。
    def forward(self, x: torch.Tensor, channel_llr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        channel_llr = torch.nan_to_num(channel_llr, nan=0.0, posinf=20.0, neginf=-20.0)
        channel_energy = channel_llr.float().abs().mean(dim=-1, keepdim=True).clamp_min(1e-4).to(channel_llr.dtype)
        channel_llr = channel_llr / channel_energy
        prior_logits = self._sanitize_logits(self.block_code_head(x))
        gate = self.block_channel_gate(x)
        # 输出的是 bit=1 的 logits；标准 LLR 正值代表 bit=0，所以信道项需要取反。
        code_logits = self._sanitize_logits(prior_logits - gate * channel_llr)
        for update in self.block_bp_updates:
            code_logits = self._sanitize_logits(code_logits + 0.15 * torch.tanh(update(x)))
        code_sign_mean = self._bit1_logits_to_sign_mean(code_logits)
        info_sign_mean = self._soft_inverse_polar_transform(code_sign_mean).index_select(-1, self.info_indices)
        structural_info_logits = self._sanitize_logits(self._sign_mean_to_bit1_logits(info_sign_mean))
        learned_info_logits = self._sanitize_logits(
            self.info_proj(torch.cat([code_logits, channel_llr, structural_info_logits], dim=-1))
        )
        structural_gate = self.info_structural_gate(x)
        info_logits = self._sanitize_logits(learned_info_logits + 0.10 * structural_gate * structural_info_logits)
        reencoded_code_logits = self._sanitize_logits(self._build_code_logits_from_info_logits(info_logits))
        feedback_gate = self.code_feedback_gate(x)
        code_logits = self._sanitize_logits(
            code_logits + self.code_feedback_gain * feedback_gate * (reencoded_code_logits - code_logits)
        )
        refined_code_sign_mean = self._bit1_logits_to_sign_mean(code_logits)
        refined_info_sign_mean = self._soft_inverse_polar_transform(refined_code_sign_mean).index_select(
            -1,
            self.info_indices,
        )
        refined_structural_info_logits = self._sanitize_logits(
            self._sign_mean_to_bit1_logits(refined_info_sign_mean)
        )
        info_logits = self._sanitize_logits(
            info_logits + 0.10 * structural_gate * (refined_structural_info_logits - structural_info_logits)
        )
        return self._sanitize_logits(code_logits), self._sanitize_logits(info_logits)


class IntermediateFeatureRestorer(nn.Module):
    # 从载密图像中恢复接收端中间特征，为解扩和 latent 细化提供共享底座。
    def __init__(self, image_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct(image_channels, hidden_dim, stride=2),
            ResidualBlock(hidden_dim),
            ConvNormAct(hidden_dim, hidden_dim * 2, stride=2),
            ResidualBlock(hidden_dim * 2),
        )

    # 提取载密图像的中间特征图。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentRefinementHead(nn.Module):
    # 在无法完整解码全部 bitstream 时，直接从接收特征回归近似 latent。
    def __init__(self, hidden_dim: int, latent_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct(hidden_dim * 2, hidden_dim * 2),
            ResidualBlock(hidden_dim * 2),
            nn.Conv2d(hidden_dim * 2, latent_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    # 输出指定 latent 尺寸的近似重建结果。
    def forward(self, x: torch.Tensor, latent_spatial_size: tuple[int, int]) -> torch.Tensor:
        x = self.net(x)
        if x.shape[-2:] != latent_spatial_size:
            x = F.interpolate(x, size=latent_spatial_size, mode="bilinear", align_corners=False)
        return x


class ImageRefinementHead(nn.Module):
    # 在图像域联合利用接收特征、粗恢复图和载密图做残差式精修，避免直接把载密图像素拷回恢复图。
    def __init__(
        self,
        hidden_dim: int,
        image_channels: int,
        max_refine_side: int = 256,
        use_gradient_checkpointing: bool = True,
        eval_full_resolution: bool = True,
        min_decoded_blend: float = 0.0,
        analog_demod_lowpass_kernel: int = 1,
        analog_demod_residual_mix: float = 0.05,
    ) -> None:
        super().__init__()
        reduced_dim = max(hidden_dim // 2, 32)
        image_dim = max(reduced_dim // 2, 32)
        fusion_dim = max(hidden_dim, 64)
        self.feature_tower = nn.Sequential(
            ConvNormAct(hidden_dim * 2, hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Conv2d(hidden_dim, reduced_dim, kernel_size=3, padding=1, bias=False),
            build_spatial_norm(reduced_dim),
            nn.SiLU(),
            ResidualBlock(reduced_dim),
        )
        self.decoded_tower = nn.Sequential(
            ConvNormAct(image_channels, image_dim),
            ResidualBlock(image_dim),
        )
        self.cover_hint_tower = nn.Sequential(
            ConvNormAct(image_channels, image_dim),
            ResidualBlock(image_dim),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(reduced_dim + image_dim * 2 + image_channels, fusion_dim, kernel_size=3, padding=1, bias=False),
            build_spatial_norm(fusion_dim),
            nn.SiLU(),
            ResidualBlock(fusion_dim),
            ResidualBlock(fusion_dim),
            ResidualBlock(fusion_dim),
        )
        self.blend_head = nn.Conv2d(fusion_dim, image_channels, kernel_size=3, padding=1)
        self.residual_head = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, kernel_size=3, padding=1, bias=False),
            build_spatial_norm(fusion_dim),
            nn.SiLU(),
            ResidualBlock(fusion_dim),
            nn.Conv2d(fusion_dim, image_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        reveal_dim = max(reduced_dim, 64)
        self.wavelet_reveal_body = nn.Sequential(
            ConvNormAct(image_channels * 16, reveal_dim),
            ResidualBlock(reveal_dim),
            ResidualBlock(reveal_dim),
            nn.Conv2d(reveal_dim, reveal_dim, kernel_size=3, padding=1, bias=False),
            build_spatial_norm(reveal_dim),
            nn.SiLU(),
        )
        self.wavelet_reveal_delta = nn.Conv2d(reveal_dim, image_channels * 4, kernel_size=3, padding=1)
        self.wavelet_reveal_gate = nn.Conv2d(reveal_dim, image_channels * 4, kernel_size=3, padding=1)
        self.analog_gain_head = nn.Conv2d(fusion_dim, 2, kernel_size=1)
        self.max_refine_side = max(0, int(max_refine_side))
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.eval_full_resolution = bool(eval_full_resolution)
        self.min_decoded_blend = float(max(0.0, min(1.0, min_decoded_blend)))
        self.analog_demod_lowpass_kernel = max(1, int(analog_demod_lowpass_kernel))
        self.analog_demod_residual_mix = float(max(0.0, min(1.0, analog_demod_residual_mix)))
        if self.analog_demod_lowpass_kernel % 2 == 0:
            self.analog_demod_lowpass_kernel += 1
        self.wavelet = LeGall53Wavelet2D()
        nn.init.zeros_(self.blend_head.weight)
        nn.init.constant_(self.blend_head.bias, 0.0)
        nn.init.zeros_(self.residual_head[-2].weight)
        nn.init.zeros_(self.residual_head[-2].bias)
        nn.init.zeros_(self.wavelet_reveal_delta.weight)
        nn.init.zeros_(self.wavelet_reveal_delta.bias)
        nn.init.zeros_(self.wavelet_reveal_gate.weight)
        nn.init.constant_(self.wavelet_reveal_gate.bias, -1.5)
        nn.init.zeros_(self.analog_gain_head.weight)
        nn.init.zeros_(self.analog_gain_head.bias)

    # 计算图像精修阶段真正工作的分辨率，默认把大图卷积限制在较小边长以内。
    def _resolve_working_size(
        self,
        feature_size: tuple[int, int],
        output_size: tuple[int, int],
    ) -> tuple[int, int]:
        if not self.training and self.eval_full_resolution:
            return output_size
        if self.max_refine_side <= 0:
            return output_size
        output_height, output_width = output_size
        max_output_side = max(output_height, output_width)
        if max_output_side <= self.max_refine_side:
            return output_size
        scale = self.max_refine_side / float(max_output_side)
        working_height = max(feature_size[0], int(round(output_height * scale)))
        working_width = max(feature_size[1], int(round(output_width * scale)))
        return min(output_height, working_height), min(output_width, working_width)

    # 在需要反向传播且张量足够大的时候，对重卷积子图启用 checkpoint 以降低峰值显存。
    def _maybe_checkpoint(self, module: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
        if not self.training or not self.use_gradient_checkpointing or not tensor.requires_grad:
            return module(tensor)
        return checkpoint(module, tensor, use_reentrant=False)

    # 生成与发送端一致的高频载波，用于从 LH/HL/HH 差分中解调低频残差信号。
    def _build_analog_carriers(
        self,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = reference.shape[-2:]
        yy = torch.arange(height, device=reference.device, dtype=torch.long).view(1, 1, height, 1)
        xx = torch.arange(width, device=reference.device, dtype=torch.long).view(1, 1, 1, width)
        carrier_lh = torch.where((xx % 2) == 0, 1.0, -1.0)
        carrier_hl = torch.where((yy % 2) == 0, 1.0, -1.0)
        carrier_hh = torch.where(((xx + yy) % 2) == 0, 1.0, -1.0)
        return (
            carrier_lh.to(device=reference.device, dtype=reference.dtype),
            carrier_hl.to(device=reference.device, dtype=reference.dtype),
            carrier_hh.to(device=reference.device, dtype=reference.dtype),
        )

    # 对乘回 carrier 后的模拟残差信号做匹配低通，抑制未调制纹理和 QIM 稀疏尖刺。
    def _matched_lowpass(self, tensor: torch.Tensor) -> torch.Tensor:
        kernel = int(self.analog_demod_lowpass_kernel)
        if kernel <= 1:
            return tensor
        pad = kernel // 2
        padded = F.pad(tensor, (pad, pad, pad, pad), mode="reflect")
        return F.avg_pool2d(padded, kernel_size=kernel, stride=1)

    # 从载密图与粗恢复图的高频子带差分中解析式恢复低频/颜色残差补偿。
    def _demodulate_analog_residuals(
        self,
        decoded_image: torch.Tensor,
        cover_image: torch.Tensor,
        carrier_reference_image: torch.Tensor | None,
        output_size: tuple[int, int],
        inverse_companding: bool = False,
        lowfreq_sender_gain: float = 1.0,
        lowfreq_sender_clip: float = 0.20,
        detail_sender_gain: float = 1.0,
        detail_sender_clip: float = 0.10,
        detail_hh_ratio: float = 1.0,
        lowfreq_hh_ratio: float = 1.0,
        ll_direct_sender_gain: float = 0.0,
        ll_direct_decode_mix: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference_image = decoded_image if carrier_reference_image is None else carrier_reference_image
        decoded_bands = self.wavelet.dwt(reference_image.clamp(0.0, 1.0))
        cover_bands = self.wavelet.dwt(cover_image.clamp(0.0, 1.0))
        carrier_lh, carrier_hl, carrier_hh = self._build_analog_carriers(cover_bands.lh)
        raw_lh = cover_bands.lh - decoded_bands.lh
        raw_hl = cover_bands.hl - decoded_bands.hl
        raw_hh = cover_bands.hh - decoded_bands.hh
        residual_lh = raw_lh * carrier_lh / 0.60
        residual_hl = raw_hl * carrier_hl / 0.60
        hh_lowfreq_ratio = float(max(0.0, min(1.0, lowfreq_hh_ratio)))
        residual_hh = raw_hh * carrier_hh / max(1e-4, 0.35 * hh_lowfreq_ratio)
        matched_lh = self._matched_lowpass(residual_lh)
        matched_hl = self._matched_lowpass(residual_hl)
        if hh_lowfreq_ratio > 1e-6:
            matched_hh = self._matched_lowpass(residual_hh)
            lowfreq_payload_estimate = (matched_lh + matched_hl + 0.50 * matched_hh) / 2.50
        else:
            lowfreq_payload_estimate = (matched_lh + matched_hl) / 2.00
        carrier_lowfreq_payload = torch.nan_to_num(
            lowfreq_payload_estimate,
            nan=0.0,
            posinf=0.25,
            neginf=-0.25,
        )
        direct_mix = float(max(0.0, min(1.0, ll_direct_decode_mix)))
        direct_gain = float(max(0.0, ll_direct_sender_gain))
        if direct_mix > 0.0 and direct_gain > 1e-6:
            direct_lowfreq_payload = torch.nan_to_num(
                (cover_bands.ll - decoded_bands.ll) / direct_gain,
                nan=0.0,
                posinf=0.25,
                neginf=-0.25,
            )
            lowfreq_payload = (1.0 - direct_mix) * carrier_lowfreq_payload + direct_mix * direct_lowfreq_payload
        else:
            lowfreq_payload = carrier_lowfreq_payload
        lowfreq_payload_limit = max(1e-4, float(lowfreq_sender_gain) * float(lowfreq_sender_clip))
        lowfreq_payload = lowfreq_payload.clamp(-0.985 * lowfreq_payload_limit, 0.985 * lowfreq_payload_limit)
        detail_lh = raw_lh - 0.60 * lowfreq_payload * carrier_lh
        detail_hl = raw_hl - 0.60 * lowfreq_payload * carrier_hl
        detail_hh = raw_hh - 0.35 * hh_lowfreq_ratio * lowfreq_payload * carrier_hh
        hh_detail_ratio = float(max(0.0, min(1.0, detail_hh_ratio)))
        if hh_detail_ratio > 1e-6:
            detail_hh = detail_hh / hh_detail_ratio
        else:
            detail_hh = torch.zeros_like(detail_hh)

        def inverse_tanh_payload(payload: torch.Tensor, sender_gain: float, sender_clip: float) -> torch.Tensor:
            gain = float(max(1e-4, sender_gain))
            clip = float(max(1e-4, sender_clip))
            normalized = (payload / (gain * clip)).clamp(-0.985, 0.985)
            return 0.5 * clip * (torch.log1p(normalized) - torch.log1p(-normalized))

        if inverse_companding:
            lowfreq_residual = inverse_tanh_payload(
                lowfreq_payload,
                sender_gain=lowfreq_sender_gain,
                sender_clip=lowfreq_sender_clip,
            )
            lowfreq_residual = torch.tanh(lowfreq_residual / 0.32) * 0.32
            detail_lh = inverse_tanh_payload(detail_lh, sender_gain=detail_sender_gain, sender_clip=detail_sender_clip)
            detail_hl = inverse_tanh_payload(detail_hl, sender_gain=detail_sender_gain, sender_clip=detail_sender_clip)
            detail_hh = inverse_tanh_payload(detail_hh, sender_gain=detail_sender_gain, sender_clip=detail_sender_clip)
        else:
            lowfreq_residual = torch.tanh(lowfreq_payload / max(0.20, float(lowfreq_sender_clip))) * max(
                0.20,
                float(lowfreq_sender_clip),
            )

        def stabilize_detail(detail: torch.Tensor) -> torch.Tensor:
            if detail.shape[1] >= 3:
                neutral = detail.mean(dim=1, keepdim=True)
                detail = neutral + 0.45 * (detail - neutral)
            if inverse_companding:
                detail_limit = 0.18
            else:
                detail_limit = 0.10
            return torch.tanh(detail / detail_limit) * detail_limit

        detail_lh = stabilize_detail(detail_lh)
        detail_hl = stabilize_detail(detail_hl)
        detail_hh = stabilize_detail(detail_hh)
        lowfreq_image = self.wavelet.idwt(
            type(decoded_bands)(
                ll=lowfreq_residual,
                lh=torch.zeros_like(decoded_bands.lh),
                hl=torch.zeros_like(decoded_bands.hl),
                hh=torch.zeros_like(decoded_bands.hh),
                original_size=output_size,
                padded_size=decoded_bands.padded_size,
            )
        )
        detail_image = self.wavelet.idwt(
            type(decoded_bands)(
                ll=torch.zeros_like(decoded_bands.ll),
                lh=detail_lh,
                hl=detail_hl,
                hh=detail_hh,
                original_size=output_size,
                padded_size=decoded_bands.padded_size,
            )
        )
        residual_mix = self.analog_demod_residual_mix
        return (
            (residual_mix * lowfreq_image).to(device=decoded_image.device, dtype=decoded_image.dtype),
            (residual_mix * detail_image).to(device=decoded_image.device, dtype=decoded_image.dtype),
        )

    def _flatten_wavelet_bands(self, bands) -> torch.Tensor:
        return torch.cat([bands.ll, bands.lh, bands.hl, bands.hh], dim=1)

    # 在小波域学习载密图相对粗恢复图的可逆残差信息，补足固定解调公式难以覆盖的细节。
    def _predict_wavelet_reveal_residual(
        self,
        decoded_image: torch.Tensor,
        cover_image: torch.Tensor,
        analog_lowfreq_residual: torch.Tensor,
        analog_detail_residual: torch.Tensor,
        output_size: tuple[int, int],
        reveal_strength: float,
    ) -> torch.Tensor:
        if reveal_strength <= 0.0:
            return torch.zeros_like(decoded_image)
        decoded_bands = self.wavelet.dwt(decoded_image.clamp(0.0, 1.0))
        cover_bands = self.wavelet.dwt(cover_image.clamp(0.0, 1.0))
        lowfreq_bands = self.wavelet.dwt(torch.nan_to_num(analog_lowfreq_residual).clamp(-1.0, 1.0))
        detail_bands = self.wavelet.dwt(torch.nan_to_num(analog_detail_residual).clamp(-1.0, 1.0))
        decoded_flat = self._flatten_wavelet_bands(decoded_bands)
        cover_delta = self._flatten_wavelet_bands(cover_bands) - decoded_flat
        lowfreq_flat = self._flatten_wavelet_bands(lowfreq_bands)
        detail_flat = self._flatten_wavelet_bands(detail_bands)
        reveal_input = torch.cat([decoded_flat, cover_delta, lowfreq_flat, detail_flat], dim=1)
        reveal_input = torch.nan_to_num(reveal_input, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        reveal_features = self._maybe_checkpoint(self.wavelet_reveal_body, reveal_input)
        raw_delta = torch.tanh(self.wavelet_reveal_delta(reveal_features))
        reveal_gate = torch.sigmoid(self.wavelet_reveal_gate(reveal_features))
        ll_delta, lh_delta, hl_delta, hh_delta = (raw_delta * reveal_gate).chunk(4, dim=1)
        base_strength = float(max(0.0, min(0.35, reveal_strength)))
        ll_scale = max(0.02, min(0.28, base_strength * 1.80))
        hf_scale = max(0.01, min(0.16, base_strength * 1.15))
        residual_bands = type(decoded_bands)(
            ll=ll_scale * ll_delta,
            lh=hf_scale * lh_delta,
            hl=hf_scale * hl_delta,
            hh=hf_scale * hh_delta,
            original_size=output_size,
            padded_size=decoded_bands.padded_size,
        )
        return self.wavelet.idwt(residual_bands).to(device=decoded_image.device, dtype=decoded_image.dtype)

    # 输出精修后的恢复图，以 decoded image 为主，只让 cover image 参与残差引导。
    def forward(
        self,
        restored_features: torch.Tensor,
        decoded_image: torch.Tensor,
        cover_image: torch.Tensor,
        carrier_reference_image: torch.Tensor | None,
        output_size: tuple[int, int],
        residual_scale: float,
        lowfreq_decode_gain: float = 0.0,
        detail_decode_gain: float = 0.0,
        inverse_companding: bool = False,
        lowfreq_sender_gain: float = 1.0,
        lowfreq_sender_clip: float = 0.20,
        detail_sender_gain: float = 1.0,
        detail_sender_clip: float = 0.10,
        detail_hh_ratio: float = 1.0,
        lowfreq_hh_ratio: float = 1.0,
        ll_direct_sender_gain: float = 0.0,
        ll_direct_decode_mix: float = 0.0,
        disable_cover_guidance: bool = False,
        cover_guidance_strength: float = 1.0,
    ) -> torch.Tensor:
        lowfreq_gain = float(max(0.0, min(8.0, lowfreq_decode_gain)))
        detail_gain = float(max(0.0, min(10.0, detail_decode_gain)))
        if residual_scale <= 0.0 and lowfreq_gain <= 0.0 and detail_gain <= 0.0:
            return decoded_image.clamp(0.0, 1.0)
        guidance_strength = 0.0 if disable_cover_guidance else float(max(0.0, min(1.0, cover_guidance_strength)))
        cover_guidance_disabled = guidance_strength <= 1e-6
        features = self._maybe_checkpoint(self.feature_tower, restored_features)
        working_size = self._resolve_working_size(
            feature_size=(int(features.shape[-2]), int(features.shape[-1])),
            output_size=output_size,
        )
        if features.shape[-2:] != working_size:
            features = F.interpolate(features, size=working_size, mode="bilinear", align_corners=False)
        analog_lowfreq_residual, analog_detail_residual = self._demodulate_analog_residuals(
            decoded_image=decoded_image,
            cover_image=cover_image,
            carrier_reference_image=carrier_reference_image,
            output_size=output_size,
            inverse_companding=inverse_companding,
            lowfreq_sender_gain=lowfreq_sender_gain,
            lowfreq_sender_clip=lowfreq_sender_clip,
            detail_sender_gain=detail_sender_gain,
            detail_sender_clip=detail_sender_clip,
            detail_hh_ratio=detail_hh_ratio,
            lowfreq_hh_ratio=lowfreq_hh_ratio,
            ll_direct_sender_gain=ll_direct_sender_gain,
            ll_direct_decode_mix=ll_direct_decode_mix,
        )
        decoded_for_fusion = decoded_image
        cover_for_fusion = cover_image
        carrier_reference_for_fusion = decoded_image if carrier_reference_image is None else carrier_reference_image
        lowfreq_for_fusion = analog_lowfreq_residual
        detail_for_fusion = analog_detail_residual
        if working_size != output_size:
            decoded_for_fusion = F.interpolate(decoded_image, size=working_size, mode="bilinear", align_corners=False)
            cover_for_fusion = F.interpolate(cover_image, size=working_size, mode="bilinear", align_corners=False)
            carrier_reference_for_fusion = F.interpolate(
                carrier_reference_for_fusion,
                size=working_size,
                mode="bilinear",
                align_corners=False,
            )
            lowfreq_for_fusion = F.interpolate(
                analog_lowfreq_residual,
                size=working_size,
                mode="bilinear",
                align_corners=False,
            )
            detail_for_fusion = F.interpolate(
                analog_detail_residual,
                size=working_size,
                mode="bilinear",
                align_corners=False,
            )
        if cover_guidance_disabled:
            cover_for_fusion = decoded_for_fusion
            carrier_reference_for_fusion = decoded_for_fusion
            lowfreq_for_fusion = torch.zeros_like(lowfreq_for_fusion)
            detail_for_fusion = torch.zeros_like(detail_for_fusion)
        elif guidance_strength < 1.0:
            cover_for_fusion = carrier_reference_for_fusion + guidance_strength * (
                cover_for_fusion - carrier_reference_for_fusion
            )
            lowfreq_for_fusion = guidance_strength * lowfreq_for_fusion
            detail_for_fusion = guidance_strength * detail_for_fusion
        cover_residual_hint = torch.nan_to_num(
            cover_for_fusion - carrier_reference_for_fusion,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        cover_low = F.avg_pool2d(cover_for_fusion, kernel_size=5, stride=1, padding=2)
        decoded_low = F.avg_pool2d(carrier_reference_for_fusion, kernel_size=5, stride=1, padding=2)
        detail_hint = (cover_for_fusion - cover_low) - (carrier_reference_for_fusion - decoded_low)
        cover_residual_hint = torch.nan_to_num(
            0.35 * cover_residual_hint
            + 0.35 * detail_hint
            + 0.45 * lowfreq_for_fusion
            + 0.35 * detail_for_fusion,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        cover_hint_scale = 0.06 + 0.14 * max(0.0, 1.0 - self.min_decoded_blend)
        cover_hint_scale *= guidance_strength
        cover_residual_hint = cover_hint_scale * torch.tanh(cover_residual_hint / 0.16)
        decoded_features = self._maybe_checkpoint(self.decoded_tower, decoded_for_fusion)
        if cover_guidance_disabled:
            cover_hint_features = torch.zeros_like(decoded_features)
        else:
            cover_hint_features = self._maybe_checkpoint(self.cover_hint_tower, cover_residual_hint)
        fusion_input = torch.cat([features, decoded_features, cover_hint_features, cover_residual_hint], dim=1)
        fused = self._maybe_checkpoint(self.fusion, fusion_input)
        decoded_gate = torch.sigmoid(self.blend_head(fused))
        if self.min_decoded_blend > 0.0:
            decoded_gate = self.min_decoded_blend + (1.0 - self.min_decoded_blend) * decoded_gate
        residual_strength = float(max(0.0, residual_scale))
        residual = residual_strength * torch.tanh(self.residual_head(fused))
        analog_gain = 0.75 + 0.50 * torch.sigmoid(self.analog_gain_head(F.adaptive_avg_pool2d(fused, 1)))
        lowfreq_gain_adjust, detail_gain_adjust = analog_gain.chunk(2, dim=1)
        if working_size != output_size:
            decoded_gate = F.interpolate(decoded_gate, size=output_size, mode="bilinear", align_corners=False)
            residual = F.interpolate(residual, size=output_size, mode="bilinear", align_corners=False)
        residual_gate_floor = min(0.18, max(0.02, 0.40 * (1.0 - self.min_decoded_blend)))
        residual_gate = torch.clamp(1.0 - decoded_gate, min=residual_gate_floor, max=0.35)
        # 解析式解调出的低频和细节残差已经与发送端的模拟嵌入尺度对齐。
        # 这里不能再次强行 tanh 裁剪或乘固定小权重，否则会把恢复图从高保真残差补偿退化成模糊预览图。
        lowfreq_weight = min(2.4, max(0.0, lowfreq_gain))
        detail_weight = min(2.4, max(0.0, detail_gain))
        if cover_guidance_disabled:
            analog_recovery = torch.zeros_like(decoded_image)
            wavelet_recovery = torch.zeros_like(decoded_image)
        else:
            analog_recovery = guidance_strength * (
                lowfreq_weight * analog_lowfreq_residual + detail_weight * analog_detail_residual
            )
            wavelet_recovery = guidance_strength * self._predict_wavelet_reveal_residual(
                decoded_image=decoded_image,
                cover_image=cover_image,
                analog_lowfreq_residual=analog_lowfreq_residual,
                analog_detail_residual=analog_detail_residual,
                output_size=output_size,
                reveal_strength=residual_strength,
            )
        residual_limit = 0.08 + 0.12 * max(0.0, 1.0 - self.min_decoded_blend)
        wavelet_recovery = torch.tanh(wavelet_recovery / 0.08) * 0.08
        residual = torch.tanh(residual / residual_limit) * residual_limit
        refined = decoded_image + analog_recovery + 0.25 * wavelet_recovery + residual_gate * residual
        return refined.clamp(0.0, 1.0)


class Receiver(nn.Module):
    # 构建接收端主模块，完成分组解扩、Mamba LLR 细化、bit 恢复与 latent 重建。
    def __init__(
        self,
        image_channels: int,
        hidden_dim: int,
        code_length: int,
        chips_per_symbol: int,
        mamba_dim: int,
        num_layers: int,
        info_length: int,
        design_snr_db: float,
        crc_length: int,
        latent_channels: int,
        bit_depth: int = 4,
        channel_bit_depths: tuple[int, ...] | list[int] | None = None,
        bitstream_order: str = "channel",
        base_layer_channels: int | None = None,
        decode_block_chunk_size: int = 8,
        llr_init_scale: float = 6.0,
        llr_max_scale: float = 24.0,
        llr_clamp: float = 20.0,
        bp_iterations: int = 2,
        chip_seed: int = 20260520,
        confidence_gate_threshold: float = 0.35,
        confidence_gate_floor: float = 0.0,
        semantic_fusion_temperature: float = 0.75,
        hard_decode_prior_weight: float = 0.35,
        hard_decode_retry_weight: float = 0.0,
        hard_reconstruction_ratio: float = 0.0,
        symbol_hint_mix: float = 0.30,
        full_decode_feature_mix: float = 0.10,
        full_decode_feature_training_threshold: float = 0.20,
        full_decode_feature_min_mix_ratio: float = 0.02,
        allow_feature_correction_on_external_hard_decode: bool = False,
        image_refine_strength: float = 0.08,
        image_refine_min_decoded_blend: float = 0.0,
        analog_lowfreq_decode_gain: float = 0.0,
        analog_detail_decode_gain: float | None = None,
        analog_inverse_companding: bool = False,
        analog_lowfreq_sender_gain: float = 1.0,
        analog_lowfreq_sender_clip: float = 0.20,
        analog_detail_sender_gain: float = 1.0,
        analog_detail_sender_clip: float = 0.10,
        analog_detail_hh_ratio: float = 1.0,
        analog_lowfreq_hh_ratio: float = 1.0,
        analog_ll_direct_sender_gain: float = 0.0,
        analog_ll_direct_decode_mix: float = 0.0,
        analog_demod_lowpass_kernel: int = 1,
        image_refine_max_side: int = 256,
        image_refine_checkpointing: bool = True,
        image_refine_eval_full_resolution: bool = True,
        analog_demod_residual_mix: float = 0.05,
        bypass_image_refiner_on_external_hard_decode: bool = False,
        bypass_image_refiner_on_full_hard_decode: bool = False,
        disable_cover_guidance_in_refiner: bool = False,
        cover_guidance_strength_in_refiner: float = 1.0,
        external_llr_cover_guidance_strength: float | None = None,
        train_decode_blocks: int = 32,
        eval_decode_blocks: int = 128,
        group_size: int = 4,
        latent_clip_value: float = 2.5,
        max_group_count: int = 128,
        pilot_calibration_enabled: bool = True,
        pilot_target_abs: float = 1.6,
        strict_external_llr_hard_decode: bool = False,
        bridge_external_llr_for_metrics_only: bool = False,
        external_hard_decode_crc_concealment: bool = False,
        external_llr_adapter_gain_limit: float = 0.25,
        external_llr_adapter_bias_limit: float = 0.10,
        external_llr_adapter_residual_mix: float = 0.20,
        external_llr_adapter_delta_scale_limit: float = 0.0,
    ) -> None:
        super().__init__()
        self.code_length = code_length
        self.chips_per_symbol = chips_per_symbol
        self.group_size = max(1, int(group_size))
        self.wavelet = LeGall53Wavelet2D()
        self.restorer = IntermediateFeatureRestorer(image_channels * 4, hidden_dim)
        self.clean_hf_predictor = nn.Sequential(
            ConvNormAct(image_channels * 4, hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Conv2d(hidden_dim, image_channels * 3, kernel_size=3, padding=1),
        )
        self.residual_restorer = IntermediateFeatureRestorer(image_channels * 4, hidden_dim)
        self.residual_feature_gain = 0.80
        self.despreader = GroupedNeuralCDMADespreader(
            input_channels=hidden_dim * 2,
            hidden_dim=hidden_dim,
            code_length=code_length,
            chips_per_symbol=chips_per_symbol,
            group_size=self.group_size,
            chip_seed=chip_seed,
            max_groups=max_group_count,
        )
        self.correlator = GroupAwareChipCorrelator(
            code_length,
            chips_per_symbol,
            chip_seed=chip_seed,
            group_size=self.group_size,
        )
        self.llr_scaler = LearnableLLRScaler(
            init_scale=llr_init_scale,
            max_scale=llr_max_scale,
            clamp_value=llr_clamp,
        )
        self.external_llr_adapter = ExternalLLRAdapter(
            code_length=code_length,
            clamp_value=llr_clamp,
            init_scale=1.0,
            adaptive_gain_limit=external_llr_adapter_gain_limit,
            adaptive_bias_limit=external_llr_adapter_bias_limit,
            residual_mix=external_llr_adapter_residual_mix,
            delta_scale_limit=external_llr_adapter_delta_scale_limit,
        )
        self.external_symbol_calibrator = SymbolDirectionCalibrator()
        self.robust_external_llr_adapter = ExternalLLRAdapter(
            code_length=code_length,
            clamp_value=llr_clamp,
            init_scale=1.0,
            adaptive_gain_limit=external_llr_adapter_gain_limit,
            adaptive_bias_limit=external_llr_adapter_bias_limit,
            residual_mix=external_llr_adapter_residual_mix,
            delta_scale_limit=external_llr_adapter_delta_scale_limit,
        )
        self.robust_external_llr_adapter.load_state_dict(self.external_llr_adapter.state_dict())
        self.robust_external_symbol_calibrator = SymbolDirectionCalibrator()
        self.robust_external_symbol_calibrator.load_state_dict(self.external_symbol_calibrator.state_dict())
        self.symbol_calibrator = SymbolDirectionCalibrator()
        self.mamba_dim = mamba_dim
        self.block_token_norm = nn.LayerNorm(code_length)
        self.block_token_proj = nn.Sequential(
            nn.Linear(code_length, mamba_dim),
            nn.GELU(),
            nn.Linear(mamba_dim, mamba_dim),
        )
        self.mamba = BidirectionalMambaDecoder(mamba_dim, num_layers=num_layers)
        self.hard_polar_decoder = PolarCRCCodec(
            code_length=code_length,
            info_length=info_length,
            design_snr_db=design_snr_db,
            crc_length=crc_length,
        )
        self.frozen_pilot_calibrator = FrozenPilotLLRCalibrator(
            code_length=code_length,
            info_indices=self.hard_polar_decoder.info_indices,
            enabled=pilot_calibration_enabled,
            target_abs=pilot_target_abs,
            clamp_value=llr_clamp,
        )
        self.bp_decoder = NeuralBPDecoder(
            mamba_dim,
            code_length=code_length,
            info_length=info_length,
            info_indices=self.hard_polar_decoder.info_indices,
            bp_iterations=bp_iterations,
        )
        self.payload_length = self.hard_polar_decoder.payload_length
        self.hard_decode_prior_weight = float(hard_decode_prior_weight)
        self.hard_decode_retry_weight = float(hard_decode_retry_weight)
        self.hard_reconstruction_ratio = float(max(0.0, min(1.0, hard_reconstruction_ratio)))
        self.failed_crc_hard_gate = 0.22
        self.strict_external_llr_hard_decode = bool(strict_external_llr_hard_decode)
        self.bridge_external_llr_for_metrics_only = bool(bridge_external_llr_for_metrics_only)
        self.external_hard_decode_crc_concealment = bool(external_hard_decode_crc_concealment)
        self.symbol_hint_mix = float(max(0.0, min(1.0, symbol_hint_mix)))
        self.full_decode_feature_mix = float(max(0.0, min(1.0, full_decode_feature_mix)))
        self.full_decode_feature_training_threshold = float(
            max(0.0, min(1.0, full_decode_feature_training_threshold))
        )
        self.full_decode_feature_min_mix_ratio = float(
            max(0.0, min(1.0, full_decode_feature_min_mix_ratio))
        )
        self.allow_feature_correction_on_external_hard_decode = bool(
            allow_feature_correction_on_external_hard_decode
        )
        self.bit_condition = nn.Linear(info_length, hidden_dim * 2)
        self.state_condition = nn.Sequential(
            nn.LayerNorm(code_length + mamba_dim),
            nn.Linear(code_length + mamba_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        self.feature_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=1),
            nn.Sigmoid(),
        )
        self.confidence_gate_threshold = confidence_gate_threshold
        self.confidence_gate_floor = float(max(0.0, min(0.95, confidence_gate_floor)))
        self.semantic_fusion_temperature = semantic_fusion_temperature
        self.latent_head = LatentRefinementHead(hidden_dim, latent_channels)
        self.image_refiner = ImageRefinementHead(
            hidden_dim,
            image_channels,
            min_decoded_blend=image_refine_min_decoded_blend,
            max_refine_side=image_refine_max_side,
            use_gradient_checkpointing=image_refine_checkpointing,
            eval_full_resolution=image_refine_eval_full_resolution,
            analog_demod_lowpass_kernel=analog_demod_lowpass_kernel,
            analog_demod_residual_mix=analog_demod_residual_mix,
        )
        self.bypass_image_refiner_on_external_hard_decode = bool(
            bypass_image_refiner_on_external_hard_decode
        )
        self.bypass_image_refiner_on_full_hard_decode = bool(
            bypass_image_refiner_on_full_hard_decode
        )
        self.disable_cover_guidance_in_refiner = bool(disable_cover_guidance_in_refiner)
        self.cover_guidance_strength_in_refiner = float(max(0.0, min(1.0, cover_guidance_strength_in_refiner)))
        self.external_llr_cover_guidance_strength = (
            None
            if external_llr_cover_guidance_strength is None
            else float(max(0.0, min(1.0, external_llr_cover_guidance_strength)))
        )
        self.image_refine_strength = float(max(0.0, image_refine_strength))
        self.image_refine_min_decoded_blend = float(max(0.0, min(1.0, image_refine_min_decoded_blend)))
        self.analog_lowfreq_decode_gain = float(max(0.0, min(8.0, analog_lowfreq_decode_gain)))
        if analog_detail_decode_gain is None:
            analog_detail_decode_gain = 1.20 * self.analog_lowfreq_decode_gain
        self.analog_detail_decode_gain = float(max(0.0, min(10.0, analog_detail_decode_gain)))
        self.analog_inverse_companding = bool(analog_inverse_companding)
        self.analog_lowfreq_sender_gain = float(max(1e-4, analog_lowfreq_sender_gain))
        self.analog_lowfreq_sender_clip = float(max(1e-4, analog_lowfreq_sender_clip))
        self.analog_detail_sender_gain = float(max(1e-4, analog_detail_sender_gain))
        self.analog_detail_sender_clip = float(max(1e-4, analog_detail_sender_clip))
        self.analog_detail_hh_ratio = float(max(0.0, min(1.0, analog_detail_hh_ratio)))
        self.analog_lowfreq_hh_ratio = float(max(0.0, min(1.0, analog_lowfreq_hh_ratio)))
        self.analog_ll_direct_sender_gain = float(max(0.0, min(1.0, analog_ll_direct_sender_gain)))
        self.analog_ll_direct_decode_mix = float(max(0.0, min(1.0, analog_ll_direct_decode_mix)))
        self.bitstream_restorer = LatentBitstreamRestorer(
            bit_depth=bit_depth,
            latent_clip_value=latent_clip_value,
            channel_bit_depths=channel_bit_depths,
            bitstream_order=bitstream_order,
            base_layer_channels=base_layer_channels,
        )
        self.decode_block_chunk_size = max(1, decode_block_chunk_size)
        self.train_decode_blocks = train_decode_blocks
        self.eval_decode_blocks = eval_decode_blocks

    # 同时提取图像恢复特征和通信残差特征，通信分支优先关注高频残差而不是完整自然纹理。
    def _extract_feature_streams(
        self,
        cover_image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        cover_bands = self.wavelet.dwt(cover_image)
        flat_bands = cover_bands.flatten()
        restored_features = self.restorer(flat_bands)
        clean_lh, clean_hl, clean_hh = self.clean_hf_predictor(flat_bands).chunk(3, dim=1)
        residual_flat = torch.cat(
            [
                torch.zeros_like(cover_bands.ll),
                cover_bands.lh - clean_lh,
                cover_bands.hl - clean_hl,
                cover_bands.hh - clean_hh,
            ],
            dim=1,
        )
        residual_features = self.residual_restorer(residual_flat)
        comm_features = restored_features + self.residual_feature_gain * residual_features
        return restored_features, comm_features, (clean_lh, clean_hl, clean_hh)

    # 将组内局部符号提示还原成按 block 排列的符号提示序列。
    def _group_symbol_hints_to_blocks(
        self,
        group_symbol_hints: torch.Tensor,
        total_blocks: int,
        available_group_indices: torch.Tensor | None = None,
        block_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, num_groups, group_size, code_length = group_symbol_hints.shape
        if group_size != self.group_size:
            raise ValueError(
                f"group_symbol_hints group_size mismatch: expected {self.group_size}, got {group_size}."
            )
        if code_length != self.code_length:
            raise ValueError(
                f"group_symbol_hints code_length mismatch: expected {self.code_length}, got {code_length}."
            )
        if available_group_indices is None:
            available_group_indices = torch.arange(num_groups, device=group_symbol_hints.device, dtype=torch.long)
        else:
            available_group_indices = available_group_indices.to(device=group_symbol_hints.device, dtype=torch.long)
        local_offsets = torch.arange(self.group_size, device=group_symbol_hints.device, dtype=torch.long).view(1, self.group_size)
        global_block_indices = available_group_indices.view(-1, 1) * self.group_size + local_offsets
        full_hints = group_symbol_hints.reshape(batch, num_groups * self.group_size, self.code_length)
        flat_indices = global_block_indices.flatten()
        valid_mask = flat_indices < total_blocks
        valid_indices = flat_indices[valid_mask]
        full_hints = full_hints[:, valid_mask]
        if valid_indices.numel() != total_blocks:
            padded_hints = full_hints.new_zeros(batch, total_blocks, self.code_length)
            if valid_indices.numel() > 0:
                padded_hints.index_copy_(1, valid_indices, full_hints)
            full_hints = padded_hints
        if block_indices is None:
            return full_hints[:, :total_blocks]
        selected_block_indices = block_indices.to(device=group_symbol_hints.device, dtype=torch.long)
        validate_index_tensor(selected_block_indices, total_blocks, "symbol_hint_block_indices")
        return full_hints.index_select(1, selected_block_indices)

    # 按平均绝对值归一化符号提示，避免与相关解扩分支的量纲严重失配。
    def _normalize_symbol_hint(self, symbol_hint: torch.Tensor) -> torch.Tensor:
        safe_symbol_hint = torch.nan_to_num(symbol_hint, nan=0.0, posinf=1.0, neginf=-1.0)
        scale = safe_symbol_hint.float().abs().mean(dim=-1, keepdim=True).clamp_min(1e-4).to(safe_symbol_hint.dtype)
        return safe_symbol_hint / scale

    # 根据训练或评估阶段决定实际送入 Mamba 的 block 子集，减轻显存与时间压力。
    def _select_decode_indices(
        self,
        num_blocks: int,
        device: torch.device,
        max_decode_blocks: int | None = None,
        force_full_decode: bool = False,
    ) -> torch.Tensor | None:
        if force_full_decode:
            return None
        if max_decode_blocks is None:
            max_decode_blocks = self.train_decode_blocks if self.training else self.eval_decode_blocks
        if max_decode_blocks is None or max_decode_blocks <= 0 or max_decode_blocks >= num_blocks:
            return None
        if self.training:
            total_groups = max(1, math.ceil(num_blocks / self.group_size))
            sampled_group_count = max(1, math.ceil(max_decode_blocks / self.group_size))
            sampled_groups = torch.randperm(total_groups, device=device)[:sampled_group_count]
            sampled_groups = torch.sort(sampled_groups).values
            selected_blocks: list[torch.Tensor] = []
            for group_index in sampled_groups.detach().cpu().tolist():
                start = int(group_index) * self.group_size
                end = min(num_blocks, start + self.group_size)
                selected_blocks.append(torch.arange(start, end, device=device, dtype=torch.long))
            block_indices = torch.cat(selected_blocks, dim=0)
            return block_indices[:max_decode_blocks]
        return torch.arange(max_decode_blocks, device=device, dtype=torch.long)

    # 用双向 Mamba 和神经 BP 细化每个 block 的 LLR 与 logits。
    def _decode_llr_blocks(
        self,
        llr_signal: torch.Tensor,
        block_indices: torch.Tensor | None = None,
        total_blocks: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, num_blocks, code_length = llr_signal.shape
        block_llr = torch.nan_to_num(llr_signal, nan=0.0, posinf=20.0, neginf=-20.0)
        feature_llr = block_llr / block_llr.float().abs().mean(dim=-1, keepdim=True).clamp_min(1e-4).to(block_llr.dtype)
        block_tokens = self.block_token_norm(feature_llr)
        block_tokens = self.block_token_proj(block_tokens)
        if block_indices is None:
            position_embedding = build_sinusoidal_position_embedding(num_blocks, self.mamba_dim, block_tokens.device, block_tokens.dtype)
        else:
            all_positions = build_sinusoidal_position_embedding(
                int(total_blocks or num_blocks),
                self.mamba_dim,
                block_tokens.device,
                block_tokens.dtype,
            )
            validated_block_indices = block_indices.to(device=block_tokens.device, dtype=torch.long)
            validate_index_tensor(validated_block_indices, all_positions.shape[1], "mamba_position_indices")
            position_embedding = all_positions.index_select(1, validated_block_indices)
        block_tokens = block_tokens + position_embedding
        block_state = self.mamba(block_tokens)
        if num_blocks > self.decode_block_chunk_size:
            code_chunks = []
            bit_chunks = []
            for start in range(0, num_blocks, self.decode_block_chunk_size):
                end = min(start + self.decode_block_chunk_size, num_blocks)
                code_logits, bit_logits = self.bp_decoder(block_state[:, start:end], block_llr[:, start:end])
                code_chunks.append(code_logits)
                bit_chunks.append(bit_logits)
            decoded_code_logits = torch.cat(code_chunks, dim=1)
            decoded_bit_logits = torch.cat(bit_chunks, dim=1)
        else:
            decoded_code_logits, decoded_bit_logits = self.bp_decoder(block_state, block_llr)
        return decoded_code_logits, decoded_bit_logits, block_state

    # 在完整码流可用时运行真实 CRC-Polar 硬译码，输出 payload、信息位和重编码码字。
    def _build_hard_decode_llr(
        self,
        llr_signal: torch.Tensor,
        decoded_code_logits: torch.Tensor | None = None,
        prior_weight: float | None = None,
    ) -> torch.Tensor:
        weight = self.hard_decode_prior_weight if prior_weight is None else float(prior_weight)
        if decoded_code_logits is None or weight <= 0:
            return llr_signal
        # decoded_code_logits 是 bit=1 logits，而 Polar SC 需要 log(P0/P1) 形式的标准 LLR。
        prior_llr = -torch.nan_to_num(decoded_code_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        prior_llr = prior_llr / prior_llr.float().abs().mean(dim=-1, keepdim=True).clamp_min(1e-4).to(prior_llr.dtype)
        return (1.0 - weight) * llr_signal + weight * prior_llr

    # 在完整码流可用时运行带学习先验的 CRC-Polar 硬译码，并输出 payload、信息位和重编码码字。
    def _hard_decode_full_stream(
        self,
        llr_signal: torch.Tensor,
        decoded_code_logits: torch.Tensor | None = None,
        decoded_bit_logits: torch.Tensor | None = None,
        prefer_strict_polar_path: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 先用原始信道 LLR 做主译码，再在失败样本上比较 retry 与 learned 候选，避免外部 LLR 路径只会“硬切”而不做稳健修复。
        candidate_outputs: list[tuple[object, torch.Tensor]] = []
        hard_output = self.hard_polar_decoder.decode(llr_signal)
        candidate_outputs.append((hard_output, hard_output.crc_pass_mask))

        if self.hard_decode_retry_weight > 0:
            retry_llr = self._build_hard_decode_llr(llr_signal, decoded_code_logits, self.hard_decode_retry_weight)
            retry_output = self.hard_polar_decoder.decode(retry_llr)
            candidate_outputs.append((retry_output, retry_output.crc_pass_mask))

        if decoded_bit_logits is not None and not prefer_strict_polar_path:
            learned_payload_bits = (decoded_bit_logits[..., : self.payload_length] >= 0).to(dtype=llr_signal.dtype)
            learned_info_bits = self.hard_polar_decoder.attach_crc(learned_payload_bits)
            learned_code_bits = self.hard_polar_decoder.encode_info_bits(learned_info_bits)
            learned_output = type(hard_output)(
                payload_bits=learned_payload_bits,
                info_bits=learned_info_bits,
                code_bits=learned_code_bits,
                crc_pass_mask=torch.zeros_like(hard_output.crc_pass_mask),
            )
            candidate_outputs.append((learned_output, torch.zeros_like(hard_output.crc_pass_mask)))

        if len(candidate_outputs) == 1:
            return (
                hard_output.payload_bits.to(dtype=llr_signal.dtype),
                hard_output.info_bits.to(dtype=llr_signal.dtype),
                hard_output.code_bits.to(dtype=llr_signal.dtype),
                hard_output.crc_pass_mask,
            )

        flat_payload_candidates = []
        flat_info_candidates = []
        flat_code_candidates = []
        flat_crc_candidates = []
        score_candidates = []
        trusted_candidates = []
        for candidate_output, trusted_mask in candidate_outputs:
            flat_payload_candidates.append(candidate_output.payload_bits.reshape(-1, candidate_output.payload_bits.shape[-1]))
            flat_info_candidates.append(candidate_output.info_bits.reshape(-1, candidate_output.info_bits.shape[-1]))
            flat_code_candidates.append(candidate_output.code_bits.reshape(-1, candidate_output.code_bits.shape[-1]))
            flat_crc_candidates.append(candidate_output.crc_pass_mask.reshape(-1))
            reliability = self._estimate_hard_reliability(
                llr_signal,
                candidate_output.code_bits,
            ).reshape(-1, 1)
            prior = self._estimate_prior_agreement(
                candidate_output.code_bits,
                decoded_code_logits,
            ).reshape(-1, 1)
            payload_prior = self._estimate_payload_prior_agreement(
                candidate_output.payload_bits,
                decoded_bit_logits,
            ).reshape(-1, 1)
            trusted_value = trusted_mask.to(dtype=llr_signal.dtype).reshape(-1, 1)
            trusted_candidates.append(trusted_value)
            score_candidates.append(1.15 * reliability + 0.30 * prior + 0.35 * payload_prior + 0.10 * trusted_value)

        stacked_payload = torch.stack(flat_payload_candidates, dim=1)
        stacked_info = torch.stack(flat_info_candidates, dim=1)
        stacked_code = torch.stack(flat_code_candidates, dim=1)
        stacked_crc = torch.stack(flat_crc_candidates, dim=1)
        stacked_scores = torch.stack(score_candidates, dim=1).squeeze(-1)
        stacked_trusted = torch.stack(trusted_candidates, dim=1).squeeze(-1) > 0.5

        best_any = stacked_scores.argmax(dim=1)
        has_trusted = stacked_trusted.any(dim=1)
        trusted_scores = stacked_scores.masked_fill(~stacked_trusted, -1e9)
        best_trusted = trusted_scores.argmax(dim=1)
        best_index = torch.where(has_trusted, best_trusted, best_any)
        sample_index = torch.arange(best_index.shape[0], device=best_index.device)

        selected_payload = stacked_payload[sample_index, best_index].view_as(hard_output.payload_bits)
        selected_info = stacked_info[sample_index, best_index].view_as(hard_output.info_bits)
        selected_code = stacked_code[sample_index, best_index].view_as(hard_output.code_bits)
        selected_crc = stacked_crc[sample_index, best_index].view_as(hard_output.crc_pass_mask)
        # Keep the candidate selected by the decoder for link-level metrics.
        # CRC-based reconstruction repair is deliberately kept out of this
        # record: it is a downstream image-recovery operation, not a received
        # codeword.  Returning ``selected_code`` here makes Info BER, Code BER,
        # and CRC admission refer to the same pre-repair candidate.
        return (
            selected_payload.to(dtype=llr_signal.dtype),
            selected_info.to(dtype=llr_signal.dtype),
            selected_code.to(dtype=llr_signal.dtype),
            selected_crc,
        )

    # 根据 CRC 通过情况对硬解码结果和软预测做融合，减少错误码包对 latent 重建的破坏。
    def _fuse_payload_estimates(
        self,
        soft_payload_bits: torch.Tensor,
        hard_payload_bits: torch.Tensor | None,
        crc_pass_mask: torch.Tensor | None,
        llr_signal: torch.Tensor | None = None,
        hard_code_bits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hard_payload_bits is None or crc_pass_mask is None:
            return soft_payload_bits
        crc_mask = crc_pass_mask.to(dtype=soft_payload_bits.dtype).unsqueeze(-1)
        soft_hard_gap = (hard_payload_bits - soft_payload_bits).abs().mean(dim=-1, keepdim=True)
        consistency_gate = torch.exp(-6.0 * soft_hard_gap).to(dtype=soft_payload_bits.dtype)
        effective_gate = crc_mask * consistency_gate
        if llr_signal is not None and hard_code_bits is not None and self.failed_crc_hard_gate > 0:
            reliability_gate = self._estimate_hard_reliability(llr_signal, hard_code_bits)
            fallback_gate = (1.0 - crc_mask) * self.failed_crc_hard_gate * reliability_gate * consistency_gate.sqrt()
            effective_gate = (effective_gate + fallback_gate).clamp(0.0, 1.0)
        return effective_gate * hard_payload_bits + (1.0 - effective_gate) * soft_payload_bits

    # 为 latent 重建选择 payload：CRC 通过的 block 直接使用硬译码结果，未通过时再退回软硬融合。
    def _select_reconstruction_payload_bits(
        self,
        soft_payload_bits: torch.Tensor,
        hard_payload_bits: torch.Tensor | None,
        crc_pass_mask: torch.Tensor | None,
        llr_signal: torch.Tensor | None = None,
        hard_code_bits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hard_payload_bits is None or crc_pass_mask is None:
            return soft_payload_bits
        fallback_payload = self._fuse_payload_estimates(
            soft_payload_bits=soft_payload_bits,
            hard_payload_bits=hard_payload_bits,
            crc_pass_mask=crc_pass_mask,
            llr_signal=llr_signal,
            hard_code_bits=hard_code_bits,
        )
        crc_mask = crc_pass_mask.to(dtype=soft_payload_bits.dtype).unsqueeze(-1)
        if llr_signal is None or hard_code_bits is None:
            return crc_mask * hard_payload_bits + (1.0 - crc_mask) * fallback_payload
        reliability_gate = self._estimate_hard_reliability(llr_signal, hard_code_bits)
        soft_hard_gap = (hard_payload_bits - soft_payload_bits).abs().mean(dim=-1, keepdim=True)
        consistency_gate = torch.exp(-8.0 * soft_hard_gap).to(dtype=soft_payload_bits.dtype)
        trusted_fallback_mask = (
            (1.0 - crc_mask)
            * (reliability_gate >= 0.985).to(dtype=soft_payload_bits.dtype)
            * (consistency_gate >= 0.985).to(dtype=soft_payload_bits.dtype)
        )
        effective_hard_mask = torch.clamp(crc_mask + trusted_fallback_mask, 0.0, 1.0)
        return effective_hard_mask * hard_payload_bits + (1.0 - effective_hard_mask) * fallback_payload

    def _fuse_code_estimates(
        self,
        soft_code_bits: torch.Tensor,
        hard_code_bits: torch.Tensor | None,
        crc_pass_mask: torch.Tensor | None,
        llr_signal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hard_code_bits is None or crc_pass_mask is None:
            return soft_code_bits
        crc_mask = crc_pass_mask.to(dtype=soft_code_bits.dtype).unsqueeze(-1)
        soft_hard_gap = (hard_code_bits - soft_code_bits).abs().mean(dim=-1, keepdim=True)
        consistency_gate = torch.exp(-4.0 * soft_hard_gap).to(dtype=soft_code_bits.dtype)
        effective_gate = crc_mask * consistency_gate
        if llr_signal is not None and self.failed_crc_hard_gate > 0:
            reliability_gate = self._estimate_hard_reliability(llr_signal, hard_code_bits)
            fallback_gate = (1.0 - crc_mask) * (self.failed_crc_hard_gate + 0.08) * reliability_gate * consistency_gate.sqrt()
            effective_gate = (effective_gate + fallback_gate).clamp(0.0, 1.0)
        return effective_gate * hard_code_bits + (1.0 - effective_gate) * soft_code_bits

    # 根据硬判码字与接收 LLR 的一致性，估计 CRC 未通过时硬解结果仍可参考的可信度。
    def _estimate_hard_reliability(
        self,
        llr_signal: torch.Tensor,
        hard_code_bits: torch.Tensor,
    ) -> torch.Tensor:
        normalized_llr = torch.tanh(torch.nan_to_num(llr_signal, nan=0.0, posinf=20.0, neginf=-20.0) / 4.0)
        hard_symbols = 1.0 - 2.0 * hard_code_bits.to(dtype=normalized_llr.dtype)
        signed_margin = (normalized_llr * hard_symbols).mean(dim=-1, keepdim=True)
        return torch.sigmoid(6.0 * (signed_margin - 0.10))

    # 估计硬译码结果与软码字先验的一致性，在 CRC 失败时辅助选择更可信的候选。
    def _estimate_prior_agreement(
        self,
        hard_code_bits: torch.Tensor,
        decoded_code_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        if decoded_code_logits is None:
            return torch.zeros(
                hard_code_bits.shape[0],
                hard_code_bits.shape[1],
                1,
                device=hard_code_bits.device,
                dtype=hard_code_bits.dtype,
            )
        soft_code_bits = torch.sigmoid(decoded_code_logits).to(dtype=hard_code_bits.dtype)
        agreement_gap = (hard_code_bits.to(dtype=soft_code_bits.dtype) - soft_code_bits).abs().mean(dim=-1, keepdim=True)
        return torch.exp(-4.0 * agreement_gap)

    # 估计候选 payload 与软 payload 先验的一致性，避免 learned 候选只修好 payload 却把整体码字带偏。
    def _estimate_payload_prior_agreement(
        self,
        hard_payload_bits: torch.Tensor,
        decoded_bit_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        if decoded_bit_logits is None:
            return torch.zeros(
                hard_payload_bits.shape[0],
                hard_payload_bits.shape[1],
                1,
                device=hard_payload_bits.device,
                dtype=hard_payload_bits.dtype,
            )
        soft_payload_bits = torch.sigmoid(decoded_bit_logits[..., : self.payload_length]).to(dtype=hard_payload_bits.dtype)
        agreement_gap = (hard_payload_bits.to(dtype=soft_payload_bits.dtype) - soft_payload_bits).abs().mean(dim=-1, keepdim=True)
        return torch.exp(-4.5 * agreement_gap)

    # 用直通估计把 teacher 或 hard payload 混入前向重建，同时让恢复图梯度继续回传到软比特分支。
    def _straight_through_tensor_blend(
        self,
        predicted_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        blend_ratio: float,
    ) -> torch.Tensor:
        safe_ratio = float(max(0.0, min(1.0, blend_ratio)))
        if safe_ratio <= 0.0:
            return predicted_tensor
        target_value = target_tensor.to(
            device=predicted_tensor.device,
            dtype=predicted_tensor.dtype,
        )
        if target_value.shape != predicted_tensor.shape:
            raise ValueError(
                "target_tensor shape must match predicted_tensor for straight-through blend. "
                f"got target={target_value.shape}, predicted={predicted_tensor.shape}."
            )
        blended_tensor = (1.0 - safe_ratio) * predicted_tensor + safe_ratio * target_value
        return predicted_tensor + (blended_tensor - predicted_tensor).detach()

    # 将接收端恢复出的空间特征、比特统计与 Mamba 状态融合成 latent 恢复条件。
    def _build_feature_conditioning(
        self,
        restored_features: torch.Tensor,
        decoded_bits: torch.Tensor,
        llr_signal: torch.Tensor,
        mamba_state: torch.Tensor,
    ) -> torch.Tensor:
        avg_info_bits = decoded_bits.mean(dim=1)
        avg_llr = torch.tanh(llr_signal).mean(dim=1)
        avg_state = mamba_state.mean(dim=1)
        return (
            restored_features
            + self.feature_gate(restored_features) * self.semantic_fusion_temperature
            + self.bit_condition(avg_info_bits).unsqueeze(-1).unsqueeze(-1)
            + self.state_condition(torch.cat([avg_llr, avg_state], dim=1)).unsqueeze(-1).unsqueeze(-1)
        )

    # 通过特征分支预测一份 latent，用来给完整码流解码结果做细节校正。
    def _predict_feature_latent(
        self,
        restored_features: torch.Tensor,
        decoded_bits: torch.Tensor,
        llr_signal: torch.Tensor,
        mamba_state: torch.Tensor,
        latent_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        conditioned = self._build_feature_conditioning(
            restored_features=restored_features,
            decoded_bits=decoded_bits,
            llr_signal=llr_signal,
            mamba_state=mamba_state,
        )
        return self.latent_head(conditioned, latent_shape[-2:])

    # 在 full-decode 阶段优先相信 bitstream latent，只在解码不可靠时才引入少量特征 latent 修正。
    def _blend_full_decode_latent(
        self,
        decoded_latent: torch.Tensor,
        feature_latent: torch.Tensor,
        soft_payload_bits: torch.Tensor,
        fused_payload_bits: torch.Tensor | None,
        llr_signal: torch.Tensor,
        crc_pass_mask: torch.Tensor | None,
        allow_feature_correction: bool = True,
    ) -> torch.Tensor:
        if self.full_decode_feature_mix <= 0.0 or not allow_feature_correction:
            return decoded_latent
        safe_decoded_latent = torch.nan_to_num(decoded_latent)
        safe_feature_latent = torch.nan_to_num(feature_latent)
        payload_reference = fused_payload_bits if fused_payload_bits is not None else soft_payload_bits
        payload_gap = (payload_reference - soft_payload_bits).abs().mean(dim=(1, 2))
        llr_confidence = torch.tanh(llr_signal.abs().mean(dim=(1, 2)) / 3.0)
        if crc_pass_mask is None:
            crc_confidence = llr_confidence
        else:
            crc_confidence = crc_pass_mask.to(dtype=payload_gap.dtype).reshape(payload_gap.shape[0], -1).mean(dim=1)
        # 全量码包已经高置信恢复时，必须直接信任 bitstream latent，
        # 避免 feature latent 在 stage3/stage4 将本已正确的重建主干拉偏。
        stable_full_decode_mask = (
            (crc_confidence >= 0.999)
            & (llr_confidence >= 0.90)
            & (payload_gap <= 0.002)
        )
        if torch.all(stable_full_decode_mask):
            return safe_decoded_latent
        payload_consistency = 1.0 - payload_gap.clamp(0.0, 1.0)
        reliability_gate = (
            0.55 * llr_confidence
            + 0.35 * crc_confidence
            + 0.10 * payload_consistency
        ).clamp(0.0, 1.0)
        uncertainty_gate = (1.0 - reliability_gate).clamp(0.0, 1.0)
        crc_uncertainty = (1.0 - crc_confidence).clamp(0.0, 1.0)
        adaptive_mix = self.full_decode_feature_mix * (
            self.full_decode_feature_min_mix_ratio
            + 0.58 * uncertainty_gate
            + 0.25 * payload_gap.clamp(0.0, 1.0)
            + 0.15 * crc_uncertainty
        ).clamp(0.0, 1.0)
        stable_scale = torch.where(
            stable_full_decode_mask,
            torch.zeros_like(adaptive_mix),
            torch.ones_like(adaptive_mix),
        )
        adaptive_mix = adaptive_mix * stable_scale
        adaptive_mix = adaptive_mix.view(-1, 1, 1, 1)
        return safe_decoded_latent + adaptive_mix * (safe_feature_latent - safe_decoded_latent)

    # 用直通估计把 teacher 或 hard payload 混入前向重建，同时让恢复图梯度继续回传到软比特分支。
    def _straight_through_payload_blend(
        self,
        soft_payload_bits: torch.Tensor,
        target_payload_bits: torch.Tensor,
        blend_ratio: float,
    ) -> torch.Tensor:
        return self._straight_through_tensor_blend(
            predicted_tensor=soft_payload_bits,
            target_tensor=target_payload_bits,
            blend_ratio=blend_ratio,
        )

    # 鎸夊墠绔ā寮忛€夋嫨 clean 鎴?robust 澶栭儴 LLR 鏍″噯鍓嶇锛岄伩鍏嶆敾鍑荤粏鍖栧奖鍝?clean 涓婚摼璺€?
    def _apply_external_llr_frontend(
        self,
        raw_external_llr_signal: torch.Tensor,
        frontend_mode: str = "clean",
    ) -> torch.Tensor:
        resolved_mode = str(frontend_mode or "clean").strip().lower()
        if resolved_mode == "robust":
            adapter = self.robust_external_llr_adapter
            symbol_calibrator = self.robust_external_symbol_calibrator
        else:
            adapter = self.external_llr_adapter
            symbol_calibrator = self.external_symbol_calibrator
        calibrated_llr_signal = adapter(raw_external_llr_signal)
        calibrated_llr_signal = symbol_calibrator(calibrated_llr_signal)
        # External QIM LLRs are already generated in the transmitted codeword
        # domain. Treating frozen positions as pilots here can flip/scale a
        # packet based on extractor bias rather than true pilot evidence,
        # which in practice corrupts CRC bits while leaving payload bits
        # apparently intact. Keep the learned adapter/symbol calibration, but
        # skip frozen-bit post-calibration on the external-LLR path.
        return calibrated_llr_signal

    # 从载密图像恢复 payload 比特，再重建 latent 与最终恢复图像。
    def forward(
        self,
        cover_image: torch.Tensor,
        latent_shape: tuple[int, int, int],
        latent_decoder: Callable[[torch.Tensor, tuple[int, int]], torch.Tensor],
        output_size: tuple[int, int],
        carrier_reference_image: torch.Tensor | None = None,
        valid_info_bits: int | None = None,
        num_blocks: int = 1,
        max_decode_blocks: int | None = None,
        force_full_decode: bool = False,
        full_bitstream_available: bool = True,
        teacher_payload_bits: torch.Tensor | None = None,
        teacher_payload_ratio: float = 0.0,
        teacher_channel_symbols: torch.Tensor | None = None,
        teacher_channel_ratio: float = 0.0,
        external_llr_signal: torch.Tensor | None = None,
        external_llr_frontend_mode: str = "clean",
        detach_image_reconstruction: bool = False,
        force_image_refiner: bool = False,
        cover_guidance_strength_override: float | None = None,
    ) -> ReceiverOutput:
        restored_features, comm_features, _ = self._extract_feature_streams(cover_image)
        reconstruction_payload_bits: torch.Tensor | None = None
        bridge_internal_llr_signal: torch.Tensor | None = None
        bridge_internal_despread: torch.Tensor | None = None
        bridge_internal_received_chips: torch.Tensor | None = None
        bridge_internal_block_symbol_hints: torch.Tensor | None = None
        decoded_block_indices = self._select_decode_indices(
            num_blocks=num_blocks,
            device=cover_image.device,
            max_decode_blocks=max_decode_blocks,
            force_full_decode=force_full_decode,
        )
        if decoded_block_indices is not None:
            validate_index_tensor(decoded_block_indices, num_blocks, "receiver_forward_decoded_block_indices")
        total_groups = max(1, math.ceil(num_blocks / self.group_size))
        active_group_indices = None
        receive_group_count = total_groups
        if decoded_block_indices is not None:
            active_group_indices = torch.unique(
                torch.div(decoded_block_indices, self.group_size, rounding_mode="floor"),
                sorted=True,
            )
            validate_index_tensor(active_group_indices, total_groups, "receiver_active_group_indices")
            receive_group_count = max(1, int(active_group_indices.numel()))
        group_indices_for_despreader = active_group_indices
        if group_indices_for_despreader is None:
            group_indices_for_despreader = torch.arange(total_groups, device=cover_image.device, dtype=torch.long)
        need_internal_bridge_path = external_llr_signal is None or self.bridge_external_llr_for_metrics_only
        if need_internal_bridge_path:
            internal_bridge_context = (
                torch.no_grad()
                if external_llr_signal is not None and self.bridge_external_llr_for_metrics_only
                else nullcontext()
            )
            with internal_bridge_context:
                bridge_internal_received_chips, group_symbol_hints = self.despreader(
                    comm_features,
                    num_groups=receive_group_count,
                    group_indices=group_indices_for_despreader,
                    total_group_count=total_groups,
                )
                bridge_internal_block_symbol_hints = self._group_symbol_hints_to_blocks(
                    group_symbol_hints=group_symbol_hints,
                    total_blocks=num_blocks,
                    available_group_indices=group_indices_for_despreader,
                    block_indices=decoded_block_indices,
                )
                bridge_internal_despread = self.correlator(
                    bridge_internal_received_chips,
                    block_indices=decoded_block_indices,
                    total_blocks=num_blocks,
                    available_group_indices=active_group_indices,
                )
                if self.symbol_hint_mix > 0.0:
                    bridge_internal_despread = torch.lerp(
                        bridge_internal_despread,
                        self._normalize_symbol_hint(bridge_internal_block_symbol_hints).to(dtype=bridge_internal_despread.dtype),
                        self.symbol_hint_mix,
                    )
                bridge_internal_despread = self.symbol_calibrator(bridge_internal_despread)
                bridge_internal_llr_signal = self.llr_scaler(bridge_internal_despread)
                bridge_internal_llr_signal = self.frozen_pilot_calibrator(bridge_internal_llr_signal)
        if external_llr_signal is None:
            received_chips = bridge_internal_received_chips
            block_symbol_hints = bridge_internal_block_symbol_hints
            despread = bridge_internal_despread
            llr_signal = bridge_internal_llr_signal
            used_external_llr = False
            raw_external_llr_signal = None
            decoder_llr_signal = llr_signal
        else:
            if external_llr_signal.shape[1] != int(num_blocks) or external_llr_signal.shape[-1] != self.code_length:
                raise ValueError(
                    "external_llr_signal must have shape [B, num_blocks, code_length]. "
                    f"got {tuple(external_llr_signal.shape)}, expected blocks={num_blocks}, code={self.code_length}."
                )
            if decoded_block_indices is None:
                raw_external_llr_signal = external_llr_signal.to(device=cover_image.device, dtype=cover_image.dtype)
            else:
                raw_external_llr_signal = external_llr_signal.to(device=cover_image.device, dtype=cover_image.dtype).index_select(
                    1,
                    decoded_block_indices,
                )
            calibrated_llr_signal = self._apply_external_llr_frontend(
                raw_external_llr_signal,
                frontend_mode=external_llr_frontend_mode,
            )
            llr_signal = calibrated_llr_signal
            despread = llr_signal
            if self.bridge_external_llr_for_metrics_only and bridge_internal_received_chips is not None:
                block_symbol_hints = bridge_internal_block_symbol_hints
                received_chips = bridge_internal_received_chips
            else:
                block_symbol_hints = None
                received_chips = cover_image.new_zeros(
                    cover_image.shape[0],
                    receive_group_count,
                    self.code_length,
                    self.chips_per_symbol,
                )
            used_external_llr = True
            decoder_llr_signal = calibrated_llr_signal
        effective_teacher_channel_ratio = float(max(0.0, min(0.20, teacher_channel_ratio)))
        if self.training and teacher_channel_symbols is not None and effective_teacher_channel_ratio > 0.0:
            if decoded_block_indices is not None:
                teacher_channel_symbols = teacher_channel_symbols.index_select(1, decoded_block_indices)
            normalized_teacher_symbols = torch.nan_to_num(
                teacher_channel_symbols,
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            ).clamp(-1.0, 1.0)
            decoder_llr_signal = self._straight_through_tensor_blend(
                predicted_tensor=decoder_llr_signal,
                target_tensor=normalized_teacher_symbols,
                blend_ratio=effective_teacher_channel_ratio,
            )
        hard_decode_llr_signal = decoder_llr_signal if used_external_llr else llr_signal
        if used_external_llr and self.strict_external_llr_hard_decode and raw_external_llr_signal is not None:
            hard_decode_llr_signal = raw_external_llr_signal
        decoded_code_logits, decoded_bit_logits, mamba_state = self._decode_llr_blocks(
            decoder_llr_signal,
            block_indices=decoded_block_indices,
            total_blocks=num_blocks,
        )
        decoded_code_bits = torch.sigmoid(decoded_code_logits)
        decoded_bits = torch.sigmoid(decoded_bit_logits)
        soft_payload_bits = decoded_bits[..., : self.payload_length]
        metric_payload_bits = soft_payload_bits
        metric_code_bits = decoded_code_bits
        valid_bits = valid_info_bits if valid_info_bits is not None else soft_payload_bits.flatten(start_dim=1).shape[1]
        full_bitstream_decoded = full_bitstream_available and decoded_block_indices is None
        bridge_internal_decoded_code_logits = None
        bridge_internal_decoded_bits = None
        bridge_internal_soft_payload_bits = None
        bridge_internal_mamba_state = None
        if used_external_llr and self.bridge_external_llr_for_metrics_only and bridge_internal_llr_signal is not None:
            with torch.no_grad():
                (
                    bridge_internal_decoded_code_logits,
                    bridge_internal_decoded_bit_logits,
                    bridge_internal_mamba_state,
                ) = self._decode_llr_blocks(
                    bridge_internal_llr_signal,
                    block_indices=decoded_block_indices,
                    total_blocks=num_blocks,
                )
                bridge_internal_decoded_bits = torch.sigmoid(bridge_internal_decoded_bit_logits)
                bridge_internal_soft_payload_bits = bridge_internal_decoded_bits[..., : self.payload_length]
                feature_latent = self._predict_feature_latent(
                    restored_features=restored_features,
                    decoded_bits=bridge_internal_decoded_bits,
                    llr_signal=bridge_internal_llr_signal,
                    mamba_state=bridge_internal_mamba_state,
                    latent_shape=latent_shape,
                )
        else:
            feature_latent = self._predict_feature_latent(
                restored_features=restored_features,
                decoded_bits=decoded_bits,
                llr_signal=llr_signal,
                mamba_state=mamba_state,
                latent_shape=latent_shape,
            )
        hard_decoded_payload_bits = None
        hard_decoded_info_bits = None
        hard_decoded_code_bits = None
        crc_pass_mask = None
        payload_faithful_full_decode = False
        if full_bitstream_decoded:
            if used_external_llr:
                (
                    hard_decoded_payload_bits,
                    hard_decoded_info_bits,
                    hard_decoded_code_bits,
                    crc_pass_mask,
                ) = self._hard_decode_full_stream(
                    hard_decode_llr_signal,
                    decoded_code_logits=decoded_code_logits,
                    decoded_bit_logits=decoded_bit_logits,
                    prefer_strict_polar_path=self.strict_external_llr_hard_decode,
                )
                metric_payload_bits = hard_decoded_payload_bits
                metric_code_bits = hard_decoded_code_bits
                if self.bridge_external_llr_for_metrics_only:
                    if self.strict_external_llr_hard_decode:
                        if self.external_hard_decode_crc_concealment:
                            reconstruction_source_bits = (
                                bridge_internal_soft_payload_bits
                                if bridge_internal_soft_payload_bits is not None
                                else soft_payload_bits
                            )
                            reconstruction_payload_bits = self._select_reconstruction_payload_bits(
                                soft_payload_bits=reconstruction_source_bits,
                                hard_payload_bits=hard_decoded_payload_bits,
                                crc_pass_mask=crc_pass_mask,
                                llr_signal=hard_decode_llr_signal,
                                hard_code_bits=hard_decoded_code_bits,
                            )
                            if self.training:
                                reconstruction_payload_bits = self._straight_through_payload_blend(
                                    soft_payload_bits=soft_payload_bits,
                                    target_payload_bits=reconstruction_payload_bits,
                                    blend_ratio=1.0,
                                )
                        elif self.training:
                            reconstruction_payload_bits = self._straight_through_payload_blend(
                                soft_payload_bits=soft_payload_bits,
                                target_payload_bits=hard_decoded_payload_bits,
                                blend_ratio=1.0,
                            )
                        else:
                            reconstruction_payload_bits = hard_decoded_payload_bits
                        payload_faithful_full_decode = bool((not self.training) and reconstruction_payload_bits is not None)
                    elif bridge_internal_soft_payload_bits is not None:
                        reconstruction_payload_bits = bridge_internal_soft_payload_bits
                    else:
                        reconstruction_payload_bits = soft_payload_bits
                    decoded_latent = self.bitstream_restorer(
                        reconstruction_payload_bits,
                        latent_shape,
                        valid_bits,
                    )
                    external_feature_correction_allowed = bool(
                        self.allow_feature_correction_on_external_hard_decode
                        and self.full_decode_feature_mix > 0.0
                    )
                    if (
                        payload_faithful_full_decode
                        or
                        self.full_decode_feature_mix <= 0.0
                        or (
                            self.strict_external_llr_hard_decode
                            and not external_feature_correction_allowed
                        )
                    ):
                        recovered_latent = decoded_latent
                    else:
                        recovered_latent = self._blend_full_decode_latent(
                            decoded_latent=decoded_latent,
                            feature_latent=feature_latent,
                            soft_payload_bits=reconstruction_payload_bits,
                            fused_payload_bits=reconstruction_payload_bits,
                            llr_signal=bridge_internal_llr_signal if bridge_internal_llr_signal is not None else llr_signal,
                            crc_pass_mask=None,
                            allow_feature_correction=self.allow_feature_correction_on_external_hard_decode,
                        )
                else:
                    if self.training:
                        payload_for_reconstruction = self._straight_through_payload_blend(
                            soft_payload_bits=soft_payload_bits,
                            target_payload_bits=hard_decoded_payload_bits,
                            blend_ratio=1.0,
                        )
                    else:
                        payload_for_reconstruction = hard_decoded_payload_bits
                    reconstruction_payload_bits = payload_for_reconstruction
                    payload_faithful_full_decode = bool((not self.training) and reconstruction_payload_bits is not None)
                    recovered_latent = self.bitstream_restorer(payload_for_reconstruction, latent_shape, valid_bits)
                    trust_hard_decode_latent = bool(
                        payload_faithful_full_decode
                        or
                        self.bypass_image_refiner_on_external_hard_decode
                        or self.bypass_image_refiner_on_full_hard_decode
                        or self.strict_external_llr_hard_decode
                    )
                    if not payload_faithful_full_decode:
                        recovered_latent = self._blend_full_decode_latent(
                            decoded_latent=recovered_latent,
                            feature_latent=feature_latent,
                            soft_payload_bits=soft_payload_bits,
                            fused_payload_bits=payload_for_reconstruction,
                            llr_signal=hard_decode_llr_signal,
                            crc_pass_mask=crc_pass_mask,
                            allow_feature_correction=(
                                self.allow_feature_correction_on_external_hard_decode
                                and not self.bypass_image_refiner_on_external_hard_decode
                                and not self.bypass_image_refiner_on_full_hard_decode
                            ),
                        )
            elif self.training:
                if not used_external_llr:
                    payload_for_reconstruction = soft_payload_bits
                    allow_feature_correction = self.full_decode_feature_mix > self.full_decode_feature_training_threshold
                    effective_teacher_payload_ratio = float(max(0.0, min(0.35, teacher_payload_ratio)))
                    if teacher_payload_bits is not None and effective_teacher_payload_ratio > 0.0:
                        payload_for_reconstruction = self._straight_through_payload_blend(
                            soft_payload_bits=soft_payload_bits,
                            target_payload_bits=teacher_payload_bits,
                            blend_ratio=effective_teacher_payload_ratio,
                        )
                    elif self.hard_reconstruction_ratio > 0:
                        (
                            hard_decoded_payload_bits,
                            hard_decoded_info_bits,
                            hard_decoded_code_bits,
                            crc_pass_mask,
                        ) = self._hard_decode_full_stream(
                            llr_signal,
                            decoded_code_logits=decoded_code_logits,
                            decoded_bit_logits=decoded_bit_logits,
                        )
                        fused_hard_payload = self._fuse_payload_estimates(
                            soft_payload_bits=soft_payload_bits,
                            hard_payload_bits=hard_decoded_payload_bits,
                            crc_pass_mask=crc_pass_mask,
                            llr_signal=llr_signal,
                            hard_code_bits=hard_decoded_code_bits,
                        )
                        reconstruction_hard_payload = self._select_reconstruction_payload_bits(
                            soft_payload_bits=soft_payload_bits,
                            hard_payload_bits=hard_decoded_payload_bits,
                            crc_pass_mask=crc_pass_mask,
                            llr_signal=llr_signal,
                            hard_code_bits=hard_decoded_code_bits,
                        )
                        metric_payload_bits = fused_hard_payload
                        metric_code_bits = self._fuse_code_estimates(
                            soft_code_bits=decoded_code_bits,
                            hard_code_bits=hard_decoded_code_bits,
                            crc_pass_mask=crc_pass_mask,
                            llr_signal=llr_signal,
                        )
                        payload_for_reconstruction = self._straight_through_payload_blend(
                            soft_payload_bits=soft_payload_bits,
                            target_payload_bits=reconstruction_hard_payload,
                            blend_ratio=self.hard_reconstruction_ratio,
                        )
                    reconstruction_payload_bits = payload_for_reconstruction
                    recovered_latent = self.bitstream_restorer(payload_for_reconstruction, latent_shape, valid_bits)
                    recovered_latent = self._blend_full_decode_latent(
                        decoded_latent=recovered_latent,
                        feature_latent=feature_latent,
                        soft_payload_bits=soft_payload_bits,
                        fused_payload_bits=payload_for_reconstruction,
                        llr_signal=llr_signal,
                        crc_pass_mask=crc_pass_mask,
                        allow_feature_correction=allow_feature_correction,
                    )
            else:
                (
                    hard_decoded_payload_bits,
                    hard_decoded_info_bits,
                    hard_decoded_code_bits,
                    crc_pass_mask,
                ) = self._hard_decode_full_stream(
                    hard_decode_llr_signal,
                    decoded_code_logits=decoded_code_logits,
                    decoded_bit_logits=decoded_bit_logits,
                )
                fused_payload_bits = self._fuse_payload_estimates(
                    soft_payload_bits=soft_payload_bits,
                    hard_payload_bits=hard_decoded_payload_bits,
                    crc_pass_mask=crc_pass_mask,
                    llr_signal=hard_decode_llr_signal,
                    hard_code_bits=hard_decoded_code_bits,
                )
                reconstruction_payload_bits = self._select_reconstruction_payload_bits(
                    soft_payload_bits=soft_payload_bits,
                    hard_payload_bits=hard_decoded_payload_bits,
                    crc_pass_mask=crc_pass_mask,
                    llr_signal=hard_decode_llr_signal,
                    hard_code_bits=hard_decoded_code_bits,
                )
                metric_payload_bits = fused_payload_bits
                metric_code_bits = self._fuse_code_estimates(
                    soft_code_bits=decoded_code_bits,
                    hard_code_bits=hard_decoded_code_bits,
                    crc_pass_mask=crc_pass_mask,
                    llr_signal=hard_decode_llr_signal,
                )
                reconstruction_payload_bits = reconstruction_payload_bits
                recovered_latent = self.bitstream_restorer(reconstruction_payload_bits, latent_shape, valid_bits)
                payload_faithful_full_decode = bool((not self.training) and reconstruction_payload_bits is not None)
                trust_hard_decode_latent = bool(
                    payload_faithful_full_decode
                    or
                    self.bypass_image_refiner_on_full_hard_decode
                    or (used_external_llr and self.bypass_image_refiner_on_external_hard_decode)
                    or (used_external_llr and self.strict_external_llr_hard_decode)
                )
                if not payload_faithful_full_decode:
                    recovered_latent = self._blend_full_decode_latent(
                        decoded_latent=recovered_latent,
                        feature_latent=feature_latent,
                        soft_payload_bits=soft_payload_bits,
                        fused_payload_bits=fused_payload_bits,
                        llr_signal=llr_signal,
                        crc_pass_mask=crc_pass_mask,
                        allow_feature_correction=(
                            (
                                used_external_llr
                                and self.allow_feature_correction_on_external_hard_decode
                                and not self.bypass_image_refiner_on_external_hard_decode
                            )
                            or ((not used_external_llr) and (not trust_hard_decode_latent))
                        ),
                    )
        else:
            recovered_latent = feature_latent
        # 当训练阶段只需要通信链路反传时，冻结恢复图重建分支的计算图，
        # 避免 compressor decoder / image refiner 占用过多显存。
        reconstruction_context = torch.no_grad() if detach_image_reconstruction else nullcontext()
        with reconstruction_context:
            latent_for_image = recovered_latent.detach() if detach_image_reconstruction else recovered_latent
            decoded_image = latent_decoder(latent_for_image, output_size)
            # 当外部 LLR 已经把整条 payload 精确恢复时，优先直接输出 codec 解码图，
            # 避免未收敛的图像精修头在 clean/full-decode 路径上把本已正确的重建结果修坏。
            bypass_image_refiner = bool(
                (not force_image_refiner)
                and full_bitstream_decoded
                and hard_decoded_payload_bits is not None
                and (
                    payload_faithful_full_decode
                    or
                    (
                        used_external_llr
                        and self.bypass_image_refiner_on_external_hard_decode
                        and (not self.bridge_external_llr_for_metrics_only)
                    )
                    or self.bypass_image_refiner_on_full_hard_decode
                )
            )
            if bypass_image_refiner:
                restored_image = decoded_image.clamp(0.0, 1.0)
            else:
                effective_cover_guidance_strength = self.cover_guidance_strength_in_refiner
                if used_external_llr and self.external_llr_cover_guidance_strength is not None:
                    effective_cover_guidance_strength = self.external_llr_cover_guidance_strength
                if cover_guidance_strength_override is not None:
                    effective_cover_guidance_strength = float(
                        max(0.0, min(1.0, cover_guidance_strength_override))
                    )
                restored_image = self.image_refiner(
                    restored_features=restored_features.detach() if detach_image_reconstruction else restored_features,
                    decoded_image=decoded_image,
                    cover_image=cover_image.detach() if detach_image_reconstruction else cover_image,
                    carrier_reference_image=(
                        carrier_reference_image.detach() if (detach_image_reconstruction and carrier_reference_image is not None)
                        else carrier_reference_image
                    ),
                    output_size=output_size,
                    residual_scale=self.image_refine_strength,
                    lowfreq_decode_gain=self.analog_lowfreq_decode_gain,
                    detail_decode_gain=self.analog_detail_decode_gain,
                    inverse_companding=self.analog_inverse_companding,
                    lowfreq_sender_gain=self.analog_lowfreq_sender_gain,
                    lowfreq_sender_clip=self.analog_lowfreq_sender_clip,
                    detail_sender_gain=self.analog_detail_sender_gain,
                    detail_sender_clip=self.analog_detail_sender_clip,
                    detail_hh_ratio=self.analog_detail_hh_ratio,
                    lowfreq_hh_ratio=self.analog_lowfreq_hh_ratio,
                    ll_direct_sender_gain=self.analog_ll_direct_sender_gain,
                    ll_direct_decode_mix=self.analog_ll_direct_decode_mix,
                    disable_cover_guidance=self.disable_cover_guidance_in_refiner,
                    cover_guidance_strength=effective_cover_guidance_strength,
                )
        return ReceiverOutput(
            recovered_latent=recovered_latent,
            decoded_image=decoded_image,
            received_chips=received_chips,
            block_symbol_hints=block_symbol_hints,
            despread_signal=despread,
            llr_signal=llr_signal,
            decoded_code_logits=decoded_code_logits,
            decoded_bit_logits=decoded_bit_logits,
            decoded_code_bits=decoded_code_bits,
            decoded_bits=decoded_bits,
            hard_decoded_code_bits=hard_decoded_code_bits,
            hard_decoded_info_bits=hard_decoded_info_bits,
            hard_decoded_payload_bits=hard_decoded_payload_bits,
            crc_pass_mask=crc_pass_mask,
            metric_code_bits=metric_code_bits,
            metric_payload_bits=metric_payload_bits,
            reconstruction_payload_bits=reconstruction_payload_bits,
            restored_image=restored_image,
            decoded_block_indices=decoded_block_indices,
            decoded_group_indices=active_group_indices,
            info_indices=self.hard_polar_decoder.info_indices,
            raw_external_llr_signal=raw_external_llr_signal,
            decoder_llr_signal=decoder_llr_signal,
            used_external_llr=used_external_llr,
            bridge_external_llr_for_metrics_only=self.bridge_external_llr_for_metrics_only,
        )
