from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import ConvNormAct, ResidualBlock, build_spatial_norm
from models.comm import build_grouped_chip_codes, build_low_correlation_chip_codes
from models.transmitter import TransmitterOutput
from utils.wavelets import LeGall53Wavelet2D, WaveletBands


@dataclass
class GeneratorOutput:
    generated_image: torch.Tensor
    cover_image: torch.Tensor
    carrier_anchor_image: torch.Tensor | None = None

    @property
    def prior_landscape(self) -> torch.Tensor:
        return self.generated_image

    @property
    def stego_image(self) -> torch.Tensor:
        return self.cover_image

    @property
    def carrier_base_image(self) -> torch.Tensor:
        return self.generated_image


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
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / max(dim, 1)))
    embedding = torch.zeros(length, dim, device=device, dtype=dtype)
    embedding[:, 0::2] = torch.sin(position * div_term)
    if dim > 1:
        embedding[:, 1::2] = torch.cos(position * div_term)[:, : embedding[:, 1::2].shape[1]]
    return embedding.unsqueeze(0)


class PriorGenerator(nn.Module):
    # 从噪声生成自然图像先验，用作载密图的纹理补充。
    def __init__(
        self,
        latent_dim: int,
        semantic_channels: int,
        hidden_channels: int,
        carrier_channels: int,
        template_strength: float = 0.25,
        template_mode: str = "auto",
        dataset_name: str = "",
    ) -> None:
        super().__init__()
        self.carrier_channels = int(carrier_channels)
        self.template_strength = float(max(0.0, min(1.0, template_strength)))
        self.template_mode = str(template_mode or "auto").strip().lower()
        self.dataset_name = str(dataset_name or "").strip().lower()
        self.noise_proj = nn.Linear(latent_dim, hidden_channels * 32 * 32)
        self.backbone = nn.Sequential(
            ResidualBlock(hidden_channels),
            ConvNormAct(hidden_channels, hidden_channels),
            ResidualBlock(hidden_channels),
            ConvNormAct(hidden_channels, hidden_channels),
            ResidualBlock(hidden_channels),
        )
        self.to_image = nn.Sequential(
            ConvNormAct(hidden_channels, hidden_channels, stride=2, transpose=True),
            ResidualBlock(hidden_channels),
            ConvNormAct(hidden_channels, max(hidden_channels // 2, 16), stride=2, transpose=True),
            ResidualBlock(max(hidden_channels // 2, 16)),
            ConvNormAct(max(hidden_channels // 2, 16), semantic_channels, stride=2, transpose=True),
            ResidualBlock(semantic_channels),
            ConvNormAct(semantic_channels, max(semantic_channels // 2, 16), stride=2, transpose=True),
            ResidualBlock(max(semantic_channels // 2, 16)),
            nn.Conv2d(max(semantic_channels // 2, 16), carrier_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    # 根据潜向量构造低频景观骨架，为未充分训练的先验生成器提供天空、远山和地面结构。
    def _landscape_template(
        self,
        noise: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        batch = noise.shape[0]
        height, width = int(output_size[0]), int(output_size[1])
        device, dtype = noise.device, noise.dtype
        y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype).view(1, 1, height, 1)
        x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype).view(1, 1, 1, width)
        noise_code = F.pad(noise, (0, max(0, 12 - noise.shape[1])))[:, :12]
        noise_code = torch.tanh(noise_code)
        sky = torch.stack(
            [
                0.38 + 0.08 * noise_code[:, 0],
                0.58 + 0.08 * noise_code[:, 1],
                0.78 + 0.07 * noise_code[:, 2],
            ],
            dim=1,
        ).view(batch, 3, 1, 1)
        ground = torch.stack(
            [
                0.30 + 0.09 * noise_code[:, 3],
                0.43 + 0.10 * noise_code[:, 4],
                0.26 + 0.08 * noise_code[:, 5],
            ],
            dim=1,
        ).view(batch, 3, 1, 1)
        haze = torch.stack(
            [
                0.58 + 0.05 * noise_code[:, 6],
                0.62 + 0.05 * noise_code[:, 7],
                0.64 + 0.05 * noise_code[:, 8],
            ],
            dim=1,
        ).view(batch, 3, 1, 1)
        horizon = (0.48 + 0.07 * noise_code[:, 9]).view(batch, 1, 1, 1)
        ground_mix = torch.sigmoid((y - horizon) * 18.0)
        base = sky * (1.0 - ground_mix) + ground * ground_mix
        ridge = horizon + 0.07 * torch.sin(
            (x * (4.0 + noise_code[:, 10].view(batch, 1, 1, 1)) + noise_code[:, 11].view(batch, 1, 1, 1))
            * math.pi
        )
        mountain_mask = torch.exp(-((y - ridge) ** 2) / 0.006).clamp(0.0, 1.0)
        template = torch.lerp(base, haze, 0.28 * mountain_mask)
        template = template * (1.0 - 0.10 * y).clamp(0.0, 1.0)
        if self.carrier_channels != 3:
            template = template.mean(dim=1, keepdim=True).expand(batch, self.carrier_channels, height, width)
        return template.clamp(0.0, 1.0)

    # 将噪声映射成指定分辨率的先验图像。
    def forward(self, noise: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        batch = noise.shape[0]
        features = self.noise_proj(noise).view(batch, -1, 32, 32)
        features = self.backbone(features)
        image = self.to_image(features)
        if image.shape[-2:] != output_size:
            image = F.interpolate(image, size=output_size, mode="bilinear", align_corners=False)
        if self.template_strength > 0.0:
            template = self._build_structural_template(noise, output_size)
            if template is not None:
                template = template.to(device=image.device, dtype=image.dtype)
                image = torch.lerp(image, template, self.template_strength)
        return image.clamp(0.0, 1.0)


    # 涓洪潪鏅鏁版嵁闆嗘瀯閫犻€氱敤浣庨鑹插僵鍜岀紦鍙樻ā鏉匡紝閬垮厤寮虹儓鐨勫湴骞崇嚎鍜岃繙灞卞亸缃€?
    def _generic_natural_template(
        self,
        noise: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        batch = noise.shape[0]
        height, width = int(output_size[0]), int(output_size[1])
        device, dtype = noise.device, noise.dtype
        y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype).view(1, 1, height, 1)
        x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype).view(1, 1, 1, width)
        noise_code = F.pad(noise, (0, max(0, 18 - noise.shape[1])))[:, :18]
        noise_code = torch.tanh(noise_code)
        base = torch.stack(
            [
                0.46 + 0.08 * noise_code[:, 0],
                0.49 + 0.08 * noise_code[:, 1],
                0.47 + 0.08 * noise_code[:, 2],
            ],
            dim=1,
        ).view(batch, 3, 1, 1)
        grad_x = (x - 0.5).expand(batch, 1, height, width)
        grad_y = (y - 0.5).expand(batch, 1, height, width)
        color_x = torch.stack(
            [
                0.10 * noise_code[:, 3],
                0.10 * noise_code[:, 4],
                0.10 * noise_code[:, 5],
            ],
            dim=1,
        ).view(batch, 3, 1, 1)
        color_y = torch.stack(
            [
                0.08 * noise_code[:, 6],
                0.08 * noise_code[:, 7],
                0.08 * noise_code[:, 8],
            ],
            dim=1,
        ).view(batch, 3, 1, 1)
        blob1_center_x = (0.30 + 0.25 * noise_code[:, 9]).view(batch, 1, 1, 1)
        blob1_center_y = (0.32 + 0.25 * noise_code[:, 10]).view(batch, 1, 1, 1)
        blob2_center_x = (0.68 + 0.20 * noise_code[:, 11]).view(batch, 1, 1, 1)
        blob2_center_y = (0.66 + 0.20 * noise_code[:, 12]).view(batch, 1, 1, 1)
        blob1 = torch.exp(-(((x - blob1_center_x) ** 2) / 0.05 + ((y - blob1_center_y) ** 2) / 0.04))
        blob2 = torch.exp(-(((x - blob2_center_x) ** 2) / 0.06 + ((y - blob2_center_y) ** 2) / 0.05))
        blob1_color = torch.stack(
            [
                0.10 * noise_code[:, 13],
                0.10 * noise_code[:, 14],
                0.10 * noise_code[:, 15],
            ],
            dim=1,
        ).view(batch, 3, 1, 1)
        blob2_color = torch.stack(
            [
                0.08 * noise_code[:, 16],
                0.08 * noise_code[:, 17],
                0.04 * (noise_code[:, 0] + noise_code[:, 1]),
            ],
            dim=1,
        ).view(batch, 3, 1, 1)
        template = base + color_x * grad_x + color_y * grad_y + blob1_color * blob1 + blob2_color * blob2
        template = template.clamp(0.0, 1.0)
        if self.carrier_channels != 3:
            template = template.mean(dim=1, keepdim=True).expand(batch, self.carrier_channels, height, width)
        return template

    # 鏍规嵁鏁版嵁闆嗗拰妯℃澘妯″紡閫夋嫨鍚堥€傜殑缁撴瀯鍏堥獙锛岄伩鍏嶉潪鏅鏁版嵁闆嗚寮哄埗鎷夊悜鏅鍒嗗竷銆?
    def _build_structural_template(
        self,
        noise: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor | None:
        mode = self.template_mode
        if mode == "auto":
            mode = "landscape" if self.dataset_name == "landscape" else "generic"
        if mode == "none":
            return None
        if mode == "landscape":
            return self._landscape_template(noise, output_size)
        if mode == "generic":
            return self._generic_natural_template(noise, output_size)
        raise ValueError(f"Unsupported template_mode: {self.template_mode}")


class GroupAwarePayloadEmbedder(nn.Module):
    # 将分组 CDMA 序列按 code/chip 结构映射到高频小波残差，尽量保留接收端可相关解扩的结构。
    def __init__(
        self,
        image_channels: int,
        hidden_channels: int,
        residual_gain: float,
        code_length: int,
        chips_per_symbol: int,
        group_size: int = 4,
        chip_seed: int = 20260520,
        direct_template_gain: float = 0.18,
        residual_delta_clip: float = 0.018,
        chroma_residual_ratio: float = 0.08,
        local_energy_kernel: int = 15,
        local_energy_limit: float = 0.010,
        analog_residual_gain: float = 0.0,
        analog_residual_clip: float = 0.080,
        analog_residual_chroma_ratio: float = 0.35,
        analog_lowfreq_gain: float = 0.0,
        analog_lowfreq_clip: float = 0.180,
        analog_lowfreq_chroma_ratio: float = 0.55,
    ) -> None:
        super().__init__()
        code_grid_size = int(math.sqrt(code_length))
        if code_grid_size * code_grid_size != code_length:
            raise ValueError("GroupAwarePayloadEmbedder expects code_length to be a square number.")
        payload_channels = max(24, min(hidden_channels // 4, 96))
        self.payload_channels = payload_channels
        self.code_length = code_length
        self.code_grid_size = code_grid_size
        self.chips_per_symbol = chips_per_symbol
        self.group_size = max(1, int(group_size))
        self.chip_seed = int(chip_seed)
        self.chip_encoder = nn.Sequential(
            nn.Conv2d(chips_per_symbol, payload_channels, kernel_size=1, bias=False),
            build_spatial_norm(payload_channels),
            nn.SiLU(),
            ResidualBlock(payload_channels),
            nn.Conv2d(payload_channels, payload_channels, kernel_size=3, padding=1),
            build_spatial_norm(payload_channels),
            nn.SiLU(),
        )
        self.symbol_summary_encoder = nn.Sequential(
            nn.Conv2d(2, payload_channels, kernel_size=3, padding=1, bias=False),
            build_spatial_norm(payload_channels),
            nn.SiLU(),
            ResidualBlock(payload_channels),
        )
        self.group_refine = nn.Sequential(
            nn.Conv2d(payload_channels, payload_channels, kernel_size=3, padding=1),
            build_spatial_norm(payload_channels),
            nn.SiLU(),
            ResidualBlock(payload_channels),
            nn.Conv2d(payload_channels, payload_channels, kernel_size=3, padding=1),
            build_spatial_norm(payload_channels),
            nn.SiLU(),
        )
        self.payload_refine = nn.Sequential(
            ResidualBlock(payload_channels),
            nn.Conv2d(payload_channels, payload_channels, kernel_size=3, padding=1, bias=False),
            build_spatial_norm(payload_channels),
            nn.SiLU(),
        )
        self.lh_head = self._build_band_head(image_channels)
        self.hl_head = self._build_band_head(image_channels)
        self.hh_head = self._build_band_head(image_channels)
        self.image_channels = image_channels
        self.residual_gain = residual_gain
        self.direct_template_gain = float(direct_template_gain)
        self.residual_delta_clip = float(max(1e-4, residual_delta_clip))
        self.chroma_residual_ratio = float(max(0.0, min(1.0, chroma_residual_ratio)))
        self.local_energy_kernel = max(1, int(local_energy_kernel) | 1)
        self.local_energy_limit = float(max(1e-4, local_energy_limit))
        self.register_buffer(
            "chip_projection",
            self._build_chip_projection(image_channels, chips_per_symbol),
        )
        self.register_buffer(
            "chip_codes",
            build_low_correlation_chip_codes(code_length, chips_per_symbol, seed=self.chip_seed),
        )
        symbol_projection, symbol_unprojection = self._build_symbol_projection(image_channels, self.group_size)
        self.register_buffer("symbol_projection", symbol_projection)
        self.register_buffer("symbol_unprojection", symbol_unprojection)

    # 构造固定的 chip 到图像通道投影，给发送端留一条可被 matched filter 读取的显式通路。
    def _build_chip_projection(self, image_channels: int, chips_per_symbol: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(20260531)
        projection = torch.randn(image_channels, chips_per_symbol, generator=generator)
        projection = projection - projection.mean(dim=1, keepdim=True)
        return F.normalize(projection, p=2, dim=1)

    # 构造三个高频子带联合使用的符号投影与伪逆，保证组内码包在固定子空间中可分离。
    def _build_symbol_projection(self, image_channels: int, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device="cpu").manual_seed(20260531 + int(group_size) * 17)
        flat_projection = torch.randn(3 * image_channels, group_size, generator=generator)
        flat_projection = flat_projection - flat_projection.mean(dim=0, keepdim=True)
        flat_projection = F.normalize(flat_projection, p=2, dim=0)
        symbol_unprojection = torch.linalg.pinv(flat_projection)
        return flat_projection.view(3, image_channels, group_size).contiguous(), symbol_unprojection.contiguous()

    # 构造与发送端 CDMA 完全一致的组内 block 码片模板。
    def _build_group_block_chip_templates(
        self,
        group_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        num_blocks = max(1, int(group_count) * self.group_size)
        block_codes = build_grouped_chip_codes(
            num_blocks,
            self.chips_per_symbol,
            group_size=self.group_size,
            seed=self.chip_seed,
            device=device,
            dtype=dtype,
        ).view(group_count, self.group_size, self.chips_per_symbol)
        chip_codes = self.chip_codes.to(device=device, dtype=dtype)
        return F.normalize(
            chip_codes.unsqueeze(0).unsqueeze(0) * block_codes.unsqueeze(2),
            p=2,
            dim=-1,
        )

    # 将分组 CDMA 短序列解成组内 block 符号图，作为高频固定通信子空间的充分统计量。
    def _spread_to_group_symbols(self, spread: torch.Tensor) -> torch.Tensor:
        group_spread = self._reshape_group_spread(spread)
        batch, group_count, _, grid_h, grid_w = group_spread.shape
        group_chips = group_spread.view(batch, group_count, self.chips_per_symbol, self.code_length)
        group_chips = group_chips.permute(0, 1, 3, 2).contiguous()
        group_chips = group_chips - group_chips.mean(dim=-1, keepdim=True)
        chip_norm = group_chips.float().norm(p=2, dim=-1, keepdim=True).clamp_min(1e-4).to(group_chips.dtype)
        group_chips = group_chips / chip_norm
        chip_templates = self._build_group_block_chip_templates(
            group_count=group_count,
            device=group_spread.device,
            dtype=group_spread.dtype,
        )
        group_symbols = torch.einsum("bgkc,gskc->bgsk", group_chips, chip_templates)
        return group_symbols.view(batch, group_count, self.group_size, grid_h, grid_w).contiguous()

    # 构建单个高频子带的 payload 注入头，让每个子带按自身纹理自适应承载结构化扰动。
    def _build_band_head(self, image_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(payload_channels := self.payload_channels + image_channels, self.payload_channels, kernel_size=3, padding=1, bias=False),
            build_spatial_norm(self.payload_channels),
            nn.SiLU(),
            ResidualBlock(self.payload_channels),
            nn.Conv2d(self.payload_channels, image_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    # 将发送端 spread 还原成 [group, chip, code_h, code_w] 结构，避免在嵌入前就破坏可解扩信息。
    def _reshape_group_spread(self, spread: torch.Tensor) -> torch.Tensor:
        expected_length = self.code_length * self.chips_per_symbol
        if spread.dim() == 2:
            if spread.shape[1] % expected_length != 0:
                raise ValueError(
                    "Ungrouped spread length must be divisible by code_length * chips_per_symbol. "
                    f"got {spread.shape[1]}, expected multiple of {expected_length}."
                )
            spread = spread.view(spread.shape[0], -1, expected_length)
        if spread.dim() != 3:
            raise ValueError("GroupAwarePayloadEmbedder expects spread with shape [B, G, L] or [B, L].")
        if spread.shape[-1] != expected_length:
            raise ValueError(
                "Grouped spread length does not match configured code_length * chips_per_symbol. "
                f"got {spread.shape[-1]}, expected {expected_length}."
            )
        safe_spread = torch.nan_to_num(spread, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
        safe_spread = safe_spread.view(
            safe_spread.shape[0],
            safe_spread.shape[1],
            self.code_length,
            self.chips_per_symbol,
        )
        safe_spread = safe_spread.permute(0, 1, 3, 2).contiguous()
        safe_spread = safe_spread.view(
            safe_spread.shape[0],
            safe_spread.shape[1],
            self.chips_per_symbol,
            self.code_grid_size,
            self.code_grid_size,
        )
        scale = safe_spread.float().abs().mean(dim=(2, 3, 4), keepdim=True).clamp_min(1e-4).to(safe_spread.dtype)
        return safe_spread / scale

    # 将按组的码片网格编码成保留 code/chip 结构的二维 payload 特征图。
    def _spread_to_group_feature(self, spread: torch.Tensor, band_size: tuple[int, int]) -> torch.Tensor:
        group_spread = self._reshape_group_spread(spread)
        batch, group_count, _, grid_h, grid_w = group_spread.shape
        flat_group_spread = group_spread.view(batch * group_count, self.chips_per_symbol, grid_h, grid_w)
        chip_feature = self.chip_encoder(flat_group_spread)
        symbol_summary = torch.stack(
            [
                group_spread.mean(dim=2),
                group_spread.abs().mean(dim=2),
            ],
            dim=2,
        ).view(batch * group_count, 2, grid_h, grid_w)
        summary_feature = self.symbol_summary_encoder(symbol_summary)
        group_feature = chip_feature + 0.35 * summary_feature
        group_feature = self.group_refine(group_feature).view(batch, group_count, self.payload_channels, grid_h, grid_w)
        summary_feature = summary_feature.view(batch, group_count, self.payload_channels, grid_h, grid_w)
        position_embedding = build_sinusoidal_position_embedding(
            group_count,
            self.payload_channels,
            group_feature.device,
            group_feature.dtype,
        ).view(1, group_count, self.payload_channels, 1, 1)
        group_feature = group_feature + 0.20 * summary_feature + position_embedding
        if group_feature.shape[-2:] != band_size:
            group_feature = F.interpolate(
                group_feature.view(batch * group_count, self.payload_channels, grid_h, grid_w),
                size=band_size,
                mode="bilinear",
                align_corners=False,
            ).view(batch, group_count, self.payload_channels, band_size[0], band_size[1])
        band_height, band_width = band_size
        groups_per_band = max(1, int(group_count))
        groups_per_row = max(1, math.ceil(math.sqrt(groups_per_band)))
        groups_per_col = max(1, math.ceil(groups_per_band / groups_per_row))
        tile_height = max(1, math.ceil(band_height / groups_per_col))
        tile_width = max(1, math.ceil(band_width / groups_per_row))
        payload_feature = group_feature.new_zeros(batch, 3, self.payload_channels, band_height, band_width)
        payload_weight = group_feature.new_zeros(1, 3, 1, band_height, band_width)
        for group_index in range(group_count):
            slot_index = group_index
            row_index = slot_index // groups_per_row
            col_index = slot_index % groups_per_row
            row_start = row_index * tile_height
            col_start = col_index * tile_width
            row_end = min(row_start + tile_height, band_height)
            col_end = min(col_start + tile_width, band_width)
            tile_feature = F.interpolate(
                group_feature[:, group_index],
                size=(row_end - row_start, col_end - col_start),
                mode="bilinear",
                align_corners=False,
            )
            for band_index in range(3):
                payload_feature[:, band_index, :, row_start:row_end, col_start:col_end] += tile_feature
                payload_weight[:, band_index, :, row_start:row_end, col_start:col_end] += 1.0
        payload_feature = payload_feature / payload_weight.clamp_min(1.0)
        payload_feature = payload_feature + 0.05 * group_feature.mean(dim=1, keepdim=False).unsqueeze(1)
        payload_feature = self.payload_refine(
            payload_feature.view(batch * 3, self.payload_channels, band_height, band_width)
        ).view(batch, self.payload_channels * 3, band_height, band_width)
        return payload_feature

    # 计算 group 到三个高频子带 tile 的确定性布局，保证发送端、oracle 和接收端监督一致。
    def _group_tile_layout(
        self,
        group_count: int,
        band_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        band_height, band_width = band_size
        groups_per_band = max(1, int(group_count))
        groups_per_row = max(1, math.ceil(math.sqrt(groups_per_band)))
        groups_per_col = max(1, math.ceil(groups_per_band / groups_per_row))
        tile_height = max(1, math.ceil(band_height / groups_per_col))
        tile_width = max(1, math.ceil(band_width / groups_per_row))
        return groups_per_row, groups_per_col, tile_height, tile_width

    # 把 CDMA spread 通过固定投影直接变成高频模板，使残差里存在可被 oracle 读取的结构化信号。
    def spread_to_band_templates(self, spread: torch.Tensor, band_size: tuple[int, int]) -> torch.Tensor:
        group_symbols = self._spread_to_group_symbols(spread)
        batch, group_count, _, grid_h, grid_w = group_symbols.shape
        band_height, band_width = band_size
        groups_per_row, _, tile_height, tile_width = self._group_tile_layout(group_count, band_size)
        templates = group_symbols.new_zeros(batch, 3, self.image_channels, band_height, band_width)
        weights = group_symbols.new_zeros(1, 3, 1, band_height, band_width)
        projection = self.symbol_projection.to(device=group_symbols.device, dtype=group_symbols.dtype)
        for group_index in range(group_count):
            slot_index = group_index
            row_index = slot_index // groups_per_row
            col_index = slot_index % groups_per_row
            row_start = row_index * tile_height
            col_start = col_index * tile_width
            row_end = min(row_start + tile_height, band_height)
            col_end = min(col_start + tile_width, band_width)
            for band_index in range(3):
                group_rgb = torch.einsum(
                    "bshw,rs->brhw",
                    group_symbols[:, group_index],
                    projection[band_index],
                ) / math.sqrt(float(self.group_size))
                group_rgb = F.interpolate(
                    group_rgb,
                    size=(row_end - row_start, col_end - col_start),
                    mode="bilinear",
                    align_corners=False,
                )
                templates[:, band_index, :, row_start:row_end, col_start:col_end] += group_rgb
                weights[:, band_index, :, row_start:row_end, col_start:col_end] += 1.0
        return templates / weights.clamp_min(1.0)

    # 用固定投影从高频残差 tile 中反推 chip，作为发送端 oracle 预训练和诊断指标。
    def extract_oracle_chips_from_bands(
        self,
        lh_residual: torch.Tensor,
        hl_residual: torch.Tensor,
        hh_residual: torch.Tensor,
        group_count: int,
        return_band_symbols: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        bands = (lh_residual, hl_residual, hh_residual)
        band_height, band_width = lh_residual.shape[-2:]
        groups_per_row, _, tile_height, tile_width = self._group_tile_layout(group_count, (band_height, band_width))
        projection = self.symbol_projection.to(device=lh_residual.device, dtype=lh_residual.dtype)
        unprojection = self.symbol_unprojection.to(device=lh_residual.device, dtype=lh_residual.dtype)
        chip_templates = self._build_group_block_chip_templates(
            group_count=group_count,
            device=lh_residual.device,
            dtype=lh_residual.dtype,
        )
        group_chips: list[torch.Tensor] = []
        group_band_symbols: list[torch.Tensor] = []
        for group_index in range(group_count):
            slot_index = group_index
            row_index = slot_index // groups_per_row
            col_index = slot_index % groups_per_row
            row_start = row_index * tile_height
            col_start = col_index * tile_width
            row_end = min(row_start + tile_height, band_height)
            col_end = min(col_start + tile_width, band_width)
            resized_tiles = []
            per_band_symbols = []
            for band_index, band in enumerate(bands):
                tile = band[..., row_start:row_end, col_start:col_end]
                tile = F.interpolate(
                    tile,
                    size=(self.code_grid_size, self.code_grid_size),
                    mode="bilinear",
                    align_corners=False,
                )
                resized_tiles.append(tile)
                band_symbol = torch.einsum(
                    "brhw,rs->bshw",
                    tile,
                    projection[band_index],
                ) * math.sqrt(float(self.group_size))
                per_band_symbols.append(band_symbol)
            stacked_tiles = torch.cat(resized_tiles, dim=1)
            symbols = torch.einsum(
                "bqhw,sq->bshw",
                stacked_tiles,
                unprojection,
            ) * math.sqrt(float(self.group_size))
            symbols = symbols.view(symbols.shape[0], self.group_size, self.code_length)
            chips = torch.einsum(
                "bsk,skc->bkc",
                symbols,
                chip_templates[group_index],
            )
            scale = chips.float().abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-4).to(chips.dtype)
            group_chips.append(chips / scale)
            if return_band_symbols:
                band_symbols = torch.stack(per_band_symbols, dim=1)
                band_symbols = band_symbols.view(
                    band_symbols.shape[0],
                    3,
                    self.group_size,
                    self.code_length,
                )
                group_band_symbols.append(band_symbols)
        fused_chips = torch.stack(group_chips, dim=1).contiguous()
        if not return_band_symbols:
            return fused_chips
        return fused_chips, torch.stack(group_band_symbols, dim=2).contiguous()

    # 根据高频纹理强弱生成自适应掩码，在复杂区域注入更强 payload。
    def _band_texture_mask(self, band: torch.Tensor) -> torch.Tensor:
        energy = band.abs().mean(dim=1, keepdim=True)
        centered = energy - energy.mean(dim=(-2, -1), keepdim=True)
        normalized = centered / energy.float().std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-4).to(
            energy.dtype
        )
        return 0.20 + 0.55 * torch.sigmoid(normalized)

    def _suppress_chroma_spikes(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.shape[1] < 3:
            return residual
        neutral = residual.mean(dim=1, keepdim=True)
        chroma = residual - neutral
        return neutral + self.chroma_residual_ratio * chroma

    def _limit_local_energy(self, residual: torch.Tensor) -> torch.Tensor:
        if self.local_energy_kernel <= 1:
            return residual
        local_energy = residual.float().abs().mean(dim=1, keepdim=True)
        local_energy = F.avg_pool2d(
            local_energy,
            kernel_size=self.local_energy_kernel,
            stride=1,
            padding=self.local_energy_kernel // 2,
        )
        scale = (self.local_energy_limit / local_energy.clamp_min(1e-6)).clamp(max=1.0)
        return residual * scale.to(device=residual.device, dtype=residual.dtype)

    def _payload_delta(
        self,
        residual: torch.Tensor,
        template: torch.Tensor,
        texture_mask: torch.Tensor,
        spread_strength: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.nan_to_num(
            residual + self.direct_template_gain * template,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        combined = self._suppress_chroma_spikes(combined)
        delta = self.residual_gain * spread_strength * texture_mask * combined
        delta = torch.tanh(delta / self.residual_delta_clip) * self.residual_delta_clip
        return self._limit_local_energy(delta)

    # 将结构化 payload 分别注入 LH、HL、HH 子带，保持 LL 低频不承载信息。
    def forward(self, spread: torch.Tensor, generated_bands: WaveletBands) -> WaveletBands:
        payload_feature = self._spread_to_group_feature(spread, generated_bands.lh.shape[-2:])
        band_templates = self.spread_to_band_templates(spread, generated_bands.lh.shape[-2:])
        lh_feature, hl_feature, hh_feature = payload_feature.chunk(3, dim=1)
        lh_template, hl_template, hh_template = band_templates.unbind(dim=1)
        spread_strength = spread.flatten(start_dim=1).float().abs().mean(dim=1, keepdim=True).view(-1, 1, 1, 1)
        spread_strength = torch.tanh(spread_strength / 4.0)
        lh_mask = self._band_texture_mask(generated_bands.lh)
        hl_mask = self._band_texture_mask(generated_bands.hl)
        hh_mask = self._band_texture_mask(generated_bands.hh)
        lh_residual = self.lh_head(torch.cat([lh_feature, generated_bands.lh], dim=1))
        hl_residual = self.hl_head(torch.cat([hl_feature, generated_bands.hl], dim=1))
        hh_residual = self.hh_head(torch.cat([hh_feature, generated_bands.hh], dim=1))
        lh = generated_bands.lh + self._payload_delta(lh_residual, lh_template, lh_mask, spread_strength)
        hl = generated_bands.hl + self._payload_delta(hl_residual, hl_template, hl_mask, spread_strength)
        hh = generated_bands.hh + self._payload_delta(hh_residual, hh_template, hh_mask, spread_strength)
        return WaveletBands(
            ll=generated_bands.ll,
            lh=lh,
            hl=hl,
            hh=hh,
            original_size=generated_bands.original_size,
            padded_size=generated_bands.padded_size,
        )


class PolarGenerator(nn.Module):
    # 生成独立的景观先验底图，并在高频子带中嵌入 payload。
    def __init__(
        self,
        latent_dim: int,
        semantic_channels: int,
        hidden_channels: int,
        carrier_channels: int,
        code_length: int,
        chips_per_symbol: int,
        group_size: int = 4,
        chip_seed: int = 20260520,
        residual_gain: float = 0.1,
        prior_mix: float = 0.05,
        preview_bridge_strength: float | None = None,
        preview_guidance_mix: float = 0.15,
        preview_guidance_pool: int = 16,
        template_strength: float = 0.25,
        direct_template_gain: float = 0.18,
        residual_delta_clip: float = 0.018,
        chroma_residual_ratio: float = 0.08,
        local_energy_kernel: int = 15,
        local_energy_limit: float = 0.010,
        analog_residual_gain: float = 0.0,
        analog_residual_clip: float = 0.080,
        analog_residual_chroma_ratio: float = 0.35,
        analog_lowfreq_gain: float = 0.0,
        analog_lowfreq_clip: float = 0.180,
        analog_lowfreq_chroma_ratio: float = 0.55,
        analog_detail_hh_ratio: float = 1.0,
        analog_lowfreq_hh_ratio: float = 1.0,
        analog_ll_direct_gain: float = 0.0,
        analog_injection_target: str = "generated",
        source_bridge_strength: float = 1.0,
        external_carrier_source_bridge: bool = False,
        external_carrier_blend: float = 0.0,
        external_carrier_lowfreq_only: bool = True,
        preserve_source_bridge_with_external_carrier: bool = True,
        carrier_first_mode: bool = False,
        carrier_style_residual_gain: float = 0.0,
        carrier_style_color_mix: float = 0.0,
        carrier_style_lowfreq_kernel: int = 9,
        template_mode: str = "auto",
        dataset_name: str = "",
    ) -> None:
        super().__init__()
        self.prior_generator = PriorGenerator(
            latent_dim=latent_dim,
            semantic_channels=semantic_channels,
            hidden_channels=hidden_channels,
            carrier_channels=carrier_channels,
            template_strength=template_strength,
            template_mode=template_mode,
            dataset_name=dataset_name,
        )
        self.wavelet = LeGall53Wavelet2D()
        self.payload_embedder = GroupAwarePayloadEmbedder(
            image_channels=carrier_channels,
            hidden_channels=hidden_channels,
            residual_gain=residual_gain,
            code_length=code_length,
            chips_per_symbol=chips_per_symbol,
            group_size=group_size,
            chip_seed=chip_seed,
            direct_template_gain=direct_template_gain,
            residual_delta_clip=residual_delta_clip,
            chroma_residual_ratio=chroma_residual_ratio,
            local_energy_kernel=local_energy_kernel,
            local_energy_limit=local_energy_limit,
        )
        self.prior_mix = float(prior_mix)
        legacy_preview_bridge_strength = 1.0 - self.prior_mix
        if preview_bridge_strength is None:
            preview_bridge_strength = legacy_preview_bridge_strength
        self.preview_bridge_strength = float(max(0.0, min(1.0, preview_bridge_strength)))
        self.preview_guidance_mix = float(max(0.0, min(1.0, preview_guidance_mix)))
        self.preview_guidance_pool = max(1, int(preview_guidance_pool))
        self.analog_residual_gain = float(max(0.0, min(1.0, analog_residual_gain)))
        self.analog_residual_clip = float(max(1e-4, analog_residual_clip))
        self.analog_residual_chroma_ratio = float(max(0.0, min(1.0, analog_residual_chroma_ratio)))
        self.analog_lowfreq_gain = float(max(0.0, min(1.0, analog_lowfreq_gain)))
        self.analog_lowfreq_clip = float(max(1e-4, analog_lowfreq_clip))
        self.analog_lowfreq_chroma_ratio = float(max(0.0, min(1.0, analog_lowfreq_chroma_ratio)))
        self.analog_detail_hh_ratio = float(max(0.0, min(1.0, analog_detail_hh_ratio)))
        self.analog_lowfreq_hh_ratio = float(max(0.0, min(1.0, analog_lowfreq_hh_ratio)))
        self.analog_ll_direct_gain = float(max(0.0, min(1.0, analog_ll_direct_gain)))
        normalized_analog_target = str(analog_injection_target or "generated").strip().lower()
        if normalized_analog_target not in {"generated", "cover"}:
            normalized_analog_target = "generated"
        self.analog_injection_target = normalized_analog_target
        self.source_bridge_strength = float(max(0.0, min(1.0, source_bridge_strength)))
        self.external_carrier_source_bridge = bool(external_carrier_source_bridge)
        self.external_carrier_blend = float(max(0.0, min(1.0, external_carrier_blend)))
        self.external_carrier_lowfreq_only = bool(external_carrier_lowfreq_only)
        self.preserve_source_bridge_with_external_carrier = bool(
            preserve_source_bridge_with_external_carrier
        )
        self.carrier_first_mode = bool(carrier_first_mode)
        self.carrier_style_residual_gain = float(max(0.0, min(0.10, carrier_style_residual_gain)))
        self.carrier_style_color_mix = float(max(0.0, min(0.25, carrier_style_color_mix)))
        kernel_size = max(1, int(carrier_style_lowfreq_kernel))
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.carrier_style_lowfreq_kernel = kernel_size

    # 将外部自然载体只作为低频外观引导融入底图，避免整图硬插值直接破坏恢复主链的高频与桥接结构。
    def _blend_external_carrier_guidance(
        self,
        prior_image: torch.Tensor,
        carrier_image: torch.Tensor,
    ) -> torch.Tensor:
        blend = float(max(0.0, min(1.0, self.external_carrier_blend)))
        if blend <= 0.0:
            return prior_image
        if not self.external_carrier_lowfreq_only:
            return torch.lerp(prior_image, carrier_image, blend).clamp(0.0, 1.0)
        prior_bands = self.wavelet.dwt(prior_image)
        carrier_bands = self.wavelet.dwt(carrier_image)
        fused_bands = WaveletBands(
            ll=torch.lerp(prior_bands.ll, carrier_bands.ll, blend),
            lh=prior_bands.lh,
            hl=prior_bands.hl,
            hh=prior_bands.hh,
            original_size=prior_bands.original_size,
            padded_size=prior_bands.padded_size,
        )
        return self.wavelet.idwt(fused_bands).clamp(0.0, 1.0)

    # 生成三组固定正交感较强的高频载波，用于把低频残差搬移到 LH/HL/HH 子带。
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
            carrier_lh.to(dtype=reference.dtype),
            carrier_hl.to(dtype=reference.dtype),
            carrier_hh.to(dtype=reference.dtype),
        )

    # 将压缩预览图转成以低频信息为主的引导图，减少生成器直接继承原图语义。
    def _build_preview_guidance(self, preview_image: torch.Tensor) -> torch.Tensor:
        if self.preview_guidance_pool <= 1:
            lowfreq_preview = preview_image
        else:
            kernel_h = min(self.preview_guidance_pool, int(preview_image.shape[-2]))
            kernel_w = min(self.preview_guidance_pool, int(preview_image.shape[-1]))
            if kernel_h <= 1 or kernel_w <= 1:
                lowfreq_preview = preview_image
            else:
                lowfreq_preview = F.avg_pool2d(
                    preview_image,
                    kernel_size=(kernel_h, kernel_w),
                    stride=(kernel_h, kernel_w),
                )
                lowfreq_preview = F.interpolate(
                    lowfreq_preview,
                    size=preview_image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
        lowfreq_preview = lowfreq_preview.to(device=preview_image.device, dtype=preview_image.dtype)
        return torch.lerp(lowfreq_preview, preview_image, self.preview_guidance_mix)

    # 对齐外部载体尺寸与 dtype，保证 carrier-first 路径直接以自然图为主干。
    def _prepare_carrier_image(
        self,
        carrier_image: torch.Tensor,
        reference: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        aligned_carrier = carrier_image.to(device=reference.device, dtype=reference.dtype).clamp(0.0, 1.0)
        if aligned_carrier.shape[-2:] != target_size:
            aligned_carrier = F.interpolate(
                aligned_carrier,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
        return aligned_carrier

    # 构造恢复端需要的模拟残差子带，后续可注入生成底图或最终载密图。
    def _build_analog_residual_delta(
        self,
        base_bands: WaveletBands,
        preview_image: torch.Tensor,
        source_image: torch.Tensor | None,
    ) -> WaveletBands | None:
        preview_residual_bridge_enabled = self.preview_bridge_strength > 0.005 or self.carrier_first_mode
        if (
            source_image is None
            or self.source_bridge_strength <= 0.0
            or not preview_residual_bridge_enabled
            or (self.analog_residual_gain <= 0.0 and self.analog_lowfreq_gain <= 0.0)
        ):
            return None
        safe_source = source_image.to(device=base_bands.ll.device, dtype=base_bands.ll.dtype).clamp(0.0, 1.0)
        safe_preview = preview_image.to(device=base_bands.ll.device, dtype=base_bands.ll.dtype).clamp(0.0, 1.0)
        source_bands = self.wavelet.dwt(safe_source)
        preview_bands = self.wavelet.dwt(safe_preview)
        bridge_strength = base_bands.ll.new_tensor(self.source_bridge_strength)

        def limited_residual(source_band: torch.Tensor, preview_band: torch.Tensor) -> torch.Tensor:
            residual = source_band - preview_band
            if residual.shape[1] >= 3:
                neutral = residual.mean(dim=1, keepdim=True)
                residual = neutral + self.analog_residual_chroma_ratio * (residual - neutral)
            clipped = torch.tanh(residual / self.analog_residual_clip) * self.analog_residual_clip
            return bridge_strength * self.analog_residual_gain * clipped

        def limited_lowfreq_residual() -> torch.Tensor:
            residual = source_bands.ll - preview_bands.ll
            if residual.shape[1] >= 3:
                neutral = residual.mean(dim=1, keepdim=True)
                residual = neutral + self.analog_lowfreq_chroma_ratio * (residual - neutral)
            clipped = torch.tanh(residual / self.analog_lowfreq_clip) * self.analog_lowfreq_clip
            return bridge_strength * self.analog_lowfreq_gain * clipped

        lowfreq_payload = torch.zeros_like(base_bands.ll)
        if self.analog_lowfreq_gain > 0.0:
            lowfreq_payload = limited_lowfreq_residual()
        carrier_lh, carrier_hl, carrier_hh = self._build_analog_carriers(base_bands.lh)
        return WaveletBands(
            ll=bridge_strength * self.analog_ll_direct_gain * lowfreq_payload,
            lh=limited_residual(source_bands.lh, preview_bands.lh) + 0.60 * lowfreq_payload * carrier_lh,
            hl=limited_residual(source_bands.hl, preview_bands.hl) + 0.60 * lowfreq_payload * carrier_hl,
            hh=(
                self.analog_detail_hh_ratio * limited_residual(source_bands.hh, preview_bands.hh)
                + 0.35 * self.analog_lowfreq_hh_ratio * lowfreq_payload * carrier_hh
            ),
            original_size=base_bands.original_size,
            padded_size=base_bands.padded_size,
        )

    # 将模拟残差叠加到指定的小波子带，避免恢复补偿与生成底图外观强绑定。
    def _add_analog_residual_to_bands(
        self,
        base_bands: WaveletBands,
        preview_image: torch.Tensor,
        source_image: torch.Tensor | None,
    ) -> WaveletBands:
        residual_delta = self._build_analog_residual_delta(
            base_bands=base_bands,
            preview_image=preview_image,
            source_image=source_image,
        )
        if residual_delta is None:
            return base_bands
        return WaveletBands(
            ll=base_bands.ll + residual_delta.ll,
            lh=base_bands.lh + residual_delta.lh,
            hl=base_bands.hl + residual_delta.hl,
            hh=base_bands.hh + residual_delta.hh,
            original_size=base_bands.original_size,
            padded_size=base_bands.padded_size,
        )

    # 以真实自然载体为主底图，仅叠加轻微的生成式风格残差，避免中间图继承秘密图外观。
    def _compose_carrier_first_base(
        self,
        prior_image: torch.Tensor,
        carrier_image: torch.Tensor,
    ) -> torch.Tensor:
        if not self.carrier_first_mode:
            if self.external_carrier_blend > 0.0:
                return self._blend_external_carrier_guidance(prior_image, carrier_image)
            return prior_image
        styled_carrier = carrier_image
        if self.carrier_style_lowfreq_kernel > 1:
            prior_lowfreq = F.avg_pool2d(
                prior_image,
                kernel_size=self.carrier_style_lowfreq_kernel,
                stride=1,
                padding=self.carrier_style_lowfreq_kernel // 2,
            )
            carrier_lowfreq = F.avg_pool2d(
                carrier_image,
                kernel_size=self.carrier_style_lowfreq_kernel,
                stride=1,
                padding=self.carrier_style_lowfreq_kernel // 2,
            )
        else:
            prior_lowfreq = prior_image
            carrier_lowfreq = carrier_image
        style_strength = styled_carrier.new_tensor(1.0)
        if self.carrier_style_color_mix > 0.0:
            styled_carrier = styled_carrier + (
                style_strength * self.carrier_style_color_mix * (prior_lowfreq - carrier_lowfreq)
            )
        if self.carrier_style_residual_gain > 0.0:
            prior_highfreq = prior_image - prior_lowfreq
            if prior_highfreq.shape[1] >= 3:
                neutral = prior_highfreq.mean(dim=1, keepdim=True)
                prior_highfreq = neutral + 0.20 * (prior_highfreq - neutral)
            prior_highfreq = torch.tanh(prior_highfreq / 0.12) * 0.12
            styled_carrier = styled_carrier + style_strength * self.carrier_style_residual_gain * prior_highfreq
        return styled_carrier.clamp(0.0, 1.0)

    # 将原图相对离散预览图的高频残差注入载体高频子带，为接收端提供连续细节补偿。
    def _inject_analog_residual(
        self,
        generated_image: torch.Tensor,
        preview_image: torch.Tensor,
        source_image: torch.Tensor | None,
    ) -> torch.Tensor:
        generated_bands = self.wavelet.dwt(generated_image)
        fused_bands = self._add_analog_residual_to_bands(
            base_bands=generated_bands,
            preview_image=preview_image,
            source_image=source_image,
        )
        return self.wavelet.idwt(fused_bands).clamp(0.0, 1.0)

    # 使用景观先验与压缩预览图生成载体底图，再在高频子带中嵌入 payload。
    def forward(
        self,
        transmitter_output: TransmitterOutput,
        noise: torch.Tensor,
        embed_payload: bool = True,
        source_image: torch.Tensor | None = None,
        carrier_image: torch.Tensor | None = None,
    ) -> GeneratorOutput:
        target_size = transmitter_output.compression.original_size
        preview_image = transmitter_output.compression.reconstructed_image
        prior_image = self.prior_generator(noise, target_size)
        generated_image = prior_image.to(device=preview_image.device, dtype=preview_image.dtype).clamp(0.0, 1.0)
        aligned_carrier_image = None
        if carrier_image is not None:
            aligned_carrier_image = self._prepare_carrier_image(
                carrier_image=carrier_image,
                reference=generated_image,
                target_size=target_size,
            )
        if aligned_carrier_image is not None and self.carrier_first_mode:
            generated_image = self._compose_carrier_first_base(
                prior_image=generated_image,
                carrier_image=aligned_carrier_image,
            )
        else:
            if self.preview_bridge_strength > 0.0:
                preview_guidance = self._build_preview_guidance(preview_image)
                generated_image = torch.lerp(
                    generated_image,
                    preview_guidance,
                    self.preview_bridge_strength,
                ).clamp(0.0, 1.0)
            if aligned_carrier_image is not None and self.external_carrier_blend > 0.0:
                generated_image = self._blend_external_carrier_guidance(
                    generated_image,
                    aligned_carrier_image,
                )
        should_inject_source_bridge = (
            self.source_bridge_strength > 0.0
            and (
                aligned_carrier_image is None
                or self.external_carrier_source_bridge
                or self.carrier_first_mode
                or self.preserve_source_bridge_with_external_carrier
            )
        )
        inject_analog_on_cover = should_inject_source_bridge and self.analog_injection_target == "cover"
        if should_inject_source_bridge and not inject_analog_on_cover:
            generated_image = self._inject_analog_residual(
                generated_image=generated_image,
                preview_image=preview_image,
                source_image=source_image,
            )
        generated_bands = self.wavelet.dwt(generated_image)
        if not embed_payload:
            cover_bands = generated_bands
            if inject_analog_on_cover:
                cover_bands = self._add_analog_residual_to_bands(
                    base_bands=generated_bands,
                    preview_image=preview_image,
                    source_image=source_image,
                )
            cover_bands.original_size = target_size
            cover_bands.padded_size = generated_bands.padded_size
            cover_image = self.wavelet.idwt(cover_bands).clamp(0.0, 1.0)
            return GeneratorOutput(
                generated_image=generated_image,
                cover_image=cover_image,
                carrier_anchor_image=aligned_carrier_image,
            )
        fused_bands = self.payload_embedder(transmitter_output.packet.spread, generated_bands)
        if inject_analog_on_cover:
            fused_bands = self._add_analog_residual_to_bands(
                base_bands=fused_bands,
                preview_image=preview_image,
                source_image=source_image,
            )
        fused_bands.original_size = target_size
        fused_bands.padded_size = generated_bands.padded_size
        cover_image = self.wavelet.idwt(fused_bands).clamp(0.0, 1.0)
        return GeneratorOutput(
            generated_image=generated_image,
            cover_image=cover_image,
            carrier_anchor_image=aligned_carrier_image,
        )
