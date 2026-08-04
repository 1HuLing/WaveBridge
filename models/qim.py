from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.wavelets import LeGall53Wavelet2D, WaveletBands


class WaveletQIMChannel(nn.Module):
    # 在 5/3 小波高频子带中使用标量 QIM 嵌入 coded bits，并从载密图像解析出 Polar LLR。
    def __init__(
        self,
        delta: float = 0.012,
        strength: float = 1.0,
        llr_scale: float = 10.0,
        llr_clamp: float = 18.0,
        llr_polarity: float = 1.0,
        carrier_bands: str | Sequence[str] = ("lh", "hl", "hh"),
        position_mode: str = "linear",
        position_seed: int = 20260607,
        adaptive_strength: bool = True,
        texture_floor: float = 0.75,
        texture_sharpness: float = 1.8,
        texture_kernel: int = 17,
        texture_abs_threshold: float = 0.004,
        repetition_factor: int = 1,
        extract_smooth_kernel: int = 1,
        dither_enabled: bool = True,
        dither_strength: float = 0.45,
        domain: str = "wavelet",
        dct_coefficients: str | Sequence[str] | Sequence[Sequence[int]] | None = None,
        dct_channels: str = "y",
        dct_quality: int = 50,
        dct_quant_scale: float = 1.0,
        dct_parity_mode: bool = False,
    ) -> None:
        super().__init__()
        if delta <= 0:
            raise ValueError("QIM delta must be positive.")
        self.delta = float(delta)
        self.strength = float(max(0.0, min(1.0, strength)))
        self.llr_scale = float(llr_scale)
        self.llr_clamp = float(llr_clamp)
        self.llr_polarity = 1.0 if float(llr_polarity) >= 0.0 else -1.0
        self.carrier_bands = self._normalize_carrier_bands(carrier_bands)
        self.position_mode = str(position_mode).strip().lower()
        if self.position_mode not in {"linear", "permutation", "stratified"}:
            raise ValueError("QIM position_mode must be one of 'linear', 'permutation' or 'stratified'.")
        self.position_seed = int(position_seed)
        self.adaptive_strength = bool(adaptive_strength)
        self.texture_floor = float(max(0.0, min(1.0, texture_floor)))
        self.texture_sharpness = float(max(0.1, texture_sharpness))
        self.texture_kernel = max(1, int(texture_kernel) | 1)
        self.texture_abs_threshold = float(max(1e-6, texture_abs_threshold))
        self.repetition_factor = max(1, int(repetition_factor))
        self.extract_smooth_kernel = max(1, int(extract_smooth_kernel) | 1)
        self.dither_enabled = bool(dither_enabled)
        self.dither_strength = float(max(0.0, min(0.49, dither_strength)))
        self.domain = str(domain).strip().lower()
        if self.domain not in {"wavelet", "dct"}:
            raise ValueError("QIM domain must be either 'wavelet' or 'dct'.")
        self.dct_channels = str(dct_channels).strip().lower()
        if self.dct_channels not in {"y", "rgb", "ycbcr"}:
            raise ValueError("qim.dct_channels must be one of 'y', 'rgb' or 'ycbcr'.")
        self.dct_quality = int(max(1, min(100, dct_quality)))
        self.dct_quant_scale = float(max(0.25, dct_quant_scale))
        self.dct_parity_mode = bool(dct_parity_mode)
        self.dct_coefficients = self._normalize_dct_coefficients(dct_coefficients)
        self.wavelet = LeGall53Wavelet2D()
        self.register_buffer("dct_matrix", self._build_dct_matrix(8), persistent=False)

    # 用直通估计保留 QIM 前向离散格点效果，同时让梯度回传到底图系数。
    def _straight_through_lattice_project(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        strength: torch.Tensor,
    ) -> torch.Tensor:
        blended_target = source + strength.to(device=source.device, dtype=source.dtype) * (target - source)
        return source + (blended_target - source).detach()

    # 构建 8x8 正交 DCT 矩阵，用于 JPEG 对齐的 DCT-QIM 鲁棒通道。
    def _build_dct_matrix(self, size: int) -> torch.Tensor:
        matrix = torch.empty(size, size, dtype=torch.float32)
        factor = torch.pi / float(size)
        for k in range(size):
            alpha = (1.0 / size) ** 0.5 if k == 0 else (2.0 / size) ** 0.5
            for n in range(size):
                matrix[k, n] = alpha * torch.cos(torch.tensor((n + 0.5) * k * factor)).item()
        return matrix

    # 按 JPEG zig-zag 思路生成 DCT 系数坐标，优先使用低频和中频位置。
    def _zigzag_dct_coefficients(self, limit: int | None = None) -> list[tuple[int, int]]:
        coords = [(y, x) for y in range(8) for x in range(8) if not (y == 0 and x == 0)]
        coords.sort(key=lambda item: (item[0] + item[1], item[0]))
        if limit is not None:
            coords = coords[: max(1, min(int(limit), len(coords)))]
        return coords

    # 构造 JPEG 标准亮度/色度量化表，并按质量因子缩放到当前质量等级。
    def _jpeg_quant_tables(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        luma_base = torch.tensor(
            [
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99],
            ],
            device=device,
            dtype=torch.float32,
        )
        chroma_base = torch.tensor(
            [
                [17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
            ],
            device=device,
            dtype=torch.float32,
        )
        quality = max(1, min(100, int(self.dct_quality)))
        scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality

        def scale_table(base: torch.Tensor) -> torch.Tensor:
            scaled = torch.floor((base * scale + 50.0) / 100.0).clamp(1.0, 255.0)
            return scaled.to(device=device, dtype=dtype)

        return scale_table(luma_base), scale_table(chroma_base)

    # 解析 DCT-QIM 使用的 JPEG 中频系数坐标，支持 auto48/auto56/auto63 自动容量模式。
    def _normalize_dct_coefficients(
        self,
        dct_coefficients: str | Sequence[str] | Sequence[Sequence[int]] | None,
    ) -> tuple[tuple[int, int], ...]:
        if dct_coefficients is None:
            return tuple(self._zigzag_dct_coefficients(limit=56))
        elif isinstance(dct_coefficients, str):
            normalized = dct_coefficients.strip().lower()
            if normalized in {"auto", "zigzag"}:
                return tuple(self._zigzag_dct_coefficients(limit=56))
            if normalized.startswith("auto"):
                suffix = normalized.removeprefix("auto").strip()
                limit = int(suffix) if suffix else 56
                return tuple(self._zigzag_dct_coefficients(limit=limit))
            raw_items = [item.strip() for item in dct_coefficients.split(";") if item.strip()]
        else:
            raw_items = list(dct_coefficients)
        coords: list[tuple[int, int]] = []
        for item in raw_items:
            if isinstance(item, str):
                parts = [part.strip() for part in item.replace(":", ",").split(",") if part.strip()]
                if len(parts) != 2:
                    raise ValueError(f"Invalid DCT coefficient coordinate: {item}")
                y, x = int(parts[0]), int(parts[1])
            else:
                if len(item) != 2:
                    raise ValueError(f"Invalid DCT coefficient coordinate: {item}")
                y, x = int(item[0]), int(item[1])
            if y == 0 and x == 0:
                continue
            if not (0 <= y < 8 and 0 <= x < 8):
                raise ValueError(f"DCT coefficient coordinate out of range: {(y, x)}")
            coord = (y, x)
            if coord not in coords:
                coords.append(coord)
        if not coords:
            raise ValueError("DCT-QIM requires at least one non-DC coefficient.")
        return tuple(coords)

    # 将 RGB 图像转成亮度平面，DCT-QIM 默认只改亮度以对齐 JPEG 的主要保真通道。
    def _luma(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            return image
        weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (image[:, :3] * weights).sum(dim=1, keepdim=True)

    # 将 RGB 图像转成近似 JPEG 的 YCbCr 空间，便于探测彩色 DCT 载体的鲁棒性上限。
    def _rgb_to_ycbcr(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            return image
        r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
        return torch.cat([y, cb, cr], dim=1)

    # 将 YCbCr 图像转回 RGB，作为多通道 DCT-QIM 的反变换。
    def _ycbcr_to_rgb(self, ycbcr: torch.Tensor) -> torch.Tensor:
        if ycbcr.shape[1] == 1:
            return ycbcr
        y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2] - 0.5, ycbcr[:, 2:3] - 0.5
        r = y + 1.402 * cr
        g = y - 0.344136 * cb - 0.714136 * cr
        b = y + 1.772 * cb
        return torch.cat([r, g, b], dim=1)

    # 根据 dct_channels 选择 DCT-QIM 的工作图像域。
    def _dct_input(self, image: torch.Tensor) -> torch.Tensor:
        if self.dct_channels == "rgb":
            return image
        if self.dct_channels == "ycbcr":
            return self._rgb_to_ycbcr(image)
        return self._luma(image)

    # 将 DCT-QIM 域中的恢复结果映射回 RGB 图像。
    def _dct_output_to_rgb(self, original_image: torch.Tensor, source: torch.Tensor, restored: torch.Tensor) -> torch.Tensor:
        if self.dct_channels == "rgb":
            return restored
        if self.dct_channels == "ycbcr":
            return self._ycbcr_to_rgb(restored)
        luma_delta = restored - source
        return original_image + luma_delta.expand_as(original_image)

    # 把单通道或多通道图像分成 8x8 块并计算正交 DCT 系数。
    def _dct_blocks(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
        batch, channels, height, width = image.shape
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        padded = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else image
        padded_height, padded_width = int(padded.shape[-2]), int(padded.shape[-1])
        unfolded = F.unfold(padded, kernel_size=8, stride=8)
        num_blocks = int(unfolded.shape[-1])
        blocks = unfolded.view(batch, channels, 64, num_blocks).permute(0, 1, 3, 2)
        blocks = blocks.reshape(batch, channels * num_blocks, 8, 8)
        dct = self.dct_matrix.to(device=image.device, dtype=image.dtype)
        coeff = torch.matmul(dct, torch.matmul(blocks * 255.0 - 128.0, dct.t()))
        return coeff, (height, width), (padded_height, padded_width)

    # 将 DCT 系数逆变换回单通道或多通道图像。
    def _idct_blocks(
        self,
        coeff: torch.Tensor,
        original_size: tuple[int, int],
        padded_size: tuple[int, int],
    ) -> torch.Tensor:
        batch = int(coeff.shape[0])
        dct = self.dct_matrix.to(device=coeff.device, dtype=coeff.dtype)
        blocks = (torch.matmul(dct.t(), torch.matmul(coeff, dct)) + 128.0) / 255.0
        padded_height, padded_width = padded_size
        num_blocks = (padded_height // 8) * (padded_width // 8)
        channels = max(1, int(coeff.shape[1]) // max(1, num_blocks))
        blocks = blocks.reshape(batch, channels, num_blocks, 64)
        blocks = blocks.permute(0, 1, 3, 2).reshape(batch, channels * 64, num_blocks)
        restored = F.fold(blocks, output_size=padded_size, kernel_size=8, stride=8)
        height, width = original_size
        return restored[..., :height, :width]

    # 取出 DCT-QIM 使用的中频系数并展平成载体序列。
    def _flatten_dct_coefficients(self, coeff: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coords = self._normalize_dct_coefficients(self.dct_coefficients)
        if coords != self.dct_coefficients:
            self.dct_coefficients = coords
        y_indices = torch.tensor([coord[0] for coord in coords], device=coeff.device, dtype=torch.long)
        x_indices = torch.tensor([coord[1] for coord in coords], device=coeff.device, dtype=torch.long)
        selected = coeff[:, :, y_indices, x_indices]
        return selected.flatten(start_dim=1), y_indices, x_indices

    # 把修改后的 DCT-QIM 载体序列写回对应中频系数。
    def _unflatten_dct_coefficients(
        self,
        coeff: torch.Tensor,
        flat: torch.Tensor,
        y_indices: torch.Tensor,
        x_indices: torch.Tensor,
    ) -> torch.Tensor:
        updated = coeff.clone()
        selected = flat.view(coeff.shape[0], coeff.shape[1], y_indices.numel())
        updated[:, :, y_indices, x_indices] = selected
        return updated

    # 生成与 DCT 载体一一对应的 JPEG 量化步长，便于按 JPEG 量化格点做奇偶嵌入。
    def _flatten_dct_quant_steps(
        self,
        coeff: torch.Tensor,
        y_indices: torch.Tensor,
        x_indices: torch.Tensor,
        channel_count: int,
    ) -> torch.Tensor:
        luma_table, chroma_table = self._jpeg_quant_tables(coeff.device, coeff.dtype)
        per_channel_steps = []
        for channel_index in range(max(1, int(channel_count))):
            if self.dct_channels == "y":
                base_table = luma_table
            elif self.dct_channels == "ycbcr":
                base_table = luma_table if channel_index == 0 else chroma_table
            else:
                base_table = luma_table
            per_channel_steps.append(base_table[y_indices, x_indices])
        selected_steps = torch.stack(per_channel_steps, dim=0)
        dct_rows = int(coeff.shape[1])
        blocks_per_channel = max(1, dct_rows // max(1, int(channel_count)))
        tiled_steps = selected_steps.unsqueeze(1).expand(-1, blocks_per_channel, -1).reshape(dct_rows, -1)
        return tiled_steps.flatten().unsqueeze(0).expand(coeff.shape[0], -1)

    # 规范化 QIM 使用的高频子带名称，避免二进制信道和模拟 reveal 信道抢同一批 carrier。
    def _normalize_carrier_bands(self, carrier_bands: str | Sequence[str]) -> tuple[str, ...]:
        if isinstance(carrier_bands, str):
            raw_bands = [band.strip().lower() for band in carrier_bands.split(",")]
        else:
            raw_bands = [str(band).strip().lower() for band in carrier_bands]
        valid = {"ll", "lh", "hl", "hh"}
        normalized: list[str] = []
        for band in raw_bands:
            if not band:
                continue
            if band not in valid:
                raise ValueError(f"Unsupported QIM carrier band: {band}. Expected one of {sorted(valid)}.")
            if band not in normalized:
                normalized.append(band)
        if not normalized:
            raise ValueError("QIM carrier_bands must contain at least one wavelet band.")
        return tuple(normalized)

    # 按配置将选中的小波子带展平成一个确定性 carrier 序列。
    def _flatten_bands(
        self,
        bands: WaveletBands,
    ) -> tuple[torch.Tensor, dict[str, torch.Size]]:
        shapes = {
            "ll": bands.ll.shape,
            "lh": bands.lh.shape,
            "hl": bands.hl.shape,
            "hh": bands.hh.shape,
        }
        band_map = {
            "ll": bands.ll,
            "lh": bands.lh,
            "hl": bands.hl,
            "hh": bands.hh,
        }
        flat = torch.cat([band_map[name].flatten(start_dim=1) for name in self.carrier_bands], dim=1)
        return flat, shapes

    # 根据高频子带局部能量生成纹理权重，平滑区域降低 QIM 位移以减少肉眼可见网格。
    def _band_texture_mask(self, band: torch.Tensor) -> torch.Tensor:
        energy = band.float().abs().mean(dim=1, keepdim=True)
        if self.texture_kernel > 1:
            energy = F.avg_pool2d(
                energy,
                kernel_size=self.texture_kernel,
                stride=1,
                padding=self.texture_kernel // 2,
            )
        mean = energy.mean(dim=(-2, -1), keepdim=True)
        std = energy.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
        normalized = (energy - mean) / std
        relative_texture = torch.sigmoid(self.texture_sharpness * normalized)
        absolute_texture = torch.sigmoid(
            (energy - self.texture_abs_threshold) / max(self.texture_abs_threshold * 0.5, 1e-6)
        )
        texture = relative_texture * absolute_texture
        texture = self.texture_floor + (1.0 - self.texture_floor) * texture
        return texture.to(device=band.device, dtype=band.dtype).expand_as(band)

    # 按 QIM 使用的高频子带顺序展开纹理权重，与 carrier 序列保持一一对应。
    def _flatten_texture_mask(self, bands: WaveletBands) -> torch.Tensor:
        if not self.adaptive_strength:
            flat, _ = self._flatten_bands(bands)
            return torch.ones_like(flat)
        mask_map = {
            "ll": self._band_texture_mask(bands.ll),
            "lh": self._band_texture_mask(bands.lh),
            "hl": self._band_texture_mask(bands.hl),
            "hh": self._band_texture_mask(bands.hh),
        }
        return torch.cat([mask_map[name].flatten(start_dim=1) for name in self.carrier_bands], dim=1)

    # 将展平后的 carrier 序列写回被 QIM 使用的子带，其余子带保持原值不变。
    def _unflatten_bands(
        self,
        flat: torch.Tensor,
        shapes: dict[str, torch.Size],
        original_bands: WaveletBands,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        band_shapes = shapes
        band_tensors = {
            "ll": original_bands.ll,
            "lh": original_bands.lh,
            "hl": original_bands.hl,
            "hh": original_bands.hh,
        }
        split_sizes = [int(torch.tensor(band_shapes[name][1:]).prod().item()) for name in self.carrier_bands]
        split_flats = torch.split(flat, split_sizes, dim=1)
        for name, band_flat in zip(self.carrier_bands, split_flats):
            band_tensors[name] = band_flat.view(band_shapes[name])
        return band_tensors["ll"], band_tensors["lh"], band_tensors["hl"], band_tensors["hh"]

    # 在提取前做轻量平滑，抑制 JPEG block 噪声对 lattice 距离估计的扰动。
    def _prepare_extract_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.extract_smooth_kernel <= 1:
            return image
        pad = self.extract_smooth_kernel // 2
        padded = F.pad(image, (pad, pad, pad, pad), mode="reflect")
        return F.avg_pool2d(padded, kernel_size=self.extract_smooth_kernel, stride=1)

    # 生成固定的交织 carrier 位置，使发送端和接收端不需要共享图像自适应索引。
    def _carrier_positions(
        self,
        capacity: int,
        bit_count: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, int]:
        if bit_count <= 0:
            raise ValueError("bit_count must be positive.")
        effective_repeat = min(self.repetition_factor, max(1, capacity // bit_count))
        required_slots = bit_count * effective_repeat
        if required_slots > capacity:
            raise ValueError(
                "QIM carrier capacity is smaller than the coded bitstream. "
                f"capacity={capacity}, coded_bits={bit_count}, repetition_factor={effective_repeat}. "
                "Increase polar info_length or reduce transmitted payload."
            )
        if required_slots == capacity:
            return torch.arange(capacity, device=device, dtype=torch.long), effective_repeat
        if self.position_mode == "permutation":
            generator = torch.Generator(device="cpu").manual_seed(self.position_seed + capacity * 17 + bit_count)
            positions = torch.randperm(capacity, generator=generator, dtype=torch.long)[:required_slots].to(device=device)
            return positions, effective_repeat
        if self.position_mode == "stratified":
            generator = torch.Generator(device="cpu").manual_seed(
                self.position_seed + capacity * 17 + bit_count * 131 + required_slots
            )
            edges = torch.linspace(0, capacity, steps=required_slots + 1, dtype=torch.float64)
            starts = torch.floor(edges[:-1]).to(torch.long)
            ends = torch.floor(edges[1:]).to(torch.long)
            ends = torch.maximum(ends, starts + 1).clamp(max=capacity)
            widths = (ends - starts).clamp_min(1)
            jitter = torch.floor(torch.rand(required_slots, generator=generator) * widths.to(torch.float32)).to(torch.long)
            positions = (starts + jitter).clamp(0, capacity - 1)
            return positions.to(device=device, dtype=torch.long), effective_repeat
        positions = torch.linspace(0, capacity - 1, steps=required_slots, device=device).round().to(torch.long)
        return positions, effective_repeat

    # 返回某个 bit 对应量化格点中最接近当前系数的代表值。
    def _nearest_lattice(self, values: torch.Tensor, bit_value: torch.Tensor) -> torch.Tensor:
        delta = values.new_tensor(self.delta)
        offset = (bit_value.to(dtype=values.dtype) - 0.5) * (0.5 * delta)
        return torch.round((values - offset) / delta) * delta + offset

    # 生成发送端和接收端共享的减性抖动，打散规则 QIM 格点以降低统计可见性。
    def _lattice_dither(self, positions: torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.dither_enabled or self.dither_strength <= 0.0:
            return torch.zeros(batch_size, positions.numel(), device=device, dtype=dtype)
        hashed = positions.to(device=device, dtype=torch.float32)
        hashed = torch.sin(hashed * 12.9898 + float(self.position_seed) * 0.0174533) * 43758.5453
        fractional = hashed - torch.floor(hashed)
        dither = (fractional - 0.5) * (2.0 * self.dither_strength * self.delta)
        return dither.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)

    # 在抖动坐标系中返回 bit 对应的最近格点，再加回共享抖动。
    def _nearest_dithered_lattice(
        self,
        values: torch.Tensor,
        bit_value: torch.Tensor,
        dither: torch.Tensor,
    ) -> torch.Tensor:
        return self._nearest_lattice(values - dither, bit_value) + dither

    # 在 JPEG 量化坐标系中，把目标 bit 写成量化索引奇偶性。
    def _nearest_jpeg_parity_lattice(
        self,
        values: torch.Tensor,
        bit_value: torch.Tensor,
        quant_step: torch.Tensor,
    ) -> torch.Tensor:
        safe_step = quant_step.clamp_min(1e-6)
        scaled = values / safe_step
        parity_stride = max(1.0, float(round(self.delta)))
        stride = values.new_tensor(parity_stride)
        bit_offset = bit_value.to(dtype=values.dtype) * stride
        target_index = torch.round((scaled - bit_offset) / (2.0 * stride)) * (2.0 * stride) + bit_offset
        return target_index * safe_step

    # 在 JPEG 量化坐标系中读取奇偶格点距离差，输出 bit=0 与 bit=1 的判别 LLR。
    def _jpeg_parity_llr(
        self,
        values: torch.Tensor,
        quant_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        safe_step = quant_step.clamp_min(1e-6)
        scaled = values / safe_step
        parity_stride = max(1.0, float(round(self.delta)))
        stride = values.new_tensor(parity_stride)
        even_lattice = torch.round(scaled / (2.0 * stride)) * (2.0 * stride)
        odd_lattice = torch.round((scaled - stride) / (2.0 * stride)) * (2.0 * stride) + stride
        distance0 = (scaled - even_lattice).abs()
        distance1 = (scaled - odd_lattice).abs()
        llr = (distance1 - distance0) * self.llr_scale
        return llr, distance0, distance1

    # 把 Polar coded bits 直接 QIM 嵌入到高频小波系数，输出载密图像。
    def embed(self, image: torch.Tensor, coded_bits: torch.Tensor) -> torch.Tensor:
        if coded_bits.dim() != 3:
            raise ValueError("QIM embed expects coded_bits with shape [B, num_blocks, code_length].")
        if not torch.isfinite(image).all():
            raise ValueError("QIM embed received a non-finite image tensor.")
        if not torch.isfinite(coded_bits).all():
            raise ValueError("QIM embed received non-finite coded bits.")
        if self.domain == "dct":
            return self._embed_dct(image, coded_bits)
        bands = self.wavelet.dwt(image)
        flat, shapes = self._flatten_bands(bands)
        bits = coded_bits.flatten(start_dim=1).to(device=image.device, dtype=image.dtype)
        positions, effective_repeat = self._carrier_positions(flat.shape[1], bits.shape[1], image.device)
        repeated_bits = bits.repeat_interleave(effective_repeat, dim=1)
        selected = flat.index_select(1, positions)
        dither = self._lattice_dither(positions, image.shape[0], image.device, image.dtype)
        target = self._nearest_dithered_lattice(selected, repeated_bits, dither)
        selected_strength = self.strength * self._flatten_texture_mask(bands).index_select(1, positions)
        embedded = self._straight_through_lattice_project(
            source=selected,
            target=target,
            strength=selected_strength,
        )
        updated_flat = flat.clone()
        updated_flat[:, positions] = embedded
        ll, lh, hl, hh = self._unflatten_bands(updated_flat, shapes, bands)
        stego_bands = WaveletBands(
            ll=ll,
            lh=lh,
            hl=hl,
            hh=hh,
            original_size=bands.original_size,
            padded_size=bands.padded_size,
        )
        return self.wavelet.idwt(stego_bands).clamp(0.0, 1.0)

    # 在 JPEG 8x8 亮度 DCT 中频系数上做 QIM，作为抗 JPEG 的备选鲁棒通道。
    def _embed_dct(self, image: torch.Tensor, coded_bits: torch.Tensor) -> torch.Tensor:
        source = self._dct_input(image)
        coeff, original_size, padded_size = self._dct_blocks(source)
        flat, y_indices, x_indices = self._flatten_dct_coefficients(coeff)
        flat_quant_steps = self._flatten_dct_quant_steps(coeff, y_indices, x_indices, source.shape[1])
        bits = coded_bits.flatten(start_dim=1).to(device=image.device, dtype=image.dtype)
        positions, effective_repeat = self._carrier_positions(flat.shape[1], bits.shape[1], image.device)
        repeated_bits = bits.repeat_interleave(effective_repeat, dim=1)
        selected = flat.index_select(1, positions)
        if self.dct_parity_mode:
            # JPEG parity 模式必须严格贴合真实 JPEG 量化步长，不能再额外缩放格点，
            # 否则重压缩后奇偶索引会漂移，导致 raw LLR 在 attacked 图像上近似随机。
            selected_steps = flat_quant_steps.index_select(1, positions)
            target = self._nearest_jpeg_parity_lattice(selected, repeated_bits, selected_steps)
        else:
            dither = self._lattice_dither(positions, image.shape[0], image.device, image.dtype)
            target = self._nearest_dithered_lattice(selected, repeated_bits, dither)
        embedded = self._straight_through_lattice_project(
            source=selected,
            target=target,
            strength=selected.new_full(selected.shape, self.strength),
        )
        updated_flat = flat.clone()
        updated_flat[:, positions] = embedded
        updated_coeff = self._unflatten_dct_coefficients(coeff, updated_flat, y_indices, x_indices)
        restored_source = self._idct_blocks(updated_coeff, original_size, padded_size)
        return self._dct_output_to_rgb(image, source, restored_source).clamp(0.0, 1.0)

    # 从载密图像中读取 QIM 格点距离差，并输出标准 LLR：正值代表 bit=0，负值代表 bit=1。
    def extract_llr(self, image: torch.Tensor, num_blocks: int, code_length: int) -> torch.Tensor:
        if not torch.isfinite(image).all():
            raise ValueError("QIM extract_llr received a non-finite image tensor.")
        if self.domain == "dct":
            return self._extract_llr_dct(image, num_blocks, code_length)
        prepared_image = self._prepare_extract_image(image)
        bands = self.wavelet.dwt(prepared_image)
        flat, _ = self._flatten_bands(bands)
        flat_texture = self._flatten_texture_mask(bands)
        bit_count = int(num_blocks) * int(code_length)
        positions, effective_repeat = self._carrier_positions(flat.shape[1], bit_count, image.device)
        selected = flat.index_select(1, positions)
        selected_texture = flat_texture.index_select(1, positions).to(device=selected.device, dtype=selected.dtype)
        dither = self._lattice_dither(positions, image.shape[0], image.device, image.dtype)
        zeros = torch.zeros_like(selected)
        ones = torch.ones_like(selected)
        lattice0 = self._nearest_dithered_lattice(selected, zeros, dither)
        lattice1 = self._nearest_dithered_lattice(selected, ones, dither)
        distance0 = (selected - lattice0).abs()
        distance1 = (selected - lattice1).abs()
        llr = (distance1 - distance0) / max(self.delta * 0.25, 1e-6) * self.llr_scale
        llr = llr * selected.new_tensor(self.llr_polarity)
        llr = torch.nan_to_num(llr, nan=0.0, posinf=self.llr_clamp, neginf=-self.llr_clamp)
        llr = llr.clamp(-self.llr_clamp, self.llr_clamp)
        if effective_repeat > 1:
            llr = llr.view(image.shape[0], bit_count, effective_repeat)
            distance_margin = (distance1 - distance0).abs().view(image.shape[0], bit_count, effective_repeat)
            repeat_weights = selected_texture.view(image.shape[0], bit_count, effective_repeat) * (
                0.35 + distance_margin / max(self.delta, 1e-6)
            )
            repeat_weights = repeat_weights / repeat_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            llr = (llr * repeat_weights).sum(dim=-1)
        return llr.view(image.shape[0], int(num_blocks), int(code_length))

    # 从 JPEG 对齐的亮度 DCT 中频系数中提取 QIM LLR。
    def _extract_llr_dct(self, image: torch.Tensor, num_blocks: int, code_length: int) -> torch.Tensor:
        # parity 模式直接在 JPEG 重载后的像素上做 DCT，避免额外平滑破坏量化索引奇偶性。
        prepared_image = image if self.dct_parity_mode else self._prepare_extract_image(image)
        source = self._dct_input(prepared_image)
        coeff, _original_size, _padded_size = self._dct_blocks(source)
        flat, y_indices, x_indices = self._flatten_dct_coefficients(coeff)
        flat_quant_steps = self._flatten_dct_quant_steps(coeff, y_indices, x_indices, source.shape[1])
        bit_count = int(num_blocks) * int(code_length)
        positions, effective_repeat = self._carrier_positions(flat.shape[1], bit_count, image.device)
        selected = flat.index_select(1, positions)
        if self.dct_parity_mode:
            selected_steps = flat_quant_steps.index_select(1, positions)
            llr, distance0, distance1 = self._jpeg_parity_llr(selected, selected_steps)
        else:
            dither = self._lattice_dither(positions, image.shape[0], image.device, image.dtype)
            zeros = torch.zeros_like(selected)
            ones = torch.ones_like(selected)
            lattice0 = self._nearest_dithered_lattice(selected, zeros, dither)
            lattice1 = self._nearest_dithered_lattice(selected, ones, dither)
            distance0 = (selected - lattice0).abs()
            distance1 = (selected - lattice1).abs()
            llr = (distance1 - distance0) / max(self.delta * 0.25, 1e-6) * self.llr_scale
        llr = llr * selected.new_tensor(self.llr_polarity)
        llr = torch.nan_to_num(llr, nan=0.0, posinf=self.llr_clamp, neginf=-self.llr_clamp)
        llr = llr.clamp(-self.llr_clamp, self.llr_clamp)
        if effective_repeat > 1:
            llr = llr.view(image.shape[0], bit_count, effective_repeat)
            distance_margin = (distance1 - distance0).abs().view(image.shape[0], bit_count, effective_repeat)
            repeat_weights = 0.35 + distance_margin / max(self.delta, 1e-6)
            repeat_weights = repeat_weights / repeat_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            llr = (llr * repeat_weights).sum(dim=-1)
        return llr.view(image.shape[0], int(num_blocks), int(code_length))
