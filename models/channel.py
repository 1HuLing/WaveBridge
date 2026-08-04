from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableRobustChannel(nn.Module):
    # 构建可微信道增强模块，用于训练接收端抵抗噪声、模糊、缩放和类 JPEG 量化扰动。
    def __init__(
        self,
        noise_std: float = 0.01,
        blur_prob: float = 0.3,
        resize_prob: float = 0.3,
        quantize_prob: float = 0.3,
        jpeg_levels: int = 32,
        eval_quantize: bool = False,
        eval_noise_std: float = 0.0,
        dct_jpeg_prob: float = 0.0,
        eval_dct_jpeg: bool = False,
        dct_quality: int = 50,
        yuv_keep_weights: tuple[int, int, int] = (25, 9, 9),
        contrast_prob: float = 0.0,
        contrast_factor_range: tuple[float, float] = (0.92, 1.08),
        rotation_prob: float = 0.0,
        rotation_degree: float = 1.0,
        resize_scale_range: tuple[float, float] = (0.75, 0.75),
        blur_kernel_sizes: tuple[int, ...] = (3, 5),
        blur_sigma: float = 1.0,
        blur_force_gaussian: bool = False,
        resize_down_modes: tuple[str, ...] = ("bicubic", "bilinear"),
        resize_up_modes: tuple[str, ...] = ("bicubic", "bilinear"),
        resize_second_pass_prob: float = 0.35,
        sample_single_attack: bool = False,
        residual_detach_mode: bool = True,
    ) -> None:
        super().__init__()
        self.noise_std = noise_std
        self.blur_prob = blur_prob
        self.resize_prob = resize_prob
        self.quantize_prob = quantize_prob
        self.jpeg_levels = jpeg_levels
        self.eval_quantize = bool(eval_quantize)
        self.eval_noise_std = float(max(0.0, eval_noise_std))
        self.dct_jpeg_prob = float(max(0.0, min(1.0, dct_jpeg_prob)))
        self.eval_dct_jpeg = bool(eval_dct_jpeg)
        self.dct_quality = int(max(5, min(95, dct_quality)))
        self.yuv_keep_weights = tuple(max(1, min(64, int(value))) for value in yuv_keep_weights)
        self.contrast_prob = float(max(0.0, min(1.0, contrast_prob)))
        contrast_min, contrast_max = contrast_factor_range
        contrast_min = float(contrast_min)
        contrast_max = float(contrast_max)
        if contrast_min > contrast_max:
            contrast_min, contrast_max = contrast_max, contrast_min
        self.contrast_factor_range = (max(0.50, contrast_min), min(1.50, contrast_max))
        self.rotation_prob = float(max(0.0, min(1.0, rotation_prob)))
        self.rotation_degree = float(max(0.0, rotation_degree))
        resize_min, resize_max = resize_scale_range
        resize_min = float(resize_min)
        resize_max = float(resize_max)
        if resize_min > resize_max:
            resize_min, resize_max = resize_max, resize_min
        self.resize_scale_range = (max(0.25, resize_min), min(1.0, resize_max))
        blur_kernel_sizes = tuple(int(max(1, int(value)) | 1) for value in blur_kernel_sizes) or (3, 5)
        self.blur_kernel_sizes = blur_kernel_sizes
        self.blur_sigma = float(max(1e-3, blur_sigma))
        self.blur_force_gaussian = bool(blur_force_gaussian)
        self.resize_down_modes = tuple(str(mode).strip().lower() for mode in resize_down_modes if str(mode).strip()) or ("bicubic", "bilinear")
        self.resize_up_modes = tuple(str(mode).strip().lower() for mode in resize_up_modes if str(mode).strip()) or ("bicubic", "bilinear")
        self.resize_second_pass_prob = float(max(0.0, min(1.0, resize_second_pass_prob)))
        self.sample_single_attack = bool(sample_single_attack)
        self.residual_detach_mode = bool(residual_detach_mode)
        self.register_buffer("dct_matrix", self._build_dct_matrix(8), persistent=False)

    # 构建 8x8 正交 DCT 矩阵，用于轻量模拟 JPEG50 的块频域量化。
    def _build_dct_matrix(self, size: int) -> torch.Tensor:
        matrix = torch.empty(size, size, dtype=torch.float32)
        factor = math.pi / float(size)
        for k in range(size):
            alpha = math.sqrt(1.0 / size) if k == 0 else math.sqrt(2.0 / size)
            for n in range(size):
                matrix[k, n] = alpha * math.cos((n + 0.5) * k * factor)
        return matrix

    # 使用直通估计模拟量化压缩，使前向产生压缩效果但反向仍可传梯度。
    def _straight_through_quantize(self, images: torch.Tensor) -> torch.Tensor:
        levels = max(2, self.jpeg_levels)
        quantized = torch.round(images * (levels - 1)) / float(levels - 1)
        return images + (quantized - images).detach()

    # 将 RGB 转成 YUV，使 JPEG 代理信道能按亮度/色度分别保留低频。
    def _rgb_to_yuv(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] == 1:
            return images
        r, g, b = images[:, 0:1], images[:, 1:2], images[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v = 0.615 * r - 0.51499 * g - 0.10001 * b
        return torch.cat([y, u, v], dim=1)

    # 将 YUV 转回 RGB，用于 JPEG 低频保留后的图像重建。
    def _yuv_to_rgb(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] == 1:
            return images
        y, u, v = images[:, 0:1], images[:, 1:2], images[:, 2:3]
        r = y + 1.13983 * v
        g = y - 0.39465 * u - 0.58060 * v
        b = y + 2.03211 * u
        return torch.cat([r, g, b], dim=1)

    # 构建 JPEG-style 低频保留 mask，参考 UDH/StegaStyleGAN 的 YUV DCT 噪声层。
    def _jpeg_keep_mask(self, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        coords = [(y, x) for y in range(8) for x in range(8)]
        coords.sort(key=lambda item: (item[0] + item[1], item[0]))
        mask = torch.zeros(channels, 8, 8, device=device, dtype=dtype)
        for channel in range(channels):
            keep = self.yuv_keep_weights[min(channel, len(self.yuv_keep_weights) - 1)]
            for y, x in coords[:keep]:
                mask[channel, y, x] = 1.0
        return mask.view(1, 1, channels, 8, 8)

    # 用 YUV-DCT 低频保留和软量化模拟 JPEG50，比单纯均匀量化更接近最终评估信道。
    def _dct_jpeg_proxy(self, images: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = images.shape
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        work = self._rgb_to_yuv(images.clamp(0.0, 1.0))
        padded = F.pad(work, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else work
        padded_height, padded_width = padded.shape[-2:]
        blocks = F.unfold(padded, kernel_size=8, stride=8)
        blocks = blocks.transpose(1, 2).reshape(batch, -1, channels, 8, 8)
        dct = self.dct_matrix.to(device=images.device, dtype=images.dtype)
        coeff = torch.matmul(dct, torch.matmul(blocks - 0.5, dct.t()))
        keep_mask = self._jpeg_keep_mask(channels, images.device, images.dtype)
        yy = torch.arange(8, device=images.device, dtype=images.dtype).view(1, 1, 1, 8, 1)
        xx = torch.arange(8, device=images.device, dtype=images.dtype).view(1, 1, 1, 1, 8)
        freq = (xx + yy) / 14.0
        quality_scale = 50.0 / float(self.dct_quality)
        quant_step = (0.004 + 0.055 * freq.pow(1.35)) * quality_scale
        quantized_coeff = torch.round(coeff / quant_step.clamp_min(1e-4)) * quant_step
        masked_coeff = quantized_coeff * keep_mask
        quantized_coeff = coeff + (masked_coeff - coeff).detach()
        restored_blocks = torch.matmul(dct.t(), torch.matmul(quantized_coeff, dct)) + 0.5
        restored_blocks = restored_blocks.reshape(batch, -1, channels * 64).transpose(1, 2)
        restored_yuv = F.fold(restored_blocks, output_size=(padded_height, padded_width), kernel_size=8, stride=8)
        restored_yuv = restored_yuv[..., :height, :width]
        return self._yuv_to_rgb(restored_yuv).clamp(0.0, 1.0)

    # 使用深度可分离均值滤波模拟轻微压缩或上传平台造成的平滑。
    def _blur(self, images: torch.Tensor) -> torch.Tensor:
        kernel_size = self.blur_kernel_sizes[0]
        if len(self.blur_kernel_sizes) > 1:
            index = int(torch.randint(0, len(self.blur_kernel_sizes), (1,), device=images.device).item())
            kernel_size = self.blur_kernel_sizes[index]
        radius = kernel_size // 2
        if self.blur_force_gaussian:
            coords = torch.arange(-radius, radius + 1, device=images.device, dtype=images.dtype)
            kernel_1d = torch.exp(-(coords ** 2) / (2.0 * (self.blur_sigma ** 2)))
        else:
            if kernel_size <= 3:
                kernel_1d = images.new_tensor([1.0, 2.0, 1.0], dtype=images.dtype)
            elif kernel_size <= 5:
                kernel_1d = images.new_tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=images.dtype)
            else:
                coords = torch.arange(-radius, radius + 1, device=images.device, dtype=images.dtype)
                kernel_1d = torch.exp(-(coords ** 2) / (2.0 * (self.blur_sigma ** 2)))
        kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-6)
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d.view(1, 1, kernel_2d.shape[0], kernel_2d.shape[1]).repeat(images.shape[1], 1, 1, 1)
        padding = kernel_2d.shape[-1] // 2
        return F.conv2d(images, kernel_2d, padding=padding, groups=images.shape[1])

    # 随机下采样再上采样，模拟社交平台缩放带来的信息损失。
    def _resize_roundtrip(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        scale_min, scale_max = self.resize_scale_range
        if abs(scale_max - scale_min) < 1e-6:
            scale = scale_min
        else:
            scale = float(torch.empty((), device=images.device).uniform_(scale_min, scale_max).item())
        down_size = (max(8, int(height * scale)), max(8, int(width * scale)))
        down_mode = self.resize_down_modes[0]
        if len(self.resize_down_modes) > 1:
            down_mode = self.resize_down_modes[int(torch.randint(0, len(self.resize_down_modes), (1,), device=images.device).item())]
        up_mode = self.resize_up_modes[0]
        if len(self.resize_up_modes) > 1:
            up_mode = self.resize_up_modes[int(torch.randint(0, len(self.resize_up_modes), (1,), device=images.device).item())]
        resized = F.interpolate(
            images,
            size=down_size,
            mode=down_mode,
            align_corners=False,
            antialias=True,
        )
        restored = F.interpolate(resized, size=(height, width), mode=up_mode, align_corners=False)
        if torch.rand((), device=images.device) < self.resize_second_pass_prob:
            secondary_scale = min(scale, 0.75)
            secondary_size = (max(8, int(height * secondary_scale)), max(8, int(width * secondary_scale)))
            restored = F.interpolate(
                restored,
                size=secondary_size,
                mode=self.resize_down_modes[0],
                align_corners=False,
                antialias=True,
            )
            restored = F.interpolate(restored, size=(height, width), mode=self.resize_up_modes[0], align_corners=False)
        return restored.clamp(0.0, 1.0)

    def _contrast_adjust(self, images: torch.Tensor) -> torch.Tensor:
        contrast_min, contrast_max = self.contrast_factor_range
        if abs(contrast_max - contrast_min) < 1e-6:
            factor = contrast_min
        else:
            factor = float(torch.empty((), device=images.device).uniform_(contrast_min, contrast_max).item())
        mean = images.mean(dim=(-2, -1), keepdim=True)
        return (mean + factor * (images - mean)).clamp(0.0, 1.0)

    def _rotate_roundtrip(self, images: torch.Tensor) -> torch.Tensor:
        if self.rotation_degree <= 0.0:
            return images
        angle = float(
            torch.empty((), device=images.device).uniform_(-self.rotation_degree, self.rotation_degree).item()
        )
        theta = math.radians(angle)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        transform = images.new_tensor(
            [
                [cos_theta, -sin_theta, 0.0],
                [sin_theta, cos_theta, 0.0],
            ]
        ).unsqueeze(0).repeat(images.shape[0], 1, 1)
        grid = F.affine_grid(transform, size=images.size(), align_corners=False)
        return F.grid_sample(
            images,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).clamp(0.0, 1.0)

    # 把攻击结果改写成 stop-gradient 残差注入，贴近 RFNNS 的攻击层训练方式。
    def _apply_residual_detach(self, source: torch.Tensor, attacked: torch.Tensor) -> torch.Tensor:
        attacked = attacked.to(device=source.device, dtype=source.dtype).clamp(0.0, 1.0)
        if not self.residual_detach_mode:
            return attacked
        return source + (attacked - source).detach()

    # 对载密图像施加一组轻量可微扰动，提升鲁棒恢复能力。
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not self.training:
            output = images
            if self.eval_noise_std > 0:
                output = output + torch.randn_like(output) * self.eval_noise_std
            if self.eval_quantize:
                output = self._straight_through_quantize(output.clamp(0.0, 1.0))
            if self.eval_dct_jpeg:
                output = self._dct_jpeg_proxy(output.clamp(0.0, 1.0))
            return output.clamp(0.0, 1.0)

        output = images
        if self.sample_single_attack:
            attack_ops: list[tuple[float, callable]] = []
            if self.noise_std > 0:
                attack_ops.append(
                    (
                        1.0,
                        lambda x: self._apply_residual_detach(x, x + torch.randn_like(x) * self.noise_std),
                    )
                )
            if self.blur_prob > 0:
                attack_ops.append((self.blur_prob, lambda x: self._apply_residual_detach(x, self._blur(x))))
            if self.contrast_prob > 0:
                attack_ops.append(
                    (self.contrast_prob, lambda x: self._apply_residual_detach(x, self._contrast_adjust(x)))
                )
            if self.resize_prob > 0:
                attack_ops.append(
                    (self.resize_prob, lambda x: self._apply_residual_detach(x, self._resize_roundtrip(x)))
                )
            if self.rotation_prob > 0:
                attack_ops.append(
                    (self.rotation_prob, lambda x: self._apply_residual_detach(x, self._rotate_roundtrip(x)))
                )
            if self.quantize_prob > 0:
                attack_ops.append(
                    (self.quantize_prob, lambda x: self._apply_residual_detach(x, self._straight_through_quantize(x)))
                )
            if self.dct_jpeg_prob > 0:
                attack_ops.append((self.dct_jpeg_prob, lambda x: self._apply_residual_detach(x, self._dct_jpeg_proxy(x))))
            if attack_ops:
                weights = images.new_tensor([max(0.0, float(weight)) for weight, _ in attack_ops], dtype=torch.float32)
                if float(weights.sum().item()) > 0.0:
                    attack_index = int(torch.multinomial(weights, num_samples=1).item())
                    output = attack_ops[attack_index][1](output)
        else:
            if self.noise_std > 0:
                output = self._apply_residual_detach(output, output + torch.randn_like(output) * self.noise_std)
            if torch.rand((), device=images.device) < self.blur_prob:
                output = self._apply_residual_detach(output, self._blur(output))
            if torch.rand((), device=images.device) < self.contrast_prob:
                output = self._apply_residual_detach(output, self._contrast_adjust(output))
            if torch.rand((), device=images.device) < self.resize_prob:
                output = self._apply_residual_detach(output, self._resize_roundtrip(output))
            if torch.rand((), device=images.device) < self.rotation_prob:
                output = self._apply_residual_detach(output, self._rotate_roundtrip(output))
            output = output.clamp(0.0, 1.0)
            if torch.rand((), device=images.device) < self.quantize_prob:
                output = self._apply_residual_detach(output, self._straight_through_quantize(output))
            if torch.rand((), device=images.device) < self.dct_jpeg_prob:
                output = self._apply_residual_detach(output, self._dct_jpeg_proxy(output))
        return output.clamp(0.0, 1.0)
