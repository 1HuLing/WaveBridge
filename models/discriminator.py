from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.blocks import ConvNormAct
from utils.wavelets import LeGall53Wavelet2D


@dataclass
class DiscriminatorOutput:
    spatial_logits: torch.Tensor
    spectral_logits: torch.Tensor


class SpatialDiscriminator(nn.Module):
    # 构建空域判别器，判断图像在像素空间中的真实性。
    def __init__(self, image_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct(image_channels, base_channels, stride=2),
            ConvNormAct(base_channels, base_channels * 2, stride=2),
            ConvNormAct(base_channels * 2, base_channels * 4, stride=2),
            nn.Conv2d(base_channels * 4, 1, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    # 输出输入图像在空域判别器中的真伪分数。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpectralDiscriminator(nn.Module):
    # 构建频域判别器，约束生成图像的小波频谱分布。
    def __init__(self, image_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct(image_channels * 4, base_channels, stride=2),
            ConvNormAct(base_channels, base_channels * 2, stride=2),
            ConvNormAct(base_channels * 2, base_channels * 4, stride=2),
            nn.Conv2d(base_channels * 4, 1, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.wavelet = LeGall53Wavelet2D()

    # 先进行小波分解，再输出输入图像的频域真伪分数。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bands = self.wavelet.dwt(x).flatten()
        return self.net(bands)


class MultiDomainDiscriminator(nn.Module):
    # 组合空域与频域两个判别器，形成多域对抗约束。
    def __init__(self, image_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        self.spatial = SpatialDiscriminator(image_channels, base_channels)
        self.spectral = SpectralDiscriminator(image_channels, base_channels)

    # 同时返回空域与频域两个判别结果。
    def forward(self, x: torch.Tensor) -> DiscriminatorOutput:
        return DiscriminatorOutput(
            spatial_logits=self.spatial(x),
            spectral_logits=self.spectral(x),
        )
