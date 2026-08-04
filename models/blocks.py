from __future__ import annotations

import torch
import torch.nn as nn


def build_spatial_norm(num_channels: int) -> nn.Module:
    # 为卷积特征图构建更适合小 batch 训练的归一化层。
    if num_channels <= 0:
        raise ValueError("num_channels must be positive.")
    group_count = min(16, num_channels)
    while num_channels % group_count != 0 and group_count > 1:
        group_count -= 1
    return nn.GroupNorm(group_count, num_channels)


class ConvNormAct(nn.Module):
    # 构建卷积/反卷积 + 归一化 + 激活函数的基础模块。
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        transpose: bool = False,
    ) -> None:
        super().__init__()
        if transpose:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=stride - 1,
                    bias=False,
                ),
                build_spatial_norm(out_channels),
                nn.SiLU(),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                build_spatial_norm(out_channels),
                nn.SiLU(),
            )

    # 对输入特征执行一次基础卷积块变换。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    # 构建带残差连接的卷积块，用于增强特征表达能力。
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            build_spatial_norm(channels),
        )
        self.act = nn.SiLU()

    # 执行残差映射并保留输入主干信息。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class MLP(nn.Module):
    # 构建两层前馈感知机，用于序列或全连接特征映射。
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    # 对输入向量执行前馈非线性变换。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
