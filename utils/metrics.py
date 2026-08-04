from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# 计算两张图像或两个张量之间的均方误差。
def mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, y)


# 根据均方误差计算峰值信噪比，用于评估重建图像质量。
def psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> float:
    value = F.mse_loss(x, y).item()
    if value == 0:
        return float("inf")
    return 10.0 * math.log10((max_val**2) / value)
