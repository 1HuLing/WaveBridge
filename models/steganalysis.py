from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _finite_image_batch(images: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _finite_logits(logits: torch.Tensor, limit: float = 20.0) -> torch.Tensor:
    return torch.nan_to_num(logits, nan=0.0, posinf=limit, neginf=-limit).clamp(-limit, limit)


class SRNetConvBlock(nn.Module):
    # 构建 SRNet 风格残差卷积块，用于捕获隐写残差特征。
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=True)

    # 执行残差卷积特征提取。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(x) + self.shortcut(x))


class TrainableSteganalyzer(nn.Module):
    # 构建训练期隐写分析器，输出 cover/stego 二分类 logits。
    def __init__(self, channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.high_pass = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        kernel = torch.tensor([[-1.0, 2.0, -1.0], [2.0, -4.0, 2.0], [-1.0, 2.0, -1.0]])
        with torch.no_grad():
            self.high_pass.weight.copy_(kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.high_pass.weight.requires_grad_(False)
        self.features = nn.Sequential(
            SRNetConvBlock(channels, base_channels),
            SRNetConvBlock(base_channels, base_channels * 2, stride=2),
            SRNetConvBlock(base_channels * 2, base_channels * 4, stride=2),
            SRNetConvBlock(base_channels * 4, base_channels * 8, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(base_channels * 8, 2)

    # 对输入图像进行高通残差提取并输出二分类 logits。
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        residual = self.high_pass(_finite_image_batch(images))
        return _finite_logits(self.classifier(self.features(residual)))


# 临时冻结或解冻模块参数，便于生成器通过隐写分析器反传但不更新其权重。
def set_requires_grad(module: nn.Module | None, requires_grad: bool) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


# 计算隐写分析器对真实图像和载密图像的监督分类损失。
def steganalyzer_classification_loss(
    steganalyzer: TrainableSteganalyzer,
    real_images: torch.Tensor,
    stego_images: torch.Tensor,
) -> torch.Tensor:
    joint_images = torch.cat([_finite_image_batch(real_images), _finite_image_batch(stego_images.detach())], dim=0)
    joint_logits = _finite_logits(steganalyzer(joint_images))
    real_logits, stego_logits = torch.split(
        joint_logits,
        [real_images.shape[0], stego_images.shape[0]],
        dim=0,
    )
    real_labels = torch.zeros(real_logits.shape[0], device=real_logits.device, dtype=torch.long)
    stego_labels = torch.ones(stego_logits.shape[0], device=stego_logits.device, dtype=torch.long)
    return 0.5 * (F.cross_entropy(real_logits, real_labels) + F.cross_entropy(stego_logits, stego_labels))


# 计算当前隐写分析器把载密图像识别为 stego 的比例。
def steganalyzer_detection_rate(steganalyzer: TrainableSteganalyzer, stego_images: torch.Tensor) -> float:
    with torch.no_grad():
        predictions = _finite_logits(steganalyzer(_finite_image_batch(stego_images))).argmax(dim=1)
        return float((predictions == 1).float().mean().item())
