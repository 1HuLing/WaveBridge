from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class WaveletBands:
    ll: torch.Tensor
    lh: torch.Tensor
    hl: torch.Tensor
    hh: torch.Tensor
    original_size: tuple[int, int] | None = None
    padded_size: tuple[int, int] | None = None

    # 将四个 5/3 小波子带在通道维拼接，方便后续卷积网络统一处理。
    def flatten(self) -> torch.Tensor:
        return torch.cat([self.ll, self.lh, self.hl, self.hh], dim=1)


class LeGall53Wavelet2D:
    # 初始化二维 LeGall 5/3 小波变换，使用可微 lifting 结构实现。
    def __init__(self) -> None:
        self.name = "legall53"

    # 沿指定维度取偶数位样本，避免使用 index_select 触发大张量索引反传。
    def _even_samples(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        slices = [slice(None)] * x.dim()
        slices[dim] = slice(0, None, 2)
        return x[tuple(slices)]

    # 沿指定维度取奇数位样本，保持与偶数位样本同样的切片语义。
    def _odd_samples(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        slices = [slice(None)] * x.dim()
        slices[dim] = slice(1, None, 2)
        return x[tuple(slices)]

    # 将偶数位和奇数位子序列重新交错回原始排列，替代 index_copy_ 的写回逻辑。
    def _interleave(self, even: torch.Tensor, odd: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.stack((even, odd), dim=dim + 1).flatten(dim, dim + 1)

    # 沿指定维度执行一维 5/3 小波分解，返回低频近似和高频细节。
    def _analysis_1d(self, x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[dim] % 2 != 0:
            raise ValueError("LeGall 5/3 wavelet requires even spatial sizes.")
        even = self._even_samples(x, dim)
        odd = self._odd_samples(x, dim)
        even_next = torch.cat([even.narrow(dim, 1, even.shape[dim] - 1), even.narrow(dim, even.shape[dim] - 1, 1)], dim=dim)
        detail = odd - 0.5 * (even + even_next)
        detail_prev = torch.cat([detail.narrow(dim, 0, 1), detail.narrow(dim, 0, detail.shape[dim] - 1)], dim=dim)
        approx = even + 0.25 * (detail_prev + detail)
        return approx, detail

    # 沿指定维度执行一维 5/3 小波逆变换，恢复交错排列的原始信号。
    def _synthesis_1d(self, approx: torch.Tensor, detail: torch.Tensor, dim: int) -> torch.Tensor:
        detail_prev = torch.cat([detail.narrow(dim, 0, 1), detail.narrow(dim, 0, detail.shape[dim] - 1)], dim=dim)
        even = approx - 0.25 * (detail_prev + detail)
        even_next = torch.cat([even.narrow(dim, 1, even.shape[dim] - 1), even.narrow(dim, even.shape[dim] - 1, 1)], dim=dim)
        odd = detail + 0.5 * (even + even_next)
        return self._interleave(even, odd, dim)

    # 对输入图像执行二维 5/3 小波变换，输出 LL、LH、HL、HH 四个子带。
    def dwt(self, x: torch.Tensor) -> WaveletBands:
        original_size = (x.shape[-2], x.shape[-1])
        pad_h = original_size[0] % 2
        pad_w = original_size[1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        padded_size = (x.shape[-2], x.shape[-1])
        low_h, high_h = self._analysis_1d(x, dim=3)
        ll, lh = self._analysis_1d(low_h, dim=2)
        hl, hh = self._analysis_1d(high_h, dim=2)
        return WaveletBands(ll=ll, lh=lh, hl=hl, hh=hh, original_size=original_size, padded_size=padded_size)

    # 根据四个 5/3 小波子带执行逆变换，恢复到图像空间。
    def idwt(self, bands: WaveletBands) -> torch.Tensor:
        low_h = self._synthesis_1d(bands.ll, bands.lh, dim=2)
        high_h = self._synthesis_1d(bands.hl, bands.hh, dim=2)
        reconstructed = self._synthesis_1d(low_h, high_h, dim=3)
        if bands.original_size is not None:
            original_h, original_w = bands.original_size
            reconstructed = reconstructed[..., :original_h, :original_w]
        return reconstructed
