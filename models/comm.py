from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class PolarSemanticPacket:
    payload_bits: torch.Tensor
    info_bits: torch.Tensor
    coded_bits: torch.Tensor
    symbols: torch.Tensor
    spread: torch.Tensor
    valid_info_bits: int
    num_blocks: int
    payload_length: int
    crc_length: int
    info_length: int
    source_num_blocks: int | None = None
    group_size: int = 1
    group_count: int | None = None


# 判断输入值是否为 2 的整数次幂，极化码长度必须满足这个条件。
def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


# 检查索引张量是否落在目标维度范围内，避免 CUDA 侧只抛出模糊的 device-side assert。
def validate_index_tensor(indices: torch.Tensor, upper_bound: int, name: str) -> None:
    if indices.numel() == 0:
        return
    if upper_bound <= 0:
        raise ValueError(f"{name} upper_bound must be positive, got {upper_bound}.")
    flat_indices = indices.detach()
    min_index = int(flat_indices.min().item())
    max_index = int(flat_indices.max().item())
    if min_index < 0 or max_index >= upper_bound:
        raise ValueError(
            f"{name} out of range: min={min_index}, max={max_index}, valid=[0, {upper_bound - 1}]."
        )


# 为码片及其反码生成同一个唯一键，避免正负等价码片重复占用码本。
def _canonical_code_key(candidate: torch.Tensor) -> bytes:
    positive_key = candidate.numpy().tobytes()
    negative_key = (-candidate).numpy().tobytes()
    return positive_key if positive_key < negative_key else negative_key


@lru_cache(maxsize=64)
# 在 CPU 上缓存确定性的低相关码片表，避免训练时反复构造码本。
def _cached_low_correlation_chip_codes(num_codes: int, chips_per_code: int, seed: int) -> torch.Tensor:
    if chips_per_code <= 0:
        raise ValueError("chips_per_code must be positive.")
    max_inverse_safe_codes = 2 ** max(chips_per_code - 1, 0)
    if num_codes > max_inverse_safe_codes:
        raise ValueError(
            f"Cannot build {num_codes} inverse-safe binary CDMA codes with only {chips_per_code} chips. "
            f"Increase chips_per_symbol to at least {max(1, math.ceil(math.log2(num_codes)) + 1)}."
        )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    codes: list[torch.Tensor] = []
    seen: set[bytes] = set()

    if chips_per_code <= 20:
        target_weights = sorted(
            range(chips_per_code + 1),
            key=lambda value: (abs(value - chips_per_code / 2.0), value),
        )
        for weight in target_weights:
            candidates_for_weight: list[torch.Tensor] = []
            for positions in itertools.combinations(range(chips_per_code), weight):
                candidate = -torch.ones(chips_per_code, dtype=torch.int8)
                if positions:
                    candidate[list(positions)] = 1
                key = _canonical_code_key(candidate)
                if key in seen:
                    continue
                seen.add(key)
                candidates_for_weight.append(candidate.to(torch.float32))
            if candidates_for_weight:
                order = torch.randperm(len(candidates_for_weight), generator=generator)
                for index in order.tolist():
                    codes.append(candidates_for_weight[index])
                    if len(codes) >= num_codes:
                        break
            if len(codes) >= num_codes:
                break

    batch_size = max(4096, num_codes * 2)
    while len(codes) < num_codes:
        candidates = torch.randint(0, 2, (batch_size, chips_per_code), generator=generator, dtype=torch.int8)
        candidates = candidates * 2 - 1
        order = torch.argsort(candidates.sum(dim=1).abs())
        for candidate in candidates.index_select(0, order):
            key = _canonical_code_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            codes.append(candidate.to(torch.float32))
            if len(codes) >= num_codes:
                break
    return F.normalize(torch.stack(codes, dim=0), p=2, dim=1)


# 按设备和精度返回确定性的低相关 CDMA 码片表。
def build_low_correlation_chip_codes(
    num_codes: int,
    chips_per_code: int,
    seed: int = 20260520,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    codebook = _cached_low_correlation_chip_codes(num_codes, chips_per_code, seed)
    return codebook.to(device=device, dtype=dtype)


# 按组生成 block 级 CDMA 码本，让同组码包可分离，不同组也有独立扰码。
def build_grouped_chip_codes(
    num_codes: int,
    chips_per_code: int,
    group_size: int = 4,
    seed: int = 20260520,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if num_codes <= 0:
        raise ValueError("num_codes must be positive.")
    group_size = max(1, int(group_size))
    local_count = min(group_size, num_codes)
    group_count = (num_codes + group_size - 1) // group_size
    local_codebook = build_low_correlation_chip_codes(
        local_count,
        chips_per_code,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    group_codebook = build_low_correlation_chip_codes(
        group_count,
        chips_per_code,
        seed=seed + 104729,
        device=device,
        dtype=dtype,
    )
    indices = torch.arange(num_codes, device=device, dtype=torch.long)
    local_indices = indices.remainder(local_count)
    group_indices = torch.div(indices, group_size, rounding_mode="floor")
    validate_index_tensor(local_indices, local_codebook.shape[0], "local_indices")
    validate_index_tensor(group_indices, group_codebook.shape[0], "group_indices")
    grouped = local_codebook.index_select(0, local_indices) * group_codebook.index_select(0, group_indices)
    return F.normalize(grouped, p=2, dim=1)


class PolarReliabilityMapper(nn.Module):
    # 初始化极化码可靠性映射器，用高斯近似筛选最可靠的信息位位置。
    def __init__(self, code_length: int, info_length: int, design_snr_db: float) -> None:
        super().__init__()
        if not is_power_of_two(code_length):
            raise ValueError("code_length must be a power of two for polar coding.")
        if code_length != 4096:
            raise ValueError("This project fixes the polar code length to 4096.")
        if info_length > code_length:
            raise ValueError("info_length must be smaller than or equal to code_length.")

        self.code_length = code_length
        self.info_length = info_length
        reliability = self._gaussian_approximation_reliability(code_length, info_length, design_snr_db)
        info_indices = torch.sort(torch.argsort(reliability, descending=True)[:info_length]).values
        mask = torch.zeros(code_length, dtype=torch.bool)
        mask[info_indices] = True
        self.register_buffer("info_indices", info_indices)
        self.register_buffer("frozen_mask", ~mask)
        self.register_buffer("channel_reliability", reliability)

    # 计算高斯近似中的 phi 函数，用于极化信道可靠性递推。
    def _phi(self, value: torch.Tensor) -> torch.Tensor:
        value = value.clamp_min(1e-8)
        return torch.exp(-0.4527 * value.pow(0.86) + 0.0218).clamp(1e-8, 1.0 - 1e-8)

    # 计算高斯近似中的 phi 反函数，把概率映射回 LLR 均值。
    def _phi_inverse(self, value: torch.Tensor) -> torch.Tensor:
        value = value.clamp(1e-8, 1.0 - 1e-8)
        base = ((0.0218 - torch.log(value)) / 0.4527).clamp_min(0.0)
        return base.pow(1.0 / 0.86)

    # 使用 AWGN 高斯近似构造法估计每个极化子信道的可靠性。
    def _gaussian_approximation_reliability(
        self,
        code_length: int,
        info_length: int,
        design_snr_db: float,
    ) -> torch.Tensor:
        rate = info_length / code_length
        ebn0 = 10.0 ** (design_snr_db / 10.0)
        means = torch.tensor([4.0 * rate * ebn0], dtype=torch.float32)
        for _ in range(code_length.bit_length() - 1):
            phi_value = self._phi(means)
            bad_channel = self._phi_inverse(1.0 - (1.0 - phi_value).pow(2.0))
            good_channel = 2.0 * means
            means = torch.stack([bad_channel, good_channel], dim=1).flatten()
        return means

    # 把信息位放入最可靠的极化码字位置，其余冻结位填 0。
    def forward(self, info_bits: torch.Tensor) -> torch.Tensor:
        original_shape = info_bits.shape[:-1]
        flat_bits = info_bits.reshape(-1, info_bits.shape[-1])
        if flat_bits.shape[1] != self.info_length:
            raise ValueError(
                f"PolarReliabilityMapper expected info length {self.info_length}, got {flat_bits.shape[1]}."
            )
        validate_index_tensor(self.info_indices.to(device=flat_bits.device), self.code_length, "polar_info_indices")
        u = torch.zeros(flat_bits.shape[0], self.code_length, device=info_bits.device, dtype=info_bits.dtype)
        expanded_indices = self.info_indices.to(device=flat_bits.device).unsqueeze(0).expand(flat_bits.shape[0], -1)
        u.scatter_(1, expanded_indices, flat_bits[:, : self.info_length])
        return u.view(*original_shape, self.code_length)


class PolarTransform(nn.Module):
    # 初始化快速极化变换模块，避免显式构造巨大生成矩阵。
    def __init__(self, code_length: int) -> None:
        super().__init__()
        if not is_power_of_two(code_length):
            raise ValueError("code_length must be a power of two for polar transform.")
        self.code_length = code_length

    # 执行极化蝶形异或变换，输出完整编码比特。
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        original_shape = u.shape[:-1]
        x = u.reshape(-1, self.code_length)
        stage = 1
        while stage < self.code_length:
            x = x.view(x.shape[0], -1, stage * 2)
            left = x[:, :, :stage]
            right = x[:, :, stage:]
            x = torch.cat([(left + right).remainder(2.0), right], dim=-1)
            x = x.view(-1, self.code_length)
            stage *= 2
        return x.view(*original_shape, self.code_length)


class BPSKModulator(nn.Module):
    # 把二进制比特映射为 BPSK 双极性符号。
    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        # 标准 LLR 约定：正相关代表 bit=0，负相关代表 bit=1。
        return 1.0 - bits * 2.0


class DifferentiableCDMA(nn.Module):
    # 初始化可微 CDMA 模块，使用固定低相关码片和 block 级分组扰码。
    def __init__(
        self,
        code_length: int,
        chips_per_symbol: int = 16,
        chip_seed: int = 20260520,
        group_size: int = 4,
    ) -> None:
        super().__init__()
        self.code_length = code_length
        self.chips_per_symbol = chips_per_symbol
        self.chip_seed = chip_seed
        self.group_size = max(1, int(group_size))
        fixed_codes = build_low_correlation_chip_codes(code_length, chips_per_symbol, seed=chip_seed)
        self.register_buffer("fixed_codes", fixed_codes)

    # 为每个码包生成 block 级低相关扰码，提升多码包可分离性。
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

    # 将码包按 group_size 分组叠加，既缩短嵌入序列，又保留组内低相关码片的可分离性。
    def _groupwise_superpose(self, spread: torch.Tensor) -> torch.Tensor:
        batch, num_blocks, spread_length = spread.shape
        group_count = (num_blocks + self.group_size - 1) // self.group_size
        padded_blocks = group_count * self.group_size
        if padded_blocks > num_blocks:
            spread = F.pad(spread, (0, 0, 0, padded_blocks - num_blocks))
        spread = spread.view(batch, group_count, self.group_size, spread_length)
        spread = spread.sum(dim=2) / math.sqrt(float(self.group_size))
        scale = spread.float().std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4).to(spread.dtype)
        return spread / scale

    # 根据 BPSK 符号生成 CDMA 扩频序列；开启叠加时执行组内叠加而不是全量码包混叠。
    def forward(self, symbols: torch.Tensor, superpose: bool = False) -> torch.Tensor:
        chip_codes = self.fixed_codes
        if symbols.dim() == 2:
            spread = symbols.unsqueeze(-1) * chip_codes.unsqueeze(0)
            spread = spread.flatten(start_dim=1)
            scale = spread.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4).to(spread.dtype)
            spread = spread / scale
        else:
            block_codes = self._build_block_chip_codes(symbols.shape[1], symbols.device, symbols.dtype)
            block_chip_codes = F.normalize(
                chip_codes.to(device=symbols.device, dtype=symbols.dtype).unsqueeze(0)
                * block_codes.unsqueeze(1),
                p=2,
                dim=-1,
            )
            spread = symbols.unsqueeze(-1) * block_chip_codes.unsqueeze(0)
            spread = spread.flatten(start_dim=-2)
            scale = spread.float().std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4).to(spread.dtype)
            spread = spread / scale
            if superpose:
                spread = self._groupwise_superpose(spread)
        return torch.nan_to_num(spread, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
