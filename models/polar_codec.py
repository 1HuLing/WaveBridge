from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.comm import PolarReliabilityMapper, PolarTransform, is_power_of_two


@dataclass
class PolarHardDecodeOutput:
    info_bits: torch.Tensor
    payload_bits: torch.Tensor
    code_bits: torch.Tensor
    crc_pass_mask: torch.Tensor


# 根据 CRC 阶数返回默认多项式，便于发送端和接收端共享统一的校验规则。
def default_crc_polynomial(crc_length: int) -> int:
    if crc_length == 0:
        return 0
    defaults = {
        8: 0x07,
        16: 0x1021,
        24: 0x864CFB,
        32: 0x04C11DB7,
    }
    if crc_length not in defaults:
        raise ValueError(f"Unsupported CRC length: {crc_length}.")
    return defaults[crc_length]


# 将整数形式的 CRC 多项式展开成按最高次幂到常数项排列的 0/1 系数序列。
def polynomial_to_bits(polynomial: int, crc_length: int) -> list[int]:
    if crc_length <= 0:
        return [1]
    if polynomial <= 0:
        raise ValueError("CRC polynomial must be positive.")
    coefficients = [1]
    for shift in range(crc_length - 1, -1, -1):
        coefficients.append((polynomial >> shift) & 1)
    if coefficients[-1] != 1:
        raise ValueError("CRC polynomial must include a non-zero constant term.")
    return coefficients


# 对单个比特序列执行 GF(2) 多项式长除法，返回固定长度 CRC 余数。
def crc_remainder_bits(message_bits: list[int], polynomial_bits: list[int], crc_length: int) -> list[int]:
    if crc_length == 0:
        return []
    working = message_bits + [0] * crc_length
    poly_length = len(polynomial_bits)
    for start in range(len(message_bits)):
        if working[start] == 0:
            continue
        for offset in range(poly_length):
            working[start + offset] ^= polynomial_bits[offset]
    return working[-crc_length:]


class PolarCRCCodec(nn.Module):
    # 构建真实的 CRC-Polar 硬编解码器，发送端负责拼接 CRC，接收端负责 SC 硬译码和 CRC 校验。
    def __init__(
        self,
        code_length: int,
        info_length: int,
        design_snr_db: float,
        crc_length: int = 16,
        crc_polynomial: int | None = None,
    ) -> None:
        super().__init__()
        if not is_power_of_two(code_length):
            raise ValueError("code_length must be a power of two.")
        if info_length <= 0 or info_length > code_length:
            raise ValueError("info_length must be in (0, code_length].")
        if crc_length < 0 or crc_length >= info_length:
            raise ValueError("crc_length must be in [0, info_length).")

        self.code_length = code_length
        self.info_length = info_length
        self.crc_length = crc_length
        self.payload_length = info_length - crc_length
        self.crc_polynomial = 0 if crc_length == 0 else (
            crc_polynomial if crc_polynomial is not None else default_crc_polynomial(crc_length)
        )

        self.mapper = PolarReliabilityMapper(code_length, info_length, design_snr_db)
        self.transform = PolarTransform(code_length)
        self.register_buffer("info_indices", self.mapper.info_indices.detach().clone())
        self.register_buffer("frozen_mask", self.mapper.frozen_mask.detach().clone())
        self.register_buffer("crc_matrix", self._build_crc_generator_matrix(self.payload_length, crc_length, self.crc_polynomial))

    # 为指定 payload 长度构建 CRC 生成矩阵，后续即可用矩阵乘法批量计算 CRC。
    def _build_crc_generator_matrix(
        self,
        payload_length: int,
        crc_length: int,
        polynomial: int,
    ) -> torch.Tensor:
        if crc_length == 0:
            return torch.zeros(payload_length, 0, dtype=torch.float32)
        polynomial_bits = polynomial_to_bits(polynomial, crc_length)
        rows = []
        for bit_index in range(payload_length):
            basis = [0] * payload_length
            basis[bit_index] = 1
            rows.append(crc_remainder_bits(basis, polynomial_bits, crc_length))
        return torch.tensor(rows, dtype=torch.float32)

    # 用预先缓存的 CRC 生成矩阵批量计算 payload 对应的 CRC 校验位。
    def compute_crc_bits(self, payload_bits: torch.Tensor) -> torch.Tensor:
        if self.crc_length == 0:
            shape = (*payload_bits.shape[:-1], 0)
            return payload_bits.new_zeros(shape)
        if payload_bits.shape[-1] != self.payload_length:
            raise ValueError(
                f"Expected payload length {self.payload_length}, got {payload_bits.shape[-1]}."
            )
        flat_payload = payload_bits.reshape(-1, self.payload_length)
        matrix = self.crc_matrix.to(device=payload_bits.device, dtype=payload_bits.dtype)
        crc_bits = torch.matmul(flat_payload, matrix).remainder(2.0)
        return crc_bits.view(*payload_bits.shape[:-1], self.crc_length)

    # 将原始 payload 比特补上 CRC，形成可送入极化可靠位映射器的 K 比特信息块。
    def attach_crc(self, payload_bits: torch.Tensor) -> torch.Tensor:
        if payload_bits.shape[-1] != self.payload_length:
            raise ValueError(
                f"Expected payload length {self.payload_length}, got {payload_bits.shape[-1]}."
            )
        crc_bits = self.compute_crc_bits(payload_bits)
        return torch.cat([payload_bits, crc_bits], dim=-1)

    # 将信息位映射到极化码字，方便发送端和接收端统一复用同一套编码逻辑。
    def encode_info_bits(self, info_bits: torch.Tensor) -> torch.Tensor:
        if info_bits.shape[-1] != self.info_length:
            raise ValueError(
                f"Expected info length {self.info_length}, got {info_bits.shape[-1]}."
            )
        return self.transform(self.mapper(info_bits))

    # 从信息块中拆出 payload 部分，供 latent 反量化和 BER 评估使用。
    def extract_payload(self, info_bits: torch.Tensor) -> torch.Tensor:
        if info_bits.shape[-1] != self.info_length:
            raise ValueError(
                f"Expected info length {self.info_length}, got {info_bits.shape[-1]}."
            )
        return info_bits[..., : self.payload_length]

    # 对硬判决信息块执行 CRC 校验，返回每个码包是否通过校验。
    def check_crc(self, info_bits: torch.Tensor) -> torch.Tensor:
        if self.crc_length == 0:
            return torch.ones(info_bits.shape[:-1], device=info_bits.device, dtype=torch.bool)
        payload_bits = self.extract_payload(info_bits)
        received_crc = info_bits[..., self.payload_length :]
        computed_crc = self.compute_crc_bits(payload_bits)
        return torch.all(received_crc == computed_crc, dim=-1)

    # 执行极化码译码中的 f 合并，用 min-sum 近似保持数值稳定。
    def _f_combine(self, left_llr: torch.Tensor, right_llr: torch.Tensor) -> torch.Tensor:
        return torch.sign(left_llr) * torch.sign(right_llr) * torch.minimum(left_llr.abs(), right_llr.abs())

    # 执行极化码译码中的 g 合并，把左分支硬判决反馈到右分支 LLR。
    def _g_combine(
        self,
        left_llr: torch.Tensor,
        right_llr: torch.Tensor,
        left_bits: torch.Tensor,
    ) -> torch.Tensor:
        return right_llr + (1.0 - 2.0 * left_bits) * left_llr

    # 递归执行批量 SC 硬译码，同时返回叶子决策和上层 g 合并所需的 partial sums。
    def _sc_decode_recursive(
        self,
        llr: torch.Tensor,
        frozen_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = llr.shape[-1]
        if length == 1:
            if bool(frozen_mask[0].item()):
                leaf_bits = llr.new_zeros(llr.shape[0], 1)
            else:
                leaf_bits = (llr < 0).to(llr.dtype)
            return leaf_bits, leaf_bits

        half = length // 2
        left_llr = self._f_combine(llr[:, :half], llr[:, half:])
        left_u, left_beta = self._sc_decode_recursive(left_llr, frozen_mask[:half])
        right_llr = self._g_combine(llr[:, :half], llr[:, half:], left_beta)
        right_u, right_beta = self._sc_decode_recursive(right_llr, frozen_mask[half:])
        u_hat = torch.cat([left_u, right_u], dim=-1)
        beta = torch.cat([(left_beta + right_beta).remainder(2.0), right_beta], dim=-1)
        return u_hat, beta

    # 对输入 LLR 执行真实的 Polar SC 硬译码，并恢复 info、payload 与重编码后的码字。
    def decode(self, llr: torch.Tensor) -> PolarHardDecodeOutput:
        if llr.shape[-1] != self.code_length:
            raise ValueError(f"Expected code length {self.code_length}, got {llr.shape[-1]}.")

        original_shape = llr.shape[:-1]
        flat_llr = llr.reshape(-1, self.code_length).float()
        frozen_mask = self.frozen_mask.to(device=llr.device)
        with torch.no_grad():
            u_hat, _ = self._sc_decode_recursive(flat_llr, frozen_mask)
            info_bits = u_hat.index_select(1, self.info_indices.to(device=llr.device))
            payload_bits = self.extract_payload(info_bits)
            code_bits = self.encode_info_bits(info_bits)
            crc_pass_mask = self.check_crc(info_bits)

        return PolarHardDecodeOutput(
            info_bits=info_bits.view(*original_shape, self.info_length).to(dtype=llr.dtype),
            payload_bits=payload_bits.view(*original_shape, self.payload_length).to(dtype=llr.dtype),
            code_bits=code_bits.view(*original_shape, self.code_length).to(dtype=llr.dtype),
            crc_pass_mask=crc_pass_mask.view(*original_shape),
        )
