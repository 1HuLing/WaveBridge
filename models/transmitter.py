from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from models.comm import (
    BPSKModulator,
    DifferentiableCDMA,
    PolarSemanticPacket,
)
from models.compressor import CompressionOutput, LatentTensorPacketizer
from models.polar_codec import PolarCRCCodec


@dataclass
class TransmitterOutput:
    compression: CompressionOutput
    packet: PolarSemanticPacket


class Transmitter(nn.Module):
    # 构建发送端主模块，把压缩 latent 转成极化码包和分组 CDMA 扩频序列。
    def __init__(
        self,
        code_length: int,
        info_length: int,
        design_snr_db: float,
        crc_length: int = 16,
        chips_per_symbol: int = 16,
        bit_depth: int = 4,
        chip_seed: int = 20260520,
        superpose_blocks: bool = False,
        group_size: int = 4,
        latent_clip_value: float = 2.5,
        channel_bit_depths: tuple[int, ...] | list[int] | None = None,
        target_payload_bpp: float | None = None,
        bitstream_order: str = "channel",
        base_layer_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.packetizer = LatentTensorPacketizer(
            bit_depth=bit_depth,
            clip_value=latent_clip_value,
            channel_bit_depths=channel_bit_depths,
            bitstream_order=bitstream_order,
            base_layer_channels=base_layer_channels,
        )
        self.crc_codec = PolarCRCCodec(
            code_length=code_length,
            info_length=info_length,
            design_snr_db=design_snr_db,
            crc_length=crc_length,
        )
        self.modulator = BPSKModulator()
        self.cdma = DifferentiableCDMA(
            code_length,
            chips_per_symbol=chips_per_symbol,
            chip_seed=chip_seed,
            group_size=group_size,
        )
        self.info_length = info_length
        self.payload_length = self.crc_codec.payload_length
        self.crc_length = crc_length
        self.superpose_blocks = superpose_blocks
        self.target_payload_bpp = None
        if target_payload_bpp is not None and float(target_payload_bpp) > 0.0:
            self.target_payload_bpp = float(target_payload_bpp)

    # 将量化 latent 打包成极化码信息块，再生成 BPSK 与分组 CDMA 序列。
    def forward(self, compression: CompressionOutput) -> TransmitterOutput:
        max_valid_bits = None
        if self.target_payload_bpp is not None:
            height, width = compression.original_size
            max_valid_bits = max(1, int(math.floor(float(height * width) * self.target_payload_bpp)))
        payload_bits, valid_info_bits, num_blocks, _quantized_latent = self.packetizer(
            compression.latent,
            self.payload_length,
            max_valid_bits=max_valid_bits,
        )
        transmitted_quantized_latent = self.packetizer.bits_to_latent(
            payload_bits,
            compression.latent_shape,
            valid_info_bits,
        )
        encode_info_bits = self.crc_codec.attach_crc(payload_bits)
        coded_bits = self.crc_codec.encode_info_bits(encode_info_bits)
        symbols = self.modulator(coded_bits)
        spread = self.cdma(symbols, superpose=self.superpose_blocks)
        if not self.superpose_blocks:
            spread = spread.flatten(start_dim=1)
        packet_group_count = (num_blocks + self.cdma.group_size - 1) // self.cdma.group_size
        compression.quantized_latent = transmitted_quantized_latent
        return TransmitterOutput(
            compression=compression,
            packet=PolarSemanticPacket(
                payload_bits=payload_bits,
                info_bits=encode_info_bits,
                coded_bits=coded_bits,
                symbols=symbols,
                spread=spread,
                valid_info_bits=valid_info_bits,
                num_blocks=num_blocks,
                payload_length=self.payload_length,
                crc_length=self.crc_length,
                info_length=self.info_length,
                source_num_blocks=num_blocks,
                group_size=self.cdma.group_size,
                group_count=packet_group_count,
            ),
        )
