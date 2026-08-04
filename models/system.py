from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
from typing import Any, Dict

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.channel import DifferentiableRobustChannel
from models.compressor import LearnedImageCompressor, resolve_latent_channel_configuration
from models.generator import GeneratorOutput, PolarGenerator
from models.qim import WaveletQIMChannel
from models.receiver import Receiver, ReceiverOutput
from models.transmitter import Transmitter, TransmitterOutput
from utils.config import resolve_project_path
from utils.datasets import IMAGE_SUFFIXES


@dataclass
class SystemOutput:
    transmitter: TransmitterOutput
    generator: GeneratorOutput
    receiver: ReceiverOutput


class StegoNaturalizer(nn.Module):
    # 对 QIM 后的最终载密图做极小幅度高频自然化修正，初始输出严格等于输入载密图。
    def __init__(
        self,
        channels: int,
        hidden_channels: int = 24,
        max_delta: float = 0.0015,
        highpass_kernel: int = 5,
        enabled: bool = False,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.max_delta = float(max(0.0, max_delta))
        self.highpass_kernel = max(1, int(highpass_kernel) | 1)
        hidden_channels = max(8, int(hidden_channels))
        self.input_proj = nn.Conv2d(channels * 2, hidden_channels, kernel_size=3, padding=1)
        self.depthwise = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels),
            nn.GELU(),
        )
        self.output_proj = nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    # 去除残差的局部低频分量，避免自然化模块改变整体亮度和颜色统计。
    def _highpass_residual(self, residual: torch.Tensor) -> torch.Tensor:
        if self.highpass_kernel <= 1:
            return residual
        pad = self.highpass_kernel // 2
        padded = F.pad(residual, (pad, pad, pad, pad), mode="reflect")
        local_mean = F.avg_pool2d(padded, kernel_size=self.highpass_kernel, stride=1)
        return residual - local_mean

    # 根据 QIM 载密图和 QIM 前底图预测受限残差，并保持像素范围合法。
    def forward(self, stego_image: torch.Tensor, base_image: torch.Tensor | None = None) -> torch.Tensor:
        if (not self.enabled) or self.max_delta <= 0.0:
            return stego_image
        if base_image is None:
            base_image = stego_image
        q_distortion = torch.nan_to_num(stego_image - base_image, nan=0.0, posinf=1.0, neginf=-1.0)
        features = torch.cat([stego_image, q_distortion], dim=1)
        residual = self.output_proj(self.depthwise(self.input_proj(features)))
        residual = torch.tanh(residual) * self.max_delta
        residual = self._highpass_residual(residual).clamp(-self.max_delta, self.max_delta)
        return (stego_image + residual).clamp(0.0, 1.0)


class WaveBridegeSystem(nn.Module):
    # 组装压缩器、发送端、景观先验生成器和接收端，形成新的 latent 传输闭环。
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        channels = config["data"]["channels"]
        code_length = config["transmitter"]["polar_code"]["code_length"]
        info_length = config["transmitter"]["polar_code"]["info_length"]
        design_snr_db = config["transmitter"]["polar_code"]["design_snr_db"]
        crc_length = config["transmitter"]["polar_code"].get("crc_length", 16)
        latent_dim = config["gan"]["latent_dim"]
        chips_per_symbol = config["transmitter"]["chips_per_symbol"]
        group_size = config["transmitter"].get("group_size", 4)
        bit_depth = config["transmitter"].get("bit_depth", 4)
        compressor_cfg = config.get("compressor", {})
        transmitted_channels = int(compressor_cfg.get("latent_channels", channels * 4))
        ll_latent_channels, hf_latent_channels, latent_channel_bit_depths = resolve_latent_channel_configuration(
            latent_channels=transmitted_channels,
            bit_depth=bit_depth,
            ll_latent_channels=compressor_cfg.get("ll_latent_channels"),
            hf_latent_channels=compressor_cfg.get("hf_latent_channels"),
            ll_bit_depth=compressor_cfg.get("ll_bit_depth"),
            hf_bit_depth=compressor_cfg.get("hf_bit_depth"),
        )
        transmitted_channels = ll_latent_channels + hf_latent_channels
        base_layer_channels = int(compressor_cfg.get("base_layer_channels", ll_latent_channels))
        latent_hidden_channels = compressor_cfg.get("hidden_channels", 64)
        latent_clip_value = compressor_cfg.get("clip_value", 2.5)
        latent_sanitize_limit_factor = float(compressor_cfg.get("latent_sanitize_limit_factor", 1.25))
        bitstream_order = compressor_cfg.get("bitstream_order", config["transmitter"].get("bitstream_order", "channel"))
        codec_layout_version = str(compressor_cfg.get("codec_layout_version", "")).strip().lower()
        rgb_anchor_ll_mode = str(compressor_cfg.get("rgb_anchor_ll_mode", "blend")).strip().lower()
        if rgb_anchor_ll_mode == "residual_base" and codec_layout_version != "rgb_anchor_residual_base_v2":
            # 历史高指标 checkpoint 在当前代码路径下并没有稳定复现 residual-base 解码，
            # 兼容模式里继续沿用已验证过更稳的 blend 低频融合语义。
            rgb_anchor_ll_mode = "blend"
        if not codec_layout_version:
            # 旧版高指标 checkpoint 的 latent 动态范围远大于当前新代码里的保守裁剪范围。
            # 这里放宽净化上限，避免在解码前把大部分 legacy latent 直接夹坏。
            latent_sanitize_limit_factor = max(latent_sanitize_limit_factor, 8.0)
        self.compressor = LearnedImageCompressor(
            image_channels=channels,
            hidden_channels=latent_hidden_channels,
            latent_channels=transmitted_channels,
            bit_depth=bit_depth,
            clip_value=latent_clip_value,
            downsample_stages=int(compressor_cfg.get("downsample_stages", 3)),
            ll_band_scale=float(compressor_cfg.get("ll_band_scale", 0.75)),
            hf_band_scale=float(compressor_cfg.get("hf_band_scale", 0.35)),
            ll_band_bias=float(compressor_cfg.get("ll_band_bias", 0.40)),
            ll_latent_channels=ll_latent_channels,
            hf_latent_channels=hf_latent_channels,
            ll_bit_depth=compressor_cfg.get("ll_bit_depth"),
            hf_bit_depth=compressor_cfg.get("hf_bit_depth"),
            soft_preview_quant=compressor_cfg.get("soft_preview_quant", True),
            soft_preview_blend=float(compressor_cfg.get("soft_preview_blend", 1.0)),
            bitstream_order=bitstream_order,
            base_layer_channels=base_layer_channels,
            hyperprior_channels=int(compressor_cfg.get("hyperprior_channels", max(32, latent_hidden_channels // 2))),
            use_attention=bool(compressor_cfg.get("use_attention", True)),
            use_hyperprior_modulation=bool(compressor_cfg.get("use_hyperprior_modulation", True)),
            use_post_refiner=bool(compressor_cfg.get("use_post_refiner", True)),
            post_refiner_scale=float(compressor_cfg.get("post_refiner_scale", 0.05)),
            soft_quant_noise_scale=float(compressor_cfg.get("soft_quant_noise_scale", 1.0)),
            base_layer_soft_quant_scale=float(compressor_cfg.get("base_layer_soft_quant_scale", 1.0)),
            enhancement_soft_quant_scale=float(compressor_cfg.get("enhancement_soft_quant_scale", 1.0)),
            use_rgb_anchor=bool(compressor_cfg.get("use_rgb_anchor", False)),
            rgb_anchor_range=float(compressor_cfg.get("rgb_anchor_range", 0.95)),
            rgb_anchor_blend=float(compressor_cfg.get("rgb_anchor_blend", 0.70)),
            rgb_anchor_max_blend=float(compressor_cfg.get("rgb_anchor_max_blend", 0.98)),
            rgb_anchor_image_blend=float(compressor_cfg.get("rgb_anchor_image_blend", 0.0)),
            rgb_anchor_ll_mode=rgb_anchor_ll_mode,
            ll_anchor_channels=int(compressor_cfg.get("ll_anchor_channels", 0)),
            ll_anchor_range=float(compressor_cfg.get("ll_anchor_range", 1.50)),
            ll_anchor_blend=float(compressor_cfg.get("ll_anchor_blend", 0.88)),
            ll_anchor_max_blend=float(compressor_cfg.get("ll_anchor_max_blend", 0.98)),
            latent_sanitize_limit_factor=latent_sanitize_limit_factor,
        )
        self.transmitter = Transmitter(
            code_length=code_length,
            info_length=info_length,
            design_snr_db=design_snr_db,
            crc_length=crc_length,
            chips_per_symbol=chips_per_symbol,
            bit_depth=bit_depth,
            chip_seed=config["transmitter"].get("chip_seed", 20260520),
            superpose_blocks=config["transmitter"].get("superpose_blocks", False),
            group_size=group_size,
            latent_clip_value=latent_clip_value,
            channel_bit_depths=latent_channel_bit_depths,
            target_payload_bpp=config["transmitter"].get("target_bpp", config["transmitter"].get("payload_bpp")),
            bitstream_order=bitstream_order,
            base_layer_channels=base_layer_channels,
        )
        active_dataset_name = str(config.get("datasets", {}).get("active", "")).strip().lower()
        self.generator = PolarGenerator(
            latent_dim=latent_dim,
            semantic_channels=config["gan"]["semantic_channels"],
            hidden_channels=config["gan"]["hidden_channels"],
            carrier_channels=channels,
            code_length=code_length,
            chips_per_symbol=chips_per_symbol,
            group_size=group_size,
            chip_seed=config["transmitter"].get("chip_seed", 20260520),
            residual_gain=config["gan"]["residual_gain"],
            prior_mix=config["gan"].get("prior_mix", 0.05),
            preview_bridge_strength=config["gan"].get("preview_bridge_strength"),
            preview_guidance_mix=config["gan"].get("preview_guidance_mix", 0.15),
            preview_guidance_pool=config["gan"].get("preview_guidance_pool", 16),
            template_strength=config["gan"].get("template_strength", 0.25),
            direct_template_gain=config["gan"].get("direct_template_gain", 0.18),
            residual_delta_clip=config["gan"].get("residual_delta_clip", 0.018),
            chroma_residual_ratio=config["gan"].get("chroma_residual_ratio", 0.08),
            local_energy_kernel=config["gan"].get("local_energy_kernel", 15),
            local_energy_limit=config["gan"].get("local_energy_limit", 0.010),
            analog_residual_gain=config["gan"].get("analog_residual_gain", 0.0),
            analog_residual_clip=config["gan"].get("analog_residual_clip", 0.080),
            analog_residual_chroma_ratio=config["gan"].get("analog_residual_chroma_ratio", 0.35),
            analog_lowfreq_gain=config["gan"].get("analog_lowfreq_gain", 0.0),
            analog_lowfreq_clip=config["gan"].get("analog_lowfreq_clip", 0.180),
            analog_lowfreq_chroma_ratio=config["gan"].get("analog_lowfreq_chroma_ratio", 0.55),
            analog_detail_hh_ratio=config["gan"].get("analog_detail_hh_ratio", 1.0),
            analog_lowfreq_hh_ratio=config["gan"].get("analog_lowfreq_hh_ratio", 1.0),
            analog_ll_direct_gain=config["gan"].get("analog_ll_direct_gain", 0.0),
            analog_injection_target=config["gan"].get("analog_injection_target", "generated"),
            source_bridge_strength=config["gan"].get("source_bridge_strength", 1.0),
            external_carrier_source_bridge=config["gan"].get("external_carrier_source_bridge", False),
            external_carrier_blend=config["gan"].get(
                "external_carrier_blend",
                config.get("carrier_bank", {}).get("external_blend", 0.0),
            ),
            external_carrier_lowfreq_only=config["gan"].get("external_carrier_lowfreq_only", True),
            preserve_source_bridge_with_external_carrier=config["gan"].get(
                "preserve_source_bridge_with_external_carrier",
                True,
            ),
            carrier_first_mode=config["gan"].get("carrier_first_mode", False),
            carrier_style_residual_gain=config["gan"].get("carrier_style_residual_gain", 0.0),
            carrier_style_color_mix=config["gan"].get("carrier_style_color_mix", 0.0),
            carrier_style_lowfreq_kernel=config["gan"].get("carrier_style_lowfreq_kernel", 9),
            template_mode=config["gan"].get("template_mode", "auto"),
            dataset_name=active_dataset_name,
        )
        carrier_bank_cfg = config.get("carrier_bank", {})
        receiver_cfg = dict(config["receiver"])
        legacy_receiver_refine_compat = (
            (not codec_layout_version)
            and ("analog_demod_residual_mix" not in receiver_cfg)
            and float(receiver_cfg.get("image_refine_strength", 0.0)) > 0.0
            and not bool(receiver_cfg.get("bypass_image_refiner_on_full_hard_decode", False))
        )
        if legacy_receiver_refine_compat:
            # 旧版高指标恢复 checkpoint 没有显式保存后续新增的 demod residual mix 字段。
            # 当前默认值 0.05 会把模拟残差补偿压到极弱，导致 restored 基本退化成 decoded。
            receiver_cfg["analog_demod_residual_mix"] = 1.0
            # 旧版恢复链没有和当前的 LL direct 混合一起标定；继续沿用默认 1.0 会显著拉低恢复 PSNR。
            receiver_cfg["analog_ll_direct_decode_mix"] = 0.0
        if bool(carrier_bank_cfg.get("enabled", False)) and (
            bool(config.get("gan", {}).get("carrier_first_mode", False))
            or bool(config.get("gan", {}).get("external_carrier_source_bridge", False))
            or float(config.get("gan", {}).get("external_carrier_blend", 0.0)) > 0.0
        ) and bool(
            carrier_bank_cfg.get("protect_receiver_refiner", True)
        ):
            # Natural carrier mode should not let carrier appearance dominate recovery, but
            # turning cover guidance fully off can collapse restored_image back to decoded_image.
            # Preserve the checkpoint/runtime guidance unless the config explicitly disables it.
            receiver_cfg["disable_cover_guidance_in_refiner"] = bool(
                carrier_bank_cfg.get(
                    "disable_cover_guidance_in_refiner",
                    receiver_cfg.get("disable_cover_guidance_in_refiner", False),
                )
            )
            if receiver_cfg["disable_cover_guidance_in_refiner"]:
                receiver_cfg["cover_guidance_strength_in_refiner"] = 0.0
            else:
                receiver_cfg["cover_guidance_strength_in_refiner"] = float(
                    carrier_bank_cfg.get(
                        "cover_guidance_strength_in_refiner",
                        receiver_cfg.get("cover_guidance_strength_in_refiner", 1.0),
                    )
                )
            if bool(carrier_bank_cfg.get("force_receiver_refiner", True)):
                receiver_cfg["bypass_image_refiner_on_external_hard_decode"] = False
                receiver_cfg["bypass_image_refiner_on_full_hard_decode"] = False
            else:
                receiver_cfg["bypass_image_refiner_on_external_hard_decode"] = bool(
                    receiver_cfg.get("bypass_image_refiner_on_external_hard_decode", False)
                )
                receiver_cfg["bypass_image_refiner_on_full_hard_decode"] = bool(
                    receiver_cfg.get("bypass_image_refiner_on_full_hard_decode", False)
                )
        qim_cfg = config.get("qim", {})
        self.qim_enabled = bool(qim_cfg.get("enabled", False))
        self.qim_clean_carrier_mode = str(qim_cfg.get("clean_carrier_mode", "none")).strip().lower()
        if self.qim_clean_carrier_mode not in {"none", "dummy_bits"}:
            raise ValueError("qim.clean_carrier_mode must be either 'none' or 'dummy_bits'.")
        self.qim_clean_carrier_seed = int(qim_cfg.get("clean_carrier_seed", 20260607))
        self.qim_channel = WaveletQIMChannel(
            delta=float(qim_cfg.get("delta", 0.012)),
            strength=float(qim_cfg.get("strength", 1.0)),
            llr_scale=float(qim_cfg.get("llr_scale", 10.0)),
            llr_clamp=float(qim_cfg.get("llr_clamp", receiver_cfg.get("llr_clamp", 18.0))),
            llr_polarity=float(qim_cfg.get("llr_polarity", 1.0)),
            carrier_bands=qim_cfg.get("carrier_bands", ("lh", "hl", "hh")),
            position_mode=qim_cfg.get("position_mode", "linear"),
            position_seed=int(qim_cfg.get("position_seed", config["transmitter"].get("chip_seed", 20260520))),
            adaptive_strength=bool(qim_cfg.get("adaptive_strength", True)),
            texture_floor=float(qim_cfg.get("texture_floor", 0.75)),
            texture_sharpness=float(qim_cfg.get("texture_sharpness", 1.8)),
            texture_kernel=int(qim_cfg.get("texture_kernel", 17)),
            texture_abs_threshold=float(qim_cfg.get("texture_abs_threshold", 0.004)),
            repetition_factor=int(qim_cfg.get("repetition_factor", 1)),
            extract_smooth_kernel=int(qim_cfg.get("extract_smooth_kernel", 1)),
            dither_enabled=bool(qim_cfg.get("dither_enabled", True)),
            dither_strength=float(qim_cfg.get("dither_strength", 0.45)),
            domain=qim_cfg.get("domain", "wavelet"),
            dct_coefficients=qim_cfg.get("dct_coefficients"),
            dct_channels=qim_cfg.get("dct_channels", "y"),
            dct_quality=int(qim_cfg.get("dct_quality", 50)),
            dct_quant_scale=float(qim_cfg.get("dct_quant_scale", 1.0)),
            dct_parity_mode=bool(qim_cfg.get("dct_parity_mode", False)),
        )
        robust_qim_cfg = dict(qim_cfg)
        robust_qim_cfg.update(config.get("robust_qim", {}))
        self.robust_qim_enabled = bool(robust_qim_cfg.get("enabled", False))
        self.robust_qim_clean_fusion_weight = float(
            max(0.0, min(1.0, robust_qim_cfg.get("clean_fusion_weight", 0.0)))
        )
        self.robust_qim_attack_fusion_weight = float(
            max(0.0, min(1.0, robust_qim_cfg.get("attack_fusion_weight", 1.0)))
        )
        self.robust_qim_embed_enabled = bool(robust_qim_cfg.get("embed_enabled", False))
        self.robust_qim_embed_alpha = float(
            max(0.0, min(1.0, robust_qim_cfg.get("embed_alpha", 0.0)))
        )
        self.robust_qim_channel = (
            WaveletQIMChannel(
                delta=float(robust_qim_cfg.get("delta", 1.0)),
                strength=float(robust_qim_cfg.get("strength", 1.0)),
                llr_scale=float(robust_qim_cfg.get("llr_scale", 36.0)),
                llr_clamp=float(robust_qim_cfg.get("llr_clamp", receiver_cfg.get("llr_clamp", 18.0))),
                llr_polarity=float(robust_qim_cfg.get("llr_polarity", qim_cfg.get("llr_polarity", 1.0))),
                carrier_bands=robust_qim_cfg.get("carrier_bands", qim_cfg.get("carrier_bands", ("lh", "hl", "hh"))),
                position_mode=robust_qim_cfg.get("position_mode", qim_cfg.get("position_mode", "stratified")),
                position_seed=int(
                    robust_qim_cfg.get(
                        "position_seed",
                        int(qim_cfg.get("position_seed", config["transmitter"].get("chip_seed", 20260520))) + 7919,
                    )
                ),
                adaptive_strength=bool(robust_qim_cfg.get("adaptive_strength", False)),
                texture_floor=float(robust_qim_cfg.get("texture_floor", 1.0)),
                texture_sharpness=float(robust_qim_cfg.get("texture_sharpness", 1.8)),
                texture_kernel=int(robust_qim_cfg.get("texture_kernel", 17)),
                texture_abs_threshold=float(robust_qim_cfg.get("texture_abs_threshold", 0.004)),
                repetition_factor=int(robust_qim_cfg.get("repetition_factor", 1)),
                extract_smooth_kernel=int(robust_qim_cfg.get("extract_smooth_kernel", 1)),
                dither_enabled=bool(robust_qim_cfg.get("dither_enabled", False)),
                dither_strength=float(robust_qim_cfg.get("dither_strength", 0.0)),
                domain=robust_qim_cfg.get("domain", "dct"),
                dct_coefficients=robust_qim_cfg.get("dct_coefficients", "auto56"),
                dct_channels=robust_qim_cfg.get("dct_channels", "y"),
                dct_quality=int(robust_qim_cfg.get("dct_quality", 50)),
                dct_quant_scale=float(robust_qim_cfg.get("dct_quant_scale", 1.0)),
                dct_parity_mode=bool(robust_qim_cfg.get("dct_parity_mode", True)),
            )
            if self.robust_qim_enabled
            else None
        )
        stego_naturalizer_cfg = config.get("stego_naturalizer", {})
        self.stego_naturalizer_apply_before_qim = bool(
            stego_naturalizer_cfg.get("apply_before_qim", False)
        )
        self.stego_naturalizer = StegoNaturalizer(
            channels=channels,
            hidden_channels=int(stego_naturalizer_cfg.get("hidden_channels", 24)),
            max_delta=float(stego_naturalizer_cfg.get("max_delta", 0.0015)),
            highpass_kernel=int(stego_naturalizer_cfg.get("highpass_kernel", 5)),
            enabled=bool(stego_naturalizer_cfg.get("enabled", False)),
        )
        self.receiver = Receiver(
            image_channels=channels,
            hidden_dim=receiver_cfg["hidden_dim"],
            code_length=code_length,
            chips_per_symbol=chips_per_symbol,
            mamba_dim=receiver_cfg["mamba_dim"],
            num_layers=receiver_cfg["num_layers"],
            info_length=info_length,
            design_snr_db=design_snr_db,
            crc_length=crc_length,
            latent_channels=transmitted_channels,
            bit_depth=bit_depth,
            channel_bit_depths=latent_channel_bit_depths,
            decode_block_chunk_size=receiver_cfg.get("decode_block_chunk_size", 8),
            base_layer_channels=base_layer_channels,
            llr_init_scale=receiver_cfg.get("llr_init_scale", 6.0),
            llr_max_scale=receiver_cfg.get("llr_max_scale", 24.0),
            llr_clamp=receiver_cfg.get("llr_clamp", 20.0),
            bp_iterations=receiver_cfg.get("bp_iterations", 2),
            chip_seed=config["transmitter"].get("chip_seed", 20260520),
            confidence_gate_threshold=receiver_cfg.get("confidence_gate_threshold", 0.35),
            confidence_gate_floor=receiver_cfg.get("confidence_gate_floor", 0.0),
            semantic_fusion_temperature=receiver_cfg.get("semantic_fusion_temperature", 0.75),
            hard_decode_prior_weight=receiver_cfg.get("hard_decode_prior_weight", 0.35),
            hard_decode_retry_weight=receiver_cfg.get("hard_decode_retry_weight", 0.0),
            hard_reconstruction_ratio=receiver_cfg.get("hard_reconstruction_ratio", 0.0),
            symbol_hint_mix=receiver_cfg.get("symbol_hint_mix", 0.30),
            full_decode_feature_mix=receiver_cfg.get("full_decode_feature_mix", 0.10),
            full_decode_feature_training_threshold=receiver_cfg.get(
                "full_decode_feature_training_threshold",
                0.20,
            ),
            full_decode_feature_min_mix_ratio=receiver_cfg.get(
                "full_decode_feature_min_mix_ratio",
                0.02,
            ),
            allow_feature_correction_on_external_hard_decode=receiver_cfg.get(
                "allow_feature_correction_on_external_hard_decode",
                False,
            ),
            image_refine_strength=receiver_cfg.get("image_refine_strength", 0.08),
            image_refine_min_decoded_blend=receiver_cfg.get("image_refine_min_decoded_blend", 0.0),
            analog_lowfreq_decode_gain=receiver_cfg.get("analog_lowfreq_decode_gain", 0.0),
            analog_detail_decode_gain=receiver_cfg.get("analog_detail_decode_gain"),
            analog_inverse_companding=receiver_cfg.get("analog_inverse_companding", False),
            analog_lowfreq_sender_gain=config["gan"].get("analog_lowfreq_gain", 1.0),
            analog_lowfreq_sender_clip=config["gan"].get("analog_lowfreq_clip", 0.20),
            analog_detail_sender_gain=config["gan"].get("analog_residual_gain", 1.0),
            analog_detail_sender_clip=config["gan"].get("analog_residual_clip", 0.10),
            analog_detail_hh_ratio=config["gan"].get("analog_detail_hh_ratio", 1.0),
            analog_lowfreq_hh_ratio=config["gan"].get("analog_lowfreq_hh_ratio", 1.0),
            analog_ll_direct_sender_gain=config["gan"].get("analog_ll_direct_gain", 0.0),
            analog_ll_direct_decode_mix=receiver_cfg.get("analog_ll_direct_decode_mix", 0.0),
            analog_demod_lowpass_kernel=receiver_cfg.get("analog_demod_lowpass_kernel", 1),
            analog_demod_residual_mix=receiver_cfg.get("analog_demod_residual_mix", 0.05),
            image_refine_max_side=receiver_cfg.get("image_refine_max_side", 256),
            image_refine_checkpointing=receiver_cfg.get("image_refine_checkpointing", True),
            image_refine_eval_full_resolution=receiver_cfg.get(
                "image_refine_eval_full_resolution",
                False if not codec_layout_version else True,
            ),
            bypass_image_refiner_on_external_hard_decode=receiver_cfg.get(
                "bypass_image_refiner_on_external_hard_decode",
                False,
            ),
            bypass_image_refiner_on_full_hard_decode=receiver_cfg.get(
                "bypass_image_refiner_on_full_hard_decode",
                False,
            ),
            disable_cover_guidance_in_refiner=receiver_cfg.get(
                "disable_cover_guidance_in_refiner",
                False,
            ),
            cover_guidance_strength_in_refiner=receiver_cfg.get(
                "cover_guidance_strength_in_refiner",
                1.0,
            ),
            external_llr_cover_guidance_strength=receiver_cfg.get(
                "external_llr_cover_guidance_strength",
                None,
            ),
            train_decode_blocks=receiver_cfg.get("train_decode_blocks", 32),
            eval_decode_blocks=receiver_cfg.get("eval_decode_blocks", 128),
            group_size=group_size,
            latent_clip_value=latent_clip_value,
            max_group_count=receiver_cfg.get("max_group_count", 128),
            pilot_calibration_enabled=receiver_cfg.get("pilot_calibration_enabled", True),
            pilot_target_abs=receiver_cfg.get("pilot_target_abs", 1.6),
            strict_external_llr_hard_decode=receiver_cfg.get("strict_external_llr_hard_decode", False),
            bridge_external_llr_for_metrics_only=receiver_cfg.get(
                "bridge_external_llr_for_metrics_only",
                False,
            ),
            external_hard_decode_crc_concealment=receiver_cfg.get(
                "external_hard_decode_crc_concealment",
                False,
            ),
            external_llr_adapter_gain_limit=receiver_cfg.get("external_llr_adapter_gain_limit", 0.25),
            external_llr_adapter_bias_limit=receiver_cfg.get("external_llr_adapter_bias_limit", 0.10),
            external_llr_adapter_residual_mix=receiver_cfg.get("external_llr_adapter_residual_mix", 0.20),
            external_llr_adapter_delta_scale_limit=receiver_cfg.get("external_llr_adapter_delta_scale_limit", 0.0),
            bitstream_order=bitstream_order,
        )
        robust_cfg = config.get("robust_channel", {})
        self.robust_channel = (
            DifferentiableRobustChannel(
                noise_std=robust_cfg.get("noise_std", 0.0),
                blur_prob=robust_cfg.get("blur_prob", 0.0),
                resize_prob=robust_cfg.get("resize_prob", 0.0),
                quantize_prob=robust_cfg.get("quantize_prob", 0.0),
                jpeg_levels=robust_cfg.get("jpeg_levels", 32),
                eval_quantize=robust_cfg.get("eval_quantize", False),
                eval_noise_std=robust_cfg.get("eval_noise_std", 0.0),
                dct_jpeg_prob=robust_cfg.get("dct_jpeg_prob", 0.0),
                eval_dct_jpeg=robust_cfg.get("eval_dct_jpeg", False),
                dct_quality=robust_cfg.get("dct_quality", 50),
                yuv_keep_weights=tuple(robust_cfg.get("yuv_keep_weights", (25, 9, 9))),
                contrast_prob=robust_cfg.get("contrast_prob", 0.0),
                contrast_factor_range=tuple(robust_cfg.get("contrast_factor_range", (0.92, 1.08))),
                rotation_prob=robust_cfg.get("rotation_prob", 0.0),
                rotation_degree=robust_cfg.get("rotation_degree", 1.0),
                resize_scale_range=tuple(robust_cfg.get("resize_scale_range", (0.75, 0.75))),
                blur_kernel_sizes=tuple(robust_cfg.get("blur_kernel_sizes", (3, 5))),
                blur_sigma=robust_cfg.get("blur_sigma", 1.0),
                blur_force_gaussian=robust_cfg.get("blur_force_gaussian", False),
                resize_down_modes=tuple(robust_cfg.get("resize_down_modes", ("bicubic", "bilinear"))),
                resize_up_modes=tuple(robust_cfg.get("resize_up_modes", ("bicubic", "bilinear"))),
                resize_second_pass_prob=robust_cfg.get("resize_second_pass_prob", 0.35),
                sample_single_attack=robust_cfg.get("sample_single_attack", False),
                residual_detach_mode=robust_cfg.get("residual_detach_mode", True),
            )
            if robust_cfg.get("enabled", False)
            else nn.Identity()
        )
        self.latent_dim = latent_dim
        noise_cfg = config.get("noise", {})
        self.noise_mode = noise_cfg.get("mode", "random")
        self.noise_std = noise_cfg.get("std", 1.0)
        training_cfg = config.get("training", {})
        self.teacher_payload_ratio = float(training_cfg.get("teacher_payload_ratio", 0.0))
        self.teacher_channel_ratio = float(training_cfg.get("teacher_channel_ratio", 0.0))
        self.register_buffer("landscape_anchor", self._build_landscape_anchor(latent_dim, noise_cfg.get("seed", 2026)))
        self.carrier_bank_enabled = bool(carrier_bank_cfg.get("enabled", False))
        self.carrier_bank_forward_into_generator = bool(carrier_bank_cfg.get("forward_into_generator", True))
        self.carrier_bank_paths = self._load_carrier_bank_paths(config, carrier_bank_cfg)
        self.carrier_bank_seed = int(carrier_bank_cfg.get("seed", 20260617))
        self.carrier_bank_stride = max(1, int(carrier_bank_cfg.get("stride", 9973)))
        self.carrier_bank_random_train = bool(carrier_bank_cfg.get("random_train", True))
        if self.carrier_bank_enabled and not self.carrier_bank_paths:
            raise FileNotFoundError("carrier_bank.enabled is true but no carrier images were found.")

    # 加载自然载体图库路径，用真实自然图作为载密底图，避免低质量随机生成图主导 FID/BRISQUE。
    def _load_carrier_bank_paths(self, config: Dict[str, Any], carrier_bank_cfg: Dict[str, Any]) -> list[Path]:
        if not bool(carrier_bank_cfg.get("enabled", False)):
            return []
        explicit_dirs = carrier_bank_cfg.get("dirs") or []
        if isinstance(explicit_dirs, (str, Path)):
            explicit_dirs = [explicit_dirs]
        candidate_dirs: list[Path] = []
        for raw_dir in explicit_dirs:
            resolved = resolve_project_path(raw_dir, base_dir=config.get("project_root"))
            if resolved is not None and resolved.exists():
                candidate_dirs.append(resolved)
        dataset_name = str(carrier_bank_cfg.get("dataset", "")).strip().lower()
        dataset_cfg = config.get("datasets", {}).get(dataset_name, {}) if dataset_name else {}
        for field_name in ("train_dir", "val_dir", "test_dir"):
            raw_dir = dataset_cfg.get(field_name)
            if not raw_dir:
                continue
            resolved = resolve_project_path(raw_dir, base_dir=config.get("project_root"))
            if resolved is not None and resolved.exists():
                candidate_dirs.append(resolved)
        seen_dirs: set[Path] = set()
        image_paths: list[Path] = []
        max_images = int(carrier_bank_cfg.get("max_images", 4096))
        for image_dir in candidate_dirs:
            image_dir = image_dir.resolve()
            if image_dir in seen_dirs or not image_dir.is_dir():
                continue
            seen_dirs.add(image_dir)
            for image_path in sorted(image_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                    image_paths.append(image_path)
                    if max_images > 0 and len(image_paths) >= max_images:
                        return image_paths
        return image_paths

    # 按批次确定性采样自然载体图，训练时可随机扰动，验证时保持可复现。
    def sample_carrier_bank_images(
        self,
        reference: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor | None:
        if (not self.carrier_bank_enabled) or not self.carrier_bank_paths:
            return None
        batch_size = int(reference.shape[0])
        height, width = int(reference.shape[-2]), int(reference.shape[-1])
        mode = "RGB" if int(reference.shape[1]) == 3 else "L"
        base_seed = int(self.carrier_bank_seed)
        if self.training and self.carrier_bank_random_train:
            base_seed += int(torch.randint(0, 2**30 - 1, (1,), device=reference.device).item())
        tensors: list[torch.Tensor] = []
        for batch_index in range(batch_size):
            image_path: Path | None = None
            available_count = len(self.carrier_bank_paths)
            for attempt_index in range(available_count):
                if self.training and self.carrier_bank_random_train:
                    rng = random.Random(base_seed + batch_index * self.carrier_bank_stride + attempt_index)
                    path_index = rng.randrange(available_count)
                else:
                    noise_value = float(noise[batch_index].detach().float().mean().cpu().item()) if noise.numel() > 0 else 0.0
                    path_index = int(
                        abs(noise_value) * 1000003
                        + base_seed
                        + batch_index * self.carrier_bank_stride
                        + attempt_index
                    )
                    path_index %= available_count
                candidate_path = self.carrier_bank_paths[path_index]
                if candidate_path.exists():
                    image_path = candidate_path
                    break
            if image_path is None:
                raise FileNotFoundError(
                    "carrier_bank could not find any existing image files at runtime. "
                    "Please refresh the configured carrier directories."
                )
            with Image.open(image_path) as image:
                image = image.convert(mode)
                image = image.resize((width, height), Image.Resampling.BICUBIC)
                array = np.asarray(image, dtype=np.uint8).copy()
            if mode == "RGB":
                tensor = torch.from_numpy(array).permute(2, 0, 1).float().div(255.0)
            else:
                tensor = torch.from_numpy(array).view(height, width, 1).permute(2, 0, 1).float().div(255.0)
                if int(reference.shape[1]) == 3:
                    tensor = tensor.repeat(3, 1, 1)
            tensors.append(tensor)
        return torch.stack(tensors, dim=0).to(device=reference.device, dtype=reference.dtype)

    # 构建固定的景观 latent anchor，让生成器围绕同一类自然场景稳定采样。
    def _build_landscape_anchor(self, latent_dim: int, seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        anchor = torch.randn(1, latent_dim, generator=generator)
        ramp = torch.linspace(-1.0, 1.0, latent_dim).unsqueeze(0)
        seasonal = torch.sin(torch.linspace(0.0, 6.28318530718, latent_dim)).unsqueeze(0)
        anchor = 0.6 * anchor + 0.3 * ramp + 0.1 * seasonal
        return anchor / anchor.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)

    # 根据配置生成随机噪声或围绕景观 anchor 的可复现噪声。
    def sample_noise(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.noise_mode == "landscape":
            anchor = self.landscape_anchor.to(device=device, dtype=dtype).expand(batch_size, -1)
            if self.training and self.noise_std > 0:
                return anchor + torch.randn_like(anchor) * self.noise_std
            return anchor
        return torch.randn(batch_size, self.latent_dim, device=device, dtype=dtype)

    # 生成与真实 QIM payload 同尺寸的随机 dummy bits，用于构造统计分布匹配的 clean carrier。
    def sample_qim_dummy_bits(self, coded_bits: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(
            self.qim_clean_carrier_seed + int(coded_bits.shape[1]) * 1009 + int(coded_bits.shape[2])
        )
        dummy = torch.randint(
            0,
            2,
            coded_bits.shape,
            generator=generator,
            dtype=torch.int64,
        )
        return dummy.to(device=coded_bits.device, dtype=coded_bits.dtype)

    # 统一应用 QIM 后载密图自然化，保证训练、验证和评估使用同一条最终载密链路。
    def apply_stego_naturalizer(
        self,
        stego_image: torch.Tensor,
        base_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.stego_naturalizer(stego_image, base_image)

    # 按场景提取 clean QIM 或 robust auxiliary QIM 的 LLR，避免鲁棒分支覆盖 clean 主链路。
    def extract_qim_llr(
        self,
        image: torch.Tensor,
        num_blocks: int,
        code_length: int,
        prefer_robust: bool = False,
    ) -> torch.Tensor:
        clean_llr = self.qim_channel.extract_llr(image, num_blocks=num_blocks, code_length=code_length)
        if not self.robust_qim_enabled or self.robust_qim_channel is None:
            return clean_llr
        robust_llr = self.robust_qim_channel.extract_llr(image, num_blocks=num_blocks, code_length=code_length)
        fusion_weight = self.robust_qim_attack_fusion_weight if prefer_robust else self.robust_qim_clean_fusion_weight
        if fusion_weight <= 0.0:
            return clean_llr
        clean_energy = clean_llr.float().abs().mean(dim=-1, keepdim=True).clamp_min(1e-6).to(clean_llr.dtype)
        robust_energy = robust_llr.float().abs().mean(dim=-1, keepdim=True).clamp_min(1e-6).to(robust_llr.dtype)
        normalized_clean_llr = clean_llr / clean_energy
        normalized_robust_llr = robust_llr / robust_energy
        if fusion_weight >= 1.0:
            if prefer_robust:
                return robust_llr
            return robust_llr
        if prefer_robust:
            dominant_weight = max(fusion_weight, 0.80)
            clean_support = torch.where(
                clean_llr * robust_llr > 0,
                clean_llr,
                torch.zeros_like(clean_llr),
            )
            support_weight = (1.0 - dominant_weight) * 0.35
            return robust_llr * dominant_weight + clean_support * support_weight
        return clean_llr * (1.0 - fusion_weight) + robust_llr * fusion_weight

    # 在 clean QIM 主链路上叠加弱强度鲁棒辅助嵌入，让 robust extractor 在攻击后仍有真实可读信号。
    def apply_robust_qim_aux_embed(
        self,
        clean_qim_cover: torch.Tensor,
        embed_source: torch.Tensor,
        coded_bits: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not self.robust_qim_enabled
            or not self.robust_qim_embed_enabled
            or self.robust_qim_channel is None
            or self.robust_qim_embed_alpha <= 0.0
        ):
            return clean_qim_cover
        robust_aux_cover = self.robust_qim_channel.embed(
            embed_source,
            coded_bits.detach(),
        )
        robust_residual = robust_aux_cover - embed_source
        return (clean_qim_cover + self.robust_qim_embed_alpha * robust_residual).clamp(0.0, 1.0)

    # 将发射端真实可传输的量化 latent 重新解码为 preview，保证训练目标与真实传输链路一致。
    def align_transmitted_preview(
        self,
        transmitter_output: TransmitterOutput,
        preserve_encoder_grad: bool = True,
    ) -> TransmitterOutput:
        compression = transmitter_output.compression
        transmitted_latent = compression.quantized_latent
        if preserve_encoder_grad and self.training:
            transmitted_latent = compression.latent + (transmitted_latent - compression.latent).detach()
        compression.quantized_latent = transmitted_latent
        compression.reconstructed_image = self.compressor.decode_latent(
            transmitted_latent,
            compression.original_size,
        )
        return transmitter_output

    # 校验发送端码包数量和分组数量，避免接收端解码与评估统计静默错位。
    def _validate_packet_dimensions(self, packet) -> None:
        num_blocks = int(packet.num_blocks)
        if num_blocks <= 0:
            raise ValueError(f"packet.num_blocks must be positive, got {packet.num_blocks}.")
        for tensor_name in ("payload_bits", "info_bits", "coded_bits", "symbols"):
            tensor = getattr(packet, tensor_name)
            if tensor.dim() < 3:
                raise ValueError(f"packet.{tensor_name} must have shape [B, num_blocks, ...], got {tuple(tensor.shape)}.")
            if int(tensor.shape[1]) != num_blocks:
                raise ValueError(
                    f"packet.{tensor_name} block count mismatch: "
                    f"shape[1]={int(tensor.shape[1])}, packet.num_blocks={num_blocks}."
                )
        group_size = max(1, int(packet.group_size))
        expected_group_count = (num_blocks + group_size - 1) // group_size
        if packet.group_count is not None and int(packet.group_count) != expected_group_count:
            raise ValueError(
                f"packet.group_count mismatch: got {packet.group_count}, expected {expected_group_count}."
            )
        if packet.source_num_blocks is not None:
            source_num_blocks = int(packet.source_num_blocks)
            if source_num_blocks <= 0 or source_num_blocks > num_blocks:
                raise ValueError(
                    f"packet.source_num_blocks must be in [1, num_blocks], "
                    f"got {source_num_blocks}, num_blocks={num_blocks}."
                )

    # 执行图像压缩、编码、生成载密图与接收恢复的完整前向流程。
    def forward(
        self,
        image: torch.Tensor,
        noise: torch.Tensor | None = None,
        max_decode_blocks: int | None = None,
        force_full_decode: bool = False,
        detach_image_reconstruction: bool = False,
    ) -> SystemOutput:
        if noise is None:
            noise = self.sample_noise(image.shape[0], image.device, image.dtype)
        compression_output = self.compressor(image)
        compression_tensors = [
            compression_output.latent,
            compression_output.quantized_latent,
            compression_output.reconstructed_image,
        ]
        if any(not torch.isfinite(tensor).all() for tensor in compression_tensors):
            raise FloatingPointError("Compressor produced non-finite tensors; aborting forward pass.")
        transmitter_output = self.align_transmitted_preview(self.transmitter(compression_output))
        self._validate_packet_dimensions(transmitter_output.packet)
        carrier_anchor_image = self.sample_carrier_bank_images(image, noise)
        gan_cfg = self.config.get("gan", {})
        carrier_generator_required = (
            self.carrier_bank_forward_into_generator
            or bool(gan_cfg.get("carrier_first_mode", False))
            or float(gan_cfg.get("external_carrier_blend", 0.0)) > 0.0
        )
        carrier_input_image = (
            carrier_anchor_image
            if self.carrier_bank_enabled and carrier_generator_required
            else None
        )
        generator_output = self.generator(
            transmitter_output,
            noise,
            embed_payload=not self.qim_enabled,
            source_image=image,
            carrier_image=carrier_input_image,
        )
        if carrier_anchor_image is not None and generator_output.carrier_anchor_image is None:
            generator_output = GeneratorOutput(
                generated_image=generator_output.generated_image,
                cover_image=generator_output.cover_image,
                carrier_anchor_image=carrier_anchor_image,
            )
        receiver_input_image = generator_output.cover_image
        external_llr_signal = None
        if self.qim_enabled:
            qim_base_image = generator_output.cover_image
            if self.stego_naturalizer_apply_before_qim:
                qim_base_image = self.apply_stego_naturalizer(qim_base_image, qim_base_image)
            qim_embed_source = qim_base_image
            qim_cover_image = self.qim_channel.embed(
                qim_embed_source,
                transmitter_output.packet.coded_bits.detach(),
            )
            qim_cover_image = self.apply_robust_qim_aux_embed(
                qim_cover_image,
                qim_cover_image,
                transmitter_output.packet.coded_bits,
            )
            if not self.stego_naturalizer_apply_before_qim:
                qim_cover_image = self.apply_stego_naturalizer(qim_cover_image, qim_base_image)
            generated_image = generator_output.generated_image
            if self.qim_clean_carrier_mode == "dummy_bits":
                dummy_bits = self.sample_qim_dummy_bits(transmitter_output.packet.coded_bits.detach())
                generated_image = qim_base_image
                generated_image = self.qim_channel.embed(generated_image, dummy_bits)
                if not self.stego_naturalizer_apply_before_qim:
                    generated_image = self.apply_stego_naturalizer(generated_image, qim_base_image)
            generator_output = GeneratorOutput(
                generated_image=generated_image,
                cover_image=qim_cover_image,
                carrier_anchor_image=carrier_anchor_image,
            )
            receiver_input_image = qim_cover_image
        receiver_input_image = self.robust_channel(receiver_input_image)
        if self.qim_enabled:
            external_llr_signal = self.extract_qim_llr(
                receiver_input_image,
                num_blocks=transmitter_output.packet.num_blocks,
                code_length=transmitter_output.packet.coded_bits.shape[-1],
                prefer_robust=False,
            )
        current_stage_name = str(getattr(self, "current_stage_name", "") or "").strip().lower()
        allow_receiver_carrier_reference = current_stage_name in {
            "stage4_robust_comm_refine",
        }
        receiver_carrier_reference = None
        if allow_receiver_carrier_reference and self.carrier_bank_enabled and (
            self.carrier_bank_forward_into_generator
            or bool(self.config.get("gan", {}).get("carrier_first_mode", False))
            or float(self.config.get("gan", {}).get("external_carrier_blend", 0.0)) > 0.0
        ):
            receiver_carrier_reference = (
                generator_output.carrier_anchor_image
                if generator_output.carrier_anchor_image is not None
                else generator_output.generated_image
            )
        receiver_output = self.receiver(
            receiver_input_image,
            latent_shape=transmitter_output.compression.latent_shape,
            latent_decoder=self.compressor.decode_latent,
            output_size=transmitter_output.compression.original_size,
            carrier_reference_image=receiver_carrier_reference,
            valid_info_bits=transmitter_output.packet.valid_info_bits,
            num_blocks=transmitter_output.packet.num_blocks,
            max_decode_blocks=max_decode_blocks,
            force_full_decode=force_full_decode,
            full_bitstream_available=True,
            teacher_payload_bits=(
                transmitter_output.packet.payload_bits.detach()
                if self.training and self.teacher_payload_ratio > 0.0
                else None
            ),
            teacher_payload_ratio=self.teacher_payload_ratio if self.training else 0.0,
            teacher_channel_symbols=(
                transmitter_output.packet.symbols.detach()
                if self.training and self.teacher_channel_ratio > 0.0
                else None
            ),
            teacher_channel_ratio=self.teacher_channel_ratio if self.training else 0.0,
            external_llr_signal=external_llr_signal,
            external_llr_frontend_mode="clean",
            detach_image_reconstruction=detach_image_reconstruction,
        )
        return SystemOutput(
            transmitter=transmitter_output,
            generator=generator_output,
            receiver=receiver_output,
        )

