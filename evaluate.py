from __future__ import annotations

import argparse
import csv
import json
import os
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import Inception_V3_Weights, inception_v3

from models.system import WaveBridegeSystem
from utils.config import load_config, resolve_project_path
from utils.datasets import build_eval_loader, get_active_dataset_config
from utils.runtime import load_checkpoint_config, load_model_checkpoint, resolve_device, set_seed


DEFAULT_TARGET_PAYLOAD_BPP = 0.30
DEFAULT_PAYLOAD_BPP_TOLERANCE = 0.03
LOW_PAYLOAD_MIN_BPP = 0.20
LOW_PAYLOAD_MAX_BPP = 0.40
LOW_PAYLOAD_CARRIER_DATASET = "landscape"
LOW_PAYLOAD_MIN_LATENT_CANDIDATES = 24
LOW_PAYLOAD_LATENT_SCORE = "carrier_brisque_proxy"
LOW_PAYLOAD_PRIOR_MIXES = "0,0.001,0.002,0.004,0.006"
PARITY_Y_QIM_PROFILE = {
    "enabled": True,
    "carrier_bands": ["lh", "hl", "hh"],
    "prefer_hh_only": False,
    "domain": "dct",
    "dct_channels": "y",
    "dct_coefficients": "auto32",
    "dct_parity_mode": True,
    "dct_quant_scale": 1.0,
    "delta": 1.0,
    "strength": 1.0,
    "llr_scale": 52.0,
    "llr_clamp": 40.0,
    "position_mode": "stratified",
    "repetition_factor": 1,
    "dither_enabled": False,
    "dither_strength": 0.0,
    "extract_smooth_kernel": 1,
}


TARGET_THRESHOLDS = {
    "PSNR_restored": (38.0, ">="),
    "SSIM_restored": (0.97, ">="),
    "LPIPS_restored": (0.05, "<="),
    "BER_info_clean": (1e-4, "<="),
    "BER_code_clean": (1e-3, "<="),
    "decoded_blocks_ratio": (1.0, ">="),
    "decoded_groups_ratio": (1.0, ">="),
    "payload_bpp": (
        (
            DEFAULT_TARGET_PAYLOAD_BPP - DEFAULT_PAYLOAD_BPP_TOLERANCE,
            DEFAULT_TARGET_PAYLOAD_BPP + DEFAULT_PAYLOAD_BPP_TOLERANCE,
        ),
        "between",
    ),
    "FID_stego": (20.0, "<"),
    "KID": (0.005, "<"),
    "BRISQUE_gap_to_real": (5.0, "<="),
    "SRNet_balanced_accuracy": ((0.45, 0.55), "between"),
    "SRNet_AUC": (0.50, "<"),
    "SRNet_EER": ((0.45, 0.55), "between"),
    "JPEG50_BER_info": (0.01, "<="),
    "Mixed_BER_info": (0.03, "<="),
    "PSNR_attack": (22.0, ">="),
    "SSIM_attack": (0.70, ">="),
    "LPIPS_attack": (0.50, "<="),
    "BER_info_attack": (0.40, "<="),
    "decoded_ratio_attack": (0.93, ">="),
}


# 深拷贝配置字典，避免评估阶段修改原始配置对象。
def clone_config(config: dict) -> dict:
    return json.loads(json.dumps(config))


# 递归更新配置字典，只覆盖明确允许沿用运行期配置的键。
def deep_update_dict(target: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update_dict(target[key], value)
        else:
            target[key] = value
    return target


# 评估时优先使用 checkpoint 原生推理结构，仅沿用当前 config 中的数据集/设备/子集等外部入口配置。
def merge_checkpoint_runtime_config(runtime_config: dict, checkpoint_config: dict | None) -> dict:
    if not isinstance(checkpoint_config, dict):
        return clone_config(runtime_config)

    merged = clone_config(checkpoint_config)
    # 评估 checkpoint 时默认保留其原生模型/通信结构，避免当前 YAML 的新实验设置破坏旧权重。
    passthrough_sections = {
        "datasets",
        "subset",
    }
    passthrough_scalars = {
        "seed",
        "device",
        "device_index",
        "project_root",
    }
    for section_name in passthrough_sections:
        runtime_section = runtime_config.get(section_name)
        if isinstance(runtime_section, dict):
            merged[section_name] = clone_config(runtime_section)
    runtime_data_cfg = runtime_config.get("data")
    if isinstance(runtime_data_cfg, dict):
        merged.setdefault("data", {})
        for key in ("image_size", "channels", "keep_original_size"):
            if key in runtime_data_cfg:
                merged["data"][key] = runtime_data_cfg[key]
    runtime_training_cfg = runtime_config.get("training")
    if isinstance(runtime_training_cfg, dict):
        merged.setdefault("training", {})
        for key in ("batch_size", "num_workers", "cpu_threads", "cpu_interop_threads", "target_bpp"):
            if key in runtime_training_cfg:
                merged["training"][key] = runtime_training_cfg[key]
    runtime_transmitter_cfg = runtime_config.get("transmitter")
    if isinstance(runtime_transmitter_cfg, dict):
        # Payload sweep changes only the number of valid payload bits used at
        # evaluation time. Architecture-defining fields stay checkpoint-native.
        merged.setdefault("transmitter", {})
        for key in ("target_bpp", "payload_bpp"):
            if key in runtime_transmitter_cfg:
                merged["transmitter"][key] = runtime_transmitter_cfg[key]
    runtime_carrier_cfg = runtime_config.get("carrier_bank")
    if isinstance(runtime_carrier_cfg, dict) and bool(runtime_carrier_cfg.get("enabled", False)):
        merged["carrier_bank"] = clone_config(runtime_carrier_cfg)
        runtime_gan_cfg = runtime_config.get("gan")
        if isinstance(runtime_gan_cfg, dict):
            merged.setdefault("gan", {})
            for key in ("external_carrier_blend", "external_carrier_source_bridge", "source_bridge_strength"):
                if key in runtime_gan_cfg:
                    merged["gan"][key] = runtime_gan_cfg[key]
    runtime_robust_qim_cfg = runtime_config.get("robust_qim")
    if isinstance(runtime_robust_qim_cfg, dict) and bool(runtime_robust_qim_cfg.get("enabled", False)):
        merged["robust_qim"] = clone_config(runtime_robust_qim_cfg)
    for key in passthrough_scalars:
        if key in runtime_config:
            merged[key] = runtime_config[key]
    return merged


# 评估时优先保留 checkpoint 的原生结构，只从当前运行配置继承数据集、设备和必要的评估入口参数。
def merge_checkpoint_runtime_config(
    runtime_config: dict,
    checkpoint_config: dict | None,
    preserve_runtime_eval_modules: bool = True,
    preserve_checkpoint_payload_structure: bool = False,
) -> dict:
    if not isinstance(checkpoint_config, dict):
        return clone_config(runtime_config)

    merged = clone_config(checkpoint_config)
    passthrough_sections = {
        "datasets",
        "subset",
    }
    passthrough_scalars = {
        "seed",
        "device",
        "device_index",
        "project_root",
    }
    for section_name in passthrough_sections:
        runtime_section = runtime_config.get(section_name)
        if isinstance(runtime_section, dict):
            merged[section_name] = clone_config(runtime_section)
    runtime_data_cfg = runtime_config.get("data")
    if isinstance(runtime_data_cfg, dict):
        merged.setdefault("data", {})
        for key in ("image_size", "channels", "keep_original_size"):
            if key in runtime_data_cfg:
                merged["data"][key] = runtime_data_cfg[key]
    runtime_training_cfg = runtime_config.get("training")
    if isinstance(runtime_training_cfg, dict):
        merged.setdefault("training", {})
        for key in ("batch_size", "num_workers", "cpu_threads", "cpu_interop_threads", "target_bpp"):
            if key in runtime_training_cfg:
                merged["training"][key] = runtime_training_cfg[key]
    runtime_transmitter_cfg = runtime_config.get("transmitter")
    if isinstance(runtime_transmitter_cfg, dict):
        merged.setdefault("transmitter", {})
        for key in ("target_bpp", "payload_bpp"):
            if key in runtime_transmitter_cfg:
                merged["transmitter"][key] = runtime_transmitter_cfg[key]

    if preserve_checkpoint_payload_structure:
        checkpoint_training_cfg = checkpoint_config.get("training")
        merged_training_cfg = merged.setdefault("training", {})
        if isinstance(checkpoint_training_cfg, dict) and "target_bpp" in checkpoint_training_cfg:
            merged_training_cfg["target_bpp"] = checkpoint_training_cfg["target_bpp"]
        checkpoint_transmitter_cfg = checkpoint_config.get("transmitter")
        merged_transmitter_cfg = merged.setdefault("transmitter", {})
        if isinstance(checkpoint_transmitter_cfg, dict):
            for key in ("target_bpp", "payload_bpp"):
                if key in checkpoint_transmitter_cfg:
                    merged_transmitter_cfg[key] = checkpoint_transmitter_cfg[key]
        checkpoint_carrier_cfg = checkpoint_config.get("carrier_bank")
        if isinstance(checkpoint_carrier_cfg, dict):
            merged["carrier_bank"] = clone_config(checkpoint_carrier_cfg)
        checkpoint_gan_cfg = checkpoint_config.get("gan")
        if isinstance(checkpoint_gan_cfg, dict):
            merged.setdefault("gan", {})
            for key in (
                "external_carrier_blend",
                "external_carrier_source_bridge",
                "source_bridge_strength",
                "carrier_first_mode",
            ):
                if key in checkpoint_gan_cfg:
                    merged["gan"][key] = checkpoint_gan_cfg[key]
        checkpoint_robust_qim_cfg = checkpoint_config.get("robust_qim")
        if isinstance(checkpoint_robust_qim_cfg, dict):
            merged["robust_qim"] = clone_config(checkpoint_robust_qim_cfg)

    if preserve_runtime_eval_modules:
        runtime_carrier_cfg = runtime_config.get("carrier_bank")
        if isinstance(runtime_carrier_cfg, dict) and bool(runtime_carrier_cfg.get("enabled", False)):
            merged["carrier_bank"] = clone_config(runtime_carrier_cfg)
            runtime_gan_cfg = runtime_config.get("gan")
            if isinstance(runtime_gan_cfg, dict):
                merged.setdefault("gan", {})
                for key in (
                    "external_carrier_blend",
                    "external_carrier_source_bridge",
                    "source_bridge_strength",
                    "external_carrier_lowfreq_only",
                    "preserve_source_bridge_with_external_carrier",
                    "analog_injection_target",
                ):
                    if key in runtime_gan_cfg:
                        merged["gan"][key] = runtime_gan_cfg[key]
        runtime_robust_qim_cfg = runtime_config.get("robust_qim")
        if isinstance(runtime_robust_qim_cfg, dict) and bool(runtime_robust_qim_cfg.get("enabled", False)):
            merged["robust_qim"] = clone_config(runtime_robust_qim_cfg)
        runtime_noise_cfg = runtime_config.get("noise")
        checkpoint_noise_cfg = checkpoint_config.get("noise")
        # 仅在 checkpoint 自身已包含对应噪声/扰动配置时，才允许当前运行配置覆写。
        if isinstance(runtime_noise_cfg, dict) and isinstance(checkpoint_noise_cfg, dict):
            merged["noise"] = clone_config(runtime_noise_cfg)

    for key in passthrough_scalars:
        if key in runtime_config:
            merged[key] = runtime_config[key]
    return merged


def checkpoint_supports_runtime_eval_modules(
    checkpoint_config: dict | None,
    checkpoint_path: str | Path | None,
) -> dict[str, bool]:
    support = {
        "carrier_bank": False,
        "robust_qim": False,
        "stego_naturalizer": False,
    }
    if isinstance(checkpoint_config, dict):
        carrier_cfg = checkpoint_config.get("carrier_bank")
        robust_qim_cfg = checkpoint_config.get("robust_qim")
        stego_cfg = checkpoint_config.get("stego_naturalizer")
        support["carrier_bank"] = isinstance(carrier_cfg, dict) and bool(carrier_cfg)
        support["robust_qim"] = isinstance(robust_qim_cfg, dict) and bool(robust_qim_cfg)
        support["stego_naturalizer"] = isinstance(stego_cfg, dict) and bool(stego_cfg)
    resolved_checkpoint = resolve_project_path(checkpoint_path) if checkpoint_path is not None else None
    if resolved_checkpoint is None or not resolved_checkpoint.exists():
        return support
    try:
        checkpoint = torch.load(resolved_checkpoint, map_location="cpu")
    except Exception:
        return support
    state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        return support
    keys = tuple(str(key) for key in state_dict.keys())
    support["carrier_bank"] = support["carrier_bank"] or any(
        "carrier_bank" in key
        or "external_carrier" in key
        or "stego_naturalizer" in key
        or "generator.prior_generator" in key
        for key in keys
    )
    support["robust_qim"] = support["robust_qim"] or any(
        key.startswith("receiver.comm_decoder")
        or key.startswith("receiver.external_llr_adapter")
        or key.startswith("receiver.symbol_calibrator")
        or key.startswith("receiver.bp_decoder")
        or key.startswith("receiver.mamba")
        or "robust_qim" in key
        or ".qim" in key
        for key in keys
    )
    support["stego_naturalizer"] = support["stego_naturalizer"] or any(
        "stego_naturalizer" in key for key in keys
    )
    return support


def metric_passes_target(value: float | int | bool | str, threshold: float | tuple[float, float], direction: str) -> bool:
    # 根据目标方向判断单个指标是否达标；NaN/Inf 一律视为未达标。
    if isinstance(value, bool):
        numeric_value = 1.0 if value else 0.0
    else:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return False
    if not math.isfinite(numeric_value):
        return False
    if direction == "between":
        lower, upper = threshold
        return float(lower) <= numeric_value <= float(upper)
    if direction == ">=":
        return numeric_value >= float(threshold)
    if direction == ">":
        return numeric_value > float(threshold)
    if direction == "<=":
        return numeric_value <= float(threshold)
    if direction == "<":
        return numeric_value < float(threshold)
    raise ValueError(f"Unsupported target direction: {direction}")


def finite_float_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def resolve_config_target_bpp(config: dict) -> float | None:
    transmitter_cfg = config.get("transmitter", {})
    if isinstance(transmitter_cfg, dict):
        for key in ("target_bpp", "payload_bpp"):
            value = finite_float_or_none(transmitter_cfg.get(key))
            if value is not None and value > 0.0:
                return value
    training_cfg = config.get("training", {})
    if isinstance(training_cfg, dict):
        value = finite_float_or_none(training_cfg.get("target_bpp"))
        if value is not None and value > 0.0:
            return value
    return None


def is_low_payload_target_bpp(target_bpp: float | None) -> bool:
    return target_bpp is not None and LOW_PAYLOAD_MIN_BPP <= float(target_bpp) < LOW_PAYLOAD_MAX_BPP


def resolve_effective_target_bpp(
    args: argparse.Namespace,
    config: dict,
    checkpoint_config: dict | None = None,
) -> float | None:
    target_bpp = finite_float_or_none(getattr(args, "target_bpp", None))
    if target_bpp is not None and target_bpp > 0.0:
        return target_bpp
    target_bpp = resolve_config_target_bpp(config)
    if target_bpp is not None and target_bpp > 0.0:
        return target_bpp
    if isinstance(checkpoint_config, dict):
        target_bpp = resolve_config_target_bpp(checkpoint_config)
        if target_bpp is not None and target_bpp > 0.0:
            return target_bpp
    return None


def cli_option_explicitly_set(argv_tokens: list[str], option: str) -> bool:
    prefix = f"{option}="
    return any(token == option or token.startswith(prefix) for token in argv_tokens)


def any_cli_option_explicitly_set(argv_tokens: list[str], *options: str) -> bool:
    return any(cli_option_explicitly_set(argv_tokens, option) for option in options)


def apply_low_payload_eval_defaults(
    args: argparse.Namespace,
    config: dict,
    checkpoint_config: dict | None = None,
    argv_tokens: list[str] | None = None,
) -> None:
    argv_tokens = argv_tokens or []
    effective_target_bpp = resolve_effective_target_bpp(args, config, checkpoint_config)
    if not is_low_payload_target_bpp(effective_target_bpp):
        return

    applied: list[str] = []
    if not any_cli_option_explicitly_set(argv_tokens, "--enable-carrier-bank", "--disable-carrier-bank"):
        if not bool(getattr(args, "enable_carrier_bank", False)):
            setattr(args, "enable_carrier_bank", True)
            applied.append("enable_carrier_bank=True")
    if not cli_option_explicitly_set(argv_tokens, "--carrier-bank-dataset"):
        if str(getattr(args, "carrier_bank_dataset", "") or "").strip().lower() != LOW_PAYLOAD_CARRIER_DATASET:
            setattr(args, "carrier_bank_dataset", LOW_PAYLOAD_CARRIER_DATASET)
            applied.append(f"carrier_bank_dataset={LOW_PAYLOAD_CARRIER_DATASET}")
    if not any_cli_option_explicitly_set(argv_tokens, "--enable-robust-qim", "--disable-robust-qim"):
        if not bool(getattr(args, "enable_robust_qim", False)):
            setattr(args, "enable_robust_qim", True)
            applied.append("enable_robust_qim=True")
    if not cli_option_explicitly_set(argv_tokens, "--latent-select-candidates"):
        upgraded_candidates = max(int(getattr(args, "latent_select_candidates", 1) or 1), LOW_PAYLOAD_MIN_LATENT_CANDIDATES)
        if upgraded_candidates != int(getattr(args, "latent_select_candidates", 1) or 1):
            setattr(args, "latent_select_candidates", upgraded_candidates)
            applied.append(f"latent_select_candidates={upgraded_candidates}")
    if not cli_option_explicitly_set(argv_tokens, "--latent-select-score"):
        if str(getattr(args, "latent_select_score", "") or "").strip().lower() != LOW_PAYLOAD_LATENT_SCORE:
            setattr(args, "latent_select_score", LOW_PAYLOAD_LATENT_SCORE)
            applied.append(f"latent_select_score={LOW_PAYLOAD_LATENT_SCORE}")
    if not cli_option_explicitly_set(argv_tokens, "--latent-select-prior-mixes"):
        if str(getattr(args, "latent_select_prior_mixes", "") or "") != LOW_PAYLOAD_PRIOR_MIXES:
            setattr(args, "latent_select_prior_mixes", LOW_PAYLOAD_PRIOR_MIXES)
            applied.append(f"latent_select_prior_mixes={LOW_PAYLOAD_PRIOR_MIXES}")
    if not cli_option_explicitly_set(argv_tokens, "--natural-reference-dataset"):
        current_reference_dataset = getattr(args, "natural_reference_dataset", None)
        if current_reference_dataset in {None, "", "secret"}:
            setattr(args, "natural_reference_dataset", LOW_PAYLOAD_CARRIER_DATASET)
            applied.append(f"natural_reference_dataset={LOW_PAYLOAD_CARRIER_DATASET}")

    print(f"[eval] Low-payload profile detected (target_bpp={float(effective_target_bpp):g}).", flush=True)
    if applied:
        print(f"[eval] Applied low-payload eval defaults: {', '.join(applied)}", flush=True)
    else:
        print("[eval] Low-payload eval defaults already satisfied.", flush=True)


def requested_decoder_qim_profile() -> str:
    return str(os.environ.get("WAVEBRIDEGE_STAGE4_DECODER_QIM_PROFILE", "") or "").strip().lower().replace("-", "_")


def requested_parity_y_full_profile() -> bool:
    return requested_decoder_qim_profile() in {"parity_y", "y_parity", "jpeg_parity_y", "robust_parity_y"}


def requested_parity_y_hybrid_profile() -> bool:
    return requested_decoder_qim_profile() in {
        "hybrid_parity_y",
        "parity_y_hybrid",
        "robust_aux_parity_y",
        "parity_y_aux",
    }


def robust_aux_embed_alpha() -> float:
    raw_value = str(os.environ.get("WAVEBRIDEGE_STAGE4_ROBUST_AUX_EMBED_ALPHA", "") or "").strip()
    if not raw_value:
        return 1.0
    try:
        value = float(raw_value)
    except ValueError:
        return 1.0
    if not math.isfinite(value):
        return 1.0
    return float(max(0.0, min(1.0, value)))


def external_crc_concealment_enabled(default: bool = True) -> bool:
    raw_value = str(os.environ.get("WAVEBRIDEGE_STAGE4_EXTERNAL_CRC_CONCEALMENT", "") or "").strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def attack_ber_source(default: str = "auto") -> str:
    raw_value = str(os.environ.get("WAVEBRIDEGE_ATTACK_BER_SOURCE", "") or "").strip().lower()
    if raw_value in {"", "auto"}:
        return "auto" if default not in {"strict_qim", "strict_frontend", "receiver"} else default
    if raw_value in {"strict_qim", "strict", "qim"}:
        return "strict_qim"
    if raw_value in {"strict_frontend", "frontend", "calibrated"}:
        return "strict_frontend"
    if raw_value in {"receiver", "receiver_path", "model"}:
        return "receiver"
    return "auto" if default not in {"strict_qim", "strict_frontend", "receiver"} else default


def strict_attack_llr_signal(
    model: WaveBridegeSystem,
    external_llr_signal: torch.Tensor | None,
    *,
    frontend_mode: str = "robust",
) -> torch.Tensor | None:
    if external_llr_signal is None:
        return None
    source = attack_ber_source()
    if source != "strict_frontend":
        return external_llr_signal
    apply_frontend = getattr(model.receiver, "_apply_external_llr_frontend", None)
    if not callable(apply_frontend):
        return external_llr_signal
    return apply_frontend(external_llr_signal, frontend_mode=frontend_mode)


def prefer_robust_qim_for_attack_metrics() -> bool:
    return requested_parity_y_full_profile() or requested_parity_y_hybrid_profile()


def apply_eval_qim_profile(config: dict) -> None:
    if not (requested_parity_y_full_profile() or requested_parity_y_hybrid_profile()):
        return
    if requested_parity_y_full_profile():
        config["qim"] = clone_config(PARITY_Y_QIM_PROFILE)
    robust_profile = clone_config(PARITY_Y_QIM_PROFILE)
    robust_profile["clean_fusion_weight"] = 0.0
    robust_profile["attack_fusion_weight"] = 1.0
    if requested_parity_y_hybrid_profile():
        robust_profile["embed_enabled"] = True
        robust_profile["embed_alpha"] = robust_aux_embed_alpha()
    receiver_cfg = config.setdefault("receiver", {})
    receiver_cfg["external_hard_decode_crc_concealment"] = external_crc_concealment_enabled(True)
    config["robust_qim"] = robust_profile
    if requested_parity_y_full_profile() or requested_parity_y_hybrid_profile():
        receiver_cfg["strict_external_llr_hard_decode"] = True
        receiver_cfg["bridge_external_llr_for_metrics_only"] = True
        receiver_cfg["full_decode_feature_mix"] = 0.0
        receiver_cfg["full_decode_feature_min_mix_ratio"] = 0.0


def apply_qim_profile_to_model(model: WaveBridegeSystem) -> None:
    if not (requested_parity_y_full_profile() or requested_parity_y_hybrid_profile()):
        return

    def apply_channel_runtime(channel, cfg: dict) -> None:
        if channel is None:
            return
        for key, value in cfg.items():
            if not hasattr(channel, key):
                continue
            if key == "dct_coefficients" and hasattr(channel, "_normalize_dct_coefficients"):
                value = channel._normalize_dct_coefficients(value)
            setattr(channel, key, value)

    if requested_parity_y_full_profile():
        apply_channel_runtime(getattr(model, "qim_channel", None), PARITY_Y_QIM_PROFILE)
    apply_channel_runtime(getattr(model, "robust_qim_channel", None), PARITY_Y_QIM_PROFILE)
    model.robust_qim_enabled = bool(getattr(model, "robust_qim_channel", None) is not None)
    model.robust_qim_attack_fusion_weight = 1.0
    model.robust_qim_clean_fusion_weight = 0.0
    if requested_parity_y_hybrid_profile():
        model.robust_qim_embed_enabled = True
        model.robust_qim_embed_alpha = robust_aux_embed_alpha()
    if hasattr(model.receiver, "external_hard_decode_crc_concealment"):
        model.receiver.external_hard_decode_crc_concealment = external_crc_concealment_enabled(True)
    if hasattr(model.receiver, "strict_external_llr_hard_decode"):
        model.receiver.strict_external_llr_hard_decode = True
    if hasattr(model.receiver, "bridge_external_llr_for_metrics_only"):
        model.receiver.bridge_external_llr_for_metrics_only = True
    if hasattr(model.receiver, "full_decode_feature_mix"):
        model.receiver.full_decode_feature_mix = 0.0
    if hasattr(model.receiver, "full_decode_feature_min_mix_ratio"):
        model.receiver.full_decode_feature_min_mix_ratio = 0.0


def resolve_payload_target_window(data: dict[str, object]) -> tuple[float, float, float, float]:
    target_bpp = finite_float_or_none(data.get("payload_target_bpp"))
    if target_bpp is None or target_bpp <= 0.0:
        target_bpp = finite_float_or_none(data.get("target_payload_bpp"))
    if target_bpp is None or target_bpp <= 0.0:
        target_bpp = DEFAULT_TARGET_PAYLOAD_BPP
    tolerance = finite_float_or_none(data.get("payload_bpp_tolerance"))
    if tolerance is None or tolerance < 0.0:
        tolerance = DEFAULT_PAYLOAD_BPP_TOLERANCE
    lower = finite_float_or_none(data.get("payload_bpp_window_lower"))
    upper = finite_float_or_none(data.get("payload_bpp_window_upper"))
    if lower is None or upper is None or upper < lower:
        lower = max(0.0, target_bpp - tolerance)
        upper = target_bpp + tolerance
    return target_bpp, tolerance, lower, upper


def build_target_pass_fail(data: dict[str, float | int | bool | str]) -> dict[str, dict[str, float | str | bool]]:
    # 生成最终论文目标的逐项 pass/fail 表，避免人工核对时漏掉 KID、鲁棒 BER 或解码比例。
    report: dict[str, dict[str, float | str | bool]] = {}
    all_passed = True
    payload_target_bpp, payload_tolerance, payload_lower, payload_upper = resolve_payload_target_window(data)
    for metric_name, (threshold, direction) in TARGET_THRESHOLDS.items():
        value = data.get(metric_name, float("nan"))
        if metric_name == "payload_bpp":
            passed = metric_passes_target(value, (payload_lower, payload_upper), "between")
            target_text = (
                f"{payload_lower:g} <= value <= {payload_upper:g} "
                f"(target {payload_target_bpp:g}±{payload_tolerance:g})"
            )
        else:
            passed = metric_passes_target(value, threshold, direction)
            if direction == "between":
                lower, upper = threshold
                target_text = f"{float(lower):g} <= value <= {float(upper):g}"
            else:
                target_text = f"{direction} {float(threshold):g}"
        all_passed = all_passed and passed
        report[metric_name] = {
            "value": float(value) if isinstance(value, (int, float)) else str(value),
            "target": target_text,
            "passed": passed,
        }
    report["ALL_TARGETS_PASSED"] = {
        "value": "all metrics",
        "target": "all passed",
        "passed": all_passed,
    }
    return report


def evaluation_backend_reliability(data: dict[str, object]) -> dict[str, bool | str]:
    # 标记关键自然度与检测指标是否来自正式后端，避免 fallback/proxy 数值被误当作论文主结果达标证据。
    fid_backend = str(data.get("fid_backend", "") or "")
    brisque_backend = str(data.get("brisque_backend", "") or "")
    srnet_backend = str(data.get("srnet_detector", "") or "")
    official_srnet_backend = str(data.get("official_srnet_backend", "") or "")
    fid_ok = (
        "clean-fid+legacy-inception-pretrained-kid" in fid_backend
        or fid_backend == "legacy-inception-pretrained"
    )
    brisque_ok = brisque_backend in {"imquality.brisque", "brisque"}
    srnet_ok = srnet_backend.startswith("srnet_") and "proxy" not in srnet_backend and "fallback_to_stat" not in srnet_backend
    official_srnet_ok = official_srnet_backend == "official_srnet"
    detector_ok = srnet_ok
    all_ok = fid_ok and brisque_ok and detector_ok
    return {
        "fid_backend_reliable": fid_ok,
        "brisque_backend_reliable": brisque_ok,
        "srnet_backend_reliable": srnet_ok,
        "official_srnet_backend_reliable": official_srnet_ok,
        "detector_backend_reliable": detector_ok,
        "all_critical_backends_reliable": all_ok,
        "fid_backend": fid_backend,
        "brisque_backend": brisque_backend,
        "srnet_backend": srnet_backend,
        "official_srnet_backend": official_srnet_backend,
    }


# 将 DataLoader 返回的张量或原始尺寸图像列表统一拆成可逐项送入模型的小批次。
def assert_target_metric_fields_present(data: dict[str, object]) -> None:
    # 确保最终目标中的每个指标都有可导出的字段，避免评估结果无法证明目标是否达成。
    missing = [metric_name for metric_name in TARGET_THRESHOLDS if metric_name not in data]
    if missing:
        raise KeyError(f"Missing target metric field(s): {', '.join(missing)}")


def iter_image_microbatches(images) -> list[torch.Tensor]:
    if isinstance(images, torch.Tensor):
        if images.dim() == 3:
            return [images.unsqueeze(0)]
        if images.dim() != 4:
            raise ValueError(
                "iter_image_microbatches expects image tensors with shape [C, H, W] or [B, C, H, W], "
                f"got shape {tuple(images.shape)}."
            )
        return [image.unsqueeze(0) for image in images]
    microbatches: list[torch.Tensor] = []
    for image in images:
        microbatches.append(image.unsqueeze(0) if image.dim() == 3 else image)
    return microbatches


# 根据接收端实际解码码块索引裁剪发送端目标比特。
def select_decoded_reference(reference: torch.Tensor, block_indices: torch.Tensor | None) -> torch.Tensor:
    if block_indices is None:
        return reference
    return reference.index_select(1, block_indices.to(device=reference.device))


# 构造当前参与评估码包的原始 block 位置，用于排除末包 padding 位。
def selected_block_positions(
    block_count: int,
    device: torch.device,
    block_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    if block_indices is None:
        return torch.arange(block_count, device=device, dtype=torch.long)
    return block_indices.to(device=device, dtype=torch.long)


# 只统计真实 payload 位，避免 padding 位污染 BER。
def payload_valid_mask_from_packet(
    packet,
    reference: torch.Tensor,
    block_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    payload_length = int(packet.payload_length)
    block_positions = selected_block_positions(reference.shape[1], reference.device, block_indices)
    bit_positions = torch.arange(payload_length, device=reference.device, dtype=torch.long)
    flat_positions = block_positions.view(1, -1, 1) * payload_length + bit_positions.view(1, 1, -1)
    mask = flat_positions < int(packet.valid_info_bits)
    return mask.expand(reference.shape[0], -1, -1)


# 优先取真实硬译码输出做评估；若当前只解了部分码包，则回退到 soft 预测结果。
def select_metric_bit_pairs(model_output) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    block_indices = model_output.receiver.decoded_block_indices
    if (
        getattr(model_output.receiver, "used_external_llr", False)
        and model_output.receiver.hard_decoded_payload_bits is not None
        and model_output.receiver.hard_decoded_code_bits is not None
    ):
        return (
            model_output.transmitter.packet.payload_bits.detach(),
            model_output.receiver.hard_decoded_payload_bits.detach(),
            model_output.transmitter.packet.coded_bits.detach(),
            model_output.receiver.hard_decoded_code_bits.detach(),
        )
    target_info_bits = select_decoded_reference(model_output.transmitter.packet.payload_bits.detach(), block_indices)
    metric_payload_bits = getattr(model_output.receiver, "metric_payload_bits", None)
    if metric_payload_bits is not None:
        predicted_info_bits = metric_payload_bits.detach()
    elif model_output.receiver.hard_decoded_payload_bits is not None:
        target_info_bits = model_output.transmitter.packet.payload_bits.detach()
        predicted_info_bits = model_output.receiver.hard_decoded_payload_bits.detach()
    else:
        predicted_info_bits = model_output.receiver.decoded_bits.detach()[..., : target_info_bits.shape[-1]]

    target_code_bits = select_decoded_reference(model_output.transmitter.packet.coded_bits.detach(), block_indices)
    metric_code_bits = getattr(model_output.receiver, "metric_code_bits", None)
    if metric_code_bits is not None:
        predicted_code_bits = metric_code_bits.detach()
    elif model_output.receiver.hard_decoded_code_bits is not None:
        target_code_bits = model_output.transmitter.packet.coded_bits.detach()
        predicted_code_bits = model_output.receiver.hard_decoded_code_bits.detach()
    else:
        predicted_code_bits = model_output.receiver.decoded_code_bits.detach()
    return target_info_bits, predicted_info_bits, target_code_bits, predicted_code_bits


def summarize_decode_counts(model_output) -> tuple[int, int, int, int]:
    packet = model_output.transmitter.packet
    decoded_block_indices = model_output.receiver.decoded_block_indices
    source_blocks = int(packet.source_num_blocks or packet.num_blocks)
    total_blocks = source_blocks
    group_size = max(1, int(packet.group_size))
    total_groups = int(math.ceil(source_blocks / group_size))
    if decoded_block_indices is None:
        decoded_blocks = source_blocks
        decoded_groups = total_groups
    else:
        valid_decoded_indices = decoded_block_indices.to(
            device=decoded_block_indices.device,
            dtype=torch.long,
        )
        valid_decoded_indices = valid_decoded_indices[valid_decoded_indices < source_blocks]
        decoded_blocks = int(valid_decoded_indices.numel())
        decoded_group_indices = torch.unique(
            valid_decoded_indices // group_size
        )
        decoded_groups = int(decoded_group_indices.numel())
    return decoded_blocks, total_blocks, decoded_groups, total_groups


@dataclass
class EvaluationResult:
    psnr_decoded: float
    ssim_decoded: float
    psnr_restored: float
    ssim_restored: float
    ssim_restored_y: float
    mae_restored: float
    rmse_restored: float
    stego_psnr_to_source: float
    stego_ssim_to_source: float
    stego_psnr_to_carrier: float
    stego_ssim_to_carrier: float
    stego_ssim_to_carrier_y: float
    fid_cover: float
    fid_carrier: float
    kid_stego: float
    kid_carrier: float
    brisque_real: float
    brisque_stego: float
    brisque_gap_to_real: float
    lpips_restored: float
    statistical_detection_rate: float
    statistical_stego_detection_rate: float
    statistical_false_positive_rate: float
    statistical_detection_advantage: float
    statistical_anti_detection_rate: float
    srnet_detection_rate: float
    srnet_balanced_accuracy: float
    srnet_auc: float
    srnet_eer: float
    srnet_stego_detection_rate: float
    srnet_false_positive_rate: float
    srnet_detection_advantage: float
    srnet_anti_detection_rate: float
    official_srnet_detection_rate: float
    official_srnet_anti_detection_rate: float
    official_srnet_test_loss: float
    official_srnet_backend: str
    mean_detection_advantage: float
    anti_detection_rate: float
    payload_bpp: float
    payload_target_bpp: float
    payload_bpp_tolerance: float
    ber_info: float
    ber_code: float
    jpeg50_ber_info: float
    jpeg50_ber_code: float
    mixed_ber_info: float
    mixed_ber_code: float
    attack_psnr_restored: float
    attack_ssim_restored: float
    attack_lpips_restored: float
    attack_ber_info: float
    attack_ber_code: float
    attack_decoded_ratio: float
    decoded_blocks: float
    total_blocks: float
    decoded_block_ratio: float
    decoded_groups: float
    total_groups: float
    decoded_group_ratio: float
    decoded_ratio: float
    num_images: int
    fid_backend: str
    statistical_detector: str
    srnet_detector: str
    detector_samples: int
    detector_size: int
    full_decode: bool = True
    max_images: int | None = None
    checkpoint_path: str | None = None
    data_path: str | None = None
    split_path: str | None = None
    detector_test_ratio: float = 0.3
    srnet_epochs: int = 0
    srnet_arch: str = "deep"
    srnet_weights: str | None = None
    srnet_batch_size: int | None = None
    fid_requested_backend: str = "auto"
    clean_fid_mode: str = "clean"
    brisque_backend: str = "proxy-nss"
    stat_requested_backend: str = "auto"
    active_dataset: str = "eval"
    image_size: int = 0
    clean_eval: bool = True
    latent_select_candidates: int = 4
    latent_select_used_ratio: float = 0.0
    latent_select_avg_index: float = 0.0
    latent_select_score: str = "cover_brisque_proxy"
    latent_select_prior_checkpoint: str | None = None
    latent_select_prior_mixes: str = ""
    latent_select_min_psnr: float = 38.0
    latent_select_min_ssim: float = 0.97
    comparison_carrier_psnr: float | None = None
    comparison_carrier_ssim: float | None = None
    comparison_carrier_lpips: float | None = None
    comparison_carrier_mae: float | None = None
    comparison_carrier_rmse: float | None = None
    comparison_recovery_psnr: float | None = None
    comparison_recovery_ssim: float | None = None
    stego_secret_distinct_ok: bool = False
    stego_secret_distinct_ratio: float = 0.0
    natural_reference_dataset: str = "secret"
    natural_reference_dirs: str = ""
    natural_reference_count: int = 0
    comparison_lpips_or_ber: str | None = None
    comparison_protocol: str = "carrier=generated_image_vs_stego_image,recovery=original_vs_restored,ssim=rgb"


# 将评估结果转成导出字典，并补充历史字段兼容。
def evaluation_result_to_dict(result: EvaluationResult) -> dict[str, float | int]:
    data = asdict(result)
    data["PSNR_decoded"] = data["psnr_decoded"]
    data["SSIM_decoded"] = data["ssim_decoded"]
    data["PSNR_restored"] = data["psnr_restored"]
    data["SSIM_restored"] = data["ssim_restored"]
    data["SSIM_restored_Y"] = data["ssim_restored_y"]
    data["LPIPS_restored"] = data["lpips_restored"]
    data["MAE_restored"] = data["mae_restored"]
    data["RMSE_restored"] = data["rmse_restored"]
    data["BER_info"] = data["ber_info"]
    data["BER_code"] = data["ber_code"]
    data["BER_info_clean"] = data["ber_info"]
    data["BER_code_clean"] = data["ber_code"]
    data["JPEG50_BER_info"] = data["jpeg50_ber_info"]
    data["JPEG50_BER_code"] = data["jpeg50_ber_code"]
    data["Mixed_BER_info"] = data["mixed_ber_info"]
    data["Mixed_BER_code"] = data["mixed_ber_code"]
    data["PSNR_attack"] = data["attack_psnr_restored"]
    data["SSIM_attack"] = data["attack_ssim_restored"]
    data["LPIPS_attack"] = data["attack_lpips_restored"]
    data["BER_info_attack"] = data["attack_ber_info"]
    data["BER_code_attack"] = data["attack_ber_code"]
    data["decoded_ratio_attack"] = data["attack_decoded_ratio"]
    data["FID_stego"] = data["fid_cover"]
    data["FID_carrier"] = data["fid_carrier"]
    data["KID"] = data["kid_stego"]
    data["KID_stego"] = data["kid_stego"]
    data["KID_carrier"] = data["kid_carrier"]
    data["BRISQUE_real"] = data["brisque_real"]
    data["BRISQUE_stego"] = data["brisque_stego"]
    data["BRISQUE_gap_to_real"] = data["brisque_gap_to_real"]
    data["psnr_stego_secret"] = data["stego_psnr_to_source"]
    data["ssim_stego_secret"] = data["stego_ssim_to_source"]
    data["psnr_stego_carrier"] = data["stego_psnr_to_carrier"]
    data["ssim_stego_carrier"] = data["stego_ssim_to_carrier"]
    data["ssim_stego_carrier_y"] = data["stego_ssim_to_carrier_y"]
    data["clean_carrier_delta_psnr"] = data["stego_psnr_to_carrier"]
    data["clean_carrier_delta_ssim"] = data["stego_ssim_to_carrier"]
    data["clean_carrier_delta_ssim_y"] = data["stego_ssim_to_carrier_y"]
    data["PSNR_stego_vs_source"] = data["stego_psnr_to_source"]
    data["SSIM_stego_vs_source"] = data["stego_ssim_to_source"]
    data["comparison_display_carrier_psnr"] = data["stego_psnr_to_source"]
    data["comparison_display_carrier_ssim"] = data["stego_ssim_to_source"]
    data["Stego_secret_distinct_ok"] = data["stego_secret_distinct_ok"]
    data["Stego_secret_distinct_ratio"] = data["stego_secret_distinct_ratio"]
    data["natural_reference_dataset"] = data["natural_reference_dataset"]
    data["natural_reference_dirs"] = data["natural_reference_dirs"]
    data["natural_reference_count"] = data["natural_reference_count"]
    data["PSNR_stego_vs_carrier"] = data["stego_psnr_to_carrier"]
    data["SSIM_stego_vs_carrier"] = data["stego_ssim_to_carrier"]
    data["SSIM_Y_stego_vs_carrier"] = data["stego_ssim_to_carrier_y"]
    data["PSNR_stego_vs_clean_carrier"] = data["stego_psnr_to_carrier"]
    data["SSIM_stego_vs_clean_carrier"] = data["stego_ssim_to_carrier"]
    data["SSIM_Y_stego_vs_clean_carrier"] = data["stego_ssim_to_carrier_y"]
    data["stego_carrier_psnr"] = data["stego_psnr_to_carrier"]
    data["stego_carrier_ssim"] = data["stego_ssim_to_carrier"]
    data["stego_carrier_ssim_y"] = data["stego_ssim_to_carrier_y"]
    data["PSNR(C)"] = data["stego_psnr_to_carrier"]
    data["SSIM(C)"] = data["stego_ssim_to_carrier"]
    data["SSIM(C)_Y"] = data["stego_ssim_to_carrier_y"]
    data["statistical_detection_rate_down"] = data["statistical_detection_rate"]
    data["statistical_stego_detection_rate"] = data["statistical_stego_detection_rate"]
    data["statistical_false_positive_rate"] = data["statistical_false_positive_rate"]
    data["statistical_detection_advantage_down"] = data["statistical_detection_advantage"]
    data["latent_select_candidates"] = data["latent_select_candidates"]
    data["latent_select_used_ratio"] = data["latent_select_used_ratio"]
    data["latent_select_avg_index"] = data["latent_select_avg_index"]
    data["latent_select_score"] = data["latent_select_score"]
    data["latent_select_prior_checkpoint"] = data["latent_select_prior_checkpoint"]
    data["latent_select_prior_mixes"] = data["latent_select_prior_mixes"]
    data["latent_select_min_psnr"] = data["latent_select_min_psnr"]
    data["latent_select_min_ssim"] = data["latent_select_min_ssim"]
    data["srnet_detection_rate_down"] = data["srnet_detection_rate"]
    data["SRNet_balanced_accuracy"] = data["srnet_balanced_accuracy"]
    data["SRNet_AUC"] = data["srnet_auc"]
    data["SRNet_EER"] = data["srnet_eer"]
    data["srnet_stego_detection_rate"] = data["srnet_stego_detection_rate"]
    data["srnet_false_positive_rate"] = data["srnet_false_positive_rate"]
    data["srnet_detection_advantage_down"] = data["srnet_detection_advantage"]
    data["official_srnet_detection_rate"] = data["official_srnet_detection_rate"]
    data["official_srnet_anti_detection_rate"] = data["official_srnet_anti_detection_rate"]
    data["official_srnet_test_loss"] = data["official_srnet_test_loss"]
    data["official_srnet_backend"] = data["official_srnet_backend"]
    data["mean_detection_advantage_down"] = data["mean_detection_advantage"]
    data["ccfa_anti_detection_pass"] = (
        data["statistical_detection_advantage"] < 0.1
        and data["srnet_detection_advantage"] < 0.1
    )
    data["anti_detection_rate_up"] = data["anti_detection_rate"]
    data["payload_capacity_bpp"] = data["payload_bpp"]
    data["payload_target_bpp"] = data["payload_target_bpp"]
    data["target_payload_bpp"] = data["payload_target_bpp"]
    data["payload_bpp_tolerance"] = data["payload_bpp_tolerance"]
    _payload_target_bpp, _payload_tolerance, payload_window_lower, payload_window_upper = resolve_payload_target_window(data)
    data["payload_bpp_window_lower"] = payload_window_lower
    data["payload_bpp_window_upper"] = payload_window_upper
    data["decoded_blocks/total_blocks"] = (
        f"{int(round(data['decoded_blocks']))}/{int(round(data['total_blocks']))}"
    )
    data["decoded_groups/total_groups"] = (
        f"{int(round(data['decoded_groups']))}/{int(round(data['total_groups']))}"
    )
    data["decoded_blocks_ratio"] = data["decoded_block_ratio"]
    data["decoded_groups_ratio"] = data["decoded_group_ratio"]
    data["detector_samples"] = data["detector_samples"]
    data["detector_size"] = data["detector_size"]
    data["comparison_carrier_psnr"] = data["comparison_carrier_psnr"]
    data["comparison_carrier_ssim"] = data["comparison_carrier_ssim"]
    data["comparison_recovery_psnr"] = data["comparison_recovery_psnr"]
    data["comparison_recovery_ssim"] = data["comparison_recovery_ssim"]
    data["comparison_lpips_or_ber"] = data["comparison_lpips_or_ber"]
    data["comparison_protocol"] = data["comparison_protocol"]
    reliability = evaluation_backend_reliability(data)
    data.update(reliability)
    assert_target_metric_fields_present(data)
    target_report = build_target_pass_fail(data)
    data["ALL_TARGETS_PASSED"] = (
        bool(target_report["ALL_TARGETS_PASSED"]["passed"])
        and bool(data.get("all_critical_backends_reliable", False))
    )
    return data


# 按截图中的实验维度组织指标，便于论文表格和日志直接读取。
def evaluation_result_to_groups(result: EvaluationResult) -> dict[str, dict[str, float | int | str | bool | dict]]:
    data = evaluation_result_to_dict(result)
    target_pass_fail = build_target_pass_fail(data)
    target_pass_fail["ALL_TARGETS_PASSED"]["passed"] = bool(data.get("ALL_TARGETS_PASSED", False))
    return {
        "target_pass_fail": target_pass_fail,
        "backend_reliability": {
            "all_critical_backends_reliable": data.get("all_critical_backends_reliable", False),
            "fid_backend_reliable": data.get("fid_backend_reliable", False),
            "brisque_backend_reliable": data.get("brisque_backend_reliable", False),
            "srnet_backend_reliable": data.get("srnet_backend_reliable", False),
            "official_srnet_backend_reliable": data.get("official_srnet_backend_reliable", False),
            "detector_backend_reliable": data.get("detector_backend_reliable", False),
            "fid_backend": data.get("fid_backend", ""),
            "brisque_backend": data.get("brisque_backend", ""),
            "srnet_backend": data.get("srnet_backend", ""),
            "official_srnet_backend": data.get("official_srnet_backend", ""),
        },
        "recovery_quality": {
            "PSNR_restored": data["PSNR_restored"],
            "SSIM_restored": data["SSIM_restored"],
            "SSIM_restored_Y": data["SSIM_restored_Y"],
            "LPIPS_restored": data["LPIPS_restored"],
            "MAE_restored": data["MAE_restored"],
            "RMSE_restored": data["RMSE_restored"],
        },
        "stego_quality_and_naturalness": {
            "PSNR_stego_vs_clean_carrier": data["PSNR_stego_vs_clean_carrier"],
            "SSIM_stego_vs_clean_carrier": data["SSIM_stego_vs_clean_carrier"],
            "SSIM_Y_stego_vs_clean_carrier": data["SSIM_Y_stego_vs_clean_carrier"],
            "clean_carrier_delta_psnr": data["clean_carrier_delta_psnr"],
            "clean_carrier_delta_ssim": data["clean_carrier_delta_ssim"],
            "clean_carrier_delta_ssim_y": data["clean_carrier_delta_ssim_y"],
            "PSNR(C)_stego_vs_clean_carrier": data["PSNR(C)"],
            "SSIM(C)_stego_vs_clean_carrier": data["SSIM(C)"],
            "SSIM(C)_Y_stego_vs_clean_carrier": data["SSIM(C)_Y"],
            "FID_stego": data["FID_stego"],
            "FID_carrier": data["FID_carrier"],
            "KID_stego": data["KID_stego"],
            "KID_carrier": data["KID_carrier"],
            "BRISQUE_real": data["BRISQUE_real"],
            "BRISQUE_stego": data["BRISQUE_stego"],
            "BRISQUE_gap_to_real": data["BRISQUE_gap_to_real"],
            "PSNR_stego_vs_source": data["PSNR_stego_vs_source"],
            "SSIM_stego_vs_source": data["SSIM_stego_vs_source"],
        },
        "anti_steganalysis": {
            "statistical_detection_rate_down": data["statistical_detection_rate_down"],
            "statistical_stego_detection_rate": data["statistical_stego_detection_rate"],
            "statistical_false_positive_rate": data["statistical_false_positive_rate"],
            "statistical_detection_advantage_down": data["statistical_detection_advantage_down"],
            "statistical_anti_detection_rate": data["statistical_anti_detection_rate"],
            "srnet_detection_rate_down": data["srnet_detection_rate_down"],
            "SRNet_balanced_accuracy": data["SRNet_balanced_accuracy"],
            "SRNet_AUC": data["SRNet_AUC"],
            "SRNet_EER": data["SRNet_EER"],
            "srnet_stego_detection_rate": data["srnet_stego_detection_rate"],
            "srnet_false_positive_rate": data["srnet_false_positive_rate"],
            "srnet_detection_advantage_down": data["srnet_detection_advantage_down"],
            "mean_detection_advantage_down": data["mean_detection_advantage_down"],
            "ccfa_anti_detection_pass": data["ccfa_anti_detection_pass"],
            "srnet_anti_detection_rate": data["srnet_anti_detection_rate"],
            "anti_detection_rate_up": data["anti_detection_rate_up"],
        },
        "communication_and_capacity": {
            "payload_target_bpp": data["payload_target_bpp"],
            "payload_bpp_tolerance": data["payload_bpp_tolerance"],
            "payload_bpp_window_lower": data["payload_bpp_window_lower"],
            "payload_bpp_window_upper": data["payload_bpp_window_upper"],
            "payload_bpp": data["payload_bpp"],
            "BER_info_clean": data["BER_info_clean"],
            "BER_code_clean": data["BER_code_clean"],
            "JPEG50_BER_info": data["JPEG50_BER_info"],
            "JPEG50_BER_code": data["JPEG50_BER_code"],
            "Mixed_BER_info": data["Mixed_BER_info"],
            "Mixed_BER_code": data["Mixed_BER_code"],
            "PSNR_attack": data["PSNR_attack"],
            "SSIM_attack": data["SSIM_attack"],
            "LPIPS_attack": data["LPIPS_attack"],
            "BER_info_attack": data["BER_info_attack"],
            "BER_code_attack": data["BER_code_attack"],
            "decoded_ratio_attack": data["decoded_ratio_attack"],
            "decoded_ratio": data["decoded_ratio"],
            "decoded_block_ratio": data["decoded_block_ratio"],
            "decoded_group_ratio": data["decoded_group_ratio"],
            "decoded_blocks_ratio": data["decoded_blocks_ratio"],
            "decoded_groups_ratio": data["decoded_groups_ratio"],
            "decoded_blocks/total_blocks": data["decoded_blocks/total_blocks"],
            "decoded_groups/total_groups": data["decoded_groups/total_groups"],
            "num_images": data["num_images"],
        },
        "evaluation_protocol": {
            "full_decode": data["full_decode"],
            "max_images": data["max_images"],
            "checkpoint_path": data["checkpoint_path"],
            "data_path": data["data_path"],
            "split_path": data["split_path"],
            "detector_test_ratio": data["detector_test_ratio"],
            "srnet_epochs": data["srnet_epochs"],
            "srnet_arch": data["srnet_arch"],
            "srnet_weights": data["srnet_weights"],
            "srnet_batch_size": data["srnet_batch_size"],
            "fid_backend": data["fid_backend"],
            "fid_requested_backend": data["fid_requested_backend"],
            "clean_fid_mode": data["clean_fid_mode"],
            "brisque_backend": data["brisque_backend"],
            "statistical_detector": data["statistical_detector"],
            "stat_requested_backend": data["stat_requested_backend"],
            "clean_eval": data["clean_eval"],
            "latent_select_candidates": data["latent_select_candidates"],
            "latent_select_used_ratio": data["latent_select_used_ratio"],
            "latent_select_avg_index": data["latent_select_avg_index"],
            "latent_select_score": data["latent_select_score"],
            "latent_select_prior_checkpoint": data["latent_select_prior_checkpoint"],
            "latent_select_prior_mixes": data["latent_select_prior_mixes"],
            "srnet_detector": data["srnet_detector"],
            "payload_target_bpp": data["payload_target_bpp"],
            "payload_bpp_tolerance": data["payload_bpp_tolerance"],
            "payload_bpp_window_lower": data["payload_bpp_window_lower"],
            "payload_bpp_window_upper": data["payload_bpp_window_upper"],
        },
    }


# 生成一份人类可读的 Markdown 评估摘要，方便快速查看四类指标。
def format_grouped_metrics_markdown(groups: dict[str, dict[str, float | int | str | bool | dict]]) -> str:
    lines = ["# WaveBridege Evaluation Metrics", ""]
    for group_name, metrics in groups.items():
        lines.append(f"## {group_name}")
        for metric_name, value in metrics.items():
            if isinstance(value, dict) and {"value", "target", "passed"}.issubset(value.keys()):
                status = "PASS" if bool(value["passed"]) else "FAIL"
                metric_value = value["value"]
                if isinstance(metric_value, float):
                    metric_value = f"{metric_value:.6f}"
                lines.append(f"- {metric_name}: {status} ({metric_value}, target {value['target']})")
            elif isinstance(value, float):
                lines.append(f"- {metric_name}: {value:.6f}")
            else:
                lines.append(f"- {metric_name}: {value}")
        lines.append("")
    return "\n".join(lines)


# 写入单行 CSV，保持字段顺序稳定，方便 Comparison 的论文表格脚本直接读取。
def write_single_row_csv(path: Path, row: dict[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)


def write_multi_row_csv(path: Path, rows: list[dict[str, object]]) -> None:
    # 将多行结果写成联合字段集合的 CSV，避免不同 attack 行字段不一致时触发写盘异常。
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_target_pass_fail_csv(path: Path, target_pass_fail: dict[str, dict[str, object]]) -> None:
    # 将最终目标逐项审计结果落成 CSV，方便和论文表格/Comparison 项目直接核对。
    rows = [
        {
            "metric": metric_name,
            "value": item.get("value"),
            "target": item.get("target"),
            "passed": item.get("passed"),
        }
        for metric_name, item in target_pass_fail.items()
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value", "target", "passed"])
        writer.writeheader()
        writer.writerows(rows)


def write_rows_json(path: Path, rows: list[dict[str, object]]) -> None:
    # 将表格行按 Comparison 统一评估脚本常用的 JSON 数组形式写盘。
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_rows_bundle(path: Path, rows: list[dict[str, object]]) -> None:
    # 同时导出 CSV 和 JSON，便于 Comparison 目录下的后处理脚本直接复用。
    if not rows:
        raise ValueError(f"write_rows_bundle expects at least one row, got 0 for {path}.")
    if len(rows) == 1:
        write_single_row_csv(path, rows[0])
    else:
        write_multi_row_csv(path, rows)
    write_rows_json(path.with_suffix(".json"), rows)


def write_table_bundle(path: Path, row: dict[str, object]) -> None:
    # 论文表风格结果同时导出 CSV、JSON 和 Markdown，减少后续手工整理。
    write_rows_bundle(path, [row])
    write_table_markdown(path.with_suffix(".md"), row)


def write_table_markdown(path: Path, row: dict[str, object]) -> None:
    # 将单行论文表导出为 Markdown，便于直接放入实验记录或论文草稿。
    headers = list(row.keys())
    values: list[str] = []
    for header in headers:
        value = row[header]
        if isinstance(value, float):
            if math.isnan(value):
                values.append("nan")
            elif abs(value) >= 10.0:
                values.append(f"{value:.2f}")
            else:
                values.append(f"{value:.6f}")
        else:
            values.append(str(value))
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        "| " + " | ".join(values) + " |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_comparison_main_row(data: dict[str, object], method: str = "WaveBridege (ours)") -> dict[str, object]:
    # 生成与 Comparison 主表同字段的单行结果，便于与你的多个 baseline 直接并排对比。
    carrier_psnr = data.get(
        "comparison_display_carrier_psnr",
        data.get("PSNR_stego_vs_source", data.get("comparison_carrier_psnr", data.get("PSNR(C)"))),
    )
    carrier_ssim = data.get(
        "comparison_display_carrier_ssim",
        data.get("SSIM_stego_vs_source", data.get("comparison_carrier_ssim", data.get("SSIM(C)"))),
    )
    recovery_psnr = data.get("comparison_recovery_psnr", data.get("PSNR_restored"))
    recovery_ssim = data.get("comparison_recovery_ssim", data.get("SSIM_restored"))
    lpips_or_ber = data.get("comparison_lpips_or_ber")
    if not lpips_or_ber:
        lpips_value = data.get("LPIPS_restored")
        ber_value = data.get("BER_info_clean")
        lpips_text = "nan" if lpips_value is None else f"{float(lpips_value):.3f}"
        ber_text = "nan" if ber_value is None else f"{float(ber_value):.3f}"
        lpips_or_ber = f"LPIPS {lpips_text} / BER {ber_text}"
    return {
        "Method": method,
        "Carrier PSNR": carrier_psnr,
        "Carrier SSIM": carrier_ssim,
        "Recovery PSNR": recovery_psnr,
        "Recovery SSIM": recovery_ssim,
        "LPIPS / BER": lpips_or_ber,
    }


def normalize_comparison_dataset_label(dataset_name: str) -> str:
    # 统一 Comparison 评估表中的数据集命名，避免一个数据集出现多种别名。
    normalized = str(dataset_name).strip()
    lowered = normalized.lower()
    if "div2k" in lowered:
        return "DIV2K"
    if "boss" in lowered:
        return "BOSSBase-quarter"
    if "alaska" in lowered:
        return "ALASKA2"
    return normalized if normalized else "eval"


def comparison_dataset_key(dataset_label: str) -> str:
    # 为 table 文件名生成稳定 key，保证和 Comparison 目录中的命名习惯一致。
    lowered = dataset_label.lower()
    if "div2k" in lowered:
        return "div2k"
    if "boss" in lowered:
        return "bossbase"
    if "alaska" in lowered:
        return "alaska2"
    return lowered.replace("#", "").replace(" ", "_").replace("-", "_") or "eval"


def resolve_paper_srnet_detection_rate(data: dict[str, object]) -> tuple[float, str]:
    # 论文表优先采用官方 SRNet 结果；没有官方结果时回退到 Comparison 代理检测器。
    try:
        official_rate = float(data.get("official_srnet_detection_rate", float("nan")))
    except (TypeError, ValueError):
        official_rate = float("nan")
    if math.isfinite(official_rate):
        return official_rate, "official_srnet"

    try:
        proxy_rate = float(data.get("srnet_detection_rate", float("nan")))
    except (TypeError, ValueError):
        proxy_rate = float("nan")
    return proxy_rate, "comparison_proxy"


def build_comparison_paper_rows(
    data: dict[str, object],
    method: str = "WaveBridege",
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    # 按 Comparison/build_paper_tables.py 的字段格式生成 table1~4 所需的单方法表行。
    dataset_label = normalize_comparison_dataset_label(str(data.get("active_dataset", data.get("dataset", "eval"))))
    srnet_detection_rate, srnet_source = resolve_paper_srnet_detection_rate(data)
    anti_detection_rate = (
        float(1.0 - srnet_detection_rate)
        if math.isfinite(srnet_detection_rate)
        else float(data.get("anti_detection_rate", float("nan")))
    )
    common_notes = (
        f"dataset={dataset_label}; "
        f"checkpoint={data.get('checkpoint_path')}; "
        f"carrier=source_vs_stego_image; "
        f"internal_carrier_delta=generated_image_vs_stego_image; "
        f"recovery=original_vs_restored; "
        f"srnet_source={srnet_source}"
    )
    main_row = {
        "method": method,
        "dataset": dataset_label,
        "payload_bpp": data.get("payload_bpp"),
        "carrier_psnr": data.get(
            "comparison_display_carrier_psnr",
            data.get("PSNR_stego_vs_source", data.get("comparison_carrier_psnr", data.get("PSNR(C)"))),
        ),
        "carrier_ssim": data.get(
            "comparison_display_carrier_ssim",
            data.get("SSIM_stego_vs_source", data.get("comparison_carrier_ssim", data.get("SSIM(C)"))),
        ),
        "psnr_restored": data.get("PSNR_restored"),
        "ssim_restored": data.get("SSIM_restored"),
        "lpips_restored": data.get("LPIPS_restored"),
        "ber_info_clean": data.get("BER_info_clean"),
        "ber_code_clean": data.get("BER_code_clean"),
        "decoded_blocks_ratio": data.get("decoded_blocks_ratio"),
        "decoded_groups_ratio": data.get("decoded_groups_ratio"),
        "fid_stego": data.get("FID_stego"),
        "kid": data.get("KID"),
        "brisque_gap_to_real": data.get("BRISQUE_gap_to_real"),
        "srnet_detection_rate": srnet_detection_rate,
        "anti_detection_rate": anti_detection_rate,
        "notes": common_notes,
    }
    robustness_row = {
        "method": method,
        "dataset": dataset_label,
        "attack": "jpeg75+gaussian0.01",
        "psnr_restored_attack": data.get("PSNR_attack"),
        "ssim_restored_attack": data.get("SSIM_attack"),
        "lpips_restored_attack": data.get("LPIPS_attack"),
        "ber_info_attack": data.get("BER_info_attack"),
        "ber_code_attack": data.get("BER_code_attack"),
        "decoded_ratio_attack": data.get("decoded_ratio_attack"),
        "jpeg50_ber_info": data.get("JPEG50_BER_info"),
        "jpeg50_ber_code": data.get("JPEG50_BER_code"),
        "mixed_ber_info": data.get("Mixed_BER_info"),
        "mixed_ber_code": data.get("Mixed_BER_code"),
        "notes": common_notes,
    }
    detection_row = {
        "method": method,
        "dataset": dataset_label,
        "statistical_detection_rate": data.get("statistical_detection_rate"),
        "srnet_detection_rate": srnet_detection_rate,
        "srnet_balanced_accuracy": data.get("SRNet_balanced_accuracy"),
        "srnet_auc": data.get("SRNet_AUC"),
        "srnet_eer": data.get("SRNet_EER"),
        "anti_detection_rate": anti_detection_rate,
        "detector_samples": data.get("detector_samples", data.get("num_images")),
        "notes": common_notes,
    }
    return main_row, robustness_row, detection_row


def build_goal_metrics_row(data: dict[str, object], method: str = "WaveBridege") -> dict[str, object]:
    # 将当前线程 goal 中要求的最终指标压成单行，便于和 Comparison 结果一起归档核对。
    return {
        "method": method,
        "dataset": normalize_comparison_dataset_label(str(data.get("active_dataset", data.get("dataset", "eval")))),
        "payload_bpp": data.get("payload_bpp"),
        "PSNR_restored": data.get("PSNR_restored"),
        "SSIM_restored": data.get("SSIM_restored"),
        "LPIPS_restored": data.get("LPIPS_restored"),
        "BER_info_clean": data.get("BER_info_clean"),
        "BER_code_clean": data.get("BER_code_clean"),
        "decoded_blocks_ratio": data.get("decoded_blocks_ratio"),
        "decoded_groups_ratio": data.get("decoded_groups_ratio"),
        "FID_stego": data.get("FID_stego"),
        "KID": data.get("KID"),
        "BRISQUE_gap_to_real": data.get("BRISQUE_gap_to_real"),
        "SRNet_balanced_accuracy": data.get("SRNet_balanced_accuracy"),
        "SRNet_AUC": data.get("SRNet_AUC"),
        "SRNet_EER": data.get("SRNet_EER"),
        "JPEG50_BER_info": data.get("JPEG50_BER_info"),
        "Mixed_BER_info": data.get("Mixed_BER_info"),
        "PSNR_attack": data.get("PSNR_attack"),
        "SSIM_attack": data.get("SSIM_attack"),
        "LPIPS_attack": data.get("LPIPS_attack"),
        "BER_info_attack": data.get("BER_info_attack"),
        "decoded_ratio_attack": data.get("decoded_ratio_attack"),
        "Stego_secret_distinct_ok": data.get("Stego_secret_distinct_ok"),
        "target_pass_all": data.get("ALL_TARGETS_PASSED"),
        "checkpoint_path": data.get("checkpoint_path"),
    }


def build_stego_naturalness_row(data: dict[str, object], method: str = "WaveBridege") -> dict[str, object]:
    # 单独导出载密图自然度与分布相似度，便于和恢复质量、通信可靠性分开分析。
    return {
        "method": method,
        "dataset": normalize_comparison_dataset_label(str(data.get("active_dataset", data.get("dataset", "eval")))),
        "FID_stego": data.get("FID_stego"),
        "FID_carrier": data.get("FID_carrier"),
        "KID_stego": data.get("KID_stego"),
        "KID_carrier": data.get("KID_carrier"),
        "BRISQUE_real": data.get("BRISQUE_real"),
        "BRISQUE_stego": data.get("BRISQUE_stego"),
        "BRISQUE_gap_to_real": data.get("BRISQUE_gap_to_real"),
        "PSNR_stego_vs_clean_carrier": data.get("PSNR_stego_vs_clean_carrier"),
        "SSIM_stego_vs_clean_carrier": data.get("SSIM_stego_vs_clean_carrier"),
        "SSIM_Y_stego_vs_clean_carrier": data.get("SSIM_Y_stego_vs_clean_carrier"),
        "PSNR_stego_vs_source": data.get("PSNR_stego_vs_source"),
        "SSIM_stego_vs_source": data.get("SSIM_stego_vs_source"),
        "Stego_secret_distinct_ok": data.get("Stego_secret_distinct_ok"),
        "checkpoint_path": data.get("checkpoint_path"),
    }


# 导出与服务器 Comparison 表格脚本兼容的四类实验表。
def save_comparison_compatible_tables(data: dict[str, object], output: Path) -> None:
    method = "WaveBridege"
    dataset_name = normalize_comparison_dataset_label(str(data.get("active_dataset", data.get("dataset", "eval"))))
    dataset_key = comparison_dataset_key(dataset_name)
    status = "ok"
    common_notes = (
        f"dataset={dataset_name}; "
        f"full_decode={data.get('full_decode')}; "
        f"checkpoint={data.get('checkpoint_path')}; "
        f"carrier=source_vs_stego_image; "
        f"internal_carrier_delta=generated_image_vs_stego_image; "
        f"recovery=original_vs_restored"
    )
    quality_row = {
        "method": method,
        "status": status,
        "dataset": dataset_name,
        "payload_bpp": data.get("payload_bpp"),
        "carrier_psnr": data.get(
            "comparison_display_carrier_psnr",
            data.get("PSNR_stego_vs_source", data.get("comparison_carrier_psnr", data.get("PSNR(C)"))),
        ),
        "carrier_ssim": data.get(
            "comparison_display_carrier_ssim",
            data.get("SSIM_stego_vs_source", data.get("comparison_carrier_ssim", data.get("SSIM(C)"))),
        ),
        "carrier_lpips": data.get("comparison_carrier_lpips", float("nan")),
        "carrier_mae": data.get("comparison_carrier_mae", float("nan")),
        "carrier_rmse": data.get("comparison_carrier_rmse", float("nan")),
        "psnr_restored": data.get("PSNR_restored"),
        "ssim_restored": data.get("SSIM_restored"),
        "lpips_restored": data.get("LPIPS_restored"),
        "mae": data.get("MAE_restored"),
        "rmse": data.get("RMSE_restored"),
        "fid_stego": data.get("FID_stego"),
        "kid": data.get("KID"),
        "brisque_gap_to_real": data.get("BRISQUE_gap_to_real"),
        "notes": common_notes,
    }
    comm_row = {
        "method": method,
        "status": status,
        "dataset": dataset_name,
        "payload_bpp": data.get("payload_bpp"),
        "ber_info": data.get("BER_info_clean"),
        "ber_code": data.get("BER_code_clean"),
        "jpeg50_ber_info": data.get("JPEG50_BER_info"),
        "jpeg50_ber_code": data.get("JPEG50_BER_code"),
        "mixed_ber_info": data.get("Mixed_BER_info"),
        "mixed_ber_code": data.get("Mixed_BER_code"),
        "decoded_blocks": int(round(float(data.get("decoded_blocks", 0.0)))),
        "total_blocks": int(round(float(data.get("total_blocks", 0.0)))),
        "decoded_block_ratio": data.get("decoded_block_ratio"),
        "decoded_groups": int(round(float(data.get("decoded_groups", 0.0)))),
        "total_groups": int(round(float(data.get("total_groups", 0.0)))),
        "decoded_group_ratio": data.get("decoded_group_ratio"),
        "decoded_blocks_ratio": data.get("decoded_blocks_ratio"),
        "decoded_groups_ratio": data.get("decoded_groups_ratio"),
        "notes": common_notes,
    }
    robustness_rows = [
        {
            "method": method,
            "status": status,
            "dataset": dataset_name,
            "attack": "clean",
            "ber_info": data.get("BER_info_clean"),
            "ber_code": data.get("BER_code_clean"),
            "decoded_blocks_ratio": data.get("decoded_blocks_ratio"),
            "decoded_groups_ratio": data.get("decoded_groups_ratio"),
            "notes": common_notes,
        },
        {
            "method": method,
            "status": status,
            "dataset": dataset_name,
            "attack": "jpeg50",
            "ber_info": data.get("JPEG50_BER_info"),
            "ber_code": data.get("JPEG50_BER_code"),
            "notes": common_notes,
        },
        {
            "method": method,
            "status": status,
            "dataset": dataset_name,
            "attack": "mixed",
            "ber_info": data.get("Mixed_BER_info"),
            "ber_code": data.get("Mixed_BER_code"),
            "notes": common_notes,
        },
        {
            "method": method,
            "status": status,
            "dataset": dataset_name,
            "psnr_restored_attack": data.get("PSNR_attack"),
            "ssim_restored_attack": data.get("SSIM_attack"),
            "lpips_restored_attack": data.get("LPIPS_attack"),
            "ber_info_attack": data.get("BER_info_attack"),
            "ber_code_attack": data.get("BER_code_attack"),
            "decoded_ratio_attack": data.get("decoded_ratio_attack"),
            "attack": "jpeg75+gaussian0.01",
            "notes": common_notes,
        },
    ]
    detection_row = {
        "method": method,
        "status": status,
        "dataset": dataset_name,
        "statistical_detection_rate": data.get("statistical_detection_rate"),
        "srnet_detection_rate": data.get("srnet_detection_rate"),
        "srnet_balanced_accuracy": data.get("SRNet_balanced_accuracy"),
        "srnet_auc": data.get("SRNet_AUC"),
        "srnet_eer": data.get("SRNet_EER"),
        "anti_detection_rate": data.get("anti_detection_rate"),
        "detector_samples": data.get("detector_samples", data.get("num_images")),
        "official_srnet_detection_rate": data.get("official_srnet_detection_rate", float("nan")),
        "official_srnet_anti_detection_rate": data.get("official_srnet_anti_detection_rate", float("nan")),
        "official_srnet_backend": data.get("official_srnet_backend", ""),
        "notes": "srnet_detection_rate uses Comparison proxy unless official SRNet is explicitly enabled and available",
    }
    naturalness_row = build_stego_naturalness_row(data, method=method)
    goal_metrics_row = build_goal_metrics_row(data, method=method)
    write_rows_bundle(output / "secret_recovery_quality.csv", [quality_row])
    write_rows_bundle(output / "communication_reliability.csv", [comm_row])
    write_rows_bundle(output / "robustness.csv", robustness_rows)
    write_rows_bundle(output / "anti_detection.csv", [detection_row])
    write_rows_bundle(output / "stego_naturalness.csv", [naturalness_row])
    write_table_bundle(output / "goal_metrics.csv", goal_metrics_row)
    table_main_row, table_robustness_row, table_detection_row = build_comparison_paper_rows(data, method=method)
    if dataset_key == "div2k":
        write_table_bundle(output / "table1_div2k_main.csv", table_main_row)
        write_table_bundle(output / "table3_div2k_robustness.csv", table_robustness_row)
        write_table_bundle(output / "table4_div2k_official_srnet.csv", table_detection_row)
    elif dataset_key == "alaska2":
        write_table_bundle(output / "table2_alaska2_generalization.csv", table_main_row)
        write_table_bundle(output / "table3_alaska2_robustness.csv", table_robustness_row)
        write_table_bundle(output / "table4_alaska2_official_srnet.csv", table_detection_row)
        write_table_bundle(output / "table4_alaska2_antidetection.csv", table_detection_row)
    elif dataset_key == "bossbase":
        write_table_bundle(output / "table2_bossbase_generalization.csv", table_main_row)
        write_table_bundle(output / "table3_bossbase_robustness.csv", table_robustness_row)
        write_table_bundle(output / "table4_bossbase_official_srnet.csv", table_detection_row)
        write_table_bundle(output / "table4_bossbase_antidetection.csv", table_detection_row)
    else:
        write_table_bundle(output / f"table_main_{dataset_key}.csv", table_main_row)
        write_table_bundle(output / f"table_robustness_{dataset_key}.csv", table_robustness_row)
        write_table_bundle(output / f"table_detection_{dataset_key}.csv", table_detection_row)
    comparison_main_row = build_comparison_main_row(data)
    write_table_bundle(output / "comparison_table_like_image.csv", comparison_main_row)


# 最终 clean 指标必须关闭训练阶段的 robust eval 扰动；JPEG50/Mixed 指标会在评估函数中单独攻击。
def disable_eval_robust_channel(config: dict) -> None:
    robust_cfg = config.setdefault("robust_channel", {})
    robust_cfg["eval_quantize"] = False
    robust_cfg["eval_dct_jpeg"] = False
    robust_cfg["eval_noise_std"] = 0.0


# 将评估图像列表展开成单张张量列表，便于导出到 clean-fid 或传统检测器。
def flatten_image_batches(images: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
    if isinstance(images, torch.Tensor):
        return [image.detach().cpu() for image in images]
    flattened: list[torch.Tensor] = []
    for batch in images:
        if batch.dim() == 3:
            flattened.append(batch.detach().cpu())
        else:
            flattened.extend(image.detach().cpu() for image in batch)
    return flattened


def stack_flattened_images(images: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    # 把批次列表统一堆叠成 [N,C,H,W]，供 BRISQUE/FID 这类逐图指标复用。
    flattened = flatten_image_batches(images)
    if not flattened:
        return torch.empty(0, 3, 0, 0)
    return torch.stack(flattened, dim=0)


def stack_flattened_images_resized(
    images: torch.Tensor | list[torch.Tensor],
    size: int,
) -> torch.Tensor:
    flattened = flatten_image_batches(images)
    if not flattened:
        return torch.empty(0, 3, size, size)
    resized = []
    for image in flattened:
        image = image.float().clamp(0.0, 1.0)
        if image.dim() != 3:
            raise ValueError(f"Expected image tensors with shape [C,H,W], got {tuple(image.shape)}.")
        resized.append(
            F.interpolate(
                image.unsqueeze(0),
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        )
    return torch.stack(resized, dim=0)


# 按目录列表收集真实自然图像路径，供 FID/KID/BRISQUE 与检测指标使用统一参考集。
def collect_reference_image_paths(image_dirs: list[str | Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for raw_dir in image_dirs:
        resolved_dir = resolve_project_path(raw_dir)
        if resolved_dir is None or (not resolved_dir.exists()) or (not resolved_dir.is_dir()):
            continue
        for path in sorted(resolved_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".pgm", ".tif", ".tiff"}:
                continue
            canonical = path.resolve()
            if canonical in seen:
                continue
            seen.add(canonical)
            paths.append(canonical)
    return paths


# 从配置或命令行解析评估用真实自然参考集目录，默认优先复用 carrier bank，其次使用当前数据集训练/验证目录。
def resolve_natural_reference_dirs(
    config: dict,
    explicit_dirs: str | None = None,
    dataset_override: str | None = None,
) -> tuple[list[Path], str]:
    candidate_dirs: list[str | Path] = []
    dataset_label = str(dataset_override or "").strip().lower()
    if explicit_dirs:
        candidate_dirs.extend([item.strip() for item in str(explicit_dirs).split(",") if item.strip()])
    carrier_cfg = config.get("carrier_bank", {})
    if not candidate_dirs and isinstance(carrier_cfg, dict):
        raw_dirs = carrier_cfg.get("dirs") or []
        if isinstance(raw_dirs, (str, Path)):
            raw_dirs = [raw_dirs]
        candidate_dirs.extend(raw_dirs)
        if not dataset_label and carrier_cfg.get("dataset"):
            dataset_label = str(carrier_cfg.get("dataset")).strip().lower()
    if not dataset_label:
        dataset_label = str(config.get("datasets", {}).get("active", "eval")).strip().lower()
    datasets_cfg = config.get("datasets", {})
    dataset_cfg = datasets_cfg.get(dataset_label, {}) if isinstance(datasets_cfg, dict) else {}
    if not candidate_dirs and isinstance(dataset_cfg, dict):
        for field_name in ("train_dir", "val_dir", "test_dir"):
            value = dataset_cfg.get(field_name)
            if value:
                candidate_dirs.append(value)
    resolved_paths = collect_reference_image_paths(candidate_dirs)
    return resolved_paths, dataset_label or "eval"


# 将真实自然参考图像读取为张量列表，数量不足时循环补齐，保证和评估样本数一致。
def load_natural_reference_images(
    image_paths: list[Path],
    image_size: tuple[int, int] | None,
    channels: int,
    needed: int,
) -> list[torch.Tensor]:
    if not image_paths:
        return []
    mode = "RGB" if int(channels) == 3 else "L"
    target_height, target_width = image_size if image_size is not None else (0, 0)
    tensors: list[torch.Tensor] = []
    usable_paths = image_paths[:]
    repeats = max(1, math.ceil(max(1, needed) / max(1, len(usable_paths))))
    expanded_paths = (usable_paths * repeats)[: max(1, needed)]
    for path in expanded_paths:
        with Image.open(path) as image:
            image = image.convert(mode)
            if target_height > 0 and target_width > 0:
                image = image.resize((int(target_width), int(target_height)), Image.Resampling.BICUBIC)
            array = np.asarray(image, dtype=np.uint8).copy()
        if mode == "RGB":
            tensor = torch.from_numpy(array).permute(2, 0, 1).float().div(255.0)
        else:
            tensor = torch.from_numpy(array).view(array.shape[0], array.shape[1], 1).permute(2, 0, 1).float().div(255.0)
            if int(channels) == 3:
                tensor = tensor.repeat(3, 1, 1)
        tensors.append(tensor)
    return tensors


# 将 [0,1] 范围的张量图像转成 PNG，供 clean-fid 等目录型评测后端使用。
def export_images_to_directory(images: torch.Tensor | list[torch.Tensor], output_dir: Path, prefix: str) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    flattened = flatten_image_batches(images)
    for index, image in enumerate(flattened):
        clamped = image.detach().cpu().clamp(0.0, 1.0)
        if clamped.dim() != 3:
            raise ValueError(f"Expected image tensors with shape [C,H,W], got {tuple(clamped.shape)}.")
        if clamped.shape[0] == 1:
            array = (clamped.squeeze(0).numpy() * 255.0).round().astype("uint8")
            pil_image = Image.fromarray(array, mode="L")
        else:
            array = (clamped.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
            pil_image = Image.fromarray(array)
        pil_image.save(output_dir / f"{prefix}_{index:05d}.png")
    return len(flattened)


def tensor_to_grayscale_png(image: torch.Tensor) -> Image.Image:
    # 将 [0,1] 范围的 CHW 张量转换为官方 SRNet 使用的灰度 PNG。
    clamped = image.detach().cpu().clamp(0.0, 1.0)
    if clamped.dim() != 3:
        raise ValueError(f"Expected image tensors with shape [C,H,W], got {tuple(clamped.shape)}.")
    if clamped.shape[0] == 1:
        gray = clamped[0]
    else:
        gray = 0.299 * clamped[0] + 0.587 * clamped[1] + 0.114 * clamped[2]
    array = (gray.numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def export_official_srnet_pairs(
    carrier: torch.Tensor | list[torch.Tensor],
    cover: torch.Tensor | list[torch.Tensor],
    output_root: Path,
    image_size: int,
    split_seed: int = 0,
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2,
) -> Path:
    # 导出官方 SRNet 所需的 cover/stego 对，并生成 train/valid/test 子目录。
    covers = flatten_image_batches(carrier)
    stegos = flatten_image_batches(cover)
    pair_count = min(len(covers), len(stegos))
    if pair_count < 8:
        raise ValueError(f"Official SRNet export needs at least 8 pairs, got {pair_count}.")

    if output_root.exists():
        shutil.rmtree(output_root)
    cover_dir = output_root / "cover"
    stego_dir = output_root / "stego"
    cover_dir.mkdir(parents=True, exist_ok=True)
    stego_dir.mkdir(parents=True, exist_ok=True)

    for index in range(pair_count):
        name = f"pair_{index:05d}.png"
        carrier_image = tensor_to_grayscale_png(F.interpolate(covers[index].unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0))
        stego_image = tensor_to_grayscale_png(F.interpolate(stegos[index].unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0))
        carrier_image.save(cover_dir / name)
        stego_image.save(stego_dir / name)

    names = [f"pair_{index:05d}.png" for index in range(pair_count)]
    rng = np.random.default_rng(int(split_seed))
    rng.shuffle(names)
    train_count = max(1, int(round(pair_count * float(train_ratio))))
    valid_count = max(1, int(round(pair_count * float(valid_ratio))))
    test_count = max(1, pair_count - train_count - valid_count)
    if train_count + valid_count + test_count > pair_count:
        test_count = max(1, pair_count - train_count - valid_count)
    if train_count + valid_count + test_count < pair_count:
        train_count += pair_count - (train_count + valid_count + test_count)

    split_specs = [
        ("train", names[:train_count]),
        ("valid", names[train_count : train_count + valid_count]),
        ("test", names[train_count + valid_count : train_count + valid_count + test_count]),
    ]
    for split_name, split_names in split_specs:
        split_cover = output_root / f"{split_name}_cover"
        split_stego = output_root / f"{split_name}_stego"
        split_cover.mkdir(parents=True, exist_ok=True)
        split_stego.mkdir(parents=True, exist_ok=True)
        for name in split_names:
            shutil.copy2(cover_dir / name, split_cover / name)
            shutil.copy2(stego_dir / name, split_stego / name)
    return output_root


def choose_official_srnet_python(comparison_root: Path, explicit_python: str | None = None) -> str:
    # 为官方 SRNet 自动选择可用的 Python，优先使用显式传入值，其次尝试服务器上常见的 TensorFlow 环境。
    if explicit_python:
        return str(explicit_python)
    candidate_binaries = [
        Path("/root/miniconda3/envs/Comparison-SRNet37/bin/python"),
        Path("/root/miniconda3/envs/Comparison/bin/python"),
    ]
    for binary in candidate_binaries:
        if binary.exists():
            return str(binary)
    return "python"


def run_official_srnet_command(
    command: list[str],
    official_root: Path,
    env: dict[str, str],
) -> tuple[subprocess.CompletedProcess[str] | None, Exception | None]:
    # 执行一次官方 SRNet 命令，统一封装 stdout/stderr 与异常返回。
    try:
        completed = subprocess.run(
            command,
            cwd=str(official_root),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed, None
    except Exception as error:
        return None, error


def run_official_srnet_evaluation(
    carrier: torch.Tensor | list[torch.Tensor],
    cover: torch.Tensor | list[torch.Tensor],
    output_root: Path,
    comparison_root: Path,
    image_size: int,
    device: torch.device,
    split_seed: int = 0,
    python_executable: str | None = None,
    max_iter: int = 2000,
    force_cpu: bool = False,
) -> dict[str, float | str]:
    # 调用 Comparison 目录中的官方 SRNet 脚本，对当前 cover/stego 结果做对齐评估。
    official_root = comparison_root / "official_srnet"
    runner = official_root / "run_official_srnet.py"
    if not runner.exists():
        return {
            "detection_rate": float("nan"),
            "anti_detection_rate": float("nan"),
            "test_loss": float("nan"),
            "backend": "official_srnet_unavailable",
        }

    pairs_root = export_official_srnet_pairs(
        carrier=carrier,
        cover=cover,
        output_root=output_root / "official_srnet_pairs",
        image_size=image_size,
        split_seed=split_seed,
    )
    log_dir = output_root / "official_srnet_logs"
    result_json = output_root / "official_srnet_result.json"
    exe = choose_official_srnet_python(comparison_root, python_executable)
    train_pair_count = len(list((pairs_root / "train_cover").glob("*.png")))
    valid_pair_count = len(list((pairs_root / "valid_cover").glob("*.png")))
    test_pair_count = len(list((pairs_root / "test_cover").glob("*.png")))
    train_batch_size = max(1, min(32, train_pair_count))
    valid_batch_size = max(1, min(40, valid_pair_count))
    test_batch_size = max(1, min(50, test_pair_count))
    effective_max_iter = max(1, int(max_iter))
    train_interval = max(1, min(100, effective_max_iter))
    valid_interval = max(1, min(500, effective_max_iter))
    save_interval = max(1, min(500, effective_max_iter))
    command = [
        exe,
        str(runner),
        "--pairs-root",
        str(pairs_root),
        "--log-dir",
        str(log_dir),
        "--mode",
        "train-test",
        "--train-batch-size",
        str(train_batch_size),
        "--valid-batch-size",
        str(valid_batch_size),
        "--test-batch-size",
        str(test_batch_size),
        "--max-iter",
        str(effective_max_iter),
        "--train-interval",
        str(train_interval),
        "--valid-interval",
        str(valid_interval),
        "--save-interval",
        str(save_interval),
        "--output-json",
        str(result_json),
    ]
    env = os.environ.copy()
    if force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    elif device.type == "cuda" and device.index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device.index)
    completed, error = run_official_srnet_command(command, official_root, env)
    if error is not None and device.type == "cuda" and not force_cpu:
        cpu_env = env.copy()
        cpu_env["CUDA_VISIBLE_DEVICES"] = ""
        cpu_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        print(
            "Warning: official SRNet GPU path failed, retrying on CPU. "
            "python="
            f"{exe} train/valid/test_batch={train_batch_size}/{valid_batch_size}/{test_batch_size} "
            f"intervals={train_interval}/{valid_interval}/{save_interval}. {error}"
        )
        completed, error = run_official_srnet_command(command, official_root, cpu_env)
    try:
        if error is not None:
            raise error
        if completed.stdout:
            print(completed.stdout)
        if completed.stderr:
            print(completed.stderr)
    except Exception as error:
        print(
            "Warning: official SRNet evaluation failed, "
            "python="
            f"{exe} train/valid/test_batch={train_batch_size}/{valid_batch_size}/{test_batch_size} "
            f"intervals={train_interval}/{valid_interval}/{save_interval}. {error}"
        )
        print(f"Warning: official SRNet evaluation failed, falling back to proxy result. {error}")
        return {
            "detection_rate": float("nan"),
            "anti_detection_rate": float("nan"),
            "test_loss": float("nan"),
            "backend": "official_srnet_failed",
        }
    if not result_json.exists():
        return {
            "detection_rate": float("nan"),
            "anti_detection_rate": float("nan"),
            "test_loss": float("nan"),
            "backend": "official_srnet_missing_result",
        }
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    detection = float(payload.get("test_accuracy", float("nan")))
    return {
        "detection_rate": detection,
        "anti_detection_rate": float(1.0 - detection) if math.isfinite(detection) else float("nan"),
        "test_loss": float(payload.get("test_loss", float("nan"))),
        "backend": "official_srnet",
    }


# 按需加载 LPIPS，避免训练脚本只是导入评估模块时就因为缺少依赖直接退出。
def build_lpips_model(device: torch.device):
    try:
        import lpips
    except ModuleNotFoundError:
        print("Warning: lpips is not installed, LPIPS metric will be recorded as NaN.")
        return None
    return lpips.LPIPS(net="alex").to(device).eval()


def brisque_proxy_score(images: torch.Tensor) -> torch.Tensor:
    # 用亮度域自然统计代理近似 BRISQUE 趋势，保证离线服务器缺少依赖时也能稳定输出自然度 gap。
    luma = rgb_to_luma_y(images).clamp(0.0, 1.0)
    mean = F.avg_pool2d(luma, kernel_size=7, stride=1, padding=3)
    centered = luma - mean
    variance = F.avg_pool2d(centered.pow(2), kernel_size=7, stride=1, padding=3)
    std = variance.sqrt().clamp_min(1e-4)
    normalized = centered / std
    skew = normalized.pow(3).mean(dim=(-2, -1)).abs()
    kurt = (normalized.pow(4).mean(dim=(-2, -1)) - 3.0).abs()
    hf_h = (luma[:, :, :, 1:] - luma[:, :, :, :-1]).abs().mean(dim=(-2, -1))
    hf_v = (luma[:, :, 1:, :] - luma[:, :, :-1, :]).abs().mean(dim=(-2, -1))
    low_contrast = std.mean(dim=(-2, -1))
    score = 35.0 * skew + 22.0 * kurt + 18.0 * (hf_h + hf_v) + 12.0 / low_contrast.clamp_min(1e-3)
    return score.squeeze(1) if score.dim() > 1 else score


def try_library_brisque_score(images: torch.Tensor) -> tuple[torch.Tensor | None, str | None]:
    # 优先走标准 BRISQUE 实现；不可用时返回 None，由代理分数兜底。
    try:
        import imquality.brisque as imq_brisque  # type: ignore
    except Exception:
        imq_brisque = None
    if imq_brisque is not None:
        try:
            import inspect
            import numpy as np
            import scipy  # type: ignore
            import skimage.color as sk_color  # type: ignore
            import skimage.transform as sk_transform  # type: ignore

            if not hasattr(scipy, "ndarray"):
                scipy.ndarray = np.ndarray  # type: ignore[attr-defined]

            original_rescale = getattr(sk_transform, "rescale", None)
            if callable(original_rescale):
                signature = inspect.signature(original_rescale)
                if "multichannel" not in signature.parameters:
                    def _compat_rescale(image, scale, *args, multichannel=None, **kwargs):
                        if multichannel is not None and "channel_axis" not in kwargs:
                            kwargs["channel_axis"] = -1 if multichannel else None
                        return original_rescale(image, scale, *args, **kwargs)

                    sk_transform.rescale = _compat_rescale

            original_rgb2gray = getattr(sk_color, "rgb2gray", None)
            if callable(original_rgb2gray):
                def _compat_rgb2gray(image, *args, **kwargs):
                    if getattr(image, "ndim", 0) == 2:
                        return image
                    return original_rgb2gray(image, *args, **kwargs)

                sk_color.rgb2gray = _compat_rgb2gray
        except Exception:
            pass
        scores = []
        for image in images.detach().cpu():
            array = (image.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
            pil = Image.fromarray(array)
            try:
                scores.append(float(imq_brisque.score(pil)))
            except Exception:
                scores = []
                break
        if scores:
            return torch.tensor(scores, dtype=torch.float32), "imquality.brisque"

    try:
        import brisque as brisque_module  # type: ignore
    except Exception:
        brisque_module = None
    if brisque_module is not None and hasattr(brisque_module, "BRISQUE"):
        scorer = brisque_module.BRISQUE()
        scores = []
        for image in images.detach().cpu():
            array = (image.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
            scores.append(float(scorer.score(array)))
        return torch.tensor(scores, dtype=torch.float32), "brisque"
    return None, None


def calculate_brisque_gap_metrics(
    original: torch.Tensor | list[torch.Tensor],
    cover: torch.Tensor | list[torch.Tensor],
) -> tuple[float, float, float, str]:
    # 计算真实集与载密图的 BRISQUE 均值和 gap，优先标准实现，缺失依赖时退回代理分数。
    original_tensor = stack_flattened_images_resized(original, 256) if isinstance(original, list) else original
    cover_tensor = stack_flattened_images_resized(cover, 256) if isinstance(cover, list) else cover
    if original_tensor.dim() == 3:
        original_tensor = original_tensor.unsqueeze(0)
    if cover_tensor.dim() == 3:
        cover_tensor = cover_tensor.unsqueeze(0)

    real_scores, backend = try_library_brisque_score(original_tensor)
    stego_scores, stego_backend = try_library_brisque_score(cover_tensor)
    if real_scores is None or stego_scores is None:
        real_scores = brisque_proxy_score(original_tensor)
        stego_scores = brisque_proxy_score(cover_tensor)
        backend = "proxy-nss"
    else:
        backend = backend or stego_backend or "brisque"
    real_mean = float(real_scores.mean().item())
    stego_mean = float(stego_scores.mean().item())
    return real_mean, stego_mean, abs(stego_mean - real_mean), backend


class InceptionFeatureExtractor(nn.Module):
    # 初始化 InceptionV3 特征提取器，用于 FID 计算。
    def __init__(self) -> None:
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights, transform_input=False, aux_logits=True)
        model.fc = nn.Identity()
        model.eval()
        self.model = model

    # 提取输入图像的 2048 维 Inception 特征。
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        return self.model(resized)


# 按需构建 FID 特征提取器，避免离线环境缺少权重时整次评估中断。
def build_inception_extractor(device: torch.device) -> tuple[nn.Module | None, str]:
    try:
        return InceptionFeatureExtractor().to(device), "legacy-inception-pretrained"
    except Exception as error:
        print(
            "Warning: failed to initialize pretrained Inception feature extractor, "
            f"retrying with an uninitialized fallback. {error}"
        )
        try:
            fallback = inception_v3(weights=None, transform_input=False, aux_logits=True)
            fallback.fc = nn.Identity()
            fallback.eval()
            return fallback.to(device), "legacy-inception-random"
        except Exception as fallback_error:
            print(
                "Warning: failed to initialize fallback Inception feature extractor, "
                f"FID/KID will be recorded as NaN. {fallback_error}"
            )
            return None, "legacy-inception-unavailable"


class SRNetConvBlock(nn.Module):
    # 构建轻量 SRNet 风格卷积块，兼容快速评测与旧权重。
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
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
        self.act = nn.ReLU(inplace=True)

    # 执行轻量残差卷积特征提取。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.shortcut(x))


class SRNetDownsampleBlock(nn.Module):
    # 构建带下采样的更深层 SRNet 风格残差块，更接近论文常见检测器深度。
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.ReLU(inplace=True)

    # 执行带步长的深层残差特征提取。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.shortcut(x))


class SRNetLiteSteganalyzer(nn.Module):
    # 轻量 SRNet 风格隐写分析网络，适合快速冒烟与无预训练权重场景。
    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.high_pass = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        kernel = torch.tensor([[-1.0, 2.0, -1.0], [2.0, -4.0, 2.0], [-1.0, 2.0, -1.0]])
        with torch.no_grad():
            self.high_pass.weight.copy_(kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.high_pass.weight.requires_grad_(False)
        self.features = nn.Sequential(
            SRNetConvBlock(channels, 32),
            SRNetConvBlock(32, 64, stride=2),
            SRNetConvBlock(64, 128, stride=2),
            SRNetConvBlock(128, 256, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(256, 2)

    # 输出输入图像被判定为 cover 或 stego 的 logits。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.high_pass(x)
        return self.classifier(self.features(residual))


class SRNetSteganalyzer(nn.Module):
    # 更深的 SRNet 风格隐写分析网络，默认作为论文式评测的深度学习检测器。
    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.high_pass = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        kernel = torch.tensor([[-1.0, 2.0, -1.0], [2.0, -4.0, 2.0], [-1.0, 2.0, -1.0]])
        with torch.no_grad():
            self.high_pass.weight.copy_(kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.high_pass.weight.requires_grad_(False)
        self.features = nn.Sequential(
            SRNetConvBlock(channels, 64),
            SRNetConvBlock(64, 64),
            SRNetDownsampleBlock(64, 128, stride=2),
            SRNetConvBlock(128, 128),
            SRNetDownsampleBlock(128, 256, stride=2),
            SRNetConvBlock(256, 256),
            SRNetDownsampleBlock(256, 512, stride=2),
            SRNetConvBlock(512, 512),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(512, 2)

    # 输出输入图像被判定为 cover 或 stego 的 logits。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.high_pass(x)
        return self.classifier(self.features(residual))


# 根据命令行指定的架构创建 SRNet 风格检测器。
def build_srnet_steganalyzer(channels: int, arch: str) -> nn.Module:
    normalized_arch = str(arch or "deep").strip().lower()
    if normalized_arch == "lite":
        return SRNetLiteSteganalyzer(channels=channels)
    return SRNetSteganalyzer(channels=channels)


class StegoPairDataset(Dataset):
    # 把 cover 和 stego 图像拼成二分类数据集。
    def __init__(self, cover: torch.Tensor, stego: torch.Tensor) -> None:
        self.images = torch.cat([cover, stego], dim=0)
        self.labels = torch.cat(
            [
                torch.zeros(cover.shape[0], dtype=torch.long),
                torch.ones(stego.shape[0], dtype=torch.long),
            ],
            dim=0,
        )

    # 返回二分类样本数量。
    def __len__(self) -> int:
        return self.images.shape[0]

    # 读取单个 cover/stego 样本和标签。
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


class StegoPairListDataset(Dataset):
    # 把不同分辨率的 cover 和 stego 图像拼成二分类数据集。
    def __init__(self, cover: list[torch.Tensor], stego: list[torch.Tensor]) -> None:
        self.images = cover + stego
        self.labels = [0] * len(cover) + [1] * len(stego)

    # 返回二分类样本数量。
    def __len__(self) -> int:
        return len(self.images)

    # 读取单个 cover/stego 样本和标签。
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], torch.tensor(self.labels[index], dtype=torch.long)


# 按列表返回不同分辨率图像，避免评估期 DataLoader 强行堆叠失败。
def stego_pair_collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[list[torch.Tensor], torch.Tensor]:
    images = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch], dim=0)
    return images, labels


# 构建可复现的分层 train/test 划分，避免检测器只看到单边类别分布。
def build_stratified_subsets(
    dataset: Dataset,
    labels: list[int] | torch.Tensor,
    test_ratio: float,
    seed: int,
) -> tuple[Subset, Subset]:
    if isinstance(labels, torch.Tensor):
        label_array = labels.detach().cpu().numpy().astype(np.int64)
    else:
        label_array = np.asarray(labels, dtype=np.int64)
    unique_labels = np.unique(label_array)
    if unique_labels.size < 2:
        indices = np.arange(len(label_array), dtype=np.int64)
        return Subset(dataset, indices.tolist()), Subset(dataset, indices.tolist())

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for label in unique_labels:
        class_indices = np.where(label_array == label)[0]
        rng.shuffle(class_indices)
        class_test = int(round(class_indices.size * float(test_ratio)))
        class_test = max(1, min(class_indices.size - 1, class_test))
        test_indices.extend(class_indices[:class_test].tolist())
        train_indices.extend(class_indices[class_test:].tolist())

    if not train_indices:
        train_indices = test_indices[:]
    if not test_indices:
        test_indices = train_indices[:]
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return Subset(dataset, train_indices), Subset(dataset, test_indices)


def build_stratified_train_valid_test_subsets(
    dataset: Dataset,
    labels: list[int] | torch.Tensor,
    test_ratio: float,
    valid_ratio: float,
    seed: int,
) -> tuple[Subset, Subset, Subset]:
    if isinstance(labels, torch.Tensor):
        label_array = labels.detach().cpu().numpy().astype(np.int64)
    else:
        label_array = np.asarray(labels, dtype=np.int64)
    unique_labels = np.unique(label_array)
    if unique_labels.size < 2:
        indices = np.arange(len(label_array), dtype=np.int64).tolist()
        subset = Subset(dataset, indices)
        return subset, subset, subset

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    valid_indices: list[int] = []
    test_indices: list[int] = []
    safe_test_ratio = float(max(0.05, min(0.45, test_ratio)))
    safe_valid_ratio = float(max(0.05, min(0.40, valid_ratio)))
    for label in unique_labels:
        class_indices = np.where(label_array == label)[0]
        rng.shuffle(class_indices)
        class_count = int(class_indices.size)
        if class_count < 3:
            train_indices.extend(class_indices.tolist())
            valid_indices.extend(class_indices.tolist())
            test_indices.extend(class_indices.tolist())
            continue
        class_test = int(round(class_count * safe_test_ratio))
        class_test = max(1, min(class_count - 2, class_test))
        remaining = class_count - class_test
        class_valid = int(round(class_count * safe_valid_ratio))
        class_valid = max(1, min(remaining - 1, class_valid))
        test_indices.extend(class_indices[:class_test].tolist())
        valid_indices.extend(class_indices[class_test : class_test + class_valid].tolist())
        train_indices.extend(class_indices[class_test + class_valid :].tolist())

    if not train_indices:
        train_indices = valid_indices[:] if valid_indices else test_indices[:]
    if not valid_indices:
        valid_indices = train_indices[:]
    if not test_indices:
        test_indices = valid_indices[:]
    rng.shuffle(train_indices)
    rng.shuffle(valid_indices)
    rng.shuffle(test_indices)
    return (
        Subset(dataset, train_indices),
        Subset(dataset, valid_indices),
        Subset(dataset, test_indices),
    )


# 计算一批图像的 PSNR 均值。
def batch_psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(x, y, reduction="none").flatten(start_dim=1).mean(dim=1).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


# 使用局部窗口版 SSIM 计算一批图像的结构相似度。
def batch_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    c1 = 0.01**2
    c2 = 0.03**2
    window_size = 11
    padding = window_size // 2
    mu_x = F.avg_pool2d(x, kernel_size=window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, kernel_size=window_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(x * x, kernel_size=window_size, stride=1, padding=padding) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(y * y, kernel_size=window_size, stride=1, padding=padding) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(x * y, kernel_size=window_size, stride=1, padding=padding) - mu_x * mu_y
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / denominator.clamp_min(1e-12)
    return ssim_map.flatten(start_dim=1).mean(dim=1)


# 将 RGB 图像转换成论文评估常用的 BT.601 Y 亮度通道。
def rgb_to_luma_y(images: torch.Tensor) -> torch.Tensor:
    if images.shape[1] == 1:
        return images
    if images.shape[1] < 3:
        return images.mean(dim=1, keepdim=True)
    return 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]


# 计算 Y 通道 SSIM，和许多图像隐藏/超分论文表格口径一致。
def batch_ssim_y(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return batch_ssim(rgb_to_luma_y(x), rgb_to_luma_y(y))


# 计算恢复图相对原图的逐图 MAE。
def batch_mae(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).abs().flatten(start_dim=1).mean(dim=1)


# 计算恢复图相对原图的逐图 RMSE。
def batch_rmse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(x, y, reduction="none").flatten(start_dim=1).mean(dim=1).clamp_min(1e-12)
    return torch.sqrt(mse)


# 根据原始比特和恢复比特计算误码率。
def bit_error_rate(
    reference: torch.Tensor,
    recovered: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    recovered_bits = (recovered >= 0.5).to(reference.dtype)
    errors = (reference != recovered_bits).float()
    if mask is None:
        return errors.flatten(start_dim=1).mean(dim=1)
    typed_mask = mask.to(device=errors.device, dtype=errors.dtype)
    while typed_mask.dim() < errors.dim():
        typed_mask = typed_mask.unsqueeze(0)
    typed_mask = typed_mask.expand_as(errors)
    return (errors * typed_mask).flatten(start_dim=1).sum(dim=1) / typed_mask.flatten(start_dim=1).sum(dim=1).clamp_min(1.0)


# 根据有效信息位长度和图像像素数计算 bpp。
def payload_bpp(info_length: int, height: int, width: int, channels: int | None = None) -> float:
    return info_length / float(height * width)


# 计算图像差分残差的统计特征，用于传统统计隐写分析。
def legacy_statistical_features(images: torch.Tensor) -> torch.Tensor:
    gray = images.mean(dim=1, keepdim=True)
    dx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    dy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    lap = F.conv2d(
        gray,
        torch.tensor(
            [[[-1.0, 2.0, -1.0], [2.0, -4.0, 2.0], [-1.0, 2.0, -1.0]]],
            device=images.device,
        ).unsqueeze(0),
        padding=1,
    )
    feats = []
    for residual in (dx, dy, lap):
        flat = residual.flatten(start_dim=1)
        feats.extend(
            [
                flat.mean(dim=1),
                flat.std(dim=1),
                flat.abs().mean(dim=1),
                flat.pow(2).mean(dim=1),
            ]
        )
    return torch.stack(feats, dim=1)


def srm_like_kernels(
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    kernels = [
        [[0.0, 0.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]],
        [[-1.0, 2.0, -1.0], [2.0, -4.0, 2.0], [-1.0, 2.0, -1.0]],
    ]
    return [torch.tensor(kernel, device=device, dtype=dtype).view(1, 1, 3, 3) for kernel in kernels]


def srm_like_statistical_features(
    images: torch.Tensor,
    quant_step: float = 2.0,
    threshold: int = 4,
) -> torch.Tensor:
    if images.dim() != 4:
        raise ValueError(f"srm_like_statistical_features expects [B,C,H,W], got {tuple(images.shape)}.")
    gray = rgb_to_luma_y(images.detach().cpu()) * 255.0
    kernels = srm_like_kernels(gray.device, gray.dtype)
    bins = int(2 * threshold + 1)
    feature_groups: list[torch.Tensor] = []
    for kernel in kernels:
        residual = F.conv2d(gray, kernel, padding=1)
        quantized = torch.round(residual / float(quant_step)).clamp(-threshold, threshold).to(torch.int64)
        kernel_features: list[torch.Tensor] = []
        for batch_index in range(quantized.shape[0]):
            q = quantized[batch_index, 0]
            flat = (q + threshold).reshape(-1)
            hist = torch.bincount(flat, minlength=bins).float()
            hist = hist / hist.sum().clamp_min(1.0)
            horizontal_pairs = ((q[:, 1:] + threshold) * bins + (q[:, :-1] + threshold)).reshape(-1)
            horizontal_hist = torch.bincount(horizontal_pairs, minlength=bins * bins).float()
            horizontal_hist = horizontal_hist / horizontal_hist.sum().clamp_min(1.0)
            vertical_pairs = ((q[1:, :] + threshold) * bins + (q[:-1, :] + threshold)).reshape(-1)
            vertical_hist = torch.bincount(vertical_pairs, minlength=bins * bins).float()
            vertical_hist = vertical_hist / vertical_hist.sum().clamp_min(1.0)
            kernel_features.append(torch.cat([hist, horizontal_hist, vertical_hist], dim=0))
        feature_groups.append(torch.stack(kernel_features, dim=0))
    return torch.cat(feature_groups, dim=1)


def comparison_detector_prepare(
    images: torch.Tensor | list[torch.Tensor],
    size: int = 64,
) -> torch.Tensor:
    # 将 cover/stego 统一缩放到 Comparison 使用的检测尺寸，避免评估口径不一致。
    return stack_flattened_images_resized(images, size)


def comparison_highpass_feature(images: torch.Tensor) -> torch.Tensor:
    # 复用 Comparison 的高通残差统计特征作为传统检测器输入。
    gray = images.mean(dim=1, keepdim=True)
    lap = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        dtype=images.dtype,
        device=images.device,
    ).view(1, 1, 3, 3)
    residual = F.conv2d(gray, lap, padding=1).abs()
    flat = residual.flatten(start_dim=1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True)
    q90 = flat.quantile(0.9, dim=1, keepdim=True)
    return torch.cat([mean, std, q90], dim=1)


class ComparisonTinyResidualStegalyzer(nn.Module):
    # 使用 Comparison 的轻量残差 CNN 代理 SRNet 检测器。
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        residual = images - F.avg_pool2d(images, 3, stride=1, padding=1)
        return self.net(residual)


def comparison_statistical_detection_metrics(
    cover: torch.Tensor | list[torch.Tensor],
    stego: torch.Tensor | list[torch.Tensor],
    size: int = 64,
) -> dict[str, float]:
    # 完整复刻 Comparison 的高通阈值检测逻辑。
    covers = comparison_detector_prepare(cover, size=size)
    stegos = comparison_detector_prepare(stego, size=size)
    if covers.numel() == 0 or stegos.numel() == 0:
        return {
            "detection_rate": 0.0,
            "overall_accuracy": 0.0,
            "stego_detection_rate": 0.0,
            "false_positive_rate": 1.0,
            "backend": "comparison_highpass_threshold_empty",
        }
    cover_score = comparison_highpass_feature(covers).mean(dim=1)
    stego_score = comparison_highpass_feature(stegos).mean(dim=1)
    threshold = (cover_score.mean() + stego_score.mean()) / 2.0
    if stego_score.mean() >= cover_score.mean():
        cover_correct = cover_score < threshold
        stego_correct = stego_score >= threshold
    else:
        cover_correct = cover_score >= threshold
        stego_correct = stego_score < threshold
    overall = torch.cat([cover_correct, stego_correct]).float().mean().item()
    stego_detection = stego_correct.float().mean().item()
    false_positive = 1.0 - cover_correct.float().mean().item()
    return {
        "detection_rate": float(overall),
        "overall_accuracy": float(overall),
        "stego_detection_rate": float(stego_detection),
        "false_positive_rate": float(false_positive),
        "backend": "comparison_highpass_threshold",
    }


def comparison_srnet_proxy_detection_metrics(
    cover: torch.Tensor | list[torch.Tensor],
    stego: torch.Tensor | list[torch.Tensor],
    device: torch.device,
    epochs: int = 3,
    size: int = 64,
    seed: int = 42,
) -> dict[str, float]:
    # 复刻 Comparison 的轻量残差 CNN 代理检测器，而不是当前项目原来的 SRNet 风格训练流程。
    covers = comparison_detector_prepare(cover, size=size)
    stegos = comparison_detector_prepare(stego, size=size)
    count = min(int(covers.shape[0]), int(stegos.shape[0]))
    if count < 8:
        metrics = comparison_statistical_detection_metrics(covers, stegos, size=size)
        metrics["backend"] = "comparison_srnet_proxy_fallback_to_stat"
        return metrics
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    x = torch.cat([covers[:count], stegos[:count]], dim=0)
    y = torch.cat(
        [
            torch.zeros(count, dtype=torch.long),
            torch.ones(count, dtype=torch.long),
        ],
        dim=0,
    )
    pair_perm = torch.randperm(count)
    train_count = max(4, int(count * 0.75))
    train_pairs = pair_perm[:train_count]
    test_pairs = pair_perm[train_count:]
    if int(test_pairs.numel()) < 2:
        test_pairs = pair_perm[-2:]
        train_pairs = pair_perm[:-2]
    train_idx = torch.cat([train_pairs, train_pairs + count])
    test_idx = torch.cat([test_pairs, test_pairs + count])
    model = ComparisonTinyResidualStegalyzer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batch_size = 32
    x_train = x[train_idx].to(device)
    y_train = y[train_idx].to(device)
    for _ in range(max(1, int(epochs))):
        perm = torch.randperm(int(x_train.shape[0]), device=device)
        for start in range(0, int(perm.numel()), batch_size):
            idx = perm[start : start + batch_size]
            logits = model(x_train[idx])
            loss = F.cross_entropy(logits, y_train[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        logits = model(x[test_idx].to(device))
        scores = torch.softmax(logits, dim=1)[:, 1].cpu()
        predictions = logits.argmax(dim=1).cpu()
        labels = y[test_idx].cpu()
    metrics = binary_detection_metrics(predictions, labels)
    metrics.update(score_detection_metrics(scores, labels))
    metrics["backend"] = f"comparison_tiny_residual_cnn_{int(epochs)}e"
    return metrics


# 根据二分类预测计算总体检测率、stego 检出率和 cover 误报率。
def binary_detection_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    typed_predictions = predictions.detach().bool()
    typed_labels = labels.detach().bool()
    cover_mask = ~typed_labels
    stego_mask = typed_labels
    overall_accuracy = (typed_predictions == typed_labels).float().mean().item()
    stego_detection = (
        (typed_predictions[stego_mask] == 1).float().mean().item()
        if stego_mask.any()
        else float("nan")
    )
    false_positive = (
        (typed_predictions[cover_mask] == 1).float().mean().item()
        if cover_mask.any()
        else float("nan")
    )
    true_negative = 1.0 - false_positive if math.isfinite(false_positive) else float("nan")
    if math.isfinite(stego_detection) and math.isfinite(true_negative):
        balanced_detection = 0.5 * (stego_detection + true_negative)
    else:
        balanced_detection = overall_accuracy
    return {
        "detection_rate": float(balanced_detection),
        "overall_accuracy": float(overall_accuracy),
        "stego_detection_rate": float(stego_detection),
        "false_positive_rate": float(false_positive),
    }


# 根据连续检测分数计算 AUC、EER 和平衡准确率，便于和隐写分析论文表格对齐。
def score_detection_metrics(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    scores_np = scores.detach().float().cpu().numpy().reshape(-1)
    labels_np = labels.detach().long().cpu().numpy().reshape(-1)
    if scores_np.size < 4 or np.unique(labels_np).size < 2:
        return {
            "auc": 0.5,
            "eer": 0.5,
            "balanced_accuracy": 0.5,
        }
    try:
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve

        auc = float(roc_auc_score(labels_np, scores_np))
        fpr, tpr, _ = roc_curve(labels_np, scores_np)
        fnr = 1.0 - tpr
        eer_index = int(np.nanargmin(np.abs(fpr - fnr)))
        eer = float(0.5 * (fpr[eer_index] + fnr[eer_index]))
        predictions = (scores_np >= 0.5).astype(np.int64)
        balanced_accuracy = float(balanced_accuracy_score(labels_np, predictions))
    except Exception:
        order = np.argsort(scores_np)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(scores_np.size, dtype=np.float64) + 1.0
        positive = labels_np == 1
        negative = labels_np == 0
        pos_count = max(1, int(positive.sum()))
        neg_count = max(1, int(negative.sum()))
        auc = float((ranks[positive].sum() - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count))
        thresholds = np.unique(scores_np)
        best_gap = float("inf")
        eer = float("nan")
        for threshold in thresholds:
            predicted = scores_np >= threshold
            fp = float(np.logical_and(predicted, negative).sum()) / neg_count
            fn = float(np.logical_and(~predicted, positive).sum()) / pos_count
            gap = abs(fp - fn)
            if gap < best_gap:
                best_gap = gap
                eer = 0.5 * (fp + fn)
        predictions = scores_np >= 0.5
        tpr = float(np.logical_and(predictions, positive).sum()) / pos_count
        tnr = float(np.logical_and(~predictions, negative).sum()) / neg_count
        balanced_accuracy = 0.5 * (tpr + tnr)
    if not math.isfinite(auc):
        auc = 0.5
    if not math.isfinite(eer):
        eer = 0.5
    if not math.isfinite(balanced_accuracy):
        balanced_accuracy = 0.5
    return {
        "auc": auc,
        "eer": eer,
        "balanced_accuracy": balanced_accuracy,
    }


# 训练一个轻量统计检测器并返回平衡检测指标。
def legacy_statistical_detection_metrics(
    cover: torch.Tensor | list[torch.Tensor],
    stego: torch.Tensor | list[torch.Tensor],
    seed: int = 42,
) -> dict[str, float]:
    if isinstance(cover, list):
        cover_features = torch.cat([legacy_statistical_features(image) for image in cover], dim=0)
        stego_features = torch.cat([legacy_statistical_features(image) for image in stego], dim=0)
    else:
        cover_features = legacy_statistical_features(cover)
        stego_features = legacy_statistical_features(stego)
    features = torch.cat([cover_features, stego_features], dim=0)
    labels = torch.cat([torch.zeros(cover_features.shape[0]), torch.ones(stego_features.shape[0])], dim=0).to(features.device)
    train_size = max(2, int(0.7 * features.shape[0]))
    generator = torch.Generator(device=features.device)
    generator.manual_seed(seed)
    indices = torch.randperm(features.shape[0], device=features.device, generator=generator)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    weights = torch.zeros(features.shape[1], 1, device=features.device, requires_grad=True)
    bias = torch.zeros(1, device=features.device, requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=0.05)
    for _ in range(200):
        logits = features[train_idx] @ weights + bias
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), labels[train_idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if test_idx.numel() == 0:
        test_idx = train_idx
    with torch.no_grad():
        predictions = torch.sigmoid(features[test_idx] @ weights + bias).squeeze(1) >= 0.5
        test_labels = labels[test_idx] >= 0.5
        metrics = binary_detection_metrics(predictions, test_labels)
        metrics["backend"] = "legacy_linear_probe"
        return metrics


def srm_like_detection_metrics(
    cover: torch.Tensor | list[torch.Tensor],
    stego: torch.Tensor | list[torch.Tensor],
    seed: int = 42,
    test_ratio: float = 0.3,
) -> dict[str, float]:
    try:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
    except ModuleNotFoundError:
        print("Warning: scikit-learn is not installed, falling back to the legacy statistical detector.")
        return legacy_statistical_detection_metrics(cover, stego, seed=seed)

    if isinstance(cover, list):
        cover_features = torch.cat([srm_like_statistical_features(image) for image in cover], dim=0)
        stego_features = torch.cat([srm_like_statistical_features(image) for image in stego], dim=0)
    else:
        cover_features = srm_like_statistical_features(cover)
        stego_features = srm_like_statistical_features(stego)
    features = torch.cat([cover_features, stego_features], dim=0).cpu().numpy()
    labels = np.concatenate(
        [
            np.zeros(cover_features.shape[0], dtype=np.int64),
            np.ones(stego_features.shape[0], dtype=np.int64),
        ],
        axis=0,
    )
    if features.shape[0] < 4 or np.unique(labels).size < 2:
        return legacy_statistical_detection_metrics(cover, stego, seed=seed)

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for label in np.unique(labels):
        class_indices = np.where(labels == label)[0]
        rng.shuffle(class_indices)
        class_test = int(round(class_indices.size * float(test_ratio)))
        class_test = max(1, min(class_indices.size - 1, class_test))
        test_indices.extend(class_indices[:class_test].tolist())
        train_indices.extend(class_indices[class_test:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    classifier = make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=8000, random_state=seed),
    )
    classifier.fit(features[train_indices], labels[train_indices])
    scores = classifier.decision_function(features[test_indices])
    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    normalized_scores = (scores - score_min) / max(score_max - score_min, 1e-8)
    predictions = classifier.predict(features[test_indices])
    metrics = binary_detection_metrics(
        torch.tensor(predictions, dtype=torch.long),
        torch.tensor(labels[test_indices], dtype=torch.long),
    )
    metrics.update(
        score_detection_metrics(
            torch.tensor(normalized_scores, dtype=torch.float32),
            torch.tensor(labels[test_indices], dtype=torch.long),
        )
    )
    metrics["backend"] = "srm_like_linear_svm"
    return metrics


def statistical_detection_metrics(
    cover: torch.Tensor | list[torch.Tensor],
    stego: torch.Tensor | list[torch.Tensor],
    seed: int = 42,
    backend: str = "auto",
    test_ratio: float = 0.3,
) -> dict[str, float]:
    normalized_backend = str(backend or "auto").strip().lower()
    if normalized_backend in {"auto", "comparison", "comparison_highpass"}:
        return comparison_statistical_detection_metrics(cover, stego)
    if normalized_backend in {"auto", "srm", "srm_like"}:
        metrics = srm_like_detection_metrics(cover, stego, seed=seed, test_ratio=test_ratio)
        if normalized_backend == "auto" or metrics.get("backend") == "srm_like_linear_svm":
            return metrics
    return legacy_statistical_detection_metrics(cover, stego, seed=seed)


# 兼容旧调用：返回统计检测器的平衡检测率。
def statistical_detection_rate(
    cover: torch.Tensor | list[torch.Tensor],
    stego: torch.Tensor | list[torch.Tensor],
    seed: int = 42,
    backend: str = "auto",
    test_ratio: float = 0.3,
) -> float:
    return statistical_detection_metrics(
        cover,
        stego,
        seed=seed,
        backend=backend,
        test_ratio=test_ratio,
    )["detection_rate"]


# 训练或加载 SRNet 风格检测器，并返回平衡检测指标。
def srnet_detection_metrics(
    cover: torch.Tensor | list[torch.Tensor],
    stego: torch.Tensor | list[torch.Tensor],
    device: torch.device,
    weights_path: str | None = None,
    epochs: int = 8,
    batch_size: int = 8,
    seed: int = 42,
    arch: str = "deep",
    lr: float = 1e-4,
    test_ratio: float = 0.3,
    save_weights_path: str | None = None,
) -> dict[str, float]:
    if int(epochs) <= 0 and weights_path is None:
        metrics = comparison_srnet_proxy_detection_metrics(
            cover,
            stego,
            device=device,
            epochs=1,
            size=64,
            seed=seed,
        )
        metrics["backend"] = f"{metrics.get('backend', 'comparison_tiny_residual_cnn_1e')}_auto_fallback"
        return metrics
    normalized_arch = str(arch or "deep").strip().lower()
    if weights_path is None and normalized_arch in {"comparison", "proxy"}:
        return comparison_srnet_proxy_detection_metrics(
            cover,
            stego,
            device=device,
            epochs=epochs,
            size=64,
            seed=seed,
        )
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    channels = cover[0].shape[1] if isinstance(cover, list) else cover.shape[1]
    model = build_srnet_steganalyzer(channels=channels, arch=normalized_arch).to(device)
    backend_label = f"srnet_{normalized_arch}"
    if weights_path:
        state = torch.load(weights_path, map_location=device)
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            if normalized_arch != "lite":
                model = build_srnet_steganalyzer(channels=channels, arch="lite").to(device)
                model.load_state_dict(state_dict)
                backend_label = "srnet_lite_pretrained_fallback"
            else:
                raise
    if isinstance(cover, list):
        dataset = StegoPairListDataset([image.detach().cpu() for image in cover], [image.detach().cpu() for image in stego])
        collate_fn = stego_pair_collate
    else:
        dataset = StegoPairDataset(cover.detach().cpu(), stego.detach().cpu())
        collate_fn = None
    if weights_path is None:
        valid_ratio = min(0.2, max(0.1, float(test_ratio) * 0.5))
        train_set, valid_set, test_set = build_stratified_train_valid_test_subsets(
            dataset,
            dataset.labels,
            test_ratio=test_ratio,
            valid_ratio=valid_ratio,
            seed=seed,
        )
        loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
        best_state_dict = None
        best_valid_loss = float("inf")
        model.train()
        for _ in range(epochs):
            for images, labels in loader:
                labels = labels.to(device)
                if isinstance(images, list):
                    logits = torch.cat([model(image.to(device)) for image in images], dim=0)
                else:
                    logits = model(images.to(device))
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            model.eval()
            valid_loss_sum = 0.0
            valid_batches = 0
            with torch.no_grad():
                for images, labels in valid_loader:
                    labels = labels.to(device)
                    if isinstance(images, list):
                        logits = torch.cat([model(image.to(device)) for image in images], dim=0)
                    else:
                        logits = model(images.to(device))
                    valid_loss_sum += float(F.cross_entropy(logits, labels).item())
                    valid_batches += 1
            avg_valid_loss = valid_loss_sum / max(1, valid_batches)
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            model.train()
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        if save_weights_path is not None:
            save_path = Path(save_weights_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "arch": normalized_arch}, save_path)
    else:
        test_set = dataset
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model.eval()
    all_predictions = []
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.to(device)
            if isinstance(images, list):
                logits = torch.cat([model(image.to(device)) for image in images], dim=0)
            else:
                logits = model(images.to(device))
            scores = torch.softmax(logits, dim=1)[:, 1]
            pred = logits.argmax(dim=1)
            all_predictions.append(pred.detach().cpu())
            all_scores.append(scores.detach().cpu())
            all_labels.append(labels.detach().cpu())
    metrics = binary_detection_metrics(torch.cat(all_predictions, dim=0), torch.cat(all_labels, dim=0))
    metrics.update(score_detection_metrics(torch.cat(all_scores, dim=0), torch.cat(all_labels, dim=0)))
    metrics["backend"] = (
        f"{backend_label}_pretrained"
        if weights_path
        else f"{backend_label}_train_valid_test_{int(epochs)}e"
    )
    return metrics


def resolve_comparison_root(explicit_root: str | None = None) -> Path | None:
    # 解析 Comparison 根目录，优先使用显式参数，其次尝试常见服务器路径。
    candidate_paths = []
    if explicit_root:
        candidate_paths.append(Path(explicit_root))
    candidate_paths.extend(
        [
            Path(__file__).resolve().parent.parent / "Comparison",
            Path(__file__).resolve().parent.parent.parent / "Comparison",
            Path("/group4/Comparison"),
            Path("/root/Comparison"),
        ]
    )
    for path in candidate_paths:
        if path.exists():
            return path
    return None


# 兼容旧调用：返回 SRNet 的平衡检测率。
def srnet_detection_rate(
    cover: torch.Tensor | list[torch.Tensor],
    stego: torch.Tensor | list[torch.Tensor],
    device: torch.device,
    weights_path: str | None = None,
    epochs: int = 2,
    batch_size: int = 8,
    seed: int = 42,
    arch: str = "deep",
    lr: float = 1e-4,
    test_ratio: float = 0.3,
    save_weights_path: str | None = None,
) -> float:
    return srnet_detection_metrics(
        cover,
        stego,
        device=device,
        weights_path=weights_path,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        arch=arch,
        lr=lr,
        test_ratio=test_ratio,
        save_weights_path=save_weights_path,
    )["detection_rate"]


# 根据两组 Inception 特征计算 FID。
def calculate_fid(real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
    try:
        from scipy import linalg
    except ModuleNotFoundError:
        print("Warning: scipy is not installed, FID metric will be recorded as NaN.")
        return float("nan")
    real_np = real_features.detach().cpu().double().numpy()
    fake_np = fake_features.detach().cpu().double().numpy()
    if real_np.shape[0] < 2 or fake_np.shape[0] < 2:
        print("Warning: FID needs at least 2 images per set, FID metric will be recorded as NaN.")
        return float("nan")
    mu_real = real_np.mean(axis=0)
    mu_fake = fake_np.mean(axis=0)
    sigma_real = np.cov(real_np, rowvar=False)
    sigma_fake = np.cov(fake_np, rowvar=False)
    if sigma_real.ndim == 0:
        sigma_real = np.eye(real_np.shape[1], dtype=np.float64) * float(sigma_real)
    if sigma_fake.ndim == 0:
        sigma_fake = np.eye(fake_np.shape[1], dtype=np.float64) * float(sigma_fake)
    eps = 1e-6
    sigma_real = sigma_real + np.eye(sigma_real.shape[0], dtype=np.float64) * eps
    sigma_fake = sigma_fake + np.eye(sigma_fake.shape[0], dtype=np.float64) * eps
    covmean = linalg.sqrtm(sigma_real @ sigma_fake)
    if covmean.dtype.kind == "c":
        covmean = covmean.real
    diff = mu_real - mu_fake
    fid = float(diff.dot(diff) + sigma_real.trace() + sigma_fake.trace() - 2.0 * covmean.trace())
    return fid if math.isfinite(fid) else float("nan")


# 使用 Inception 特征上的三阶多项式核 MMD 估计 KID，数值越低代表分布越接近。
def calculate_kid(real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
    real = real_features.detach().cpu().double()
    fake = fake_features.detach().cpu().double()
    if real.shape[0] < 2 or fake.shape[0] < 2:
        return float("nan")
    dim = max(1, int(real.shape[1]))
    k_xx = ((real @ real.t()) / dim + 1.0).pow(3)
    k_yy = ((fake @ fake.t()) / dim + 1.0).pow(3)
    k_xy = ((real @ fake.t()) / dim + 1.0).pow(3)
    n = real.shape[0]
    m = fake.shape[0]
    k_xx = (k_xx.sum() - k_xx.diag().sum()) / max(1, n * (n - 1))
    k_yy = (k_yy.sum() - k_yy.diag().sum()) / max(1, m * (m - 1))
    kid = float(k_xx + k_yy - 2.0 * k_xy.mean())
    return max(0.0, kid) if math.isfinite(kid) else float("nan")


def calculate_clean_fid(
    real_images: torch.Tensor | list[torch.Tensor],
    fake_images: torch.Tensor | list[torch.Tensor],
    device: torch.device,
    mode: str = "clean",
) -> float:
    from cleanfid import fid as clean_fid

    with tempfile.TemporaryDirectory(prefix="wavebridege_clean_fid_") as temp_dir:
        temp_root = Path(temp_dir)
        real_dir = temp_root / "real"
        fake_dir = temp_root / "fake"
        real_count = export_images_to_directory(real_images, real_dir, "real")
        fake_count = export_images_to_directory(fake_images, fake_dir, "fake")
        if real_count < 2 or fake_count < 2:
            print("Warning: clean-fid needs at least 2 exported images per set, FID metric will be recorded as NaN.")
            return float("nan")
        return float(
            clean_fid.compute_fid(
                str(real_dir),
                str(fake_dir),
                mode=mode,
                num_workers=0,
                device=str(device),
            )
        )


def calculate_fid_kid_metrics(
    original: torch.Tensor | list[torch.Tensor],
    carrier: torch.Tensor | list[torch.Tensor],
    cover: torch.Tensor | list[torch.Tensor],
    device: torch.device,
    backend: str = "auto",
    clean_mode: str = "clean",
) -> tuple[float, float, float, float, str]:
    normalized_backend = str(backend or "auto").strip().lower()
    clean_fid_values: tuple[float, float] | None = None
    if normalized_backend in {"auto", "clean", "clean-fid"}:
        try:
            fid_cover = calculate_clean_fid(original, cover, device=device, mode=clean_mode)
            fid_carrier = calculate_clean_fid(original, carrier, device=device, mode=clean_mode)
            clean_fid_values = (fid_cover, fid_carrier)
        except Exception as error:
            print(f"Warning: clean-fid backend failed, falling back to legacy Inception FID. {error}")

    inception, inception_backend = build_inception_extractor(device)
    if inception is None:
        if clean_fid_values is not None:
            return (
                clean_fid_values[0],
                clean_fid_values[1],
                float("nan"),
                float("nan"),
                f"clean-fid+{inception_backend}-kid-unavailable",
            )
        return float("nan"), float("nan"), float("nan"), float("nan"), inception_backend
    with torch.inference_mode():
        real_features = torch.cat(
            [inception(image.unsqueeze(0).to(device)) for image in flatten_image_batches(original)],
            dim=0,
        )
        carrier_features = torch.cat(
            [inception(image.unsqueeze(0).to(device)) for image in flatten_image_batches(carrier)],
            dim=0,
        )
        cover_features = torch.cat(
            [inception(image.unsqueeze(0).to(device)) for image in flatten_image_batches(cover)],
            dim=0,
        )
    kid_cover = calculate_kid(real_features, cover_features)
    kid_carrier = calculate_kid(real_features, carrier_features)
    if clean_fid_values is not None:
        return (
            clean_fid_values[0],
            clean_fid_values[1],
            kid_cover,
            kid_carrier,
            f"clean-fid+{inception_backend}-kid",
        )
    return (
        calculate_fid(real_features, cover_features),
        calculate_fid(real_features, carrier_features),
        kid_cover,
        kid_carrier,
        inception_backend,
    )


def calculate_fid_metrics(
    original: torch.Tensor | list[torch.Tensor],
    carrier: torch.Tensor | list[torch.Tensor],
    cover: torch.Tensor | list[torch.Tensor],
    device: torch.device,
    backend: str = "auto",
    clean_mode: str = "clean",
) -> tuple[float, float, str]:
    fid_cover, fid_carrier, _, _, fid_backend = calculate_fid_kid_metrics(
        original,
        carrier,
        cover,
        device=device,
        backend=backend,
        clean_mode=clean_mode,
    )
    return fid_cover, fid_carrier, fid_backend


# 只加载 checkpoint 中的 prior_generator 权重，避免污染 codec、receiver 和通信链路。
def load_prior_generator_weights_for_eval(model: WaveBridegeSystem, checkpoint_path: str | Path | None) -> str | None:
    if checkpoint_path is None:
        return None
    resolved_path = resolve_project_path(checkpoint_path)
    if resolved_path is None:
        return None
    if not resolved_path.exists():
        raise FileNotFoundError(f"Prior checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(resolved_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Prior checkpoint is not a state_dict-compatible file: {checkpoint_path}")
    prefix = "generator.prior_generator."
    prior_state = {
        name[len(prefix) :]: value
        for name, value in state_dict.items()
        if isinstance(name, str) and name.startswith(prefix)
    }
    if not prior_state:
        raise ValueError(f"No generator.prior_generator weights found in prior checkpoint: {checkpoint_path}")
    incompatible = model.generator.prior_generator.load_state_dict(prior_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        missing_preview = ", ".join(incompatible.missing_keys[:5])
        unexpected_preview = ", ".join(incompatible.unexpected_keys[:5])
        print(
            "Prior generator checkpoint loaded with non-strict key mismatch: "
            f"missing={len(incompatible.missing_keys)} [{missing_preview}] "
            f"unexpected={len(incompatible.unexpected_keys)} [{unexpected_preview}]",
            flush=True,
        )
    print(f"Loaded eval prior_generator weights from: {resolved_path}", flush=True)
    return str(resolved_path)


# 解析用于评估期安全扫参的 prior_mix 列表；显式传参时严格以命令行为准。
def parse_prior_mix_candidates(raw_mixes: str | None, baseline_mix: float) -> list[float]:
    candidates: list[float] = []
    if raw_mixes is not None and str(raw_mixes).strip():
        for item in str(raw_mixes).replace(";", ",").split(","):
            stripped = item.strip()
            if not stripped:
                continue
            value = max(0.0, min(1.0, float(stripped)))
            candidates.append(value)
    else:
        candidates.append(float(baseline_mix))
    unique: list[float] = []
    for value in candidates:
        rounded = round(float(value), 8)
        if all(abs(rounded - existing) > 1e-8 for existing in unique):
            unique.append(rounded)
    return unique


# 临时修改生成器 prior_mix，用于只在评估期试探自然先验强度。
class TemporaryPriorMix:
    def __init__(self, model: WaveBridegeSystem, prior_mix: float) -> None:
        self.model = model
        self.prior_mix = float(prior_mix)
        self.original_prior_mix = float(model.generator.prior_mix)

    def __enter__(self) -> None:
        self.model.generator.prior_mix = float(self.prior_mix)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.model.generator.prior_mix = float(self.original_prior_mix)


# 为每张图构造稳定的 latent 候选种子，保证主评估和攻击评估选中同一张载密图。
def latent_selection_image_seed(base_seed: int, image_index: int) -> int:
    return int(base_seed) + 104729 + int(image_index) * 7919


# 根据当前噪声模式生成候选 latent；landscape 模式下围绕 anchor 做小范围扰动。
def sample_candidate_noise(
    model: WaveBridegeSystem,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    candidate_index: int,
) -> torch.Tensor:
    noise_mode = str(getattr(model, "noise_mode", "random"))
    if noise_mode == "landscape":
        anchor = model.landscape_anchor.to(device=device, dtype=dtype).expand(batch_size, -1)
        if int(candidate_index) == 0:
            return anchor
        generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
        generator.manual_seed(int(seed) + int(candidate_index) * 10007)
        std = max(0.05, float(getattr(model, "noise_std", 0.25)))
        return anchor + torch.randn(anchor.shape, device=device, dtype=dtype, generator=generator) * std

    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(int(seed) + int(candidate_index) * 10007)
    return torch.randn(batch_size, int(getattr(model, "latent_dim")), device=device, dtype=dtype, generator=generator)


# 计算单个候选输出的恢复质量、误码率、解码覆盖率和自然度代理分数。
def summarize_latent_candidate(
    output,
    reference_image: torch.Tensor,
    score_name: str,
) -> dict[str, float]:
    target_info_bits, predicted_info_bits, target_code_bits, predicted_code_bits = select_metric_bit_pairs(output)
    payload_mask = payload_valid_mask_from_packet(
        output.transmitter.packet,
        target_info_bits,
        output.receiver.decoded_block_indices,
    )
    decoded_blocks, total_blocks, decoded_groups, total_groups = summarize_decode_counts(output)
    safe_reference = reference_image.detach().clamp(0.0, 1.0)
    safe_cover = output.generator.cover_image.detach().clamp(0.0, 1.0)
    safe_carrier = output.generator.generated_image.detach().clamp(0.0, 1.0)
    score_images = safe_cover
    if score_name == "carrier_brisque_proxy":
        score_images = output.generator.generated_image
    elif score_name not in {"brisque_proxy", "cover_brisque_proxy"}:
        raise ValueError(f"Unsupported latent selection score: {score_name}")
    return {
        "psnr": float(batch_psnr(safe_reference, output.receiver.restored_image).mean().item()),
        "ssim": float(batch_ssim(safe_reference, output.receiver.restored_image).mean().item()),
        "ber_info": float(bit_error_rate(target_info_bits, predicted_info_bits, payload_mask).mean().item()),
        "ber_code": float(bit_error_rate(target_code_bits, predicted_code_bits).mean().item()),
        "decoded_block_ratio": float(decoded_blocks / max(total_blocks, 1)),
        "decoded_group_ratio": float(decoded_groups / max(total_groups, 1)),
        "naturalness_score": float(brisque_proxy_score(score_images.detach()).mean().item()),
        "stego_psnr_to_source": float(batch_psnr(safe_reference, safe_cover).mean().item()),
        "stego_ssim_to_source": float(batch_ssim(safe_reference, safe_cover).mean().item()),
        "stego_l1_to_source": float(F.l1_loss(safe_cover, safe_reference).item()),
        "stego_psnr_to_carrier": float(batch_psnr(safe_carrier, safe_cover).mean().item()),
        "stego_ssim_to_carrier": float(batch_ssim(safe_carrier, safe_cover).mean().item()),
    }


# 判断候选载密图是否能在不牺牲恢复链路的前提下参与自然度择优。
def latent_candidate_distinct_ok(
    metrics: dict[str, float],
    *,
    max_source_psnr: float = 15.0,
    max_source_ssim: float = 0.20,
) -> bool:
    # 判断候选载密图是否已经和秘密图明显分离。
    source_psnr = float(metrics.get("stego_psnr_to_source", float("inf")))
    source_ssim = float(metrics.get("stego_ssim_to_source", float("inf")))
    return (
        math.isfinite(source_psnr)
        and math.isfinite(source_ssim)
        and source_psnr <= float(max_source_psnr)
        and source_ssim <= float(max_source_ssim)
    )


def stego_secret_distinct_ratio(
    original: torch.Tensor | list[torch.Tensor],
    cover: torch.Tensor | list[torch.Tensor],
    *,
    max_source_psnr: float = 15.0,
    max_source_ssim: float = 0.20,
) -> float:
    # 统计逐样本“载密图是否和秘密图充分分离”的比例，避免只看均值掩盖失败样本。
    original_batches = [original] if isinstance(original, torch.Tensor) else original
    cover_batches = [cover] if isinstance(cover, torch.Tensor) else cover
    distinct_flags: list[torch.Tensor] = []
    for source_batch, stego_batch in zip(original_batches, cover_batches):
        source_psnr = batch_psnr(source_batch, stego_batch)
        source_ssim = batch_ssim(source_batch, stego_batch)
        distinct_flags.append(
            ((source_psnr <= float(max_source_psnr)) & (source_ssim <= float(max_source_ssim))).float().detach().cpu()
        )
    if not distinct_flags:
        return 0.0
    return float(torch.cat(distinct_flags, dim=0).mean().item())


def latent_candidate_separation_penalty(
    metrics: dict[str, float],
    *,
    max_source_psnr: float = 15.0,
    max_source_ssim: float = 0.20,
    min_source_l1: float = 0.10,
) -> float:
    # 将候选载密图对秘密图的残余相似度压成一个惩罚分，越小越好。
    source_psnr = float(metrics.get("stego_psnr_to_source", float("inf")))
    source_ssim = float(metrics.get("stego_ssim_to_source", float("inf")))
    source_l1 = float(metrics.get("stego_l1_to_source", 0.0))
    if not math.isfinite(source_psnr) or not math.isfinite(source_ssim) or not math.isfinite(source_l1):
        return float("inf")
    psnr_gap = max(0.0, source_psnr - float(max_source_psnr)) / max(float(max_source_psnr), 1.0)
    ssim_gap = max(0.0, source_ssim - float(max_source_ssim)) / max(1e-6, 1.0 - float(max_source_ssim))
    l1_gap = max(0.0, float(min_source_l1) - source_l1) / max(float(min_source_l1), 1e-6)
    return 1.40 * psnr_gap + 1.90 * ssim_gap + 0.65 * l1_gap


def latent_candidate_carrier_alignment_score(metrics: dict[str, float]) -> float:
    # 用生成载体与最终载密图之间的相似度刻画“自然载体风格是否保住了”。
    psnr = float(metrics.get("stego_psnr_to_carrier", float("-inf")))
    ssim = float(metrics.get("stego_ssim_to_carrier", float("-inf")))
    if not math.isfinite(psnr) or not math.isfinite(ssim):
        return float("-inf")
    normalized_psnr = min(max(psnr, 0.0), 40.0) / 40.0
    normalized_ssim = min(1.0, max(0.0, ssim))
    return 0.35 * normalized_psnr + 0.65 * normalized_ssim


def latent_candidate_restore_safe(
    metrics: dict[str, float],
    *,
    target_psnr: float,
    target_ssim: float,
    max_ber_info: float,
    max_ber_code: float,
    decoded_floor: float,
) -> bool:
    # 判断候选是否仍处于“恢复质量和 clean 通信都安全”的区域。
    psnr = float(metrics.get("psnr", float("-inf")))
    ssim = float(metrics.get("ssim", float("-inf")))
    ber_info = float(metrics.get("ber_info", float("inf")))
    ber_code = float(metrics.get("ber_code", float("inf")))
    decoded_block_ratio = float(metrics.get("decoded_block_ratio", 0.0))
    decoded_group_ratio = float(metrics.get("decoded_group_ratio", 0.0))
    return (
        math.isfinite(psnr)
        and math.isfinite(ssim)
        and math.isfinite(ber_info)
        and math.isfinite(ber_code)
        and psnr >= float(target_psnr)
        and ssim >= float(target_ssim)
        and ber_info <= float(max_ber_info)
        and ber_code <= float(max_ber_code)
        and decoded_block_ratio + 1e-12 >= float(decoded_floor)
        and decoded_group_ratio + 1e-12 >= float(decoded_floor)
    )


def latent_candidate_passes_gate(
    candidate: dict[str, float],
    baseline: dict[str, float],
    psnr_drop: float,
    ssim_drop: float,
    max_ber_info: float,
    max_ber_code: float,
    decoded_floor: float,
    min_psnr: float | None = 38.0,
    min_ssim: float | None = 0.97,
) -> bool:
    safe_psnr_target = 35.0
    if min_psnr is not None:
        safe_psnr_target = max(35.0, float(min_psnr) - 3.0)
    safe_ssim_target = 0.98
    if min_ssim is not None:
        safe_ssim_target = max(0.98, float(min_ssim) - 0.002)
    baseline_restore_safe = latent_candidate_restore_safe(
        baseline,
        target_psnr=safe_psnr_target,
        target_ssim=safe_ssim_target,
        max_ber_info=max_ber_info,
        max_ber_code=max_ber_code,
        decoded_floor=decoded_floor,
    )
    candidate_restore_safe = latent_candidate_restore_safe(
        candidate,
        target_psnr=safe_psnr_target,
        target_ssim=safe_ssim_target,
        max_ber_info=max_ber_info,
        max_ber_code=max_ber_code,
        decoded_floor=decoded_floor,
    )
    effective_psnr_drop = float(psnr_drop)
    effective_ssim_drop = float(ssim_drop)
    if baseline_restore_safe:
        effective_psnr_drop = max(effective_psnr_drop, 0.85 if candidate_restore_safe else 0.60)
        effective_ssim_drop = max(effective_ssim_drop, 0.006 if candidate_restore_safe else 0.004)
    if min_psnr is not None and baseline["psnr"] >= float(min_psnr) - 1e-12:
        if candidate["psnr"] < float(min_psnr) - 1e-12:
            return False
    elif candidate["psnr"] < baseline["psnr"] - effective_psnr_drop:
        return False
    if min_ssim is not None and baseline["ssim"] >= float(min_ssim) - 1e-12:
        if candidate["ssim"] < float(min_ssim) - 1e-12:
            return False
    elif candidate["ssim"] < baseline["ssim"] - effective_ssim_drop:
        return False
    if baseline["ber_info"] <= float(max_ber_info):
        if candidate["ber_info"] > float(max_ber_info):
            return False
    elif candidate["ber_info"] > baseline["ber_info"] + 1e-12:
        return False
    if baseline["ber_code"] <= float(max_ber_code):
        if candidate["ber_code"] > float(max_ber_code):
            return False
    elif candidate["ber_code"] > baseline["ber_code"] + 1e-12:
        return False
    block_floor = min(float(decoded_floor), baseline["decoded_block_ratio"] + 1e-12)
    group_floor = min(float(decoded_floor), baseline["decoded_group_ratio"] + 1e-12)
    if candidate["decoded_block_ratio"] + 1e-12 < block_floor:
        return False
    if candidate["decoded_group_ratio"] + 1e-12 < group_floor:
        return False
    if latent_candidate_distinct_ok(baseline) and not latent_candidate_distinct_ok(candidate):
        return False
    return True


# 在多个 latent 候选中选择自然度更好的输出；所有候选不合格时回退基线输出。
def select_output_by_latent_candidates(
    model: WaveBridegeSystem,
    images: torch.Tensor,
    full_decode: bool,
    candidates: int,
    prior_mix_candidates: list[float] | None,
    psnr_drop: float,
    ssim_drop: float,
    max_ber_info: float,
    max_ber_code: float,
    decoded_floor: float,
    score_name: str,
    seed: int,
    min_psnr: float = 38.0,
    min_ssim: float = 0.97,
) -> tuple[object, dict[str, float]]:
    candidate_count = max(1, int(candidates))
    mix_candidates = prior_mix_candidates or [float(model.generator.prior_mix)]
    safe_psnr_target = max(35.0, float(min_psnr) - 3.0 if min_psnr is not None else 35.0)
    safe_ssim_target = max(0.98, float(min_ssim) - 0.002 if min_ssim is not None else 0.98)
    if candidate_count <= 1 and len(mix_candidates) <= 1:
        selected_mix = float(mix_candidates[0])
        with TemporaryPriorMix(model, selected_mix):
            output = model(images, force_full_decode=full_decode)
        metrics = summarize_latent_candidate(output, images, score_name)
        metrics["selected_index"] = 0.0
        metrics["selected_prior_mix"] = selected_mix
        metrics["selection_used"] = 0.0
        metrics["distinct_ok"] = 1.0 if latent_candidate_distinct_ok(metrics) else 0.0
        metrics["separation_penalty"] = float(latent_candidate_separation_penalty(metrics))
        metrics["restore_safe"] = 1.0 if latent_candidate_restore_safe(
            metrics,
            target_psnr=safe_psnr_target,
            target_ssim=safe_ssim_target,
            max_ber_info=max_ber_info,
            max_ber_code=max_ber_code,
            decoded_floor=decoded_floor,
        ) else 0.0
        metrics["carrier_alignment_score"] = float(latent_candidate_carrier_alignment_score(metrics))
        return output, metrics

    best_output = None
    best_metrics: dict[str, float] | None = None
    baseline_metrics: dict[str, float] | None = None
    best_distinct_ok = False
    best_separation_penalty = float("inf")
    ordinal_index = 0
    for mix_index, prior_mix in enumerate(mix_candidates):
        with TemporaryPriorMix(model, prior_mix):
            for candidate_index in range(candidate_count):
                noise = sample_candidate_noise(
                    model,
                    batch_size=int(images.shape[0]),
                    device=images.device,
                    dtype=images.dtype,
                    seed=seed,
                    candidate_index=candidate_index,
                )
                output = model(images, noise=noise, force_full_decode=full_decode)
                metrics = summarize_latent_candidate(output, images, score_name)
                metrics["selected_index"] = float(ordinal_index)
                metrics["selected_prior_mix"] = float(prior_mix)
                metrics["selection_used"] = 1.0 if ordinal_index > 0 else 0.0
                metrics["distinct_ok"] = 1.0 if latent_candidate_distinct_ok(metrics) else 0.0
                metrics["separation_penalty"] = float(latent_candidate_separation_penalty(metrics))
                metrics["restore_safe"] = 1.0 if latent_candidate_restore_safe(
                    metrics,
                    target_psnr=safe_psnr_target,
                    target_ssim=safe_ssim_target,
                    max_ber_info=max_ber_info,
                    max_ber_code=max_ber_code,
                    decoded_floor=decoded_floor,
                ) else 0.0
                metrics["carrier_alignment_score"] = float(latent_candidate_carrier_alignment_score(metrics))
                if ordinal_index == 0:
                    best_output = output
                    best_metrics = metrics
                    baseline_metrics = metrics
                    best_distinct_ok = bool(metrics["distinct_ok"] > 0.5)
                    best_separation_penalty = float(metrics["separation_penalty"])
                    ordinal_index += 1
                    continue
                ordinal_index += 1
                assert baseline_metrics is not None
                assert best_metrics is not None
                if not latent_candidate_passes_gate(
                    metrics,
                    baseline_metrics,
                    psnr_drop=psnr_drop,
                    ssim_drop=ssim_drop,
                    max_ber_info=max_ber_info,
                    max_ber_code=max_ber_code,
                    decoded_floor=decoded_floor,
                    min_psnr=min_psnr,
                    min_ssim=min_ssim,
                ):
                    continue
                candidate_distinct_ok = bool(metrics["distinct_ok"] > 0.5)
                candidate_separation_penalty = float(metrics["separation_penalty"])
                candidate_restore_safe = bool(metrics["restore_safe"] > 0.5)
                best_restore_safe = bool(best_metrics.get("restore_safe", 0.0) > 0.5)
                if candidate_distinct_ok != best_distinct_ok:
                    if candidate_distinct_ok:
                        best_output = output
                        best_metrics = metrics
                        best_distinct_ok = candidate_distinct_ok
                        best_separation_penalty = candidate_separation_penalty
                    continue
                if candidate_restore_safe != best_restore_safe:
                    if candidate_restore_safe:
                        best_output = output
                        best_metrics = metrics
                        best_distinct_ok = candidate_distinct_ok
                        best_separation_penalty = candidate_separation_penalty
                    continue
                if (
                    candidate_restore_safe
                    and best_restore_safe
                    and abs(candidate_separation_penalty - best_separation_penalty) <= 0.02
                ):
                    candidate_carrier_alignment = float(metrics["carrier_alignment_score"])
                    best_carrier_alignment = float(best_metrics.get("carrier_alignment_score", float("-inf")))
                    if candidate_carrier_alignment > best_carrier_alignment + 5e-4:
                        best_output = output
                        best_metrics = metrics
                        best_distinct_ok = candidate_distinct_ok
                        best_separation_penalty = candidate_separation_penalty
                        continue
                    if candidate_carrier_alignment < best_carrier_alignment - 5e-4:
                        continue
                if candidate_separation_penalty + 1e-9 < best_separation_penalty:
                    best_output = output
                    best_metrics = metrics
                    best_distinct_ok = candidate_distinct_ok
                    best_separation_penalty = candidate_separation_penalty
                    continue
                if candidate_separation_penalty > best_separation_penalty + 1e-9:
                    continue
                if metrics["naturalness_score"] + 1e-9 < best_metrics["naturalness_score"]:
                    best_output = output
                    best_metrics = metrics
                    best_distinct_ok = candidate_distinct_ok
                    best_separation_penalty = candidate_separation_penalty

    assert best_output is not None and best_metrics is not None
    return best_output, best_metrics


# 运行模型生成 cover/restored 图像并收集所有评估张量。
def collect_outputs(
    model: WaveBridegeSystem,
    dataloader: DataLoader,
    device: torch.device,
    max_images: int | None,
    full_decode: bool = True,
    latent_select_candidates: int | None = None,
    latent_select_prior_mixes: list[float] | None = None,
    latent_select_psnr_drop: float = 0.05,
    latent_select_ssim_drop: float = 0.0005,
    latent_select_max_ber_info: float = 1e-4,
    latent_select_max_ber_code: float = 1e-3,
    latent_select_decoded_floor: float = 0.999999,
    latent_select_score: str = "cover_brisque_proxy",
    latent_select_seed: int = 42,
    latent_select_min_psnr: float = 38.0,
    latent_select_min_ssim: float = 0.97,
    progress_interval: int = 0,
    compute_channel_ber: bool = True,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    latent_select_candidates = max(
        1,
        int(
            latent_select_candidates
            if latent_select_candidates is not None
            else (
                LOW_PAYLOAD_MIN_LATENT_CANDIDATES
                if is_low_payload_target_bpp(resolve_config_target_bpp(getattr(model, "config", {})))
                else 4
            )
        ),
    )
    originals = []
    carriers = []
    covers = []
    decoded = []
    restored = []
    valid_info_bits = []
    image_sizes = []
    decoded_block_counts = []
    total_block_counts = []
    decoded_group_counts = []
    total_group_counts = []
    ber_info_sum = 0.0
    ber_code_sum = 0.0
    jpeg50_ber_info_sum = 0.0
    jpeg50_ber_code_sum = 0.0
    mixed_ber_info_sum = 0.0
    mixed_ber_code_sum = 0.0
    latent_selected_count = 0.0
    latent_selected_index_sum = 0.0
    seen = 0
    model.eval()
    with torch.inference_mode():
        for images in dataloader:
            for micro_images in iter_image_microbatches(images):
                if max_images is not None and seen >= max_images:
                    break
                if max_images is not None:
                    remaining = int(max_images) - seen
                    if remaining <= 0:
                        break
                    micro_images = micro_images[:remaining]
                micro_images = micro_images.to(device)
                if seen < 2:
                    print(
                        f"[EVAL] latent-select start seen={seen} "
                        f"batch={int(micro_images.shape[0])} "
                        f"candidates={latent_select_candidates} "
                        f"prior_mixes={len(latent_select_prior_mixes or []) or 1}",
                        flush=True,
                    )
                output, selection_metrics = select_output_by_latent_candidates(
                    model,
                    micro_images,
                    full_decode=full_decode,
                    candidates=latent_select_candidates,
                    prior_mix_candidates=latent_select_prior_mixes,
                    psnr_drop=latent_select_psnr_drop,
                    ssim_drop=latent_select_ssim_drop,
                    max_ber_info=latent_select_max_ber_info,
                    max_ber_code=latent_select_max_ber_code,
                    decoded_floor=latent_select_decoded_floor,
                    score_name=latent_select_score,
                    seed=latent_selection_image_seed(latent_select_seed, seen),
                    min_psnr=latent_select_min_psnr,
                    min_ssim=latent_select_min_ssim,
                )
                if seen < 2:
                    print(
                        f"[EVAL] latent-select done seen={seen} "
                        f"selected_index={selection_metrics.get('selected_index', float('nan'))} "
                        f"selection_used={selection_metrics.get('selection_used', float('nan'))}",
                        flush=True,
                    )
                batch_count = int(micro_images.shape[0])
                latent_selected_count += float(selection_metrics.get("selection_used", 0.0)) * batch_count
                latent_selected_index_sum += float(selection_metrics.get("selected_index", 0.0)) * batch_count
                originals.append(micro_images.cpu())
                carriers.append(output.generator.generated_image.cpu())
                covers.append(output.generator.cover_image.cpu())
                decoded.append(output.receiver.decoded_image.cpu())
                restored.append(output.receiver.restored_image.cpu())
                target_info_bits, predicted_info_bits, target_code_bits, predicted_code_bits = select_metric_bit_pairs(output)
                payload_mask = payload_valid_mask_from_packet(
                    output.transmitter.packet,
                    target_info_bits,
                    output.receiver.decoded_block_indices,
                )
                ber_info_sum += float(bit_error_rate(target_info_bits, predicted_info_bits, payload_mask).mean().item()) * batch_count
                ber_code_sum += float(bit_error_rate(target_code_bits, predicted_code_bits).mean().item()) * batch_count
                if compute_channel_ber:
                    jpeg_cover = jpeg_compress_tensor(output.generator.cover_image, quality=50)
                    jpeg_info_ber, jpeg_code_ber = robust_ber_from_output(model, output, jpeg_cover, full_decode=full_decode)
                    mixed_cover = mixed_channel_tensor(output.generator.cover_image, seed=seen + 2026)
                    mixed_info_ber, mixed_code_ber = robust_ber_from_output(model, output, mixed_cover, full_decode=full_decode)
                    jpeg50_ber_info_sum += jpeg_info_ber * batch_count
                    jpeg50_ber_code_sum += jpeg_code_ber * batch_count
                    mixed_ber_info_sum += mixed_info_ber * batch_count
                    mixed_ber_code_sum += mixed_code_ber * batch_count
                valid_info_bits.extend([torch.tensor(output.transmitter.packet.valid_info_bits)] * batch_count)
                decoded_block_count, total_block_count, decoded_group_count, total_group_count = summarize_decode_counts(output)
                decoded_block_counts.extend([torch.tensor(decoded_block_count)] * batch_count)
                total_block_counts.extend([torch.tensor(total_block_count)] * batch_count)
                decoded_group_counts.extend([torch.tensor(decoded_group_count)] * batch_count)
                total_group_counts.extend([torch.tensor(total_group_count)] * batch_count)
                image_sizes.extend([tuple(micro_images.shape[-2:])] * batch_count)
                seen += batch_count
                if progress_interval > 0 and seen % int(progress_interval) == 0:
                    total = "unknown" if max_images is None else str(max_images)
                    print(f"[EVAL] collected {seen}/{total} image(s)", flush=True)
                elif seen <= 2:
                    total = "unknown" if max_images is None else str(max_images)
                    print(f"[EVAL] first-pass progress {seen}/{total} image(s)", flush=True)
            if max_images is not None and seen >= max_images:
                break
    jpeg50_info_value = jpeg50_ber_info_sum / max(1, seen) if compute_channel_ber else float("nan")
    jpeg50_code_value = jpeg50_ber_code_sum / max(1, seen) if compute_channel_ber else float("nan")
    mixed_info_value = mixed_ber_info_sum / max(1, seen) if compute_channel_ber else float("nan")
    mixed_code_value = mixed_ber_code_sum / max(1, seen) if compute_channel_ber else float("nan")
    return {
        "original": originals,
        "carrier": carriers,
        "cover": covers,
        "decoded": decoded,
        "restored": restored,
        "ber_info": torch.tensor(ber_info_sum / max(1, seen), dtype=torch.float32),
        "ber_code": torch.tensor(ber_code_sum / max(1, seen), dtype=torch.float32),
        "jpeg50_ber_info": torch.tensor(jpeg50_info_value, dtype=torch.float32),
        "jpeg50_ber_code": torch.tensor(jpeg50_code_value, dtype=torch.float32),
        "mixed_ber_info": torch.tensor(mixed_info_value, dtype=torch.float32),
        "mixed_ber_code": torch.tensor(mixed_code_value, dtype=torch.float32),
        "valid_info_bits": torch.stack(valid_info_bits, dim=0),
        "decoded_block_counts": torch.stack(decoded_block_counts, dim=0),
        "total_block_counts": torch.stack(total_block_counts, dim=0),
        "decoded_group_counts": torch.stack(decoded_group_counts, dim=0),
        "total_group_counts": torch.stack(total_group_counts, dim=0),
        "image_sizes": image_sizes,
        "num_images": torch.tensor(seen, dtype=torch.long),
        "latent_select_candidates": torch.tensor(max(1, int(latent_select_candidates)), dtype=torch.long),
        "latent_select_used_ratio": torch.tensor(latent_selected_count / max(1, seen), dtype=torch.float32),
        "latent_select_avg_index": torch.tensor(latent_selected_index_sum / max(1, seen), dtype=torch.float32),
        "latent_select_score": latent_select_score,
        "latent_select_prior_mixes": ",".join(f"{value:g}" for value in (latent_select_prior_mixes or [float(model.generator.prior_mix)])),
    }


# 使用 PIL 在内存中模拟 JPEG 压缩，返回仍在 [0,1] 区间的张量。
def jpeg_compress_tensor(images: torch.Tensor, quality: int = 50) -> torch.Tensor:
    compressed = []
    for image in images.detach().cpu():
        pil = Image.fromarray((image.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8))
        with tempfile.NamedTemporaryFile(suffix=".jpg") as handle:
            pil.save(handle.name, format="JPEG", quality=int(quality))
            reloaded = Image.open(handle.name).convert("RGB")
            array = np.asarray(reloaded).astype(np.float32) / 255.0
        compressed.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(compressed, dim=0).to(device=images.device, dtype=images.dtype)


# 构造混合信道扰动：JPEG50 + 轻微高斯噪声 + 量化裁剪。
def mixed_channel_tensor(images: torch.Tensor, seed: int | None = 42) -> torch.Tensor:
    perturbed = jpeg_compress_tensor(images, quality=50)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=images.device)
        generator.manual_seed(int(seed))
    noise = torch.randn(perturbed.shape, device=images.device, dtype=images.dtype, generator=generator) * 0.003
    return torch.round((perturbed + noise).clamp(0.0, 1.0) * 255.0) / 255.0


def jpeg75_gaussian_attack_tensor(
    images: torch.Tensor,
    quality: int = 75,
    noise_std: float = 0.01,
    seed: int | None = 42,
) -> torch.Tensor:
    # 生成 BOSSBase 鲁棒表使用的 JPEG75 + Gaussian(0.01) 攻击图像。
    perturbed = jpeg_compress_tensor(images, quality=int(quality))
    generator = None
    if seed is not None:
        generator = torch.Generator(device=images.device)
        generator.manual_seed(int(seed))
    noise = torch.randn(perturbed.shape, device=images.device, dtype=images.dtype, generator=generator) * float(noise_std)
    return (perturbed + noise).clamp(0.0, 1.0)


# 在已有输出上重放接收端，计算指定信道扰动后的 payload/code BER。
def robust_ber_from_output(
    model: WaveBridegeSystem,
    model_output,
    attacked_cover: torch.Tensor,
    full_decode: bool = True,
) -> tuple[float, float]:
    # 在攻击图像上重放接收端并计算 payload/code BER。
    external_llr_signal = None
    if getattr(model, "qim_enabled", False):
        extract_qim_llr = getattr(model, "extract_qim_llr", None)
        if callable(extract_qim_llr):
            external_llr_signal = extract_qim_llr(
                attacked_cover,
                num_blocks=model_output.transmitter.packet.num_blocks,
                code_length=model_output.transmitter.packet.coded_bits.shape[-1],
                prefer_robust=prefer_robust_qim_for_attack_metrics(),
            )
        else:
            external_llr_signal = model.qim_channel.extract_llr(
                attacked_cover,
                num_blocks=model_output.transmitter.packet.num_blocks,
                code_length=model_output.transmitter.packet.coded_bits.shape[-1],
            )
    if external_llr_signal is not None and attack_ber_source() != "receiver":
        strict_llr_signal = strict_attack_llr_signal(
            model,
            external_llr_signal,
            frontend_mode="robust" if prefer_robust_qim_for_attack_metrics() else "clean",
        )
        # 鲁棒 DCT-QIM 分支的 LLR 已经与 JPEG 量化格点对齐，BER 评估直接走严格 Polar 硬解码。
        hard_payload_bits, _hard_info_bits, hard_code_bits, _crc_pass_mask = model.receiver._hard_decode_full_stream(
            strict_llr_signal,
            decoded_code_logits=None,
            decoded_bit_logits=None,
            prefer_strict_polar_path=True,
        )
        target_info_bits = model_output.transmitter.packet.payload_bits.detach()
        target_code_bits = model_output.transmitter.packet.coded_bits.detach()
        payload_mask = payload_valid_mask_from_packet(model_output.transmitter.packet, target_info_bits, None)
        ber_info = float(bit_error_rate(target_info_bits, hard_payload_bits.detach(), payload_mask).mean().item())
        ber_code = float(bit_error_rate(target_code_bits, hard_code_bits.detach()).mean().item())
        return ber_info, ber_code
    receiver_output = model.receiver(
        attacked_cover,
        latent_shape=model_output.transmitter.compression.latent_shape,
        latent_decoder=model.compressor.decode_latent,
        output_size=model_output.transmitter.compression.original_size,
        valid_info_bits=model_output.transmitter.packet.valid_info_bits,
        num_blocks=model_output.transmitter.packet.num_blocks,
        force_full_decode=full_decode,
        full_bitstream_available=True,
        external_llr_signal=external_llr_signal,
    )
    class OutputProxy:
        pass

    proxy = OutputProxy()
    proxy.transmitter = model_output.transmitter
    proxy.receiver = receiver_output
    target_info_bits, predicted_info_bits, target_code_bits, predicted_code_bits = select_metric_bit_pairs(proxy)
    payload_mask = payload_valid_mask_from_packet(
        model_output.transmitter.packet,
        target_info_bits,
        receiver_output.decoded_block_indices,
    )
    ber_info = float(bit_error_rate(target_info_bits, predicted_info_bits, payload_mask).mean().item())
    ber_code = float(bit_error_rate(target_code_bits, predicted_code_bits).mean().item())
    return ber_info, ber_code


def robust_quality_metrics_from_output(
    model: WaveBridegeSystem,
    model_output,
    attacked_cover: torch.Tensor,
    reference_image: torch.Tensor,
    full_decode: bool = True,
    lpips_model=None,
) -> dict[str, float]:
    # 计算攻击后恢复质量、BER 和解码比例，供 CCFA 鲁棒表直接导出。
    external_llr_signal = None
    if getattr(model, "qim_enabled", False):
        extract_qim_llr = getattr(model, "extract_qim_llr", None)
        if callable(extract_qim_llr):
            external_llr_signal = extract_qim_llr(
                attacked_cover,
                num_blocks=model_output.transmitter.packet.num_blocks,
                code_length=model_output.transmitter.packet.coded_bits.shape[-1],
                prefer_robust=prefer_robust_qim_for_attack_metrics(),
            )
        else:
            external_llr_signal = model.qim_channel.extract_llr(
                attacked_cover,
                num_blocks=model_output.transmitter.packet.num_blocks,
                code_length=model_output.transmitter.packet.coded_bits.shape[-1],
            )
    strict_robust_metrics: tuple[float, float, float] | None = None
    if external_llr_signal is not None and attack_ber_source() != "receiver":
        strict_llr_signal = strict_attack_llr_signal(
            model,
            external_llr_signal,
            frontend_mode="robust" if prefer_robust_qim_for_attack_metrics() else "clean",
        )
        hard_payload_bits, _hard_info_bits, hard_code_bits, hard_crc_pass_mask = model.receiver._hard_decode_full_stream(
            strict_llr_signal,
            decoded_code_logits=None,
            decoded_bit_logits=None,
            prefer_strict_polar_path=True,
        )
        target_payload_bits = model_output.transmitter.packet.payload_bits.detach()
        target_code_bits = model_output.transmitter.packet.coded_bits.detach()
        payload_mask_for_hard = payload_valid_mask_from_packet(model_output.transmitter.packet, target_payload_bits, None)
        strict_robust_metrics = (
            float(bit_error_rate(target_payload_bits, hard_payload_bits.detach(), payload_mask_for_hard).mean().item()),
            float(bit_error_rate(target_code_bits, hard_code_bits.detach()).mean().item()),
            float(hard_crc_pass_mask.float().mean().item()),
        )
    receiver_output = model.receiver(
        attacked_cover,
        latent_shape=model_output.transmitter.compression.latent_shape,
        latent_decoder=model.compressor.decode_latent,
        output_size=model_output.transmitter.compression.original_size,
        valid_info_bits=model_output.transmitter.packet.valid_info_bits,
        num_blocks=model_output.transmitter.packet.num_blocks,
        force_full_decode=full_decode,
        full_bitstream_available=True,
        external_llr_signal=external_llr_signal,
        external_llr_frontend_mode="robust",
    )
    class OutputProxy:
        pass

    proxy = OutputProxy()
    proxy.transmitter = model_output.transmitter
    proxy.receiver = receiver_output
    target_info_bits, predicted_info_bits, target_code_bits, predicted_code_bits = select_metric_bit_pairs(proxy)
    payload_mask = payload_valid_mask_from_packet(
        model_output.transmitter.packet,
        target_info_bits,
        receiver_output.decoded_block_indices,
    )
    restored = receiver_output.restored_image.detach()
    psnr_attack = float(batch_psnr(reference_image, restored).mean().item())
    ssim_attack = float(batch_ssim(reference_image, restored).mean().item())
    ssim_attack_y = float(batch_ssim_y(reference_image, restored).mean().item())
    if lpips_model is not None:
        with torch.inference_mode():
            lpips_attack = float(lpips_model(reference_image * 2.0 - 1.0, restored * 2.0 - 1.0).mean().item())
    else:
        lpips_attack = float("nan")
    return {
        "psnr_attack": psnr_attack,
        "ssim_attack": ssim_attack,
        "ssim_attack_y": ssim_attack_y,
        "lpips_attack": lpips_attack,
        "ber_info_attack": (
            strict_robust_metrics[0]
            if strict_robust_metrics is not None
            else float(bit_error_rate(target_info_bits, predicted_info_bits, payload_mask).mean().item())
        ),
        "ber_code_attack": (
            strict_robust_metrics[1]
            if strict_robust_metrics is not None
            else float(bit_error_rate(target_code_bits, predicted_code_bits).mean().item())
        ),
        "decoded_ratio_attack": (
            strict_robust_metrics[2]
            if strict_robust_metrics is not None
            else (
                float(receiver_output.crc_pass_mask.float().mean().item())
                if receiver_output.crc_pass_mask is not None
                else float("nan")
            )
        ),
    }


# 将指标结果保存为 JSON 和 CSV 文件。
def save_results(result: EvaluationResult, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = evaluation_result_to_dict(result)
    grouped_data = evaluation_result_to_groups(result)
    target_pass_fail = grouped_data["target_pass_fail"]
    (output / "metrics.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "all_method_details.json").write_text(
        json.dumps([data], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "target_pass_fail.json").write_text(
        json.dumps(target_pass_fail, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_target_pass_fail_csv(output / "target_pass_fail.csv", target_pass_fail)
    (output / "metrics_by_group.json").write_text(
        json.dumps(grouped_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "metrics_summary.md").write_text(
        format_grouped_metrics_markdown(grouped_data),
        encoding="utf-8",
    )
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data.keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerow(data)
    save_comparison_compatible_tables(data, output)


# 执行完整评估流程并返回所有实验指标。
def evaluate(args: argparse.Namespace) -> EvaluationResult:
    raw_argv_tokens = sys.argv[1:]
    args.config = str(resolve_project_path(args.config))
    args.data = None if args.data is None else str(resolve_project_path(args.data))
    args.split = None if args.split is None else str(resolve_project_path(args.split))
    args.checkpoint = None if args.checkpoint is None else str(resolve_project_path(args.checkpoint))
    args.output_dir = str(resolve_project_path(args.output_dir))
    # 为训练脚本内联构造的参数对象补齐潜变量筛选默认值，避免缺字段时直接中断整条评估链路。
    latent_select_defaults = {
        "latent_select_candidates": 4,
        "latent_select_score": "cover_brisque_proxy",
        "latent_select_prior_checkpoint": None,
        "latent_select_prior_mixes": "",
        "latent_select_psnr_drop": 0.05,
        "latent_select_ssim_drop": 0.0005,
        "latent_select_max_ber_info": 1e-4,
        "latent_select_max_ber_code": 1e-3,
        "latent_select_decoded_floor": 0.999999,
        "latent_select_min_psnr": 38.0,
        "latent_select_min_ssim": 0.97,
        "progress_interval": 0,
        "natural_reference_dataset": None,
        "natural_reference_dirs": "",
    }
    for key, value in latent_select_defaults.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    config = load_config(args.config)
    apply_low_payload_eval_defaults(args, config, argv_tokens=raw_argv_tokens)
    apply_eval_qim_profile(config)
    if args.active_dataset:
        active_dataset = str(args.active_dataset).lower().strip()
        if active_dataset not in config.get("datasets", {}):
            available = ", ".join(sorted(config.get("datasets", {}).keys()))
            raise KeyError(f"Active dataset '{active_dataset}' is not configured. Available: {available}")
        config.setdefault("datasets", {})["active"] = active_dataset
    if args.target_bpp is not None:
        target_bpp = float(args.target_bpp)
        if not math.isfinite(target_bpp) or target_bpp <= 0.0:
            raise ValueError(f"--target-bpp must be positive, got {args.target_bpp}.")
        config.setdefault("transmitter", {})["target_bpp"] = target_bpp
        config.setdefault("training", {})["target_bpp"] = target_bpp
    if args.eval_fraction is not None:
        eval_fraction = float(args.eval_fraction)
        if not math.isfinite(eval_fraction) or eval_fraction <= 0.0:
            raise ValueError(f"--eval-fraction must be positive, got {args.eval_fraction}.")
        config.setdefault("subset", {})["eval_fraction"] = min(1.0, eval_fraction)
    if bool(getattr(args, "enable_carrier_bank", False)):
        carrier_cfg = config.setdefault("carrier_bank", {})
        carrier_cfg["enabled"] = True
        carrier_cfg["dataset"] = str(getattr(args, "carrier_bank_dataset", "div2k") or "div2k").strip().lower()
        explicit_dirs = [
            item.strip()
            for item in str(getattr(args, "carrier_bank_dirs", "") or "").split(",")
            if item.strip()
        ]
        if explicit_dirs:
            carrier_cfg["dirs"] = explicit_dirs
        elif not carrier_cfg.get("dirs"):
            fallback_dataset_name = str(carrier_cfg.get("dataset", "") or "div2k").strip().lower()
            if fallback_dataset_name == "landscape":
                carrier_cfg["dirs"] = ["dataset/Landscape"]
            else:
                carrier_cfg["dirs"] = ["dataset/DIV2K/DIV2K_train_HR"]
        carrier_cfg["random_train"] = False
        carrier_cfg["external_blend"] = float(max(0.0, min(1.0, getattr(args, "carrier_bank_blend", 1.0))))
        carrier_cfg["forward_into_generator"] = bool(
            carrier_cfg.get("forward_into_generator", True)
        )
        config.setdefault("gan", {})
        config["gan"]["external_carrier_blend"] = carrier_cfg["external_blend"]
        config["gan"]["external_carrier_lowfreq_only"] = True
        config["gan"]["preserve_source_bridge_with_external_carrier"] = True
    if bool(getattr(args, "enable_robust_qim", False)):
        config.setdefault("robust_qim", {})
        config["robust_qim"].update(
            {
                "enabled": True,
                "domain": "dct",
                "dct_channels": "y",
                "dct_coefficients": "auto56",
                "dct_quality": 50,
                "dct_quant_scale": 1.0,
                "dct_parity_mode": True,
                "delta": 1.0,
                "strength": 1.0,
                "llr_scale": max(36.0, float(config["robust_qim"].get("llr_scale", 36.0))),
                "llr_clamp": max(32.0, float(config["robust_qim"].get("llr_clamp", 32.0))),
                "position_mode": "stratified",
                "repetition_factor": 1,
                "dither_enabled": False,
                "dither_strength": 0.0,
                "clean_fusion_weight": 0.0,
                "attack_fusion_weight": 1.0,
            }
        )
    apply_eval_qim_profile(config)
    checkpoint_config = load_checkpoint_config(args.checkpoint) if args.checkpoint else None
    if checkpoint_config is not None:
        config = merge_checkpoint_runtime_config(
            config,
            checkpoint_config,
            preserve_checkpoint_payload_structure=args.target_bpp is None,
        )
        checkpoint_eval_support = checkpoint_supports_runtime_eval_modules(checkpoint_config, args.checkpoint)
        apply_low_payload_eval_defaults(
            args,
            config,
            checkpoint_config=checkpoint_config,
            argv_tokens=raw_argv_tokens,
        )
        if bool(getattr(args, "enable_carrier_bank", False)) and not checkpoint_eval_support["carrier_bank"]:
            args.enable_carrier_bank = False
            print(
                "[eval] Disabled carrier-bank runtime injection because the checkpoint lacks carrier-bank/native-carrier weights.",
                flush=True,
            )
        if bool(getattr(args, "enable_robust_qim", False)) and not checkpoint_eval_support["robust_qim"]:
            args.enable_robust_qim = False
            print(
                "[eval] Disabled robust-QIM runtime injection because the checkpoint lacks robust-QIM/comm-decoder weights.",
                flush=True,
            )
        if not checkpoint_eval_support["stego_naturalizer"]:
            config.pop("stego_naturalizer", None)
        apply_eval_qim_profile(config)
        if bool(getattr(args, "enable_carrier_bank", False)):
            carrier_cfg = config.setdefault("carrier_bank", {})
            carrier_cfg["enabled"] = True
            carrier_cfg["dataset"] = str(getattr(args, "carrier_bank_dataset", "div2k") or "div2k").strip().lower()
            explicit_dirs = [
                item.strip()
                for item in str(getattr(args, "carrier_bank_dirs", "") or "").split(",")
                if item.strip()
            ]
            if explicit_dirs:
                carrier_cfg["dirs"] = explicit_dirs
            elif not carrier_cfg.get("dirs"):
                fallback_dataset_name = str(carrier_cfg.get("dataset", "") or "div2k").strip().lower()
                if fallback_dataset_name == "landscape":
                    carrier_cfg["dirs"] = ["dataset/Landscape"]
                else:
                    carrier_cfg["dirs"] = ["dataset/DIV2K/DIV2K_train_HR"]
            carrier_cfg["random_train"] = False
            carrier_cfg["external_blend"] = float(max(0.0, min(1.0, getattr(args, "carrier_bank_blend", 1.0))))
            carrier_cfg["forward_into_generator"] = bool(carrier_cfg.get("forward_into_generator", True))
            config.setdefault("gan", {})
            config["gan"]["external_carrier_blend"] = carrier_cfg["external_blend"]
            config["gan"]["external_carrier_lowfreq_only"] = True
            config["gan"]["preserve_source_bridge_with_external_carrier"] = True
            if not config["gan"].get("analog_injection_target"):
                config["gan"]["analog_injection_target"] = "cover"
            if config["gan"].get("source_bridge_strength") is None or float(
                config["gan"].get("source_bridge_strength", 0.0)
            ) <= 0.0:
                config["gan"]["source_bridge_strength"] = 1.0
        if (
            str(getattr(args, "latent_select_score", "")).strip().lower() == "cover_brisque_proxy"
            and bool(config.get("carrier_bank", {}).get("enabled", False))
        ):
            args.latent_select_score = "carrier_brisque_proxy"
        base_qim_cfg = load_config(args.config).get("qim", {})
        if bool(getattr(args, "force_current_qim_structure", False)) and isinstance(base_qim_cfg, dict) and base_qim_cfg:
            config.setdefault("qim", {})
            for key in (
                "carrier_bands",
                "prefer_hh_only",
                "domain",
                "dct_channels",
                "dct_coefficients",
                "dct_quality",
                "dct_quant_scale",
                "dct_parity_mode",
                "adaptive_strength",
                "texture_floor",
                "texture_sharpness",
                "texture_kernel",
                "texture_abs_threshold",
                "repetition_factor",
                "position_mode",
                "position_seed",
                "strength",
                "llr_scale",
                "llr_clamp",
                "dither_enabled",
                "dither_strength",
            ):
                if key in base_qim_cfg:
                    config["qim"][key] = base_qim_cfg[key]
    if args.data is None and args.split is None and args.eval_fraction is None:
        config.setdefault("subset", {})
        # 默认离线评估直接复用完整验证集，避免在验证集上再次抽样，导致与训练日志口径不一致。
        config["subset"]["eval_fraction"] = None
    if args.clean_eval:
        disable_eval_robust_channel(config)
    set_seed(config.get("seed", 42))
    device = resolve_device(
        args.device if args.device is not None else config.get("device"),
        preferred_index=config.get("device_index", 1),
    )
    config["device"] = str(device)
    config["device_index"] = device.index if device.type == "cuda" else None
    if device.type == "cuda":
        print(f"Using CUDA device: {device} - {torch.cuda.get_device_name(device)}")
    dataloader = build_eval_loader(
        config,
        image_dir=args.data,
        split_file=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = WaveBridegeSystem(config).to(device)
    if args.checkpoint:
        load_model_checkpoint(model, args.checkpoint)
    apply_qim_profile_to_model(model)
    loaded_prior_path = load_prior_generator_weights_for_eval(model, args.latent_select_prior_checkpoint)
    payload_target_bpp = finite_float_or_none(config.get("transmitter", {}).get("target_bpp"))
    if payload_target_bpp is None or payload_target_bpp <= 0.0:
        payload_target_bpp = finite_float_or_none(config.get("training", {}).get("target_bpp"))
    if payload_target_bpp is None or payload_target_bpp <= 0.0:
        payload_target_bpp = finite_float_or_none(config.get("transmitter", {}).get("payload_bpp"))
    if payload_target_bpp is None or payload_target_bpp <= 0.0:
        payload_target_bpp = DEFAULT_TARGET_PAYLOAD_BPP
    latent_prior_mixes = parse_prior_mix_candidates(args.latent_select_prior_mixes, float(model.generator.prior_mix))
    print("[EVAL] Stage 1/5: collecting clean outputs and replay BER metrics...", flush=True)
    tensors = collect_outputs(
        model,
        dataloader,
        device,
        args.max_images,
        full_decode=args.full_decode,
        latent_select_candidates=args.latent_select_candidates,
        latent_select_prior_mixes=latent_prior_mixes,
        latent_select_psnr_drop=args.latent_select_psnr_drop,
        latent_select_ssim_drop=args.latent_select_ssim_drop,
        latent_select_max_ber_info=args.latent_select_max_ber_info,
        latent_select_max_ber_code=args.latent_select_max_ber_code,
        latent_select_decoded_floor=args.latent_select_decoded_floor,
        latent_select_score=args.latent_select_score,
        latent_select_seed=config.get("seed", 42),
        latent_select_min_psnr=args.latent_select_min_psnr,
        latent_select_min_ssim=args.latent_select_min_ssim,
        progress_interval=args.progress_interval,
        compute_channel_ber=not args.skip_channel_ber,
    )
    original = [image.to(device) for image in tensors["original"]]
    carrier = [image.to(device) for image in tensors["carrier"]]
    cover = [image.to(device) for image in tensors["cover"]]
    decoded = [image.to(device) for image in tensors["decoded"]]
    restored = [image.to(device) for image in tensors["restored"]]
    active_dataset_cfg = get_active_dataset_config(config)
    natural_reference_dataset = str(
        getattr(args, "natural_reference_dataset", None)
        or config.get("carrier_bank", {}).get("dataset")
        or config.get("datasets", {}).get("active", "eval")
    ).strip().lower()
    natural_reference_paths, natural_reference_dataset = resolve_natural_reference_dirs(
        config,
        explicit_dirs=getattr(args, "natural_reference_dirs", None),
        dataset_override=natural_reference_dataset,
    )
    sample_shape = flatten_image_batches(cover)[0].shape if flatten_image_batches(cover) else torch.Size([config["data"]["channels"], config["data"]["image_size"], config["data"]["image_size"]])
    natural_reference = load_natural_reference_images(
        natural_reference_paths,
        image_size=(int(sample_shape[-2]), int(sample_shape[-1])),
        channels=int(sample_shape[0]),
        needed=max(1, len(flatten_image_batches(cover))),
    )
    if not natural_reference:
        natural_reference = flatten_image_batches(carrier)
        natural_reference_dataset = f"{natural_reference_dataset or 'eval'}_carrier_fallback"
    detector_limit = max(8, int(getattr(args, "detector_max_images", 256)))
    detector_size = int(getattr(args, "detector_size", 64))
    detector_carrier = comparison_detector_prepare(natural_reference, size=detector_size)[:detector_limit]
    detector_cover = comparison_detector_prepare(cover, size=detector_size)[:detector_limit]

    lpips_model = build_lpips_model(device)
    with torch.inference_mode():
        if lpips_model is not None:
            lpips_values = [
                lpips_model(src * 2.0 - 1.0, rec * 2.0 - 1.0)
                for src, rec in zip(original, restored)
            ]
            lpips_restored = torch.cat(lpips_values, dim=0).mean().item()
        else:
            lpips_restored = float("nan")

    attack_psnr_sum = 0.0
    attack_ssim_sum = 0.0
    attack_lpips_sum = 0.0
    attack_ber_info_sum = 0.0
    attack_ber_code_sum = 0.0
    attack_decoded_ratio_sum = 0.0
    attack_seen = 0
    if not args.skip_attack_eval:
        print("[EVAL] Stage 2/5: running jpeg75+gaussian0.01 attack replay...", flush=True)
        model.eval()
        with torch.inference_mode():
            for batch_index, images in enumerate(dataloader, start=1):
                for micro_images in iter_image_microbatches(images):
                    if args.max_images is not None and attack_seen >= args.max_images:
                        break
                    if args.max_images is not None:
                        remaining = int(args.max_images) - attack_seen
                        if remaining <= 0:
                            break
                        micro_images = micro_images[:remaining]
                    micro_images = micro_images.to(device)
                    model_output, _selection_metrics = select_output_by_latent_candidates(
                        model,
                        micro_images,
                        full_decode=args.full_decode,
                        candidates=args.latent_select_candidates,
                        prior_mix_candidates=latent_prior_mixes,
                        psnr_drop=args.latent_select_psnr_drop,
                        ssim_drop=args.latent_select_ssim_drop,
                        max_ber_info=args.latent_select_max_ber_info,
                        max_ber_code=args.latent_select_max_ber_code,
                        decoded_floor=args.latent_select_decoded_floor,
                        score_name=args.latent_select_score,
                        seed=latent_selection_image_seed(config.get("seed", 42), attack_seen),
                        min_psnr=args.latent_select_min_psnr,
                        min_ssim=args.latent_select_min_ssim,
                    )
                    attacked_cover = jpeg75_gaussian_attack_tensor(
                        model_output.generator.cover_image,
                        quality=75,
                        noise_std=0.01,
                        seed=config.get("seed", 42) + attack_seen + batch_index,
                    )
                    attack_metrics = robust_quality_metrics_from_output(
                        model,
                        model_output,
                        attacked_cover,
                        micro_images,
                        full_decode=args.full_decode,
                        lpips_model=lpips_model,
                    )
                    batch_count = int(micro_images.shape[0])
                    attack_psnr_sum += attack_metrics["psnr_attack"] * batch_count
                    attack_ssim_sum += attack_metrics["ssim_attack"] * batch_count
                    if math.isfinite(attack_metrics["lpips_attack"]):
                        attack_lpips_sum += attack_metrics["lpips_attack"] * batch_count
                    attack_ber_info_sum += attack_metrics["ber_info_attack"] * batch_count
                    attack_ber_code_sum += attack_metrics["ber_code_attack"] * batch_count
                    attack_decoded_ratio_sum += attack_metrics["decoded_ratio_attack"] * batch_count
                    attack_seen += batch_count
                    if args.progress_interval > 0 and attack_seen % int(args.progress_interval) == 0:
                        total = "unknown" if args.max_images is None else str(args.max_images)
                        print(f"[EVAL] attacked {attack_seen}/{total} image(s)", flush=True)
                if args.max_images is not None and attack_seen >= args.max_images:
                    break

    print("[EVAL] Stage 3/5: computing naturalness metrics (FID/KID/BRISQUE)...", flush=True)
    fid_cover, fid_carrier, kid_stego, kid_carrier, fid_backend = calculate_fid_kid_metrics(
        natural_reference,
        carrier,
        cover,
        device=device,
        backend=args.fid_backend,
        clean_mode=args.clean_fid_mode,
    )
    brisque_real, brisque_stego, brisque_gap_to_real, brisque_backend = calculate_brisque_gap_metrics(
        natural_reference,
        cover,
    )
    print("[EVAL] Stage 4/5: computing statistical detector and SRNet metrics...", flush=True)
    stat_metrics = statistical_detection_metrics(
        detector_carrier,
        detector_cover,
        seed=config.get("seed", 42),
        backend=args.stat_backend,
        test_ratio=args.detector_test_ratio,
    )
    srnet_metrics = srnet_detection_metrics(
        detector_carrier,
        detector_cover,
        device=device,
        weights_path=args.srnet_weights,
        epochs=args.srnet_epochs,
        batch_size=args.srnet_batch_size or args.batch_size,
        seed=config.get("seed", 42),
        arch=args.srnet_arch,
        lr=args.srnet_lr,
        test_ratio=args.detector_test_ratio,
        save_weights_path=args.srnet_save_weights,
    )
    info_length = config["transmitter"]["polar_code"]["info_length"]
    channels = config["data"]["channels"]
    configured_image_size_value = int(config.get("data", {}).get("image_size", 256) or 256)
    official_srnet_metrics = {
        "detection_rate": float("nan"),
        "anti_detection_rate": float("nan"),
        "test_loss": float("nan"),
        "backend": "official_srnet_disabled",
    }
    if bool(getattr(args, "use_official_srnet", False)):
        comparison_root = resolve_comparison_root(getattr(args, "comparison_root", None))
        if comparison_root is None:
            print("Warning: Comparison root not found, official SRNet evaluation will be skipped.")
            official_srnet_metrics["backend"] = "official_srnet_root_missing"
        else:
            official_srnet_metrics = run_official_srnet_evaluation(
                carrier=carrier,
                cover=cover,
                output_root=Path(args.output_dir),
                comparison_root=comparison_root,
                image_size=max(64, configured_image_size_value or 256),
                device=device,
                split_seed=config.get("seed", 42),
                python_executable=getattr(args, "official_srnet_python", None),
                max_iter=int(getattr(args, "official_srnet_max_iter", 2000)),
                force_cpu=bool(getattr(args, "official_srnet_force_cpu", False)),
            )
    stat_detect = stat_metrics["detection_rate"]
    srnet_detect = srnet_metrics["detection_rate"]
    stat_detection_advantage = 2.0 * abs(stat_detect - 0.5)
    srnet_detection_advantage = 2.0 * abs(srnet_detect - 0.5)
    mean_detection_advantage = 0.5 * (stat_detection_advantage + srnet_detection_advantage)
    anti_detection_rate = max(0.0, 1.0 - max(float(stat_detect), float(srnet_detect)))
    decoded_psnr_values = torch.cat([batch_psnr(src, rec) for src, rec in zip(original, decoded)], dim=0)
    decoded_ssim_values = torch.cat([batch_ssim(src, rec) for src, rec in zip(original, decoded)], dim=0)
    psnr_values = torch.cat([batch_psnr(src, rec) for src, rec in zip(original, restored)], dim=0)
    ssim_values = torch.cat([batch_ssim(src, rec) for src, rec in zip(original, restored)], dim=0)
    ssim_y_values = torch.cat([batch_ssim_y(src, rec) for src, rec in zip(original, restored)], dim=0)
    mae_values = torch.cat([batch_mae(src, rec) for src, rec in zip(original, restored)], dim=0)
    rmse_values = torch.cat([batch_rmse(src, rec) for src, rec in zip(original, restored)], dim=0)
    stego_psnr_values = torch.cat([batch_psnr(src, stego) for src, stego in zip(original, cover)], dim=0)
    stego_ssim_values = torch.cat([batch_ssim(src, stego) for src, stego in zip(original, cover)], dim=0)
    stego_carrier_psnr_values = torch.cat([batch_psnr(clean, stego) for clean, stego in zip(carrier, cover)], dim=0)
    stego_carrier_ssim_values = torch.cat([batch_ssim(clean, stego) for clean, stego in zip(carrier, cover)], dim=0)
    stego_carrier_ssim_y_values = torch.cat(
        [batch_ssim_y(clean, stego) for clean, stego in zip(carrier, cover)],
        dim=0,
    )
    stego_carrier_mae_values = torch.cat([batch_mae(clean, stego) for clean, stego in zip(carrier, cover)], dim=0)
    stego_carrier_rmse_values = torch.cat([batch_rmse(clean, stego) for clean, stego in zip(carrier, cover)], dim=0)
    if lpips_model is not None:
        with torch.inference_mode():
            stego_carrier_lpips_values = [
                lpips_model(clean * 2.0 - 1.0, stego * 2.0 - 1.0)
                for clean, stego in zip(carrier, cover)
            ]
            stego_carrier_lpips = torch.cat(stego_carrier_lpips_values, dim=0).mean().item()
    else:
        stego_carrier_lpips = float("nan")
    if "valid_info_bits" in tensors:
        valid_bits = tensors["valid_info_bits"]
        bpp_values = [
            float(valid_bits[i].item()) / float(int(size[0]) * int(size[1]))
            for i, size in enumerate(tensors["image_sizes"])
        ]
    else:
        bpp_values = [payload_bpp(info_length, int(size[0]), int(size[1]), channels) for size in tensors["image_sizes"]]
    first_image_size = tensors["image_sizes"][0] if tensors["image_sizes"] else (0, 0)
    image_size_value = int(first_image_size[0]) if int(first_image_size[0]) == int(first_image_size[1]) else 0
    result = EvaluationResult(
        psnr_decoded=float(decoded_psnr_values.mean().item()),
        ssim_decoded=float(decoded_ssim_values.mean().item()),
        psnr_restored=float(psnr_values.mean().item()),
        ssim_restored=float(ssim_values.mean().item()),
        ssim_restored_y=float(ssim_y_values.mean().item()),
        mae_restored=float(mae_values.mean().item()),
        rmse_restored=float(rmse_values.mean().item()),
        stego_psnr_to_source=float(stego_psnr_values.mean().item()),
        stego_ssim_to_source=float(stego_ssim_values.mean().item()),
        stego_psnr_to_carrier=float(stego_carrier_psnr_values.mean().item()),
        stego_ssim_to_carrier=float(stego_carrier_ssim_values.mean().item()),
        stego_ssim_to_carrier_y=float(stego_carrier_ssim_y_values.mean().item()),
        fid_cover=float(fid_cover),
        fid_carrier=float(fid_carrier),
        kid_stego=float(kid_stego),
        kid_carrier=float(kid_carrier),
        brisque_real=float(brisque_real),
        brisque_stego=float(brisque_stego),
        brisque_gap_to_real=float(brisque_gap_to_real),
        lpips_restored=float(lpips_restored),
        statistical_detection_rate=float(stat_detect),
        statistical_stego_detection_rate=float(stat_metrics["stego_detection_rate"]),
        statistical_false_positive_rate=float(stat_metrics["false_positive_rate"]),
        statistical_detection_advantage=float(stat_detection_advantage),
        statistical_anti_detection_rate=float(1.0 - stat_detect),
        srnet_detection_rate=float(srnet_detect),
        srnet_balanced_accuracy=float(srnet_metrics.get("balanced_accuracy", srnet_metrics["detection_rate"])),
        srnet_auc=float(srnet_metrics.get("auc", float("nan"))),
        srnet_eer=float(srnet_metrics.get("eer", float("nan"))),
        srnet_stego_detection_rate=float(srnet_metrics["stego_detection_rate"]),
        srnet_false_positive_rate=float(srnet_metrics["false_positive_rate"]),
        srnet_detection_advantage=float(srnet_detection_advantage),
        srnet_anti_detection_rate=float(1.0 - srnet_detect),
        official_srnet_detection_rate=float(official_srnet_metrics.get("detection_rate", float("nan"))),
        official_srnet_anti_detection_rate=float(official_srnet_metrics.get("anti_detection_rate", float("nan"))),
        official_srnet_test_loss=float(official_srnet_metrics.get("test_loss", float("nan"))),
        official_srnet_backend=str(official_srnet_metrics.get("backend", "official_srnet_disabled")),
        mean_detection_advantage=float(mean_detection_advantage),
        anti_detection_rate=float(anti_detection_rate),
        payload_bpp=float(np.mean(bpp_values)),
        payload_target_bpp=float(payload_target_bpp),
        payload_bpp_tolerance=float(DEFAULT_PAYLOAD_BPP_TOLERANCE),
        ber_info=float(tensors["ber_info"].item()),
        ber_code=float(tensors["ber_code"].item()),
        jpeg50_ber_info=float(tensors["jpeg50_ber_info"].item()),
        jpeg50_ber_code=float(tensors["jpeg50_ber_code"].item()),
        mixed_ber_info=float(tensors["mixed_ber_info"].item()),
        mixed_ber_code=float(tensors["mixed_ber_code"].item()),
        attack_psnr_restored=(
            float(attack_psnr_sum / max(1, attack_seen))
            if not args.skip_attack_eval
            else float("nan")
        ),
        attack_ssim_restored=(
            float(attack_ssim_sum / max(1, attack_seen))
            if not args.skip_attack_eval
            else float("nan")
        ),
        attack_lpips_restored=(
            float(attack_lpips_sum / max(1, attack_seen))
            if lpips_model is not None and not args.skip_attack_eval
            else float("nan")
        ),
        attack_ber_info=(
            float(attack_ber_info_sum / max(1, attack_seen))
            if not args.skip_attack_eval
            else float("nan")
        ),
        attack_ber_code=(
            float(attack_ber_code_sum / max(1, attack_seen))
            if not args.skip_attack_eval
            else float("nan")
        ),
        attack_decoded_ratio=(
            float(attack_decoded_ratio_sum / max(1, attack_seen))
            if not args.skip_attack_eval
            else float("nan")
        ),
        decoded_blocks=float(tensors["decoded_block_counts"].sum().item()),
        total_blocks=float(tensors["total_block_counts"].sum().item()),
        decoded_block_ratio=float(
            tensors["decoded_block_counts"].sum().item()
            / max(1.0, tensors["total_block_counts"].sum().item())
        ),
        decoded_groups=float(tensors["decoded_group_counts"].sum().item()),
        total_groups=float(tensors["total_group_counts"].sum().item()),
        decoded_group_ratio=float(
            tensors["decoded_group_counts"].sum().item()
            / max(1.0, tensors["total_group_counts"].sum().item())
        ),
        decoded_ratio=float(
            tensors["decoded_block_counts"].sum().item()
            / max(1.0, tensors["total_block_counts"].sum().item())
        ),
        num_images=int(tensors["num_images"].item()),
        detector_samples=int(min(detector_carrier.shape[0], detector_cover.shape[0])),
        fid_backend=str(fid_backend),
        statistical_detector=str(stat_metrics.get("backend", args.stat_backend)),
        srnet_detector=str(srnet_metrics.get("backend", args.srnet_arch)),
        detector_size=detector_size,
        full_decode=bool(args.full_decode),
        max_images=None if args.max_images is None else int(args.max_images),
        checkpoint_path=None if args.checkpoint is None else str(args.checkpoint),
        data_path=None if args.data is None else str(args.data),
        split_path=None if args.split is None else str(args.split),
        detector_test_ratio=float(args.detector_test_ratio),
        srnet_epochs=int(args.srnet_epochs),
        srnet_arch=str(args.srnet_arch),
        srnet_weights=None if args.srnet_weights is None else str(args.srnet_weights),
        srnet_batch_size=None if args.srnet_batch_size is None else int(args.srnet_batch_size),
        fid_requested_backend=str(args.fid_backend),
        clean_fid_mode=str(args.clean_fid_mode),
        brisque_backend=str(brisque_backend),
        stat_requested_backend=str(args.stat_backend),
        active_dataset=str(config.get("datasets", {}).get("active", "eval")),
        image_size=image_size_value,
        clean_eval=bool(args.clean_eval),
        latent_select_candidates=int(tensors["latent_select_candidates"].item()),
        latent_select_used_ratio=float(tensors["latent_select_used_ratio"].item()),
        latent_select_avg_index=float(tensors["latent_select_avg_index"].item()),
        latent_select_score=str(tensors["latent_select_score"]),
        latent_select_prior_checkpoint=loaded_prior_path,
        latent_select_prior_mixes=str(tensors["latent_select_prior_mixes"]),
        latent_select_min_psnr=float(args.latent_select_min_psnr),
        latent_select_min_ssim=float(args.latent_select_min_ssim),
        comparison_carrier_psnr=float(stego_carrier_psnr_values.mean().item()),
        comparison_carrier_ssim=float(stego_carrier_ssim_values.mean().item()),
        comparison_carrier_lpips=float(stego_carrier_lpips),
        comparison_carrier_mae=float(stego_carrier_mae_values.mean().item()),
        comparison_carrier_rmse=float(stego_carrier_rmse_values.mean().item()),
        comparison_recovery_psnr=float(psnr_values.mean().item()),
        comparison_recovery_ssim=float(ssim_values.mean().item()),
        stego_secret_distinct_ok=bool(stego_secret_distinct_ratio(original, cover) >= 0.999999),
        stego_secret_distinct_ratio=float(stego_secret_distinct_ratio(original, cover)),
        natural_reference_dataset=natural_reference_dataset,
        natural_reference_dirs=",".join(str(path.parent) for path in natural_reference_paths[:8]),
        natural_reference_count=int(len(natural_reference_paths)),
        comparison_lpips_or_ber=(
            f"LPIPS {float(lpips_restored):.3f} / BER {float(tensors['ber_info'].item()):.3f}"
            if math.isfinite(float(lpips_restored))
            else f"BER {float(tensors['ber_info'].item()):.3f}"
        ),
        comparison_protocol="naturalness=real_natural_set_vs_stego,detection=real_natural_set_vs_stego,recovery=original_vs_restored,carrier=generated_image_vs_stego_image,ssim=rgb",
    )
    print("[EVAL] Stage 5/5: writing result files...", flush=True)
    save_results(result, args.output_dir)
    return result


# 解析评估脚本的命令行参数。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WaveBridege metrics.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument(
        "--active-dataset",
        default=None,
        choices=["div2k", "bossbase", "alaska2", "landscape"],
        help="Override datasets.active for CCF-A multi-dataset evaluation.",
    )
    parser.add_argument("--data", default=None, help="Optional override flat image directory")
    parser.add_argument("--split", default=None, help="Optional override evaluation split file")
    parser.add_argument("--checkpoint", default="checkpoints/wavebridege_div2k_hh_chain_v2_stage2_metric_refine.pt", help="Model checkpoint")
    parser.add_argument("--output-dir", default="eval_results/div2k", help="Directory for metric outputs")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--max-images", type=int, default=1250, help="Optional max number of images")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="Print progress every N evaluated images; set 0 to disable progress logs.",
    )
    parser.add_argument("--skip-channel-ber", action="store_true", help="Skip JPEG50/Mixed BER replay in non-robust suite phases")
    parser.add_argument("--skip-attack-eval", action="store_true", help="Skip JPEG75+Gaussian attack replay in non-robust suite phases")
    parser.add_argument("--eval-fraction", type=float, default=None, help="Optional eval subset fraction for the active dataset")
    parser.add_argument("--target-bpp", type=float, default=None, help="Override transmitter.target_bpp for payload-sweep evaluation")
    parser.add_argument("--device", default=None, help="Override device, such as cuda or cpu")
    parser.add_argument(
        "--fid-backend",
        default="clean",
        choices=["auto", "clean", "legacy"],
        help="FID backend. Default uses clean-fid for stricter evaluation; 'legacy' uses the built-in Inception implementation.",
    )
    parser.add_argument(
        "--clean-fid-mode",
        default="clean",
        help="Mode passed to clean-fid when that backend is active.",
    )
    parser.add_argument(
        "--stat-backend",
        default="srm",
        choices=["auto", "srm", "legacy"],
        help="Traditional detector backend. Default uses SRM-like residual features plus a linear SVM for stricter evaluation.",
    )
    parser.add_argument(
        "--detector-test-ratio",
        type=float,
        default=0.3,
        help="Held-out ratio for statistical detector and SRNet evaluation.",
    )
    parser.add_argument("--detector-max-images", type=int, default=256, help="Max cover/stego pairs used by unified detectors")
    parser.add_argument("--detector-size", type=int, default=64, help="Detector input size aligned to Comparison unified eval")
    parser.add_argument("--srnet-weights", default=None, help="Optional pretrained SRNet weights")
    parser.add_argument(
        "--srnet-epochs",
        type=int,
        default=3,
        help="SRNet training epochs if no pretrained weights are provided; default matches Comparison unified eval.",
    )
    parser.add_argument(
        "--srnet-arch",
        default="deep",
        choices=["deep", "lite"],
        help="SRNet-style detector architecture. 'deep' is closer to a paper-style detector, 'lite' is faster.",
    )
    parser.add_argument("--srnet-batch-size", type=int, default=None, help="Optional SRNet-specific batch size override")
    parser.add_argument("--srnet-lr", type=float, default=1e-4, help="SRNet training learning rate when training from scratch")
    parser.add_argument("--srnet-save-weights", default=None, help="Optional path to save the trained SRNet detector weights")
    parser.add_argument("--use-official-srnet", action="store_true", help="Run official SRNet from Comparison if available.")
    parser.add_argument("--comparison-root", default=None, help="Optional Comparison root for official SRNet alignment.")
    parser.add_argument("--official-srnet-python", default=None, help="Python executable used to run Comparison official SRNet.")
    parser.add_argument(
        "--official-srnet-max-iter",
        type=int,
        default=2000,
        help="Max iterations used by Comparison official SRNet training when no cached detector is supplied.",
    )
    parser.add_argument(
        "--official-srnet-force-cpu",
        action="store_true",
        help="Force Comparison official SRNet to run on CPU to avoid old TensorFlow GPU incompatibilities.",
    )
    parser.add_argument("--clean-eval", dest="clean_eval", action="store_true", help="Disable robust eval perturbations for clean PSNR/SSIM/BER metrics.")
    parser.add_argument("--robust-eval", dest="clean_eval", action="store_false", help="Keep robust_channel eval perturbations active for debugging.")
    parser.add_argument(
        "--enable-carrier-bank",
        action="store_true",
        dest="enable_carrier_bank",
        help="Use a deterministic natural-image carrier bank during evaluation instead of the checkpoint-only prior carrier.",
    )
    parser.add_argument(
        "--disable-carrier-bank",
        action="store_false",
        dest="enable_carrier_bank",
        help="Explicitly disable the carrier bank during evaluation.",
    )
    parser.add_argument("--carrier-bank-dataset", default="div2k", help="Carrier-bank dataset name used when --enable-carrier-bank is set.")
    parser.add_argument(
        "--carrier-bank-dirs",
        default="",
        help="Comma-separated carrier-bank image directories. Empty uses the dataset defaults from the config.",
    )
    parser.add_argument(
        "--carrier-bank-blend",
        type=float,
        default=1.0,
        help="Blend weight for the external natural carrier; 1.0 uses the sampled carrier image directly.",
    )
    parser.add_argument(
        "--natural-reference-dataset",
        default=None,
        help="Dataset name used as the real natural reference set for FID/KID/BRISQUE and steganalysis. Default prefers carrier_bank.dataset, then datasets.active.",
    )
    parser.add_argument(
        "--natural-reference-dirs",
        default="",
        help="Comma-separated real natural image directories used as the naturalness/detection reference set. Empty falls back to the resolved dataset directories.",
    )
    parser.add_argument(
        "--enable-robust-qim",
        action="store_true",
        dest="enable_robust_qim",
        help="Enable the runtime robust DCT-QIM auxiliary channel for JPEG/Mixed BER evaluation.",
    )
    parser.add_argument(
        "--disable-robust-qim",
        action="store_false",
        dest="enable_robust_qim",
        help="Explicitly disable the runtime robust DCT-QIM auxiliary channel during evaluation.",
    )
    parser.add_argument(
        "--force-current-qim-structure",
        action="store_true",
        help="Override checkpoint QIM layout with current config; default preserves checkpoint-native QIM.",
    )
    parser.add_argument("--full-decode", dest="full_decode", action="store_true", help="Decode every polar block for final restoration metrics")
    parser.add_argument("--proxy-decode", dest="full_decode", action="store_false", help="Use the faster partial decode path for debugging.")
    parser.add_argument(
        "--latent-select-candidates",
        type=int,
        default=4,
        help="Number of latent candidates sampled per image for safe naturalness re-ranking. Default uses a stronger final-eval setting.",
    )
    parser.add_argument(
        "--latent-select-score",
        default="cover_brisque_proxy",
        choices=["brisque_proxy", "cover_brisque_proxy", "carrier_brisque_proxy"],
        help="Naturalness proxy used to rank latent candidates after recovery/BER gates pass.",
    )
    parser.add_argument(
        "--latent-select-prior-checkpoint",
        default=None,
        help="Optional checkpoint used only for generator.prior_generator during safe eval-time naturalness re-ranking.",
    )
    parser.add_argument(
        "--latent-select-prior-mixes",
        default="",
        help="Comma-separated prior_mix candidates for eval-time naturalness re-ranking, for example 0,0.001,0.002.",
    )
    parser.add_argument(
        "--latent-select-psnr-drop",
        type=float,
        default=0.03,
        help="Maximum allowed restored-PSNR drop against the first candidate during latent selection.",
    )
    parser.add_argument(
        "--latent-select-ssim-drop",
        type=float,
        default=0.0003,
        help="Maximum allowed restored-SSIM drop against the first candidate during latent selection.",
    )
    parser.add_argument(
        "--latent-select-max-ber-info",
        type=float,
        default=1e-4,
        help="Maximum clean payload BER allowed for a latent candidate.",
    )
    parser.add_argument(
        "--latent-select-max-ber-code",
        type=float,
        default=1e-3,
        help="Maximum clean code BER allowed for a latent candidate.",
    )
    parser.add_argument(
        "--latent-select-decoded-floor",
        type=float,
        default=0.999999,
        help="Minimum decoded block/group ratio required for latent candidate selection.",
    )
    parser.add_argument(
        "--latent-select-min-psnr",
        type=float,
        default=38.0,
        help="Absolute restored-PSNR floor used by safe latent selection when the baseline already meets it.",
    )
    parser.add_argument(
        "--latent-select-min-ssim",
        type=float,
        default=0.97,
        help="Absolute restored-SSIM floor used by safe latent selection when the baseline already meets it.",
    )
    parser.set_defaults(full_decode=True)
    parser.set_defaults(clean_eval=True)
    parser.set_defaults(enable_carrier_bank=False, enable_robust_qim=False)
    return parser.parse_args()


if __name__ == "__main__":
    metrics = evaluate(parse_args())
    print(json.dumps(evaluation_result_to_dict(metrics), indent=2, ensure_ascii=False))
