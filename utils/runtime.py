from __future__ import annotations

import math
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from utils.config import resolve_project_path

_CHECKPOINT_CONFIG_CACHE: dict[str, dict[str, Any] | None] = {}


def _checkpoint_shape_adaptation_enabled() -> bool:
    raw = str(os.environ.get("WAVEBRIDEGE_DISABLE_CHECKPOINT_SHAPE_ADAPTATION", "") or "").strip().lower()
    return raw not in {"1", "true", "yes", "on"}


# 检查 checkpoint 中是否存在 NaN 或 Inf，避免继续加载已经损坏的权重。
def _nonfinite_tensor_summary(state_dict: dict[str, Any], max_items: int = 8) -> list[str]:
    bad_items: list[str] = []
    for name, value in state_dict.items():
        if not torch.is_tensor(value) or not value.is_floating_point():
            continue
        finite_mask = torch.isfinite(value)
        if finite_mask.all():
            continue
        bad_count = int((~finite_mask).sum().item())
        bad_items.append(f"{name}:{bad_count}/{value.numel()}")
        if len(bad_items) >= max_items:
            break
    return bad_items


# 设置随机种子，尽量保证训练和推理过程可复现。
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 根据显卡数量选择安全的 CUDA 设备，并同步设置当前 CUDA 上下文。
def _select_cuda_device(preferred_index: int | None = 1) -> torch.device:
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        preferred_index = 1 if preferred_index is None else int(preferred_index)
        index = preferred_index if preferred_index < gpu_count else 0
        device = torch.device(f"cuda:{index}")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


# 解析命令行或配置中的设备写法，裸 cuda 默认绑定到第二张显卡。
def resolve_device(
    requested: str | int | torch.device | None = None,
    preferred_index: int = 1,
) -> torch.device:
    if isinstance(requested, int):
        return _select_cuda_device(requested)
    if isinstance(requested, torch.device):
        if requested.type == "cuda" and requested.index is None:
            return _select_cuda_device(preferred_index)
        if requested.type == "cuda" and torch.cuda.is_available():
            torch.cuda.set_device(requested)
        return requested
    if requested is None:
        return _select_cuda_device(preferred_index)

    normalized = str(requested).strip().lower()
    if normalized in {"cuda", "gpu"}:
        return _select_cuda_device(preferred_index)
    if normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        try:
            requested_index = int(normalized.split(":", maxsplit=1)[1])
        except ValueError:
            requested_index = preferred_index
        return _select_cuda_device(requested_index)
    return torch.device(normalized)


# 从 checkpoint 中读取保存时的配置，供断点续训时复用。
def clear_checkpoint_config_cache() -> None:
    _CHECKPOINT_CONFIG_CACHE.clear()


def load_checkpoint_config(checkpoint_path: str | Path) -> dict[str, Any] | None:
    resolved_path = resolve_project_path(checkpoint_path)
    if resolved_path is None:
        return None
    cache_key = str(resolved_path.resolve())
    if cache_key in _CHECKPOINT_CONFIG_CACHE:
        cached_config = _CHECKPOINT_CONFIG_CACHE[cache_key]
        return deepcopy(cached_config) if isinstance(cached_config, dict) else None
    checkpoint = torch.load(resolved_path, map_location="cpu")
    cached_config = deepcopy(checkpoint["config"]) if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict) else None
    _CHECKPOINT_CONFIG_CACHE[cache_key] = cached_config
    if isinstance(cached_config, dict):
        return deepcopy(cached_config)
    return None


# 从 checkpoint 中读取原始 state_dict，并在加载前统一做非有限值检查。
def _read_checkpoint_state_dict(
    checkpoint_path: str | Path,
    key: str = "model",
) -> tuple[Path, dict[str, torch.Tensor]]:
    resolved_path = resolve_project_path(checkpoint_path)
    if resolved_path is None or not resolved_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(resolved_path, map_location="cpu")
    state_dict = checkpoint[key] if isinstance(checkpoint, dict) and key in checkpoint else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint is not a state_dict-compatible file: {checkpoint_path}")
    bad_tensors = _nonfinite_tensor_summary(state_dict)
    if bad_tensors:
        raise ValueError(
            "Checkpoint contains non-finite tensors and is unsafe to load: "
            f"{resolved_path}. Examples: {', '.join(bad_tensors)}"
        )
    return resolved_path, state_dict


# 将旧版 post_image_refiner 的部分权重映射到当前结构，减少恢复链路断点续训时的随机初始化范围。
def _upgrade_legacy_state_dict_for_current_model(
    state_dict: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], int]:
    upgraded_state_dict = dict(state_dict)
    remap_count = 0

    def copy_if_compatible(source_key: str, target_key: str) -> None:
        nonlocal remap_count
        if source_key not in upgraded_state_dict or target_key in upgraded_state_dict or target_key not in model_state:
            return
        source_value = upgraded_state_dict[source_key]
        target_value = model_state[target_key]
        if source_value.shape != target_value.shape:
            return
        upgraded_state_dict[target_key] = source_value
        remap_count += 1

    def expand_first_conv_if_compatible(source_key: str, target_key: str) -> None:
        nonlocal remap_count
        if source_key not in upgraded_state_dict or target_key in upgraded_state_dict or target_key not in model_state:
            return
        source_value = upgraded_state_dict[source_key]
        target_value = model_state[target_key]
        if source_value.ndim != 4 or target_value.ndim != 4:
            return
        if source_value.shape[0] != target_value.shape[0] or source_value.shape[2:] != target_value.shape[2:]:
            return
        if source_value.shape[1] >= target_value.shape[1]:
            return
        expanded = target_value.detach().clone()
        expanded.zero_()
        expanded[:, : source_value.shape[1]] = source_value
        remaining_channels = target_value.shape[1] - source_value.shape[1]
        if remaining_channels > 0:
            seed_channels = max(1, min(source_value.shape[1], remaining_channels))
            tiled = source_value[:, :seed_channels].repeat(1, math.ceil(remaining_channels / seed_channels), 1, 1)
            expanded[:, source_value.shape[1] :] = tiled[:, :remaining_channels] * 0.5
        upgraded_state_dict[target_key] = expanded
        remap_count += 1

    expand_first_conv_if_compatible(
        "compressor.post_image_refiner.shallow.0.block.0.weight",
        "compressor.post_image_refiner.input_fusion.0.block.0.weight",
    )
    for suffix in ("weight", "bias"):
        copy_if_compatible(
            f"compressor.post_image_refiner.shallow.0.block.1.{suffix}",
            f"compressor.post_image_refiner.input_fusion.0.block.1.{suffix}",
        )
    legacy_to_current = {
        "compressor.post_image_refiner.shallow.1.body.0.block.0.weight":
            "compressor.post_image_refiner.input_fusion.1.body.0.block.0.weight",
        "compressor.post_image_refiner.shallow.1.body.0.block.1.weight":
            "compressor.post_image_refiner.input_fusion.1.body.0.block.1.weight",
        "compressor.post_image_refiner.shallow.1.body.0.block.1.bias":
            "compressor.post_image_refiner.input_fusion.1.body.0.block.1.bias",
        "compressor.post_image_refiner.shallow.1.body.1.weight":
            "compressor.post_image_refiner.input_fusion.1.body.1.weight",
        "compressor.post_image_refiner.shallow.1.body.2.weight":
            "compressor.post_image_refiner.input_fusion.1.body.2.weight",
        "compressor.post_image_refiner.shallow.1.body.2.bias":
            "compressor.post_image_refiner.input_fusion.1.body.2.bias",
    }
    for source_key, target_key in legacy_to_current.items():
        copy_if_compatible(source_key, target_key)
    return upgraded_state_dict, remap_count


# 针对信息长度变化导致的一维/二维权重不匹配，尽量保留前部语义并对剩余部分做温和缩放适配。
def _adapt_tensor_to_target_shape(
    source_value: Any,
    target_value: Any,
    *,
    normalized_name: str | None = None,
) -> torch.Tensor | Any | None:
    if not _checkpoint_shape_adaptation_enabled():
        return None
    if not (torch.is_tensor(source_value) and torch.is_tensor(target_value)):
        return None
    if source_value.shape == target_value.shape:
        return source_value.detach().clone()
    if source_value.dtype != target_value.dtype:
        source_value = source_value.to(dtype=target_value.dtype)
    if source_value.device != target_value.device:
        source_value = source_value.to(device=target_value.device)
    adapted = target_value.detach().clone()
    adapted.zero_()

    if source_value.ndim == 1 and target_value.ndim == 1:
        shared = min(source_value.shape[0], target_value.shape[0])
        if shared <= 0:
            return None
        adapted[:shared] = source_value[:shared]
        return adapted

    if source_value.ndim == 2 and target_value.ndim == 2:
        shared_rows = min(source_value.shape[0], target_value.shape[0])
        shared_cols = min(source_value.shape[1], target_value.shape[1])
        if shared_rows <= 0 or shared_cols <= 0:
            return None
        adapted[:shared_rows, :shared_cols] = source_value[:shared_rows, :shared_cols]
        remaining_cols = target_value.shape[1] - shared_cols
        if remaining_cols > 0 and shared_cols > 0:
            seed_cols = max(1, min(shared_cols, remaining_cols))
            tiled = source_value[:shared_rows, :seed_cols].repeat(1, math.ceil(remaining_cols / seed_cols))
            adapted[:shared_rows, shared_cols:] = tiled[:, :remaining_cols] * 0.5
        return adapted

    if (
        source_value.ndim == 4
        and target_value.ndim == 4
        and source_value.shape[0] == target_value.shape[0]
        and source_value.shape[2:] == target_value.shape[2:]
        and source_value.shape[1] < target_value.shape[1]
    ):
        adapted[:, : source_value.shape[1]] = source_value
        remaining_channels = target_value.shape[1] - source_value.shape[1]
        if remaining_channels > 0:
            seed_channels = max(1, min(source_value.shape[1], remaining_channels))
            tiled = source_value[:, :seed_channels].repeat(1, math.ceil(remaining_channels / seed_channels), 1, 1)
            adapted[:, source_value.shape[1] :] = tiled[:, :remaining_channels] * 0.5
        return adapted

    return None


# 从 checkpoint 中加载模型参数，兼容直接保存的 state_dict 或带键名的字典。
def load_model_checkpoint(model: nn.Module, checkpoint_path: str | Path, key: str = "model") -> None:
    resolved_path, state_dict = _read_checkpoint_state_dict(checkpoint_path, key=key)
    model_state = model.state_dict()
    state_dict, remap_count = _upgrade_legacy_state_dict_for_current_model(state_dict, model_state)
    filtered_state_dict: dict[str, torch.Tensor] = {}
    skipped_mismatch: list[str] = []
    adapted_mismatch: list[str] = []
    for name, value in state_dict.items():
        normalized_name = name[7:] if isinstance(name, str) and name.startswith("module.") else name
        target_name = normalized_name if normalized_name in model_state else name
        if target_name not in model_state:
            filtered_state_dict[name] = value
            continue
        target_value = model_state[target_name]
        if target_value.shape != value.shape:
            adapted_value = _adapt_tensor_to_target_shape(
                value,
                target_value,
                normalized_name=target_name,
            )
            if adapted_value is None:
                skipped_mismatch.append(target_name)
                continue
            filtered_state_dict[target_name] = adapted_value
            adapted_mismatch.append(target_name)
            continue
        filtered_state_dict[target_name] = value.detach().clone() if torch.is_tensor(value) else value
    incompatible = model.load_state_dict(filtered_state_dict, strict=False)
    if remap_count:
        print(
            "Checkpoint compatibility remap applied: "
            f"{remap_count} legacy post-refiner tensors adapted for the current model."
        )
    if adapted_mismatch:
        adapted_preview = ", ".join(adapted_mismatch[:5])
        print(
            "Checkpoint shape adaptation applied: "
            f"count={len(adapted_mismatch)} [{adapted_preview}]"
        )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        missing_preview = ", ".join(incompatible.missing_keys[:5])
        unexpected_preview = ", ".join(incompatible.unexpected_keys[:5])
        print(
            "Checkpoint loaded with non-strict key mismatch: "
            f"missing={len(incompatible.missing_keys)} [{missing_preview}] "
            f"unexpected={len(incompatible.unexpected_keys)} [{unexpected_preview}]"
        )
    if skipped_mismatch:
        skipped_preview = ", ".join(skipped_mismatch[:5])
        print(
            "Checkpoint skipped shape-mismatched tensors: "
            f"count={len(skipped_mismatch)} [{skipped_preview}]"
        )


# 只从 checkpoint 覆盖指定前缀的模块权重，便于保留通信链同时恢复高质量图像主干。
def load_partial_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    include_prefixes: tuple[str, ...] | list[str],
    *,
    exclude_prefixes: tuple[str, ...] | list[str] | None = None,
    key: str = "model",
) -> dict[str, int | str]:
    resolved_path, raw_state_dict = _read_checkpoint_state_dict(checkpoint_path, key=key)
    model_state = model.state_dict()
    state_dict, remap_count = _upgrade_legacy_state_dict_for_current_model(raw_state_dict, model_state)
    normalized_include_prefixes = tuple(
        str(prefix).strip()
        for prefix in include_prefixes
        if str(prefix).strip()
    )
    normalized_exclude_prefixes = tuple(
        str(prefix).strip()
        for prefix in (exclude_prefixes or ())
        if str(prefix).strip()
    )
    updated_state = dict(model_state)
    loaded_count = 0
    skipped_prefix = 0
    skipped_missing = 0
    skipped_mismatch = 0
    adapted_count = 0
    for name, value in state_dict.items():
        normalized_name = name[7:] if isinstance(name, str) and name.startswith("module.") else name
        if normalized_include_prefixes and not any(
            normalized_name.startswith(prefix) for prefix in normalized_include_prefixes
        ):
            skipped_prefix += 1
            continue
        if normalized_exclude_prefixes and any(
            normalized_name.startswith(prefix) for prefix in normalized_exclude_prefixes
        ):
            skipped_prefix += 1
            continue
        if normalized_name not in model_state:
            skipped_missing += 1
            continue
        target_value = model_state[normalized_name]
        if target_value.shape != value.shape:
            adapted_value = _adapt_tensor_to_target_shape(
                value,
                target_value,
                normalized_name=normalized_name,
            )
            if adapted_value is None:
                skipped_mismatch += 1
                continue
            updated_state[normalized_name] = adapted_value
            loaded_count += 1
            adapted_count += 1
            continue
        updated_state[normalized_name] = value.detach().clone() if torch.is_tensor(value) else value
        loaded_count += 1
    model.load_state_dict(updated_state, strict=True)
    return {
        "resolved_path": str(resolved_path),
        "loaded": loaded_count,
        "skipped_prefix": skipped_prefix,
        "skipped_missing": skipped_missing,
        "skipped_mismatch": skipped_mismatch,
        "adapted_count": adapted_count,
        "remap_count": remap_count,
    }


# 基于已有 checkpoint 生成一个“保留原通信链、但恢复主干来自参考 checkpoint”的新 checkpoint。
def materialize_checkpoint_with_partial_overlay(
    base_checkpoint_path: str | Path,
    overlay_checkpoint_path: str | Path,
    *,
    include_prefixes: tuple[str, ...] | list[str],
    exclude_prefixes: tuple[str, ...] | list[str] | None = None,
    key: str = "model",
    save_path: str | Path | None = None,
    config_override: dict[str, Any] | None = None,
) -> dict[str, int | str]:
    resolved_base_path = resolve_project_path(base_checkpoint_path)
    if resolved_base_path is None or not resolved_base_path.exists():
        raise FileNotFoundError(f"Base checkpoint does not exist: {base_checkpoint_path}")
    base_checkpoint = torch.load(resolved_base_path, map_location="cpu")
    _, base_state_dict = _read_checkpoint_state_dict(resolved_base_path, key=key)
    _, overlay_raw_state_dict = _read_checkpoint_state_dict(overlay_checkpoint_path, key=key)

    target_state_dict = dict(base_state_dict)
    normalized_to_base_key: dict[str, str] = {}
    for base_name in target_state_dict.keys():
        normalized_base_name = base_name[7:] if isinstance(base_name, str) and base_name.startswith("module.") else base_name
        normalized_to_base_key.setdefault(normalized_base_name, base_name)
    overlay_state_dict, remap_count = _upgrade_legacy_state_dict_for_current_model(
        overlay_raw_state_dict,
        {
            (name[7:] if isinstance(name, str) and name.startswith("module.") else name): value
            for name, value in target_state_dict.items()
        },
    )
    normalized_include_prefixes = tuple(
        str(prefix).strip()
        for prefix in include_prefixes
        if str(prefix).strip()
    )
    normalized_exclude_prefixes = tuple(
        str(prefix).strip()
        for prefix in (exclude_prefixes or ())
        if str(prefix).strip()
    )

    loaded_count = 0
    skipped_prefix = 0
    skipped_missing = 0
    skipped_mismatch = 0
    for name, value in overlay_state_dict.items():
        normalized_name = name[7:] if isinstance(name, str) and name.startswith("module.") else name
        if normalized_include_prefixes and not any(
            normalized_name.startswith(prefix) for prefix in normalized_include_prefixes
        ):
            skipped_prefix += 1
            continue
        if normalized_exclude_prefixes and any(
            normalized_name.startswith(prefix) for prefix in normalized_exclude_prefixes
        ):
            skipped_prefix += 1
            continue
        base_key = normalized_to_base_key.get(normalized_name)
        if base_key is None or base_key not in target_state_dict:
            skipped_missing += 1
            continue
        target_value = target_state_dict[base_key]
        if torch.is_tensor(target_value) and torch.is_tensor(value):
            if target_value.shape != value.shape:
                adapted_value = _adapt_tensor_to_target_shape(
                    value,
                    target_value,
                    normalized_name=normalized_name,
                )
                if adapted_value is None:
                    skipped_mismatch += 1
                    continue
                target_state_dict[base_key] = adapted_value
                loaded_count += 1
                continue
            target_state_dict[base_key] = value.detach().clone()
            loaded_count += 1
            continue
        if type(target_value) is not type(value):
            skipped_mismatch += 1
            continue
        target_state_dict[base_key] = deepcopy(value)
        loaded_count += 1

    if isinstance(base_checkpoint, dict):
        materialized_checkpoint = dict(base_checkpoint)
        materialized_checkpoint[key] = target_state_dict
    else:
        materialized_checkpoint = {key: target_state_dict}
    if isinstance(config_override, dict):
        materialized_checkpoint["config"] = deepcopy(config_override)

    if save_path is not None:
        resolved_save_path = Path(save_path)
        resolved_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(materialized_checkpoint, resolved_save_path)

    return {
        "base_path": str(resolved_base_path),
        "overlay_path": str(resolve_project_path(overlay_checkpoint_path) or overlay_checkpoint_path),
        "loaded": loaded_count,
        "skipped_prefix": skipped_prefix,
        "skipped_missing": skipped_missing,
        "skipped_mismatch": skipped_mismatch,
        "remap_count": remap_count,
        "save_path": str(save_path) if save_path is not None else "",
    }
