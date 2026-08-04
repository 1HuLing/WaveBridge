from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_paths import CHECKPOINTS_DIR, EVAL_RESULTS_DIR, CONFIGS_DIR, ensure_standard_dirs

DEFAULT_CONFIG = CONFIGS_DIR / "default.yaml"
DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "wavebridege_div2k_hh_chain_v2_stage2_metric_refine.pt"
DEFAULT_SMOKE_CHECKPOINT = CHECKPOINTS_DIR / "wavebridege_div2k_hh_chain_v2_stage2_metric_refine.pt"
DEFAULT_EVAL_DIR = EVAL_RESULTS_DIR / "div2k"
DEFAULT_SUITE_DIR = EVAL_RESULTS_DIR / "ccfa_suite"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".pgm"}
SERVER_COMPARISON_ROOT = Path("/group4/Comparison")
LOW_PAYLOAD_MIN_BPP = 0.20
LOW_PAYLOAD_MAX_BPP = 0.40
LOW_PAYLOAD_EVAL_CARRIER_DATASET = "landscape"
LOW_PAYLOAD_EVAL_MIN_LATENT_CANDIDATES = 24
LOW_PAYLOAD_EVAL_SCORE = "carrier_brisque_proxy"
LOW_PAYLOAD_EVAL_PRIOR_MIXES = "0,0.001,0.002,0.004,0.006"


# 将相对路径统一解析到项目根目录，避免远端终端工作目录变化导致找不到文件。
def resolve_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (PROJECT_ROOT / candidate).resolve()


# 创建训练和评估阶段会用到的输出目录。
def ensure_output_dirs() -> None:
    ensure_standard_dirs()
    (PROJECT_ROOT / "train_results").mkdir(parents=True, exist_ok=True)


# 安全导入可选依赖，并把失败原因返回给预检逻辑。
def try_import(module_name: str):
    try:
        return __import__(module_name), None
    except Exception as error:  # pragma: no cover - 预检分支主要在远端环境使用
        return None, str(error)


# 读取 YAML 配置，供预检阶段检查数据集和设备设置。
def load_yaml_config(config_path: Path) -> dict[str, Any]:
    yaml_module, yaml_error = try_import("yaml")
    if yaml_module is None:
        raise RuntimeError(f"PyYAML is not available: {yaml_error}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml_module.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must parse to a dictionary: {config_path}")
    return config


def finite_float_or_none(value: Any) -> float | None:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(candidate):
        return None
    return candidate


def resolve_config_target_bpp(config: dict[str, Any]) -> float | None:
    for section_name in ("transmitter", "training"):
        section = config.get(section_name, {})
        if isinstance(section, dict):
            for key in ("target_bpp", "payload_bpp"):
                value = finite_float_or_none(section.get(key))
                if value is not None and value > 0.0:
                    return value
    return None


def is_low_payload_target_bpp(target_bpp: float | None) -> bool:
    return target_bpp is not None and LOW_PAYLOAD_MIN_BPP <= float(target_bpp) < LOW_PAYLOAD_MAX_BPP


def cli_option_explicitly_set(argv_tokens: list[str], option: str) -> bool:
    prefix = f"{option}="
    return any(token == option or token.startswith(prefix) for token in argv_tokens)


def any_cli_option_explicitly_set(argv_tokens: list[str], *options: str) -> bool:
    return any(cli_option_explicitly_set(argv_tokens, option) for option in options)


def cli_option_value(argv_tokens: list[str], option: str) -> str | None:
    prefix = f"{option}="
    for index, token in enumerate(argv_tokens):
        if token.startswith(prefix):
            return token[len(prefix) :]
        if token == option and index + 1 < len(argv_tokens):
            return argv_tokens[index + 1]
    return None


# 低 payload 评估默认走更强的鲁棒性协议，但保留用户显式传参的优先级。
def apply_low_payload_eval_defaults(
    args: argparse.Namespace,
    argv_tokens: list[str],
    config: dict[str, Any] | None,
) -> None:
    explicit_target_bpp = finite_float_or_none(cli_option_value(argv_tokens, "--target-bpp"))
    config_target_bpp = resolve_config_target_bpp(config) if config is not None else None
    effective_target_bpp = explicit_target_bpp if explicit_target_bpp is not None else config_target_bpp
    if not is_low_payload_target_bpp(effective_target_bpp):
        return

    applied: list[str] = []
    if not any_cli_option_explicitly_set(argv_tokens, "--enable-carrier-bank", "--disable-carrier-bank"):
        if not bool(args.enable_carrier_bank):
            applied.append("enable_carrier_bank=True")
        args.enable_carrier_bank = True
    if not cli_option_explicitly_set(argv_tokens, "--carrier-bank-dataset"):
        if args.carrier_bank_dataset != LOW_PAYLOAD_EVAL_CARRIER_DATASET:
            applied.append(f"carrier_bank_dataset={LOW_PAYLOAD_EVAL_CARRIER_DATASET}")
        args.carrier_bank_dataset = LOW_PAYLOAD_EVAL_CARRIER_DATASET
    if not any_cli_option_explicitly_set(argv_tokens, "--enable-robust-qim", "--disable-robust-qim"):
        if not bool(args.enable_robust_qim):
            applied.append("enable_robust_qim=True")
        args.enable_robust_qim = True
    if not cli_option_explicitly_set(argv_tokens, "--eval-latent-select-candidates"):
        upgraded_candidates = max(int(args.eval_latent_select_candidates), LOW_PAYLOAD_EVAL_MIN_LATENT_CANDIDATES)
        if upgraded_candidates != int(args.eval_latent_select_candidates):
            applied.append(f"eval_latent_select_candidates={upgraded_candidates}")
        args.eval_latent_select_candidates = upgraded_candidates
    if not cli_option_explicitly_set(argv_tokens, "--eval-latent-select-score"):
        if args.eval_latent_select_score != LOW_PAYLOAD_EVAL_SCORE:
            applied.append(f"eval_latent_select_score={LOW_PAYLOAD_EVAL_SCORE}")
        args.eval_latent_select_score = LOW_PAYLOAD_EVAL_SCORE
    if not cli_option_explicitly_set(argv_tokens, "--eval-latent-select-prior-mixes"):
        if args.eval_latent_select_prior_mixes != LOW_PAYLOAD_EVAL_PRIOR_MIXES:
            applied.append(f"eval_latent_select_prior_mixes={LOW_PAYLOAD_EVAL_PRIOR_MIXES}")
        args.eval_latent_select_prior_mixes = LOW_PAYLOAD_EVAL_PRIOR_MIXES
    if explicit_target_bpp is None and config_target_bpp is not None:
        args.eval_target_bpp = float(config_target_bpp)
        applied.append(f"target_bpp={float(config_target_bpp):g}")

    print(
        f"[INFO] Low-payload eval profile detected (target_bpp={float(effective_target_bpp):g})."
    )
    if applied:
        print(f"[INFO] Applied low-payload eval defaults: {', '.join(applied)}")
    else:
        print("[INFO] Low-payload eval defaults already satisfied by explicit arguments.")


# 低 payload 的 CCF-A suite 也默认走鲁棒性优先协议，避免 run_remote 直跑时回退到弱评估设置。
def apply_low_payload_ccfa_suite_defaults(
    args: argparse.Namespace,
    argv_tokens: list[str],
    config: dict[str, Any] | None,
) -> None:
    explicit_default_bpp = finite_float_or_none(cli_option_value(argv_tokens, "--suite-default-bpp"))
    config_target_bpp = resolve_config_target_bpp(config) if config is not None else None
    effective_target_bpp = (
        explicit_default_bpp
        if explicit_default_bpp is not None
        else config_target_bpp
        if config_target_bpp is not None
        else finite_float_or_none(getattr(args, "suite_default_bpp", None))
    )
    if not is_low_payload_target_bpp(effective_target_bpp):
        return

    applied: list[str] = []
    if explicit_default_bpp is None and config_target_bpp is not None:
        if float(args.suite_default_bpp) != float(config_target_bpp):
            applied.append(f"suite_default_bpp={float(config_target_bpp):g}")
        args.suite_default_bpp = float(config_target_bpp)
    if not any_cli_option_explicitly_set(argv_tokens, "--enable-carrier-bank", "--disable-carrier-bank"):
        if not bool(args.enable_carrier_bank):
            applied.append("enable_carrier_bank=True")
        args.enable_carrier_bank = True
    if not cli_option_explicitly_set(argv_tokens, "--carrier-bank-dataset"):
        if args.carrier_bank_dataset != LOW_PAYLOAD_EVAL_CARRIER_DATASET:
            applied.append(f"carrier_bank_dataset={LOW_PAYLOAD_EVAL_CARRIER_DATASET}")
        args.carrier_bank_dataset = LOW_PAYLOAD_EVAL_CARRIER_DATASET
    if not any_cli_option_explicitly_set(argv_tokens, "--enable-robust-qim", "--disable-robust-qim"):
        if not bool(args.enable_robust_qim):
            applied.append("enable_robust_qim=True")
        args.enable_robust_qim = True
    if not cli_option_explicitly_set(argv_tokens, "--suite-latent-select-candidates"):
        upgraded_candidates = max(int(args.suite_latent_select_candidates), LOW_PAYLOAD_EVAL_MIN_LATENT_CANDIDATES)
        if upgraded_candidates != int(args.suite_latent_select_candidates):
            applied.append(f"suite_latent_select_candidates={upgraded_candidates}")
        args.suite_latent_select_candidates = upgraded_candidates
    if not cli_option_explicitly_set(argv_tokens, "--suite-latent-select-score"):
        if args.suite_latent_select_score != LOW_PAYLOAD_EVAL_SCORE:
            applied.append(f"suite_latent_select_score={LOW_PAYLOAD_EVAL_SCORE}")
        args.suite_latent_select_score = LOW_PAYLOAD_EVAL_SCORE
    if not cli_option_explicitly_set(argv_tokens, "--suite-latent-select-prior-mixes"):
        if args.suite_latent_select_prior_mixes != LOW_PAYLOAD_EVAL_PRIOR_MIXES:
            applied.append(f"suite_latent_select_prior_mixes={LOW_PAYLOAD_EVAL_PRIOR_MIXES}")
        args.suite_latent_select_prior_mixes = LOW_PAYLOAD_EVAL_PRIOR_MIXES

    print(
        f"[INFO] Low-payload CCF-A suite profile detected (target_bpp={float(effective_target_bpp):g})."
    )
    if applied:
        print(f"[INFO] Applied low-payload suite defaults: {', '.join(applied)}")
    else:
        print("[INFO] Low-payload suite defaults already satisfied by explicit arguments.")


# 统计目录中的图像数量，便于快速确认数据集是否真的挂载到了服务器。
def count_images(image_dir: Path) -> int:
    if not image_dir.exists() or not image_dir.is_dir():
        return 0
    return sum(1 for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


# 汇总当前启用数据集的目录和切分文件状态。
def build_dataset_summary(config: dict[str, Any], dataset_names: list[str] | None = None) -> list[str]:
    datasets = config.get("datasets", {})
    active_name = datasets.get("active")
    names = dataset_names or ([active_name] if active_name is not None else [])
    lines = [f"active_dataset={active_name}"]
    seen: set[str] = set()
    for dataset_name in names:
        if not dataset_name or str(dataset_name) in seen:
            continue
        seen.add(str(dataset_name))
        dataset_cfg = datasets.get(dataset_name, {})
        lines.append(f"dataset={dataset_name}")
        if not isinstance(dataset_cfg, dict):
            lines.append("dataset_config_exists=False")
            continue
        for key in ("train_dir", "val_dir", "train_split", "val_split", "test_dir", "test_split"):
            value = dataset_cfg.get(key)
            if not value:
                continue
            resolved = resolve_path(value)
            exists = resolved.exists() if resolved is not None else False
            lines.append(f"{dataset_name}.{key}={resolved} exists={exists}")
            if key.endswith("_dir") and resolved is not None and exists:
                lines.append(f"{dataset_name}.{key}_images={count_images(resolved)}")
    return lines


# 根据模式决定本次运行必须具备的 Python 依赖。
def required_modules_for_mode(mode: str) -> list[str]:
    modules = ["yaml", "torch", "torchvision", "PIL", "numpy"]
    return modules


# 评估阶段允许缺少部分可选依赖，缺失时对应指标会退化为 NaN，但不应阻塞整体流程。
def optional_modules_for_mode(mode: str) -> list[str]:
    if mode in {"eval", "ccfa-suite"}:
        return ["scipy", "lpips"]
    return []


def ccfa_suite_dataset_names(args: argparse.Namespace | None) -> list[str]:
    # Keep the preflight check aligned with the datasets used by the suite.
    defaults = {
        "main_dataset": "div2k",
        "detection_datasets": "bossbase,alaska2",
        "payload_dataset": "div2k",
        "naturalness_datasets": "landscape",
        "robustness_dataset": "div2k",
    }
    if args is not None:
        defaults["main_dataset"] = str(getattr(args, "suite_main_dataset", defaults["main_dataset"]))
        defaults["detection_datasets"] = str(getattr(args, "suite_detection_datasets", defaults["detection_datasets"]))
        defaults["payload_dataset"] = str(getattr(args, "suite_payload_dataset", defaults["payload_dataset"]))
        defaults["naturalness_datasets"] = str(
            getattr(args, "suite_naturalness_datasets", defaults["naturalness_datasets"])
        )
        defaults["robustness_dataset"] = str(getattr(args, "suite_robustness_dataset", defaults["robustness_dataset"]))

    names: list[str] = []
    for raw in (
        defaults["main_dataset"],
        defaults["detection_datasets"],
        defaults["payload_dataset"],
        defaults["naturalness_datasets"],
        defaults["robustness_dataset"],
    ):
        for item in str(raw).split(","):
            name = item.strip().lower()
            if name and name not in names:
                names.append(name)
    return names


# 检查当前 Python 环境、配置文件、数据目录和显卡设置是否满足运行要求。
def run_preflight(
    mode: str,
    config_path: Path,
    device_override: str | None,
    checkpoint_path: Path | None,
    args: argparse.Namespace | None = None,
) -> bool:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Requested mode: {mode}")
    print(f"Config path: {config_path}")

    if not config_path.exists():
        print(f"[ERROR] Config file does not exist: {config_path}")
        return False

    ok = True
    for module_name in required_modules_for_mode(mode):
        _, error = try_import(module_name)
        if error is None:
            print(f"[OK] Python module: {module_name}")
        else:
            print(f"[ERROR] Missing module {module_name}: {error}")
            ok = False
    for module_name in optional_modules_for_mode(mode):
        _, error = try_import(module_name)
        if error is None:
            print(f"[OK] Optional module: {module_name}")
        else:
            print(f"[WARN] Optional module missing {module_name}: {error}")
    if not ok:
        return False

    config = load_yaml_config(config_path)
    dataset_names = ccfa_suite_dataset_names(args) if mode == "ccfa-suite" else None
    for line in build_dataset_summary(config, dataset_names=dataset_names):
        print(f"[DATA] {line}")
    if mode == "ccfa-suite":
        datasets = config.get("datasets", {})
        missing_datasets: list[str] = []
        for dataset_name in dataset_names or []:
            dataset_cfg = datasets.get(dataset_name, {})
            candidate_dirs = [
                resolve_path(dataset_cfg.get(key))
                for key in ("test_dir", "val_dir", "train_dir")
                if isinstance(dataset_cfg, dict) and dataset_cfg.get(key)
            ]
            usable_dirs = [path for path in candidate_dirs if path is not None and path.exists() and count_images(path) > 0]
            if not usable_dirs:
                missing_datasets.append(dataset_name)
        if missing_datasets:
            print(f"[ERROR] CCF-A suite dataset(s) missing or empty: {', '.join(missing_datasets)}")
            return False

    train_dir = resolve_path(config.get("datasets", {}).get(config.get("datasets", {}).get("active"), {}).get("train_dir"))
    if mode in {"smoke", "train"} and (train_dir is None or not train_dir.exists()):
        print("[ERROR] Training dataset directory is missing.")
        return False

    if checkpoint_path is not None and mode in {"eval", "ccfa-suite"} and not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint does not exist: {checkpoint_path}")
        return False

    torch_module, _ = try_import("torch")
    requested_device = device_override or str(config.get("device", "cpu"))
    preferred_index = int(config.get("device_index", 1) or 0)
    print(f"Requested device: {requested_device}")
    print(f"Preferred device index: {preferred_index}")
    print(f"Torch version: {torch_module.__version__}")

    if requested_device.startswith("cuda"):
        if not torch_module.cuda.is_available():
            print("[ERROR] CUDA is not available in the current environment.")
            return False
        gpu_count = torch_module.cuda.device_count()
        print(f"CUDA device count: {gpu_count}")
        for index in range(gpu_count):
            print(f"[GPU] cuda:{index} -> {torch_module.cuda.get_device_name(index)}")
        if requested_device == "cuda":
            target_index = preferred_index
        else:
            try:
                target_index = int(requested_device.split(":", maxsplit=1)[1])
            except Exception:
                target_index = preferred_index
        if target_index >= gpu_count:
            print(f"[ERROR] Requested GPU index cuda:{target_index} does not exist.")
            return False
        print(f"[OK] Selected GPU: cuda:{target_index}")
    else:
        print("[WARN] Current run is not configured to use CUDA.")

    ensure_output_dirs()
    for directory in ("logs", "checkpoints", "train_results", "eval_results"):
        print(f"[OK] Output dir: {PROJECT_ROOT / directory}")
    return True


# 构造烟雾测试、正式训练和评估对应的底层 Python 命令。
def build_command(args: argparse.Namespace, extra_args: list[str]) -> list[str]:
    common_train = [
        sys.executable,
        "train.py",
        "--strategy",
        "div2k-stable",
        "--config",
        str(args.config),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
    ]
    if args.mode == "smoke":
        command = common_train + [
            "--save",
            str(args.smoke_save),
            "--epochs",
            str(args.smoke_epochs),
            "--max-train-batches",
            str(args.smoke_train_batches),
            "--max-val-batches",
            str(args.smoke_val_batches),
            "--skip-final-eval",
            "--skip-sample-export",
            "--log-file",
            str(args.smoke_log),
        ]
    elif args.mode == "train":
        command = common_train + [
            "--save",
            str(args.train_save),
            "--early-stop-patience",
            str(args.early_stop_patience),
            "--skip-final-eval",
            "--skip-sample-export",
            "--log-file",
            str(args.train_log),
        ]
    elif args.mode == "eval":
        command = [
            sys.executable,
            "evaluate.py",
            "--config",
            str(args.config),
            "--checkpoint",
            str(args.checkpoint),
            "--output-dir",
            str(args.eval_output_dir),
            "--batch-size",
            str(args.eval_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--max-images",
            str(args.eval_max_images),
            "--device",
            args.device,
            "--fid-backend",
            args.eval_fid_backend,
            "--clean-fid-mode",
            args.eval_clean_fid_mode,
            "--stat-backend",
            args.eval_stat_backend,
            "--detector-test-ratio",
            str(args.eval_detector_test_ratio),
            "--detector-max-images",
            str(args.eval_detector_max_images),
            "--detector-size",
            str(args.eval_detector_size),
            "--srnet-epochs",
            str(args.eval_srnet_epochs),
            "--srnet-arch",
            args.eval_srnet_arch,
        ]
        if args.eval_target_bpp is not None:
            command.extend(["--target-bpp", str(args.eval_target_bpp)])
        if args.eval_fraction is not None:
            command.extend(["--eval-fraction", str(args.eval_fraction)])
        if args.eval_srnet_weights:
            command.extend(["--srnet-weights", str(args.eval_srnet_weights)])
        if args.eval_srnet_batch_size is not None:
            command.extend(["--srnet-batch-size", str(args.eval_srnet_batch_size)])
        if args.eval_srnet_lr is not None:
            command.extend(["--srnet-lr", str(args.eval_srnet_lr)])
        if args.eval_srnet_save_weights:
            command.extend(["--srnet-save-weights", str(args.eval_srnet_save_weights)])
        if args.use_official_srnet:
            command.append("--use-official-srnet")
        if args.comparison_root:
            command.extend(["--comparison-root", str(args.comparison_root)])
        if args.official_srnet_python:
            command.extend(["--official-srnet-python", args.official_srnet_python])
        command.extend(["--official-srnet-max-iter", str(args.official_srnet_max_iter)])
        if args.official_srnet_force_cpu:
            command.append("--official-srnet-force-cpu")
        if args.enable_carrier_bank:
            command.append("--enable-carrier-bank")
            command.extend(["--carrier-bank-dataset", args.carrier_bank_dataset])
            if args.carrier_bank_dirs:
                command.extend(["--carrier-bank-dirs", args.carrier_bank_dirs])
            command.extend(["--carrier-bank-blend", str(args.carrier_bank_blend)])
        if args.enable_robust_qim:
            command.append("--enable-robust-qim")
        if args.force_current_qim_structure:
            command.append("--force-current-qim-structure")
        command.extend(["--latent-select-candidates", str(args.eval_latent_select_candidates)])
        command.extend(["--latent-select-score", args.eval_latent_select_score])
        if args.eval_latent_select_prior_checkpoint:
            command.extend(["--latent-select-prior-checkpoint", str(args.eval_latent_select_prior_checkpoint)])
        if args.eval_latent_select_prior_mixes:
            command.extend(["--latent-select-prior-mixes", args.eval_latent_select_prior_mixes])
        command.extend(["--latent-select-psnr-drop", str(args.eval_latent_select_psnr_drop)])
        command.extend(["--latent-select-ssim-drop", str(args.eval_latent_select_ssim_drop)])
        command.extend(["--latent-select-max-ber-info", str(args.eval_latent_select_max_ber_info)])
        command.extend(["--latent-select-max-ber-code", str(args.eval_latent_select_max_ber_code)])
        command.extend(["--latent-select-decoded-floor", str(args.eval_latent_select_decoded_floor)])
        command.extend(["--latent-select-min-psnr", str(args.eval_latent_select_min_psnr)])
        command.extend(["--latent-select-min-ssim", str(args.eval_latent_select_min_ssim)])
    elif args.mode == "ccfa-suite":
        command = [
            sys.executable,
            "scripts/ccfa_experiment_suite.py",
            "--config",
            str(args.config),
            "--checkpoint",
            str(args.checkpoint),
            "--output-root",
            str(args.suite_output_root),
            "--device",
            args.device,
            "--batch-size",
            str(args.eval_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--progress-interval",
            str(args.eval_progress_interval),
            "--phases",
            args.suite_phases,
            "--seeds",
            args.suite_seeds,
            "--main-dataset",
            args.suite_main_dataset,
            "--detection-datasets",
            args.suite_detection_datasets,
            "--payload-dataset",
            args.suite_payload_dataset,
            "--naturalness-datasets",
            args.suite_naturalness_datasets,
            "--robustness-dataset",
            args.suite_robustness_dataset,
            "--payloads",
            args.suite_payloads,
            "--default-bpp",
            str(args.suite_default_bpp),
            "--main-max-images",
            str(args.suite_main_max_images),
            "--security-max-images",
            str(args.suite_security_max_images),
            "--payload-max-images",
            str(args.suite_payload_max_images),
            "--naturalness-max-images",
            str(args.suite_naturalness_max_images),
            "--robustness-max-images",
            str(args.suite_robustness_max_images),
            "--detector-max-images",
            str(args.suite_detector_max_images),
            "--main-srnet-epochs",
            str(args.suite_main_srnet_epochs),
            "--srnet-epochs",
            str(args.suite_srnet_epochs),
            "--payload-srnet-epochs",
            str(args.suite_payload_srnet_epochs),
            "--naturalness-srnet-epochs",
            str(args.suite_naturalness_srnet_epochs),
            "--robustness-srnet-epochs",
            str(args.suite_robustness_srnet_epochs),
            "--srnet-arch",
            args.suite_srnet_arch,
            "--srnet-batch-size",
            str(args.suite_srnet_batch_size),
            "--fid-backend",
            args.suite_fid_backend,
            "--stat-backend",
            args.suite_stat_backend,
            "--latent-select-candidates",
            str(args.suite_latent_select_candidates),
        ]
        if args.enable_carrier_bank:
            command.append("--enable-carrier-bank")
            command.extend(["--carrier-bank-dataset", args.carrier_bank_dataset])
            if args.carrier_bank_dirs:
                command.extend(["--carrier-bank-dirs", args.carrier_bank_dirs])
        if args.enable_robust_qim:
            command.append("--enable-robust-qim")
        command.extend(["--latent-select-score", args.suite_latent_select_score])
        if args.suite_latent_select_prior_mixes:
            command.extend(["--latent-select-prior-mixes", args.suite_latent_select_prior_mixes])
        command.extend(["--latent-select-psnr-drop", str(args.suite_latent_select_psnr_drop)])
        command.extend(["--latent-select-ssim-drop", str(args.suite_latent_select_ssim_drop)])
        command.extend(["--latent-select-max-ber-info", str(args.suite_latent_select_max_ber_info)])
        command.extend(["--latent-select-max-ber-code", str(args.suite_latent_select_max_ber_code)])
        command.extend(["--latent-select-decoded-floor", str(args.suite_latent_select_decoded_floor)])
        command.extend(["--latent-select-min-psnr", str(args.suite_latent_select_min_psnr)])
        command.extend(["--latent-select-min-ssim", str(args.suite_latent_select_min_ssim)])
        if args.use_official_srnet:
            command.append("--use-official-srnet")
        if args.comparison_root:
            command.extend(["--comparison-root", str(args.comparison_root)])
        if args.official_srnet_python:
            command.extend(["--official-srnet-python", args.official_srnet_python])
        if args.continue_on_error:
            command.append("--continue-on-error")
        if args.dry_run:
            command.append("--dry-run")
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    return command + extra_args


# 选择服务器侧实际可用的 Python 解释器，优先复用已经装好 torch 的离线环境。
def select_runtime_python() -> str:
    candidates = [
        Path("/group4/venvs/WaveBridege/bin/python"),
        Path("/group4/venvs/comparison-offline/bin/python"),
        Path("/root/miniconda3/bin/python"),
    ]
    for candidate in candidates:
        if candidate.exists():
            if "comparison-offline" in str(candidate):
                return str(candidate)
            if "WaveBridege" in str(candidate):
                torch_dir = candidate.parent.parent / "lib" / "python3.10" / "site-packages" / "torch"
                if torch_dir.exists():
                    return str(candidate)
            if "miniconda3" in str(candidate):
                return str(candidate)
    return sys.executable


# 以当前解释器在项目根目录启动真实训练或评估命令。
def run_command(command: list[str]) -> int:
    print("Running command:")
    print(" ".join(command))
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    return int(completed.returncode)


# 解析统一入口脚本的命令行参数。
def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Remote launcher for WaveBridege.")
    parser.add_argument("mode", choices=["preflight", "smoke", "train", "eval", "ccfa-suite"], help="Run mode")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config file")
    parser.add_argument("--device", default="cuda:1", help="Target device, default is the second GPU")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip environment and dataset checks")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Checkpoint used by eval mode")
    parser.add_argument("--smoke-save", default=str(DEFAULT_SMOKE_CHECKPOINT), help="Checkpoint path for smoke mode")
    parser.add_argument("--train-save", default=str(DEFAULT_CHECKPOINT), help="Checkpoint path for train mode")
    parser.add_argument("--smoke-log", default=str(PROJECT_ROOT / "logs" / "smoke_gpu1.log"), help="Smoke log path")
    parser.add_argument("--train-log", default=str(PROJECT_ROOT / "logs" / "div2k_single.log"), help="Train log path")
    parser.add_argument("--eval-output-dir", default=str(DEFAULT_EVAL_DIR), help="Evaluation output directory")
    parser.add_argument("--suite-output-root", default=str(DEFAULT_SUITE_DIR), help="Output root for CCF-A suite mode")
    parser.add_argument(
        "--suite-phases",
        default="main,detection,payload,naturalness,robustness",
        help="Comma-separated suite phases in execution order",
    )
    parser.add_argument("--suite-seeds", default="42,43,44", help="Comma-separated evaluation seeds")
    parser.add_argument("--suite-main-dataset", default="div2k", help="Dataset for main recovery evaluation")
    parser.add_argument("--suite-detection-datasets", default="bossbase,alaska2", help="Security datasets")
    parser.add_argument("--suite-payload-dataset", default="div2k", help="Dataset for payload sweep")
    parser.add_argument("--suite-naturalness-datasets", default="landscape", help="Datasets for naturalness evaluation")
    parser.add_argument("--suite-robustness-dataset", default="div2k", help="Dataset for robustness scan")
    parser.add_argument("--suite-payloads", default="0.1,0.2,0.3,0.5", help="Payload sweep bpp values")
    parser.add_argument("--suite-default-bpp", type=float, default=0.3, help="Default bpp outside payload sweep")
    parser.add_argument("--suite-main-max-images", type=int, default=200, help="Images for main recovery evaluation")
    parser.add_argument("--suite-security-max-images", type=int, default=1000, help="Images for security evaluation")
    parser.add_argument("--suite-payload-max-images", type=int, default=200, help="Images for payload sweep")
    parser.add_argument("--suite-naturalness-max-images", type=int, default=1000, help="Images for naturalness evaluation")
    parser.add_argument("--suite-robustness-max-images", type=int, default=200, help="Images for robustness scan")
    parser.add_argument("--suite-detector-max-images", type=int, default=1000, help="Max detector pairs")
    parser.add_argument("--suite-main-srnet-epochs", type=int, default=0, help="SRNet epochs for main recovery phase")
    parser.add_argument("--suite-srnet-epochs", type=int, default=10, help="SRNet epochs for detection phase")
    parser.add_argument("--suite-payload-srnet-epochs", type=int, default=3, help="SRNet epochs during payload sweep")
    parser.add_argument("--suite-naturalness-srnet-epochs", type=int, default=0, help="SRNet epochs during naturalness")
    parser.add_argument("--suite-robustness-srnet-epochs", type=int, default=0, help="SRNet epochs during robustness")
    parser.add_argument("--suite-srnet-arch", default="deep", choices=["deep", "lite"], help="SRNet-style detector")
    parser.add_argument("--suite-srnet-batch-size", type=int, default=8, help="SRNet batch size")
    parser.add_argument("--suite-fid-backend", default="auto", choices=["auto", "clean", "legacy"], help="FID backend")
    parser.add_argument("--suite-stat-backend", default="auto", choices=["auto", "srm", "legacy"], help="Stat backend")
    parser.add_argument("--suite-latent-select-candidates", type=int, default=24, help="Naturalness candidates")
    parser.add_argument(
        "--suite-latent-select-score",
        default="carrier_brisque_proxy",
        choices=["brisque_proxy", "cover_brisque_proxy", "carrier_brisque_proxy"],
        help="Naturalness proxy for suite latent reranking",
    )
    parser.add_argument("--suite-latent-select-prior-mixes", default="0,0.001,0.002,0.004,0.006", help="Prior mixes for candidate selection")
    parser.add_argument("--suite-latent-select-psnr-drop", type=float, default=0.03, help="Allowed PSNR drop during suite latent reranking")
    parser.add_argument("--suite-latent-select-ssim-drop", type=float, default=0.0003, help="Allowed SSIM drop during suite latent reranking")
    parser.add_argument("--suite-latent-select-max-ber-info", type=float, default=1e-4, help="Max clean BER_info for suite latent reranking")
    parser.add_argument("--suite-latent-select-max-ber-code", type=float, default=1e-3, help="Max clean BER_code for suite latent reranking")
    parser.add_argument("--suite-latent-select-decoded-floor", type=float, default=0.999999, help="Min decoded ratio for suite latent reranking")
    parser.add_argument("--suite-latent-select-min-psnr", type=float, default=38.0, help="Absolute PSNR floor for suite latent reranking")
    parser.add_argument("--suite-latent-select-min-ssim", type=float, default=0.97, help="Absolute SSIM floor for suite latent reranking")
    parser.add_argument("--use-official-srnet", action="store_true", help="Run official SRNet if Comparison is available")
    parser.add_argument("--comparison-root", default=None, help="Comparison root for official SRNet")
    parser.add_argument("--official-srnet-python", default=None, help="Python executable for official SRNet")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue CCF-A suite after a failed job")
    parser.add_argument("--dry-run", action="store_true", help="Write CCF-A suite commands without running jobs")
    parser.add_argument("--smoke-epochs", type=int, default=1, help="Epochs for smoke mode")
    parser.add_argument("--smoke-train-batches", type=int, default=2, help="Train batches for smoke mode")
    parser.add_argument("--smoke-val-batches", type=int, default=1, help="Validation batches for smoke mode")
    parser.add_argument("--early-stop-patience", type=int, default=10, help="Early stopping patience for train mode")
    parser.add_argument("--eval-batch-size", type=int, default=1, help="Batch size for eval mode")
    parser.add_argument("--eval-max-images", type=int, default=32, help="Max images for eval mode")
    parser.add_argument("--eval-progress-interval", type=int, default=20, help="Print eval progress every N images")
    parser.add_argument("--target-bpp", dest="eval_target_bpp", type=float, default=None, help="Optional target bpp override for eval mode")
    parser.add_argument("--eval-fraction", type=float, default=None, help="Optional eval subset fraction")
    parser.add_argument("--eval-fid-backend", default="clean", choices=["auto", "clean", "legacy"], help="FID backend for eval mode")
    parser.add_argument("--eval-clean-fid-mode", default="clean", help="clean-fid mode for eval mode")
    parser.add_argument("--eval-stat-backend", default="srm", choices=["auto", "srm", "legacy"], help="Statistical detector backend")
    parser.add_argument("--eval-detector-test-ratio", type=float, default=0.3, help="Held-out detector split ratio")
    parser.add_argument("--eval-detector-max-images", type=int, default=256, help="Max cover/stego pairs for detectors")
    parser.add_argument("--eval-detector-size", type=int, default=64, help="Detector input size")
    parser.add_argument("--eval-srnet-weights", default=None, help="Optional SRNet weights for eval mode")
    parser.add_argument("--eval-srnet-epochs", type=int, default=3, help="SRNet epochs for eval mode")
    parser.add_argument("--eval-srnet-arch", default="deep", choices=["deep", "lite"], help="SRNet detector architecture")
    parser.add_argument("--eval-srnet-batch-size", type=int, default=None, help="Optional SRNet batch size override")
    parser.add_argument("--eval-srnet-lr", type=float, default=None, help="Optional SRNet learning rate override")
    parser.add_argument("--eval-srnet-save-weights", default=None, help="Optional path to save SRNet weights")
    parser.add_argument("--enable-carrier-bank", action="store_true", dest="enable_carrier_bank", help="Enable carrier bank during eval")
    parser.add_argument("--disable-carrier-bank", action="store_false", dest="enable_carrier_bank", help="Explicitly disable carrier bank during eval or suite runs")
    parser.add_argument("--carrier-bank-dataset", default="div2k", help="Carrier bank dataset name for eval")
    parser.add_argument("--carrier-bank-dirs", default="", help="Optional comma-separated carrier bank directories")
    parser.add_argument("--carrier-bank-blend", type=float, default=1.0, help="External carrier blend for eval")
    parser.add_argument("--enable-robust-qim", action="store_true", dest="enable_robust_qim", help="Enable robust runtime QIM during eval")
    parser.add_argument("--disable-robust-qim", action="store_false", dest="enable_robust_qim", help="Explicitly disable robust runtime QIM during eval or suite runs")
    parser.add_argument("--force-current-qim-structure", action="store_true", help="Override checkpoint QIM structure during eval")
    parser.add_argument("--eval-latent-select-candidates", type=int, default=4, help="Latent candidates for safe naturalness reranking")
    parser.add_argument("--eval-latent-select-score", default="cover_brisque_proxy", choices=["brisque_proxy", "cover_brisque_proxy", "carrier_brisque_proxy"], help="Naturalness proxy for latent reranking")
    parser.add_argument("--eval-latent-select-prior-checkpoint", default=None, help="Optional prior checkpoint for latent reranking")
    parser.add_argument("--eval-latent-select-prior-mixes", default="", help="Comma-separated prior_mix candidates for latent reranking")
    parser.add_argument("--eval-latent-select-psnr-drop", type=float, default=0.03, help="Allowed PSNR drop during latent reranking")
    parser.add_argument("--eval-latent-select-ssim-drop", type=float, default=0.0003, help="Allowed SSIM drop during latent reranking")
    parser.add_argument("--eval-latent-select-max-ber-info", type=float, default=1e-4, help="Max clean BER_info for latent reranking")
    parser.add_argument("--eval-latent-select-max-ber-code", type=float, default=1e-3, help="Max clean BER_code for latent reranking")
    parser.add_argument("--eval-latent-select-decoded-floor", type=float, default=0.999999, help="Min decoded ratio for latent reranking")
    parser.add_argument("--eval-latent-select-min-psnr", type=float, default=38.0, help="Absolute PSNR floor for latent reranking")
    parser.add_argument("--eval-latent-select-min-ssim", type=float, default=0.97, help="Absolute SSIM floor for latent reranking")
    parser.set_defaults(enable_carrier_bank=False, enable_robust_qim=False)
    return parser.parse_known_args()


# 执行入口：先做预检，再按模式分发到训练或评估脚本。
def main() -> int:
    args, extra_args = parse_args()
    raw_argv_tokens = sys.argv[1:]
    runtime_python = select_runtime_python()
    if Path(runtime_python).resolve() != Path(sys.executable).resolve():
        sys.executable = runtime_python
    args.config = resolve_path(args.config) or Path(args.config)
    args.checkpoint = resolve_path(args.checkpoint) or Path(args.checkpoint)
    args.smoke_save = resolve_path(args.smoke_save) or Path(args.smoke_save)
    args.train_save = resolve_path(args.train_save) or Path(args.train_save)
    args.smoke_log = resolve_path(args.smoke_log) or Path(args.smoke_log)
    args.train_log = resolve_path(args.train_log) or Path(args.train_log)
    args.eval_output_dir = resolve_path(args.eval_output_dir) or Path(args.eval_output_dir)
    args.suite_output_root = resolve_path(args.suite_output_root) or Path(args.suite_output_root)
    if args.comparison_root is not None:
        args.comparison_root = resolve_path(args.comparison_root) or Path(args.comparison_root)
    elif args.mode in {"eval", "ccfa-suite"} and SERVER_COMPARISON_ROOT.exists():
        args.comparison_root = SERVER_COMPARISON_ROOT
    if args.eval_srnet_weights is not None:
        args.eval_srnet_weights = resolve_path(args.eval_srnet_weights) or Path(args.eval_srnet_weights)
    if args.eval_srnet_save_weights is not None:
        args.eval_srnet_save_weights = resolve_path(args.eval_srnet_save_weights) or Path(args.eval_srnet_save_weights)
    if args.eval_latent_select_prior_checkpoint is not None:
        args.eval_latent_select_prior_checkpoint = resolve_path(args.eval_latent_select_prior_checkpoint) or Path(args.eval_latent_select_prior_checkpoint)
    if not args.use_official_srnet and args.mode in {"eval", "ccfa-suite"} and args.comparison_root is not None:
        args.use_official_srnet = True

    config_for_runtime: dict[str, Any] | None = None
    if args.mode == "eval":
        config_for_runtime = load_yaml_config(Path(args.config))
        apply_low_payload_eval_defaults(args, raw_argv_tokens, config_for_runtime)
    elif args.mode == "ccfa-suite":
        config_for_runtime = load_yaml_config(Path(args.config))
        apply_low_payload_ccfa_suite_defaults(args, raw_argv_tokens, config_for_runtime)

    ensure_output_dirs()
    if not args.skip_preflight or args.mode == "preflight":
        checkpoint_path = Path(args.checkpoint) if args.mode in {"eval", "ccfa-suite"} else None
        ok = run_preflight(
            mode=args.mode,
            config_path=Path(args.config),
            device_override=args.device,
            checkpoint_path=checkpoint_path,
            args=args,
        )
        if not ok:
            return 1
        if args.mode == "preflight":
            return 0

    command = build_command(args, extra_args)
    return_code = run_command(command)
    if args.mode == "smoke" and return_code == 0:
        smoke_checkpoint = Path(args.smoke_save)
        if smoke_checkpoint.exists():
            smoke_checkpoint.unlink()
            print(f"Removed smoke checkpoint: {smoke_checkpoint}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
