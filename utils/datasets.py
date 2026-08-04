from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from utils.config import resolve_project_path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".pgm"}

DATASET_PATH_ALIASES: dict[str, dict[str, list[str]]] = {
    "landscape": {
        "train_dir": [
            "dataset/Landscape",
            "dataset/landscape",
            "dataset/landscape_images",
        ],
        "split_dir": [
            "dataset/splits/landscape",
        ],
    },
    "alaska2": {
        "train_dir": [
            "dataset/ALASKA2/jpeg_512_color_various_qf",
            "dataset/ALASKA_v2_JPG_512_QFvarious_COLOR",
        ],
        "test_dir": [
            "dataset/ALASKA2/jpeg_512_color_various_qf",
            "dataset/ALASKA_v2_JPG_512_QFvarious_COLOR",
        ],
        "split_dir": [
            "dataset/splits/alaska2",
        ],
    },
    "div2k": {
        "train_dir": [
            "dataset/DIV2K/DIV2K_train_HR",
        ],
        "val_dir": [
            "dataset/DIV2K/DIV2K_valid_HR",
        ],
    },
    "bossbase": {
        "train_dir": [
            "dataset/BOSSbase_1.01",
        ],
        "val_dir": [
            "dataset/BOSSbase_1.01",
        ],
        "test_dir": [
            "dataset/BOSSbase_1.01",
        ],
        "train_split": [
            "dataset/splits/bossbase/train.txt",
        ],
        "val_split": [
            "dataset/splits/bossbase/val.txt",
        ],
        "test_split": [
            "dataset/splits/bossbase/val.txt",
        ],
    },
}


# 按原图尺寸返回图像批次，避免 DataLoader 在可变分辨率场景下强行堆叠失败。
def _preserve_original_size_collate(batch: list[torch.Tensor]) -> list[torch.Tensor]:
    return batch


# 为不同机器上的同一数据集提供路径别名兜底，优先解析到真实存在的目录或文件。
def _resolve_dataset_field_path(
    config: dict[str, Any],
    dataset_name: str,
    field_name: str,
    field_value: Any,
) -> str | None:
    project_root = Path(config.get("project_root", Path.cwd()))
    if isinstance(field_value, str) and field_value:
        current_path = Path(field_value)
        if field_name == "split_dir":
            if current_path.exists():
                return str(current_path)
        elif current_path.exists():
            return str(current_path)

    for candidate in DATASET_PATH_ALIASES.get(dataset_name, {}).get(field_name, []):
        resolved = resolve_project_path(candidate, base_dir=project_root)
        if resolved is None:
            continue
        if field_name == "split_dir":
            return str(resolved)
        if resolved.exists():
            return str(resolved)

    if field_value in (None, ""):
        return None
    return str(field_value)


# 规范化当前启用数据集的关键路径，避免本地与服务器目录命名差异导致训练中途退出。
def normalize_active_dataset_config_paths(config: dict[str, Any]) -> dict[str, Any]:
    active_name = str(config["datasets"]["active"]).lower()
    dataset_cfg = config["datasets"][active_name]
    for field_name in ("train_dir", "val_dir", "test_dir", "train_split", "val_split", "test_split", "split_dir"):
        normalized = _resolve_dataset_field_path(
            config=config,
            dataset_name=active_name,
            field_name=field_name,
            field_value=dataset_cfg.get(field_name),
        )
        if normalized is None:
            continue
        dataset_cfg[field_name] = normalized
    return dataset_cfg


# 按固定随机种子抽取数据子集，便于先用较小数据量快速验证训练方向。
def maybe_build_subset(
    dataset: Dataset,
    fraction: float | None,
    seed: int,
) -> Dataset:
    if fraction is None:
        return dataset
    safe_fraction = float(fraction)
    if safe_fraction >= 1.0:
        return dataset
    if safe_fraction <= 0.0:
        raise ValueError(f"subset fraction must be positive, got {safe_fraction}.")
    dataset_size = len(dataset)
    subset_size = max(1, int(dataset_size * safe_fraction))
    if subset_size >= dataset_size:
        return dataset
    rng = random.Random(int(seed))
    indices = list(range(dataset_size))
    rng.shuffle(indices)
    return Subset(dataset, sorted(indices[:subset_size]))


# 根据是否保留原分辨率以及所处阶段，构建匹配的图像预处理流程。
def build_image_transform(
    image_size: int,
    keep_original_size: bool,
    is_train: bool,
) -> transforms.Compose:
    if keep_original_size:
        return transforms.Compose([transforms.ToTensor()])
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomCrop((image_size, image_size), pad_if_needed=True, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
    return transforms.Compose(
        [
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


class FlatImageDataset(Dataset):
    # 初始化平铺图像数据集，支持 DIV2K 的 PNG 图像和 BOSSbase 的 PGM 图像。
    def __init__(
        self,
        image_dir: str | Path,
        image_size: int,
        channels: int = 3,
        split_file: str | Path | None = None,
        paths: list[Path] | None = None,
        keep_original_size: bool = True,
        is_train: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.channels = channels
        self.keep_original_size = keep_original_size
        self.is_train = is_train
        self.paths = paths if paths is not None else self._load_paths(split_file)
        self.transform = build_image_transform(
            image_size=image_size,
            keep_original_size=self.keep_original_size,
            is_train=self.is_train,
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    # 根据目录或 split 文件加载图像路径，并提前校验路径是否存在。
    def _load_paths(self, split_file: str | Path | None) -> list[Path]:
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not self.image_dir.is_dir():
            raise NotADirectoryError(f"Image path is not a directory: {self.image_dir}")

        if split_file is None:
            return sorted(path for path in self.image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)

        split_path = Path(split_file)
        if not split_path.exists():
            raise FileNotFoundError(f"Split file does not exist: {split_path}")

        names = split_path.read_text(encoding="utf-8").splitlines()
        paths = [self.image_dir / name.strip() for name in names if name.strip()]
        missing = [path for path in paths if not path.exists()]
        if missing:
            preview = ", ".join(str(path) for path in missing[:5])
            raise FileNotFoundError(f"{len(missing)} split images are missing. Examples: {preview}")

        invalid = [path for path in paths if path.suffix.lower() not in IMAGE_SUFFIXES]
        if invalid:
            preview = ", ".join(str(path) for path in invalid[:5])
            raise ValueError(f"{len(invalid)} split files use unsupported image suffixes. Examples: {preview}")
        return paths

    # 返回当前数据集中的图像数量。
    def __len__(self) -> int:
        return len(self.paths)

    # 读取单张图像，并转换成模型配置要求的通道数和张量格式。
    def __getitem__(self, index: int) -> torch.Tensor:
        mode = "RGB" if self.channels == 3 else "L"
        with Image.open(self.paths[index]) as image:
            image = image.convert(mode)
            tensor = self.transform(image)

        if tensor.shape[0] != self.channels:
            if self.channels == 3 and tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            else:
                raise ValueError(f"Unexpected channel count {tensor.shape[0]} for {self.paths[index]}")
        return tensor


# 根据配置读取当前启用的数据集配置。
def get_active_dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    active_name = config["datasets"]["active"]
    if active_name not in config["datasets"]:
        raise KeyError(f"Active dataset '{active_name}' is not configured.")
    return normalize_active_dataset_config_paths(config)


# 汇总当前启用数据集的关键路径，便于远端启动前快速确认目录和切分文件是否存在。
def describe_active_dataset_paths(config: dict[str, Any]) -> dict[str, str]:
    dataset_cfg = get_active_dataset_config(config)
    summary: dict[str, str] = {
        "active_dataset": str(config["datasets"]["active"]),
    }
    for key in ("train_dir", "val_dir", "train_split", "val_split", "test_dir", "test_split"):
        value = dataset_cfg.get(key)
        if value is None or value == "":
            continue
        path = Path(value)
        summary[key] = str(path)
        summary[f"{key}_exists"] = str(path.exists())
    return summary


# 将相对路径列表写入 split 文件，保证训练、验证和评估读取同一份固定划分。
def write_split_file(split_file: str | Path, image_dir: str | Path, paths: list[Path]) -> None:
    output = Path(split_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(image_dir)
    names = []
    for path in paths:
        try:
            names.append(path.relative_to(root).as_posix())
        except ValueError:
            names.append(path.name)
    output.write_text("\n".join(names) + "\n", encoding="utf-8")


# 在没有显式 split 文件时生成固定划分，避免训练和评估阶段各自重新切分。
def build_or_load_auto_split(
    dataset_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    keep_original_size: bool,
) -> tuple[list[Path], list[Path], str | None]:
    train_dir = dataset_cfg["train_dir"]
    split_dir = Path(dataset_cfg.get("split_dir") or Path(train_dir) / "_splits")
    train_split = dataset_cfg.get("train_split") or str(split_dir / "train.txt")
    val_split = dataset_cfg.get("val_split") or str(split_dir / "val.txt")
    seed = int(dataset_cfg.get("split_seed", 42))

    if Path(train_split).exists() and Path(val_split).exists():
        train_paths = FlatImageDataset(
            image_dir=train_dir,
            image_size=data_cfg["image_size"],
            channels=data_cfg["channels"],
            split_file=train_split,
            keep_original_size=keep_original_size,
        ).paths
        val_paths = FlatImageDataset(
            image_dir=train_dir,
            image_size=data_cfg["image_size"],
            channels=data_cfg["channels"],
            split_file=val_split,
            keep_original_size=keep_original_size,
        ).paths
        dataset_cfg["train_split"] = train_split
        dataset_cfg["val_split"] = val_split
        return train_paths, val_paths, val_split

    all_paths = FlatImageDataset(
        image_dir=train_dir,
        image_size=data_cfg["image_size"],
        channels=data_cfg["channels"],
        keep_original_size=keep_original_size,
    ).paths
    rng = random.Random(seed)
    shuffled_paths = all_paths[:]
    rng.shuffle(shuffled_paths)
    val_count = max(1, int(len(shuffled_paths) * float(dataset_cfg["auto_split_ratio"])))
    val_paths = sorted(shuffled_paths[:val_count])
    train_paths = sorted(shuffled_paths[val_count:])
    write_split_file(train_split, train_dir, train_paths)
    write_split_file(val_split, train_dir, val_paths)
    dataset_cfg["train_split"] = train_split
    dataset_cfg["val_split"] = val_split
    return train_paths, val_paths, val_split


# 根据当前启用的数据集配置构建训练和验证 DataLoader。
def build_train_val_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader | None]:
    return build_train_val_loaders_with_samplers(config)


# 根据当前启用的数据集配置构建训练和验证 DataLoader，并可选接入分布式采样器。
def build_train_val_loaders_with_samplers(
    config: dict[str, Any],
    train_sampler=None,
    val_sampler=None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader | None]:
    dataset_cfg = get_active_dataset_config(config)
    data_cfg = config["data"]
    train_cfg = config["training"]
    keep_original_size = data_cfg.get("keep_original_size", True)
    subset_cfg = config.get("subset", {})
    subset_seed = int(subset_cfg.get("seed", config.get("seed", 42)))
    train_subset_fraction = subset_cfg.get("train_fraction")
    val_subset_fraction = subset_cfg.get("val_fraction", train_subset_fraction)

    auto_split_ratio = dataset_cfg.get("auto_split_ratio")
    if auto_split_ratio is not None and not dataset_cfg.get("val_dir"):
        train_paths, val_paths, _ = build_or_load_auto_split(dataset_cfg, data_cfg, keep_original_size)
        train_dataset = FlatImageDataset(
            image_dir=dataset_cfg["train_dir"],
            image_size=data_cfg["image_size"],
            channels=data_cfg["channels"],
            paths=train_paths,
            keep_original_size=keep_original_size,
            is_train=True,
        )
        val_dataset = FlatImageDataset(
            image_dir=dataset_cfg["train_dir"],
            image_size=data_cfg["image_size"],
            channels=data_cfg["channels"],
            paths=val_paths,
            keep_original_size=keep_original_size,
            is_train=False,
        )
    else:
        train_dataset = FlatImageDataset(
            image_dir=dataset_cfg["train_dir"],
            image_size=data_cfg["image_size"],
            channels=data_cfg["channels"],
            split_file=dataset_cfg.get("train_split"),
            keep_original_size=keep_original_size,
            is_train=True,
        )
        val_dataset = None
    if val_dataset is None and dataset_cfg.get("val_dir"):
        val_dataset = FlatImageDataset(
            image_dir=dataset_cfg["val_dir"],
            image_size=data_cfg["image_size"],
            channels=data_cfg["channels"],
            split_file=dataset_cfg.get("val_split"),
            keep_original_size=keep_original_size,
            is_train=False,
        )
    train_dataset = maybe_build_subset(train_dataset, train_subset_fraction, subset_seed)
    if val_dataset is not None:
        val_dataset = maybe_build_subset(val_dataset, val_subset_fraction, subset_seed + 1)
    if distributed and train_sampler is None:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
    if distributed and val_dataset is not None and val_sampler is None:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=train_cfg.get("num_workers", 0) > 0,
        collate_fn=_preserve_original_size_collate if keep_original_size else None,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=False if val_sampler is None else False,
            sampler=val_sampler,
            drop_last=False,
            num_workers=train_cfg.get("num_workers", 0),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=train_cfg.get("num_workers", 0) > 0,
            collate_fn=_preserve_original_size_collate if keep_original_size else None,
        )
    return train_loader, val_loader


# 根据配置构建评估 DataLoader，默认使用当前数据集的验证目录。
def build_eval_loader(
    config: dict[str, Any],
    image_dir: str | Path | None = None,
    split_file: str | Path | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> DataLoader:
    dataset_cfg = get_active_dataset_config(config)
    data_cfg = config["data"]
    keep_original_size = data_cfg.get("keep_original_size", True)
    using_validation_source = False
    if image_dir is not None:
        eval_dir = image_dir
        eval_split = split_file
    else:
        test_split = dataset_cfg.get("test_split")
        if test_split:
            eval_dir = dataset_cfg.get("test_dir") or dataset_cfg.get("val_dir") or dataset_cfg["train_dir"]
            eval_split = test_split
        elif dataset_cfg.get("val_split"):
            eval_dir = dataset_cfg.get("val_dir") or dataset_cfg["train_dir"]
            eval_split = dataset_cfg.get("val_split")
            using_validation_source = True
        elif dataset_cfg.get("val_dir"):
            eval_dir = dataset_cfg["val_dir"]
            eval_split = split_file
            using_validation_source = True
        elif dataset_cfg.get("auto_split_ratio") is not None:
            _, _, generated_val_split = build_or_load_auto_split(dataset_cfg, data_cfg, keep_original_size)
            eval_dir = dataset_cfg["train_dir"]
            eval_split = generated_val_split
            using_validation_source = True
        else:
            raise ValueError(
                "Evaluation requires val_dir, val_split, test_split, auto_split_ratio, "
                "or explicit --data/--split to avoid evaluating on the training directory."
            )

    dataset = FlatImageDataset(
        image_dir=eval_dir,
        image_size=data_cfg["image_size"],
        channels=data_cfg["channels"],
        split_file=eval_split,
        keep_original_size=keep_original_size,
        is_train=False,
    )
    subset_cfg = config.get("subset", {})
    eval_fraction = subset_cfg.get("eval_fraction")
    if eval_fraction is None and image_dir is None and split_file is None and using_validation_source:
        # 默认评估直接复用完整验证集；若还想对子集评估，需显式设置 eval_fraction。
        eval_fraction = None
    elif eval_fraction is None:
        eval_fraction = subset_cfg.get("val_fraction")
    base_subset_seed = int(subset_cfg.get("seed", config.get("seed", 42)))
    if image_dir is None and split_file is None and using_validation_source and "eval_fraction" not in subset_cfg:
        # 当评估默认复用验证集来源且没有单独配置 eval_fraction 时，
        # 复用训练期验证子集的采样种子；若 eval_fraction 为空则不会再做第二次子采样。
        subset_seed = base_subset_seed + 1
    else:
        subset_seed = base_subset_seed + 2
    dataset = maybe_build_subset(dataset, eval_fraction, subset_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size or config["training"]["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=num_workers if num_workers is not None else config["training"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers if num_workers is not None else config["training"].get("num_workers", 0)) > 0,
        collate_fn=_preserve_original_size_collate if keep_original_size else None,
    )
