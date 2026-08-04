from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from src.project_paths import PROJECT_ROOT, resolve_from_project

PATH_FIELD_SUFFIXES = ("_dir", "_file", "_path", "_split", "_weights")


# 将相对路径优先解析到项目根目录，避免在不同工作目录下启动时找不到文件。
def resolve_project_path(path: str | Path | None, base_dir: str | Path | None = None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if base_dir is not None:
        base_candidate = (Path(base_dir).expanduser().resolve() / candidate).resolve()
        if base_candidate.exists():
            return base_candidate
    project_candidate = resolve_from_project(candidate)
    if project_candidate is None:
        return None
    if project_candidate.exists():
        return project_candidate
    if base_dir is not None:
        return (Path(base_dir).expanduser().resolve() / candidate).resolve()
    return project_candidate


# 递归归一化配置中的目录、文件和切分路径，保证训练和评估统一使用绝对路径。
def normalize_config_paths(node: Any, base_dir: str | Path | None = None) -> Any:
    if isinstance(node, dict):
        normalized = {}
        for key, value in node.items():
            if isinstance(value, str) and value and key.endswith(PATH_FIELD_SUFFIXES):
                normalized[key] = str(resolve_project_path(value, base_dir=base_dir))
            else:
                normalized[key] = normalize_config_paths(value, base_dir=base_dir)
        return normalized
    if isinstance(node, list):
        return [normalize_config_paths(item, base_dir=base_dir) for item in node]
    return node


# 读取 YAML 配置文件，并把其中的路径字段统一解析成绝对路径。
def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = resolve_project_path(path)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must parse to a dictionary: {config_path}")
    normalized = normalize_config_paths(config, base_dir=config_path.parent)
    normalized["config_path"] = str(config_path)
    normalized["project_root"] = str(PROJECT_ROOT)
    return normalized
