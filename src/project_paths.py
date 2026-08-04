from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PAPER_DIR = PROJECT_ROOT / "paper"
ARCHIVE_DIR = PROJECT_ROOT / "archive"
DATASET_DIR = PROJECT_ROOT / "dataset"
CONFIGS_DIR = PROJECT_ROOT / "configs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
EVAL_RESULTS_DIR = ARTIFACTS_DIR / "eval_results"
LOGS_DIR = ARTIFACTS_DIR / "logs"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# Legacy-compatible root aliases kept for older scripts and docs.
LEGACY_CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LEGACY_EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results"
LEGACY_LOGS_DIR = PROJECT_ROOT / "logs"
LEGACY_REPORTS_DIR = PROJECT_ROOT / "reports"


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def resolve_from_project(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (PROJECT_ROOT / candidate).resolve()


def ensure_standard_dirs() -> None:
    for directory in (
        ARTIFACTS_DIR,
        PAPER_DIR,
        ARCHIVE_DIR,
        CHECKPOINTS_DIR,
        EVAL_RESULTS_DIR,
        LOGS_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
