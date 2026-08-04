from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _script(path: str) -> Path:
    return PROJECT_ROOT / path


def _run(script_path: Path, script_args: list[str]) -> int:
    command = [sys.executable, str(script_path), *script_args]
    return subprocess.call(command, cwd=PROJECT_ROOT)


def _add_common_io(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args forwarded after --")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CLI for common WaveBridege core workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Launch server-style training")
    _add_common_io(train_parser)
    train_parser.add_argument("--train-save", default=None)
    train_parser.add_argument("--train-log", default=None)

    smoke_parser = subparsers.add_parser("smoke", help="Launch smoke training")
    _add_common_io(smoke_parser)
    smoke_parser.add_argument("--smoke-save", default=None)
    smoke_parser.add_argument("--smoke-log", default=None)

    eval_parser = subparsers.add_parser("eval", help="Run main evaluation")
    _add_common_io(eval_parser)
    eval_parser.add_argument("--checkpoint", default=None)
    eval_parser.add_argument("--output-dir", default=None)
    eval_parser.add_argument("--max-images", type=int, default=None)

    infer_parser = subparsers.add_parser("infer", help="Run single-image inference")
    infer_parser.add_argument("--config", default="configs/default.yaml")
    infer_parser.add_argument("--image", required=True)
    infer_parser.add_argument("--checkpoint", default=None)
    infer_parser.add_argument("--stego-output", default=None)
    infer_parser.add_argument("--restore-output", default=None)
    infer_parser.add_argument("--noise-mode", default=None)
    infer_parser.add_argument("--noise-seed", type=int, default=None)
    infer_parser.add_argument("--noise-std", type=float, default=None)
    infer_parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args forwarded after --")

    split_parser = subparsers.add_parser("split-bossbase", help="Prepare BOSSBase split files")
    split_parser.add_argument("--data-dir", default=None)
    split_parser.add_argument("--output-dir", default=None)
    split_parser.add_argument("--train-count", type=int, default=None)
    split_parser.add_argument("--val-count", type=int, default=None)
    split_parser.add_argument("--seed", type=int, default=None)
    split_parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args forwarded after --")

    return parser


def _append_optional(args_list: list[str], flag: str, value) -> None:
    if value is not None:
        args_list.extend([flag, str(value)])


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "train":
        cmd = ["train", "--config", args.config]
        _append_optional(cmd, "--device", args.device)
        _append_optional(cmd, "--train-save", args.train_save)
        _append_optional(cmd, "--train-log", args.train_log)
        cmd.extend(args.extra)
        return _run(_script("server_entry.py"), cmd)

    if args.command == "smoke":
        cmd = ["smoke", "--config", args.config]
        _append_optional(cmd, "--device", args.device)
        _append_optional(cmd, "--smoke-save", args.smoke_save)
        _append_optional(cmd, "--smoke-log", args.smoke_log)
        cmd.extend(args.extra)
        return _run(_script("server_entry.py"), cmd)

    if args.command == "eval":
        cmd = ["--config", args.config]
        _append_optional(cmd, "--device", args.device)
        _append_optional(cmd, "--checkpoint", args.checkpoint)
        _append_optional(cmd, "--output-dir", args.output_dir)
        _append_optional(cmd, "--max-images", args.max_images)
        cmd.extend(args.extra)
        return _run(_script("evaluate.py"), cmd)

    if args.command == "infer":
        cmd = ["--config", args.config, "--image", args.image]
        _append_optional(cmd, "--checkpoint", args.checkpoint)
        _append_optional(cmd, "--stego-output", args.stego_output)
        _append_optional(cmd, "--restore-output", args.restore_output)
        _append_optional(cmd, "--noise-mode", args.noise_mode)
        _append_optional(cmd, "--noise-seed", args.noise_seed)
        _append_optional(cmd, "--noise-std", args.noise_std)
        cmd.extend(args.extra)
        return _run(_script("main.py"), cmd)

    if args.command == "split-bossbase":
        cmd: list[str] = []
        _append_optional(cmd, "--data-dir", args.data_dir)
        _append_optional(cmd, "--output-dir", args.output_dir)
        _append_optional(cmd, "--train-count", args.train_count)
        _append_optional(cmd, "--val-count", args.val_count)
        _append_optional(cmd, "--seed", args.seed)
        cmd.extend(args.extra)
        return _run(_script("prepare_bossbase_split.py"), cmd)

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
