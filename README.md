# WaveBridege

WaveBridege is the cleaned core codebase for training, evaluation, and single-image inference of the WaveBridege image steganography system.

## Core structure

```text
configs/
  default.yaml
models/
utils/
src/
  project_paths.py
scripts/
  cli/unified_cli.py
  data/prepare_bossbase_split.py
  server/server_entry.py
  train/main.py
main.py
evaluate.py
train_server.py
server_entry.py
prepare_bossbase_split.py
wavebridege_cli.py
pyproject.toml
requirements.txt
README.md
```

## Main entrypoints

- `train_server.py`: main training implementation
- `evaluate.py`: main evaluation entrypoint
- `main.py`: single-image inference entrypoint
- `server_entry.py`: preflight, smoke, train, and eval wrapper
- `prepare_bossbase_split.py`: prepares BOSSBase train/val split files
- `wavebridege_cli.py`: unified command wrapper for common workflows

## Quick start

Train:

```bash
python wavebridege_cli.py train --config configs/default.yaml
```

Smoke test:

```bash
python wavebridege_cli.py smoke --config configs/default.yaml
```

Evaluate:

```bash
python wavebridege_cli.py eval --config configs/default.yaml --checkpoint checkpoints/example.pt --output-dir eval_results/example
```

Single-image inference:

```bash
python wavebridege_cli.py infer --config configs/default.yaml --image path/to/input.png --checkpoint checkpoints/example.pt
```

Prepare BOSSBase split files:

```bash
python wavebridege_cli.py split-bossbase --data-dir dataset/BOSSbase_1.01 --output-dir dataset/splits/bossbase
```

## Notes

- `configs/default.yaml` is the only retained public config in this cleaned repository.
- Relative paths are resolved against the project root.
- Non-core paper scripts, plotting helpers, compatibility-layer wrappers, temporary experiment scripts, and table-export artifacts were intentionally removed.
