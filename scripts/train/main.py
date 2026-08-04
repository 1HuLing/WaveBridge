from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from torchvision import transforms

from src.project_paths import project_path
from models.system import WaveBridegeSystem
from utils.config import load_config, resolve_project_path
from utils.metrics import psnr
from utils.runtime import load_checkpoint_config, load_model_checkpoint, resolve_device, set_seed


# 读取输入图像并按配置固定到载密图像训练分辨率。
def load_image(path: str | Path, image_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)


# 以当前运行配置为主，仅把 checkpoint 中缺失的结构字段补齐，避免旧 checkpoint 覆盖新的推理口径。
def merge_checkpoint_runtime_config(runtime_config: dict, checkpoint_config: dict | None) -> dict:
    merged = dict(runtime_config)
    if not isinstance(checkpoint_config, dict):
        return merged
    compatible_sections = {
        "compressor",
        "generator",
        "receiver",
        "transmitter",
        "qim",
        "gan",
        "noise",
        "robust_channel",
        "steganalysis",
    }
    for section_name in compatible_sections:
        checkpoint_section = checkpoint_config.get(section_name)
        runtime_section = merged.get(section_name)
        if isinstance(checkpoint_section, dict) and isinstance(runtime_section, dict):
            for key, value in checkpoint_section.items():
                if key not in runtime_section:
                    runtime_section[key] = value
    if "device" not in merged and checkpoint_config.get("device") is not None:
        merged["device"] = checkpoint_config["device"]
    if "device_index" not in merged and checkpoint_config.get("device_index") is not None:
        merged["device_index"] = checkpoint_config["device_index"]
    return merged


# 将模型输出的张量保存为图像文件。
def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    image = tensor.squeeze(0).detach().cpu().clamp(0.0, 1.0)
    transforms.ToPILImage()(image).save(path)


# 执行单张图像推理，并保存载密图和恢复图。
def run_inference(
    config_path: str,
    image_path: str,
    stego_output_path: str,
    restore_output_path: str,
    checkpoint_path: str | None = None,
    noise_mode: str | None = None,
    noise_seed: int | None = None,
    noise_std: float | None = None,
) -> None:
    config_path = str(resolve_project_path(config_path))
    image_path = str(resolve_project_path(image_path))
    stego_output_path = str(resolve_project_path(stego_output_path))
    restore_output_path = str(resolve_project_path(restore_output_path))
    checkpoint_path = None if checkpoint_path is None else str(resolve_project_path(checkpoint_path))
    config = load_config(config_path)
    checkpoint_config = load_checkpoint_config(checkpoint_path) if checkpoint_path else None
    if checkpoint_config is not None:
        config = merge_checkpoint_runtime_config(config, checkpoint_config)
    config.setdefault("noise", {})
    if noise_mode is not None:
        config["noise"]["mode"] = noise_mode
    if noise_seed is not None:
        config["noise"]["seed"] = noise_seed
    if noise_std is not None:
        config["noise"]["std"] = noise_std
    set_seed(config.get("seed", 42))
    device = resolve_device(config.get("device"), preferred_index=config.get("device_index", 1))
    config["device"] = str(device)
    config["device_index"] = device.index if device.type == "cuda" else None
    if device.type == "cuda":
        print(f"Using CUDA device: {device} - {torch.cuda.get_device_name(device)}")
    image = load_image(image_path, config["data"]["image_size"]).to(device)

    model = WaveBridegeSystem(config).to(device)
    if checkpoint_path:
        load_model_checkpoint(model, checkpoint_path)
    model.eval()

    with torch.inference_mode():
        output = model(image, force_full_decode=True)

    save_image(output.generator.stego_image, stego_output_path)
    save_image(output.receiver.restored_image, restore_output_path)
    stego_psnr = psnr(image, output.generator.stego_image)
    restored_psnr = psnr(image, output.receiver.restored_image)
    print(f"Stego image saved to: {stego_output_path}")
    print(f"Restored image saved to: {restore_output_path}")
    print(f"Stego PSNR: {stego_psnr:.4f} dB")
    print(f"Restored PSNR: {restored_psnr:.4f} dB")
    print(f"Decoded bit shape: {tuple(output.receiver.decoded_bits.shape)}")


# 解析命令行参数并提供推理入口配置。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WaveBridege single-image pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument(
        "--stego-output",
        "--cover-output",
        dest="stego_output",
        default="outputs/stego.png",
        help="Stego image output path",
    )
    parser.set_defaults(stego_output=str(project_path("output", "stego.png")))
    parser.add_argument("--restore-output", default=str(project_path("output", "restored.png")), help="Restored image output path")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path")
    parser.add_argument("--noise-mode", default=None, choices=["random", "landscape"], help="Override generator noise mode")
    parser.add_argument("--noise-seed", type=int, default=None, help="Override fixed landscape latent seed")
    parser.add_argument("--noise-std", type=float, default=None, help="Override landscape latent perturbation std")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stego_output = resolve_project_path(args.stego_output)
    restore_output = resolve_project_path(args.restore_output)
    if stego_output is not None:
        stego_output.parent.mkdir(parents=True, exist_ok=True)
        args.stego_output = str(stego_output)
    if restore_output is not None:
        restore_output.parent.mkdir(parents=True, exist_ok=True)
        args.restore_output = str(restore_output)
    run_inference(
        args.config,
        args.image,
        args.stego_output,
        args.restore_output,
        checkpoint_path=args.checkpoint,
        noise_mode=args.noise_mode,
        noise_seed=args.noise_seed,
        noise_std=args.noise_std,
    )
