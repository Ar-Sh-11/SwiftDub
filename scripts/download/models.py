#!/usr/bin/env python3
"""Download LatentSync model weights for SwiftDub (no vendor repo required).

Usage:
  python scripts/download/models.py --model latentsync
  python scripts/download/models.py --model latentsync --latentsync-version 1.5

Weights are stored under models/weights/ only. Inference code lives in src/.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402


def info(msg: str) -> None:
    print(f"\033[0;36m==> {msg}\033[0m")


def hf_download(repo_id: str, filename: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(local_dir))
    return Path(path)


def load_configs() -> dict:
    return yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())


def install_insightface_models(dest: Path) -> None:
    """Download buffalo_l face detection models for InsightFace."""
    target = dest / "models" / "buffalo_l"
    if target.exists() and any(target.iterdir()):
        info("InsightFace buffalo_l already present")
        return

    zip_path = dest / "buffalo_l.zip"
    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    info("Downloading InsightFace buffalo_l (~280 MB)")
    dest.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest / "models")
    zip_path.unlink(missing_ok=True)
    info(f"InsightFace models ready: {target}")


def install_latentsync(version: str = "1.5") -> None:
    cfg = load_configs()["latentsync"]
    ckpt = ROOT / cfg["checkpoint"]
    whisper = ROOT / cfg["whisper_checkpoint"]
    insightface_dir = settings.insightface_root

    hf_repo = f"ByteDance/LatentSync-{version}"
    whisper_name = "tiny.pt" if version == "1.5" else "small.pt"

    if not ckpt.exists():
        info(f"Downloading LatentSync {version} checkpoint (~5 GB)")
        hf_download(hf_repo, "latentsync_unet.pt", ckpt.parent)
    else:
        info(f"LatentSync checkpoint exists: {ckpt.name}")

    if not whisper.exists():
        info(f"Downloading Whisper encoder ({whisper_name})")
        hf_download(hf_repo, f"whisper/{whisper_name}", whisper.parent)
    else:
        info(f"Whisper checkpoint exists: {whisper.name}")

    install_insightface_models(insightface_dir)
    print("  LatentSync ✓")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SwiftDub model weights (weights only)")
    parser.add_argument("--model", choices=["latentsync", "insightface"], default="latentsync")
    parser.add_argument("--latentsync-version", choices=["1.5", "1.6"], default="1.5")
    args = parser.parse_args()

    if args.model == "insightface":
        install_insightface_models(settings.insightface_root)
    else:
        install_latentsync(args.latentsync_version)

    print("\nDone. Start server: python -m src.main")


if __name__ == "__main__":
    main()
