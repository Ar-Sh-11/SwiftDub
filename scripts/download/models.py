#!/usr/bin/env python3
"""Download LatentSync 1.5 repo and weights for SwiftDub.

Usage:
  python scripts/download/models.py
  python scripts/download/models.py --latentsync-version 1.5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402


def info(msg: str) -> None:
    print(f"\033[0;36m==> {msg}\033[0m")


def clone(url: str, dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        info(f"Repo exists: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    info(f"Cloning {url}")
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def hf_download(repo_id: str, filename: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    local_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(local_dir))
    return Path(path)


def load_config() -> dict:
    cfg_path = ROOT / "configs" / "models.yaml"
    return yaml.safe_load(cfg_path.read_text())["latentsync"]


def install_latentsync(version: str = "1.5") -> None:
    cfg = load_config()
    repo_dir = ROOT / cfg["repo_dir"]
    ckpt = ROOT / cfg["checkpoint"]
    whisper = ROOT / cfg["whisper_checkpoint"]

    clone(cfg["repo_url"], repo_dir)

    hf_repo = f"ByteDance/LatentSync-{version}"
    whisper_name = "tiny.pt" if version == "1.5" else "small.pt"

    if not ckpt.exists():
        info(f"Downloading LatentSync {version} checkpoint (~5.1 GB)")
        hf_download(hf_repo, "latentsync_unet.pt", ckpt.parent)

    if not whisper.exists():
        info(f"Downloading Whisper encoder ({whisper_name})")
        hf_download(hf_repo, f"whisper/{whisper_name}", whisper.parent)

    print(f"\n  LatentSync {version} ready")
    print(f"  Repo:  {repo_dir}")
    print(f"  Ckpt:  {ckpt}")
    print(f"  Start: python -m src.main")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SwiftDub LatentSync assets")
    parser.add_argument("--latentsync-version", choices=["1.5", "1.6"], default="1.5")
    args = parser.parse_args()
    info("Installing LatentSync for SwiftDub")
    install_latentsync(args.latentsync_version)


if __name__ == "__main__":
    main()
