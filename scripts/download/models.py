#!/usr/bin/env python3
"""Download model repos and weights for SwiftDub (LatentSync + MuseTalk).

Usage:
  python scripts/download/models.py --model all
  python scripts/download/models.py --model latentsync
  python scripts/download/models.py --model musetalk
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


def load_configs() -> dict:
    return yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())


def install_latentsync(version: str = "1.5") -> None:
    cfg = load_configs()["latentsync"]
    repo_dir = ROOT / cfg["repo_dir"]
    ckpt = ROOT / cfg["checkpoint"]
    whisper = ROOT / cfg["whisper_checkpoint"]

    clone(cfg["repo_url"], repo_dir)
    hf_repo = f"ByteDance/LatentSync-{version}"
    whisper_name = "tiny.pt" if version == "1.5" else "small.pt"

    if not ckpt.exists():
        info(f"Downloading LatentSync {version} checkpoint (~1.5 GB)")
        hf_download(hf_repo, "latentsync_unet.pt", ckpt.parent)
    if not whisper.exists():
        info(f"Downloading Whisper encoder ({whisper_name})")
        hf_download(hf_repo, f"whisper/{whisper_name}", whisper.parent)
    print("  LatentSync ✓")


def install_musetalk() -> None:
    cfg = load_configs()["musetalk"]
    repo_dir = ROOT / cfg["repo_dir"]
    unet = ROOT / cfg["unet_path"]
    unet_cfg = ROOT / cfg["unet_config"]

    clone(cfg["repo_url"], repo_dir)

    if not unet.exists():
        info("Downloading MuseTalk v1.5 UNet weights from HuggingFace")
        hf_download(cfg["hf_repo"], "musetalkV15/unet.pth", unet.parent)
    if not unet_cfg.exists():
        hf_download(cfg["hf_repo"], "musetalkV15/musetalk.json", unet_cfg.parent)

    # DWPose checkpoint (bundled in repo models/dwpose — verify)
    dwpose = repo_dir / "models" / "dwpose" / "dw-ll_ucoco_384.pth"
    if not dwpose.exists():
        info("[warn] DWPose weights missing at {}", dwpose)
    print("  MuseTalk ✓")


INSTALLERS = {
    "latentsync": install_latentsync,
    "musetalk": install_musetalk,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SwiftDub model assets")
    parser.add_argument("--model", choices=["all", "latentsync", "musetalk"], default="all")
    parser.add_argument("--latentsync-version", choices=["1.5", "1.6"], default="1.5")
    args = parser.parse_args()

    targets = list(INSTALLERS) if args.model == "all" else [args.model]
    for name in targets:
        info(f"Installing {name}")
        if name == "latentsync":
            install_latentsync(args.latentsync_version)
        else:
            INSTALLERS[name]()

    print("\nDone. Start server: python -m src.main")


if __name__ == "__main__":
    main()
