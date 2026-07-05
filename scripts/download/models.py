#!/usr/bin/env python3
"""Download vendor inference snapshots and model weights for SwiftDub.

Usage:
  python scripts/download/models.py --model all
  python scripts/download/models.py --model latentsync
  python scripts/download/models.py --model musetalk

Vendor code is stored under models/vendor/ as a plain directory snapshot (no .git).
Only SwiftDub is a git repo — vendor trees are gitignored runtime assets.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402


def info(msg: str) -> None:
    print(f"\033[0;36m==> {msg}\033[0m")


def vendor_snapshot(url: str, dest: Path) -> None:
    """Download upstream inference code once; strip .git so it is not a separate repo."""
    if dest.exists() and any(dest.iterdir()):
        info(f"Vendor snapshot exists: {dest.name}")
        _strip_git(dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".__tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    info(f"Fetching vendor snapshot: {url}")
    subprocess.run(["git", "clone", "--depth", "1", url, str(tmp)], check=True)
    _strip_git(tmp)
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    info(f"Vendor ready (no git remote): {dest}")


def _strip_git(path: Path) -> None:
    git_dir = path / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir)
        info(f"Removed .git from {path.name} — not a separate repo")


def hf_download(repo_id: str, filename: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(local_dir))
    return Path(path)


def load_configs() -> dict:
    return yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())


def install_latentsync(version: str = "1.5") -> None:
    cfg = load_configs()["latentsync"]
    vendor_dir = ROOT / cfg["vendor_dir"]
    ckpt = ROOT / cfg["checkpoint"]
    whisper = ROOT / cfg["whisper_checkpoint"]

    vendor_snapshot(cfg["vendor_url"], vendor_dir)
    hf_repo = f"ByteDance/LatentSync-{version}"
    whisper_name = "tiny.pt" if version == "1.5" else "small.pt"

    if not ckpt.exists():
        info(f"Downloading LatentSync {version} checkpoint (~1.5 GB)")
        hf_download(hf_repo, "latentsync_unet.pt", ckpt.parent)
    if not whisper.exists():
        info(f"Downloading Whisper encoder ({whisper_name})")
        hf_download(hf_repo, f"whisper/{whisper_name}", whisper.parent)

    _apply_patches()
    print("  LatentSync ✓")


def install_musetalk() -> None:
    cfg = load_configs()["musetalk"]
    vendor_dir = ROOT / cfg["vendor_dir"]
    unet = ROOT / cfg["unet_path"]
    unet_cfg = ROOT / cfg["unet_config"]
    whisper_dir = ROOT / cfg["whisper_dir"]
    dwpose = ROOT / cfg["dwpose_checkpoint"]
    face_parse = ROOT / cfg["face_parse_dir"]

    vendor_snapshot(cfg["vendor_url"], vendor_dir)

    if not unet.exists():
        info("Downloading MuseTalk v1.5 UNet weights from HuggingFace")
        hf_download(cfg["hf_repo"], "musetalkV15/unet.pth", unet.parent)
    if not unet_cfg.exists():
        hf_download(cfg["hf_repo"], "musetalkV15/musetalk.json", unet_cfg.parent)

    if not (whisper_dir / "config.json").exists():
        info("Downloading MuseTalk Whisper assets")
        for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
            hf_download("openai/whisper-tiny", name, whisper_dir)

    if not dwpose.exists():
        info("Downloading DWPose weights")
        hf_download("yzd-v/DWPose", "dw-ll_ucoco_384.pth", dwpose.parent)

    if not (face_parse / "79999_iter.pth").exists():
        info("Downloading face-parse-bisent weights")
        hf_download("ManyOtherFunctions/face-parse-bisent", "79999_iter.pth", face_parse)
        import urllib.request

        face_parse.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://download.pytorch.org/models/resnet18-5c106cde.pth",
            face_parse / "resnet18-5c106cde.pth",
        )

    _wire_musetalk_vendor(vendor_dir)
    _apply_patches()
    print("  MuseTalk ✓")


def _apply_patches() -> None:
    from scripts.vendor.patches import apply_all

    apply_all()


def _symlink_vendor_asset(link_path: Path, target: Path) -> None:
    target = target.resolve()
    if not target.exists():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        else:
            shutil.rmtree(link_path)
    link_path.symlink_to(target, target_is_directory=target.is_dir())


def _wire_musetalk_vendor(vendor_dir: Path) -> None:
    weights = ROOT / "models" / "weights" / "musetalk"
    for rel, name in (
        ("models/whisper", "whisper"),
        ("models/dwpose", "dwpose"),
        ("models/face-parse-bisent", "face-parse-bisent"),
    ):
        _symlink_vendor_asset(vendor_dir / rel, weights / name)


INSTALLERS = {
    "latentsync": install_latentsync,
    "musetalk": install_musetalk,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SwiftDub vendor snapshots and weights")
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
