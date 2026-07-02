#!/usr/bin/env python3
"""Download model repos and checkpoints for all SwiftDub backends.

Usage:
  python scripts/download/models.py --model all
  python scripts/download/models.py --model wav2lip
  python scripts/download/models.py --model video_retalking
  python scripts/download/models.py --model musetalk
  python scripts/download/models.py --model latentsync --latentsync-version 1.5
  python scripts/download/models.py --model sadtalker
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

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


def download(url: str, dest: Path, *, optional: bool = False) -> bool:
    if dest.exists():
        info(f"Already present: {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    info(f"Downloading → {dest.name}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SwiftDub/1.0"})
        with urllib.request.urlopen(req, timeout=300) as r, dest.open("wb") as f:
            f.write(r.read())
        return True
    except Exception as exc:
        if optional:
            print(f"  [skip] {exc}")
            return False
        raise


def hf_download(repo_id: str, filename: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    local_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(local_dir))
    return Path(path)


# ── Per-model installers ──────────────────────────────────────────────────────

def install_wav2lip() -> None:
    cfg = settings.model_configs["wav2lip"]
    repo_dir = ROOT / cfg["repo_dir"]
    ckpt = ROOT / cfg["checkpoint"]
    s3fd = ROOT / cfg["s3fd_checkpoint"]

    clone(cfg["repo_url"], repo_dir)

    if not ckpt.exists():
        import gdown
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        info("Downloading Wav2Lip GAN checkpoint (~436 MB)")
        gdown.download(id="10Iu05Modfti3pDbxCFPnofmfVlbkvrCm", output=str(ckpt), quiet=False)

    download(cfg["s3fd_url"], s3fd)
    print("  Wav2Lip ✓")


def install_video_retalking() -> None:
    cfg = settings.model_configs["video_retalking"]
    repo_dir = ROOT / cfg["repo_dir"]
    ckpt_dir = ROOT / cfg["checkpoint_dir"]

    clone(cfg["repo_url"], repo_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for item in cfg["checkpoints"]:
        dest = ckpt_dir / item["name"]
        download(item["github_release"], dest)
        if item["name"] == "BFM.zip":
            bfm_dir = ckpt_dir / "BFM"
            if not bfm_dir.exists():
                info("Extracting BFM.zip")
                with zipfile.ZipFile(dest) as zf:
                    zf.extractall(ckpt_dir)
    print("  VideoReTalking ✓")


def install_musetalk() -> None:
    cfg = settings.model_configs["musetalk"]
    repo_dir = ROOT / cfg["repo_dir"]
    unet = ROOT / cfg["unet_path"]
    unet_cfg = ROOT / cfg["unet_config"]

    clone(cfg["repo_url"], repo_dir)

    if not unet.exists():
        info("Downloading MuseTalk v1.5 weights from HuggingFace")
        hf_download(cfg["hf_repo"], "musetalkV15/unet.pth", unet.parent)
    if not unet_cfg.exists():
        hf_download(cfg["hf_repo"], "musetalkV15/musetalk.json", unet_cfg.parent)
    print("  MuseTalk ✓")


def install_latentsync(version: str = "1.5") -> None:
    cfg = settings.model_configs["latentsync"]
    repo_dir = ROOT / cfg["repo_dir"]
    ckpt = ROOT / cfg["checkpoint"]
    whisper = ROOT / cfg["whisper_checkpoint"]

    clone(cfg["repo_url"], repo_dir)

    hf_repo = f"ByteDance/LatentSync-{version}"
    whisper_name = "tiny.pt" if version == "1.5" else "small.pt"

    if not ckpt.exists():
        info(f"Downloading LatentSync {version} checkpoint")
        hf_download(hf_repo, "latentsync_unet.pt", ckpt.parent)
    if not whisper.exists():
        hf_download(hf_repo, f"whisper/{whisper_name}", whisper.parent)
    print("  LatentSync ✓")


def install_sadtalker() -> None:
    cfg = settings.model_configs["sadtalker"]
    repo_dir = ROOT / cfg["repo_dir"]
    ckpt_dir = ROOT / cfg["checkpoint_dir"]

    clone(cfg["repo_url"], repo_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    info("Downloading SadTalker weights from HuggingFace")
    files = [
        "SadTalker_V0.0.2_256.safetensors",
        "mapping_00109-model.pth.tar",
        "mapping_00229-model.pth.tar",
    ]
    for f in files:
        try:
            hf_download(cfg["hf_repo"], f, ckpt_dir)
        except Exception as exc:
            print(f"  [warn] {f}: {exc}")
    # BFM files needed by SadTalker
    bfm_dir = repo_dir / "checkpoints" / "BFM_Fitting"
    bfm_dir.mkdir(parents=True, exist_ok=True)
    print("  SadTalker ✓")


# ── CLI ───────────────────────────────────────────────────────────────────────

INSTALLERS = {
    "wav2lip":         install_wav2lip,
    "video_retalking": install_video_retalking,
    "musetalk":        install_musetalk,
    "latentsync":      install_latentsync,
    "sadtalker":       install_sadtalker,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SwiftDub model assets")
    parser.add_argument("--model", choices=["all"] + list(INSTALLERS), default="all")
    parser.add_argument("--latentsync-version", choices=["1.5", "1.6"], default="1.5")
    args = parser.parse_args()

    targets = list(INSTALLERS) if args.model == "all" else [args.model]

    for name in targets:
        info(f"Installing {name}")
        fn = INSTALLERS[name]
        if name == "latentsync":
            fn(args.latentsync_version)
        else:
            fn()

    print("\nAll done. Models at:", settings.models_dir)


if __name__ == "__main__":
    main()
