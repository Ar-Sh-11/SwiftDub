#!/usr/bin/env python3
"""Download lip-sync model repos and weights into models/."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def clone_repo(url: str, destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        print(f"Repo already present: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", url, str(destination)])


def download_wav2lip() -> None:
    clone_repo("https://github.com/Rudrabha/Wav2Lip.git", settings.wav2lip_repo)
    ckpt_dir = settings.wav2lip_checkpoint.parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not settings.wav2lip_checkpoint.exists():
        import gdown

        gdown.download(
            id="10Iu05Modfti3pDbxCFPnofmfVlbkvrCm",
            output=str(settings.wav2lip_checkpoint),
            quiet=False,
        )

    s3fd_path = (
        settings.wav2lip_repo
        / "face_detection"
        / "detection"
        / "sfd"
        / "s3fd.pth"
    )
    if not s3fd_path.exists():
        import urllib.request

        s3fd_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
        print(f"Downloading face detector weights to {s3fd_path}")
        urllib.request.urlretrieve(url, s3fd_path)


def download_musetalk() -> None:
    clone_repo("https://github.com/TMElyralab/MuseTalk.git", settings.musetalk_repo)

    from huggingface_hub import hf_hub_download

    weight_dir = settings.musetalk_unet.parent
    weight_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "musetalkV15/unet.pth": settings.musetalk_unet,
        "musetalkV15/musetalk.json": settings.musetalk_config,
    }
    for repo_path, local_path in files.items():
        if local_path.exists():
            continue
        downloaded = hf_hub_download(
            repo_id="TMElyralab/MuseTalk",
            filename=repo_path,
            local_dir=str(weight_dir.parent),
        )
        downloaded_path = Path(downloaded)
        if downloaded_path.resolve() != local_path.resolve():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded_path, local_path)

    # MuseTalk also needs dwpose / sd-vae / whisper weights on first run;
    # huggingface cache handles those automatically.


def download_latentsync(version: str = "1.5") -> None:
    clone_repo("https://github.com/bytedance/LatentSync.git", settings.latentsync_repo)

    from huggingface_hub import hf_hub_download

    repo_id = "ByteDance/LatentSync-1.5" if version == "1.5" else "ByteDance/LatentSync-1.6"
    ckpt_name = "latentsync_unet.pt"

    if not settings.latentsync_checkpoint.exists():
        hf_hub_download(
            repo_id=repo_id,
            filename=ckpt_name,
            local_dir=str(settings.latentsync_checkpoint.parent),
        )

    whisper_dir = settings.latentsync_repo / "checkpoints" / "whisper"
    whisper_dir.mkdir(parents=True, exist_ok=True)
    whisper_name = "tiny.pt" if version == "1.5" else "small.pt"
    whisper_path = whisper_dir / whisper_name
    if not whisper_path.exists():
        hf_hub_download(
            repo_id=repo_id,
            filename=f"whisper/{whisper_name}",
            local_dir=str(settings.latentsync_repo / "checkpoints"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SwiftDub model assets")
    parser.add_argument(
        "--model",
        choices=["all", "wav2lip", "musetalk", "latentsync"],
        default="all",
    )
    parser.add_argument(
        "--latentsync-version",
        choices=["1.5", "1.6"],
        default="1.5",
        help="LatentSync 1.5 needs ~8GB VRAM; 1.6 needs ~18GB",
    )
    args = parser.parse_args()

    if args.model in ("all", "wav2lip"):
        print("\n=== Wav2Lip ===")
        download_wav2lip()

    if args.model in ("all", "musetalk"):
        print("\n=== MuseTalk ===")
        download_musetalk()

    if args.model in ("all", "latentsync"):
        print("\n=== LatentSync ===")
        download_latentsync(args.latentsync_version)

    print("\nDone. Models are under:", settings.models_dir)


if __name__ == "__main__":
    main()
