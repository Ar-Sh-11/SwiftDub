#!/usr/bin/env python3
"""Download benchmark samples and guide for training datasets.

Usage:
  python scripts/download/datasets.py --split benchmark     # quick public samples
  python scripts/download/datasets.py --split hdtf          # HDTF via yt-dlp
  python scripts/download/datasets.py --split celebv        # CelebV-HQ from HF
  python scripts/download/datasets.py --split all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import settings  # noqa: E402


def info(msg: str) -> None:
    print(f"\033[0;36m==> {msg}\033[0m")


def download(url: str, dest: Path) -> bool:
    if dest.exists():
        info(f"Exists: {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    info(f"Downloading {dest.name}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SwiftDub/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, dest.open("wb") as f:
            f.write(r.read())
        return True
    except Exception as exc:
        print(f"  [warn] {exc}")
        return False


def download_benchmark() -> None:
    """Download the 3 public LatentSync demo pairs (no signup required)."""
    info("Downloading benchmark samples (LatentSync public assets)")
    base = "https://github.com/bytedance/LatentSync/raw/main/assets"
    samples = [
        ("demo1_video.mp4", "sample1_video.mp4"),
        ("demo1_audio.wav", "sample1_audio.wav"),
        ("demo2_video.mp4", "sample2_video.mp4"),
        ("demo2_audio.wav", "sample2_audio.wav"),
        ("demo3_video.mp4", "sample3_video.mp4"),
        ("demo3_audio.wav", "sample3_audio.wav"),
    ]
    dest_dir = settings.samples_dir / "benchmark"
    for remote, local in samples:
        download(f"{base}/{remote}", dest_dir / local)

    # Also extract audio from video samples for cross-model testing
    try:
        from src.utils.ffmpeg import extract_audio
        for i in range(1, 4):
            v = dest_dir / f"sample{i}_video.mp4"
            a = dest_dir / f"sample{i}_extracted_audio.wav"
            if v.exists() and not a.exists():
                extract_audio(v, a)
    except Exception as exc:
        print(f"  [warn] Audio extraction: {exc}")

    print(f"  Benchmark samples → {dest_dir}")


def download_hdtf() -> None:
    """Download a small subset of HDTF via yt-dlp (CC-BY 4.0)."""
    info("Downloading HDTF subset via yt-dlp (CC-BY 4.0)")
    dest_dir = settings.data_dir / "hdtf"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # A curated list of stable public HDTF YouTube IDs
    # (subset from MRzzm/HDTF — original CC-BY dataset)
    youtube_ids = [
        "WDA_PaBDFPg",  # news anchor clip
        "yMFiHpNE6yc",  # news anchor clip 2
        "KRavnLl-RjQ",  # interview style
    ]
    for vid_id in youtube_ids:
        out = dest_dir / f"{vid_id}.mp4"
        if out.exists():
            continue
        info(f"  yt-dlp {vid_id}")
        subprocess.run(
            ["yt-dlp", "-f", "best[ext=mp4][height<=720]", "-o", str(out),
             f"https://www.youtube.com/watch?v={vid_id}"],
            check=False,
        )
    print(f"  HDTF subset → {dest_dir}")


def download_celebv() -> None:
    """Download a small CelebV-HQ sample from HuggingFace (non-commercial research)."""
    info("Downloading CelebV-HQ sample from HuggingFace")
    try:
        from huggingface_hub import snapshot_download
        dest = settings.data_dir / "celebv_sample"
        snapshot_download(
            repo_id="AIDB-Lab/CelebV-HQ",
            repo_type="dataset",
            local_dir=str(dest),
            ignore_patterns=["*.zip", "*.tar*"],   # skip large archives
            max_workers=4,
        )
        print(f"  CelebV-HQ → {dest}")
    except Exception as exc:
        print(f"  [warn] CelebV download failed: {exc}")
        print("  Manual: https://huggingface.co/datasets/AIDB-Lab/CelebV-HQ")


def print_manual_instructions() -> None:
    print("""
Training datasets requiring manual download:
──────────────────────────────────────────
LRS3 (438h, ~60GB)
  1. Go to https://mmai.io/datasets/lip_reading/
  2. Sign the Oxford/BBC data agreement
  3. Download lrs3_trainval.zip and lrs3_test.zip
  4. Place in: data/lrs3/

VoxCeleb2 (2442h, ~145GB)
  1. Go to https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
  2. Request credentials
  3. Download vox2_mp4.zip
  4. Place in: data/voxceleb2/

AVSpeech (4700h, ~600GB)
  1. Visit: https://looking-to-listen.github.io/avspeech/download.html
  2. Download the CSV with YouTube IDs
  3. Run: python scripts/download/avspeech_scraper.py --csv avspeech.csv
""")


SPLITS = {
    "benchmark": download_benchmark,
    "hdtf":      download_hdtf,
    "celebv":    download_celebv,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SwiftDub datasets")
    parser.add_argument("--split", choices=["all"] + list(SPLITS), default="benchmark")
    args = parser.parse_args()

    targets = list(SPLITS) if args.split == "all" else [args.split]
    for name in targets:
        SPLITS[name]()
    print_manual_instructions()


if __name__ == "__main__":
    main()
