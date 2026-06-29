#!/usr/bin/env python3
"""Download public sample video/audio for pipeline testing."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.utils.ffmpeg import extract_audio  # noqa: E402

# Public samples used by open-source lip-sync projects (English dialogue, talking head).
SAMPLES = {
    "source_video.mp4": (
        "https://github.com/bytedance/LatentSync/raw/main/assets/demo1_video.mp4",
        "LatentSync demo talking-head clip",
        False,
    ),
    "dub_audio.wav": (
        "https://github.com/bytedance/LatentSync/raw/main/assets/demo1_audio.wav",
        "Matching dialogue audio for dubbing test",
        False,
    ),
    "alt_video.mp4": (
        "https://github.com/bytedance/LatentSync/raw/main/assets/demo2_video.mp4",
        "Second LatentSync demo clip",
        True,
    ),
    "alt_audio.wav": (
        "https://github.com/bytedance/LatentSync/raw/main/assets/demo2_audio.wav",
        "Second demo dialogue audio",
        True,
    ),
}


def download_file(url: str, destination: Path, *, optional: bool = False) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Already exists: {destination}")
        return True
    print(f"Downloading {url}")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "SwiftDub/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            destination.write_bytes(response.read())
        print(f"Saved {destination}")
        return True
    except Exception as exc:
        if optional:
            print(f"Skipped optional sample ({exc})")
            return False
        raise


def main() -> None:
    settings.samples_dir.mkdir(parents=True, exist_ok=True)

    for filename, (url, description, optional) in SAMPLES.items():
        print(f"\n[{description}]")
        download_file(url, settings.samples_dir / filename, optional=optional)

    source = settings.samples_dir / "source_video.mp4"
    extracted = settings.samples_dir / "source_audio.wav"
    if source.exists() and not extracted.exists():
        print("\nExtracting audio track from source video...")
        extract_audio(source, extracted)

    print("\nSample files ready in:", settings.samples_dir)
    print("Suggested test:")
    print("  python scripts/run_pipeline_demo.py --model wav2lip")


if __name__ == "__main__":
    main()
