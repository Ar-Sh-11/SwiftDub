#!/usr/bin/env python3
"""Run lip-sync pipeline on downloaded sample data."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LipSyncModel, ensure_directories, settings  # noqa: E402
from src.services.lipsync import BACKENDS  # noqa: E402
from src.services.pipeline import LipSyncPipeline, PipelineInput  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SwiftDub pipeline demo")
    parser.add_argument(
        "--model",
        choices=[m.value for m in LipSyncModel],
        default="wav2lip",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=settings.samples_dir / "source_video.mp4",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=settings.samples_dir / "dub_audio.wav",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Benchmark wav2lip, musetalk, and latentsync sequentially",
    )
    args = parser.parse_args()

    ensure_directories()

    if not args.video.exists():
        print("Sample video missing. Run: python scripts/download_sample_data.py")
        sys.exit(1)
    if not args.audio.exists():
        print("Sample audio missing. Run: python scripts/download_sample_data.py")
        sys.exit(1)

    models = [LipSyncModel(m) for m in LipSyncModel] if args.all_models else [LipSyncModel(args.model)]
    pipeline = LipSyncPipeline()

    for model in models:
        backend = BACKENDS[model]
        if not backend.is_available():
            print(f"Skipping {model.value}: not installed (run scripts/download_models.py)")
            continue

        print(f"\n=== Running {model.value} ===")
        started = time.time()
        result = pipeline.run(
            PipelineInput(
                video_path=args.video,
                audio_path=args.audio,
                model=model,
            )
        )
        elapsed = time.time() - started
        print(f"Output: {result.output_video}")
        print(f"Face box: {result.face_box}")
        print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
