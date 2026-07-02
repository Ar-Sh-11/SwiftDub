#!/usr/bin/env python3
"""Run all models on benchmark samples; save outputs to output/<model_name>/."""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ModelName, settings  # noqa: E402
from src.core.models.registry import get_backend, status  # noqa: E402
from src.core.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", choices=[m.value for m in ModelName])
    parser.add_argument("--output-root", type=Path, default=ROOT / "output")
    args = parser.parse_args()

    sample_dir = settings.samples_dir / "benchmark"
    videos = sorted(sample_dir.glob("sample*_video.mp4"))
    if not videos:
        print("No samples found. Run: python scripts/download/datasets.py --split benchmark")
        sys.exit(1)

    ready = {k: v for k, v in status().items() if v}
    target_models = args.model or list(ready.keys())
    missing = [m for m in target_models if not ready.get(m)]
    if missing:
        print(f"[warn] Not installed (skipping): {missing}")
        target_models = [m for m in target_models if m not in missing]
    if not target_models:
        print("No models ready. Run: python scripts/download/models.py --model all")
        sys.exit(1)

    results: list[dict] = []
    for model in target_models:
        out_dir = args.output_root / model
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {model} ===")

        for video in videos:
            stem = video.stem.replace("_video", "")
            audio = sample_dir / f"{stem}_audio.wav"
            if not audio.exists():
                audio = sample_dir / f"{stem}_extracted_audio.wav"
            if not audio.exists():
                print(f"  skip {video.name}: no audio")
                continue

            dest = out_dir / f"{stem}.mp4"
            job_id = f"samples_{model}_{stem}"
            print(f"  {video.name} -> {dest.relative_to(ROOT)}")
            t0 = time.time()
            try:
                result_path = asyncio.run(run_pipeline(job_id, video, audio, model))
                shutil.copy2(result_path, dest)
                elapsed = time.time() - t0
                results.append({"model": model, "sample": stem, "status": "ok", "elapsed_s": round(elapsed, 1), "output": str(dest)})
                print(f"    done in {elapsed:.1f}s")
            except Exception as exc:
                elapsed = time.time() - t0
                results.append({"model": model, "sample": stem, "status": "error", "error": str(exc), "elapsed_s": round(elapsed, 1)})
                print(f"    ERROR: {exc}")

    print("\n─── Summary ───────────────────────────────────────")
    for r in results:
        status_str = r["status"]
        extra = f" ({r['elapsed_s']}s)" if status_str == "ok" else f" — {r.get('error', '')[:80]}"
        print(f"  {r['model']:<20} {r['sample']:<12} {status_str}{extra}")
    print(f"\nOutputs → {args.output_root}")


if __name__ == "__main__":
    main()
