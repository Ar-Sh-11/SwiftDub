#!/usr/bin/env python3
"""Run all installed models on the benchmark samples and produce a metrics table.

Usage:
  python scripts/benchmark/run_benchmark.py
  python scripts/benchmark/run_benchmark.py --model wav2lip latentsync
  python scripts/benchmark/run_benchmark.py --output benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import ModelName, settings  # noqa: E402
from src.core.face import FaceSelector  # noqa: E402
from src.core.models.registry import get_backend, status  # noqa: E402
from src.core.pipeline import run_pipeline  # noqa: E402
from src.utils.ffmpeg import extract_audio  # noqa: E402


def compute_ssim(pred: Path, gt: Path) -> float:
    """Structural similarity between two videos (frame-by-frame mean)."""
    try:
        import cv2
        import numpy as np
        from skimage.metrics import structural_similarity

        cap_p = cv2.VideoCapture(str(pred))
        cap_g = cv2.VideoCapture(str(gt))
        scores = []
        while True:
            ok1, f1 = cap_p.read()
            ok2, f2 = cap_g.read()
            if not ok1 or not ok2:
                break
            f1g = cv2.cvtColor(cv2.resize(f1, (256, 256)), cv2.COLOR_BGR2GRAY)
            f2g = cv2.cvtColor(cv2.resize(f2, (256, 256)), cv2.COLOR_BGR2GRAY)
            scores.append(structural_similarity(f1g, f2g, data_range=255))
        cap_p.release()
        cap_g.release()
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return -1.0


def run_one(model: str, video: Path, audio: Path) -> dict:
    import asyncio
    t0 = time.time()
    job_id = f"bench_{model}_{video.stem}"
    output = settings.outputs_dir / f"{job_id}.mp4"
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    try:
        asyncio.run(run_pipeline(job_id, video, audio, model))
        elapsed = time.time() - t0
        return {"status": "ok", "elapsed_s": round(elapsed, 2), "output": str(output)}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "elapsed_s": round(time.time() - t0, 2)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", choices=[m.value for m in ModelName])
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    sample_dir = settings.samples_dir / "benchmark"
    videos = sorted(sample_dir.glob("sample*_video.mp4"))
    if not videos:
        print("No benchmark samples found. Run: python scripts/download/datasets.py --split benchmark")
        sys.exit(1)

    ready = {k: v for k, v in status().items() if v}
    target_models = args.model or list(ready.keys())
    missing = [m for m in target_models if not ready.get(m)]
    if missing:
        print(f"[warn] Not installed (skipping): {missing}")
        target_models = [m for m in target_models if m not in missing]

    results: dict = {}
    for model in target_models:
        results[model] = []
        for video in videos[:2]:                  # limit to 2 samples for speed
            stem = video.stem.replace("_video", "")
            audio = sample_dir / f"{stem}_audio.wav"
            if not audio.exists():
                audio = sample_dir / f"{stem}_extracted_audio.wav"
            if not audio.exists():
                continue
            print(f"  {model} × {video.name}")
            r = run_one(model, video, audio)
            r["sample"] = video.name
            results[model].append(r)

    # Print table
    print("\n─── Benchmark Results ────────────────────────────────────────")
    print(f"{'Model':<22} {'Sample':<25} {'Status':<8} {'Time(s)'}")
    print("─" * 65)
    for model, runs in results.items():
        for r in runs:
            print(f"{model:<22} {r.get('sample',''):<25} {r['status']:<8} {r['elapsed_s']}")

    out = Path(args.output)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nFull results → {out}")


if __name__ == "__main__":
    main()
