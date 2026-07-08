"""MuseTalk v1.5 inference — subprocess runner.

Calls our first-party src/models/musetalk/infer.py entry point.
No dependency on models/vendor/ — only models/weights/ required.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger

from src.config import settings
from src.utils.gpu_alloc import GPUAllocator

_gpu_alloc = GPUAllocator()


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    try:
        import imageio_ffmpeg
        ffmpeg_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
        env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")
    except Exception:
        pass
    root = str(settings.root_dir)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root + (os.pathsep + existing if existing else "")
    return env


def is_ready() -> bool:
    return (
        settings.musetalk_unet.exists()
        and settings.musetalk_unet_config.exists()
        and settings.musetalk_whisper_dir.exists()
        and settings.musetalk_dwpose.exists()
    )


def run_musetalk(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    bbox_shift: int | None = None,
    temp_dir: Path | None = None,
    gpu_id: int | None = None,
) -> float:
    """Run MuseTalk inference via our first-party infer.py. Returns elapsed seconds."""
    if not is_ready():
        raise RuntimeError(
            f"MuseTalk weights not found. UNet: {settings.musetalk_unet}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_dir = temp_dir or (settings.temp_dir / f"mt_{output_path.stem}")
    result_dir.mkdir(parents=True, exist_ok=True)
    shift = bbox_shift if bbox_shift is not None else settings.musetalk_bbox_shift

    selected_gpu = gpu_id if gpu_id is not None else _gpu_alloc.acquire()
    try:
        cmd = [
            sys.executable, "-m", "src.models.musetalk.infer",
            "--video_path", str(video_path.resolve()),
            "--audio_path", str(audio_path.resolve()),
            "--output_path", str(output_path),
            "--unet_model_path", str(settings.musetalk_unet),
            "--unet_config", str(settings.musetalk_unet_config),
            "--whisper_dir", str(settings.musetalk_whisper_dir),
            "--result_dir", str(result_dir),
            "--bbox_shift", str(shift),
            "--gpu_id", str(selected_gpu),
        ]

        logger.debug("MuseTalk cmd: {}", " ".join(cmd))
        t0 = time.time()
        result = subprocess.run(
            cmd,
            cwd=str(settings.root_dir),
            env=_base_env(),
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - t0
    finally:
        _gpu_alloc.release(selected_gpu)

    if result.returncode != 0:
        stderr = result.stderr[-2500:] if result.stderr else ""
        stdout = result.stdout[-1000:] if result.stdout else ""
        raise RuntimeError(
            f"MuseTalk failed (exit {result.returncode}):\n"
            f"STDERR: {stderr}\nSTDOUT: {stdout}"
        )

    if not output_path.exists():
        raise RuntimeError(f"MuseTalk produced no output at {output_path}")

    logger.info("MuseTalk done in {:.1f}s → {}", elapsed, output_path)
    return elapsed
