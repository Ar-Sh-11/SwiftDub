"""LatentSync 1.5 inference — subprocess runner.

Calls our first-party src/models/latentsync/infer.py entry point.
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
    """Build subprocess environment with ffmpeg and SwiftDub root on path."""
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
    """Return True when weights are present (vendor no longer required)."""
    return (
        settings.latentsync_ckpt.exists()
        and settings.latentsync_whisper.exists()
        and settings.latentsync_unet_config_path.exists()
    )


def run_latentsync(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    inference_steps: int | None = None,
    guidance_scale: float | None = None,
    seed: int | None = None,
    enable_deepcache: bool | None = None,
    temp_dir: Path | None = None,
    gpu_id: int | None = None,
) -> float:
    """Run LatentSync inference via our first-party infer.py. Returns elapsed seconds."""
    if not is_ready():
        raise RuntimeError(
            f"LatentSync weights not found. "
            f"Ckpt: {settings.latentsync_ckpt} | "
            f"Whisper: {settings.latentsync_whisper}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    steps = inference_steps if inference_steps is not None else settings.latentsync_inference_steps
    scale = guidance_scale if guidance_scale is not None else settings.latentsync_guidance_scale
    rng = seed if seed is not None else settings.latentsync_seed
    dcache = enable_deepcache if enable_deepcache is not None else settings.latentsync_enable_deepcache

    td = temp_dir or (settings.temp_dir / f"ls_{output_path.stem}")
    td.mkdir(parents=True, exist_ok=True)

    selected_gpu = gpu_id if gpu_id is not None else _gpu_alloc.acquire()
    try:
        cmd = [
            sys.executable, "-m", "src.models.latentsync.infer",
            "--unet_config_path", str(settings.latentsync_unet_config_path),
            "--inference_ckpt_path", str(settings.latentsync_ckpt),
            "--whisper_model_path", str(settings.latentsync_whisper),
            "--video_path", str(video_path.resolve()),
            "--audio_path", str(audio_path.resolve()),
            "--video_out_path", str(output_path),
            "--inference_steps", str(steps),
            "--guidance_scale", str(scale),
            "--seed", str(rng),
            "--temp_dir", str(td),
            "--gpu_id", str(selected_gpu),
        ]
        if dcache:
            cmd.append("--enable_deepcache")

        logger.debug("LatentSync cmd: {}", " ".join(cmd))
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
        stderr = result.stderr[-2000:] if result.stderr else ""
        stdout = result.stdout[-1000:] if result.stdout else ""
        raise RuntimeError(
            f"LatentSync failed (exit {result.returncode}):\n"
            f"STDERR: {stderr}\nSTDOUT: {stdout}"
        )

    if not output_path.exists():
        raise RuntimeError(f"LatentSync produced no output at {output_path}")

    logger.info("LatentSync done in {:.1f}s → {}", elapsed, output_path)
    return elapsed
