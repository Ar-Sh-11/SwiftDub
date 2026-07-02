"""LatentSync 1.5 inference — runs the model script in a subprocess.

The subprocess approach keeps GPU memory fully released after each job
and avoids import-time side effects in the parent process.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger

from src.config import settings


def _ffmpeg_env() -> dict[str, str]:
    """Env dict that ensures ffmpeg is on PATH and latentsync repo is importable."""
    env = os.environ.copy()
    try:
        import imageio_ffmpeg
        ffmpeg_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
        env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")
    except Exception:
        pass
    # The latentsync package lives inside its own repo directory
    repo = str(settings.latentsync_repo)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo + (os.pathsep + existing if existing else "")
    return env


def is_ready() -> bool:
    """Return True when model weights and repo are present."""
    return settings.latentsync_repo.exists() and settings.latentsync_ckpt.exists()


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
) -> float:
    """Run LatentSync inference and return elapsed seconds.

    Args:
        video_path: Input video file.
        audio_path: Input audio file (WAV/MP3).
        output_path: Where to write the dubbed MP4.
        inference_steps: DDIM steps (default from settings).
        guidance_scale: Classifier-free guidance (default from settings).
        seed: Random seed (-1 = random).
        enable_deepcache: Speed-up via DeepCache.
        temp_dir: Working directory for intermediate frames.

    Returns:
        Elapsed time in seconds.

    Raises:
        RuntimeError: If model is not ready or subprocess fails.
    """
    if not is_ready():
        raise RuntimeError(
            f"LatentSync weights missing. "
            f"Ckpt: {settings.latentsync_ckpt} | Repo: {settings.latentsync_repo}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    steps = inference_steps if inference_steps is not None else settings.latentsync_inference_steps
    scale = guidance_scale if guidance_scale is not None else settings.latentsync_guidance_scale
    rng   = seed if seed is not None else settings.latentsync_seed
    dcache = enable_deepcache if enable_deepcache is not None else settings.latentsync_enable_deepcache

    td = temp_dir or (settings.temp_dir / f"ls_{output_path.stem}")
    td.mkdir(parents=True, exist_ok=True)

    unet_cfg = settings.latentsync_repo / settings.latentsync_unet_config

    cmd = [
        sys.executable,
        "scripts/inference.py",
        "--unet_config_path", str(unet_cfg),
        "--inference_ckpt_path", str(settings.latentsync_ckpt),
        "--video_path", str(video_path),
        "--audio_path", str(audio_path),
        "--video_out_path", str(output_path),
        "--inference_steps", str(steps),
        "--guidance_scale", str(scale),
        "--seed", str(rng),
        "--temp_dir", str(td),
    ]
    if dcache:
        cmd.append("--enable_deepcache")

    logger.debug("LatentSync cmd: {}", " ".join(cmd))
    t0 = time.time()

    result = subprocess.run(
        cmd,
        cwd=str(settings.latentsync_repo),
        env=_ffmpeg_env(),
        capture_output=True,
        text=True,
    )

    elapsed = time.time() - t0

    if result.returncode != 0:
        stderr_tail = result.stderr[-2000:] if result.stderr else ""
        stdout_tail = result.stdout[-1000:] if result.stdout else ""
        raise RuntimeError(
            f"LatentSync failed (exit {result.returncode}):\n"
            f"STDERR: {stderr_tail}\nSTDOUT: {stdout_tail}"
        )

    if not output_path.exists():
        raise RuntimeError(f"LatentSync produced no output at {output_path}")

    logger.info("LatentSync done in {:.1f}s → {}", elapsed, output_path)
    return elapsed
