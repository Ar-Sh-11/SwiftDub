"""MuseTalk v1.5 inference — subprocess runner."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml
from loguru import logger

from src.config import settings


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    try:
        import imageio_ffmpeg
        ffmpeg_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
        env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")
    except Exception:
        pass
    repo = str(settings.musetalk_vendor)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo + (os.pathsep + existing if existing else "")
    return env


def is_ready() -> bool:
    return (
        settings.musetalk_vendor.exists()
        and settings.musetalk_unet.exists()
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
) -> float:
    if not is_ready():
        raise RuntimeError(
            f"MuseTalk not ready. Vendor: {settings.musetalk_vendor} | "
            f"UNet: {settings.musetalk_unet}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_dir = temp_dir or (settings.temp_dir / f"mt_{output_path.stem}")
    result_dir.mkdir(parents=True, exist_ok=True)
    shift = bbox_shift if bbox_shift is not None else settings.musetalk_bbox_shift

    task_cfg = {
        "task_0": {
            "video_path": str(video_path.resolve()),
            "audio_path": str(audio_path.resolve()),
        }
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(task_cfg, f)
        cfg_path = f.name

    cmd = [
        sys.executable, "-m", "scripts.inference",
        "--inference_config", cfg_path,
        "--result_dir", str(result_dir),
        "--unet_model_path", str(settings.musetalk_unet),
        "--unet_config", str(settings.musetalk_unet_config),
        "--whisper_dir", str(settings.musetalk_whisper_dir),
        "--version", "v15",
        "--bbox_shift", str(shift),
        "--gpu_id", "0",
    ]

    logger.debug("MuseTalk cmd: {}", " ".join(cmd))
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(settings.musetalk_vendor),
            env=_subprocess_env(),
            capture_output=True,
            text=True,
        )
    finally:
        Path(cfg_path).unlink(missing_ok=True)

    elapsed = time.time() - t0

    if result.returncode != 0:
        stderr_tail = result.stderr[-2500:] if result.stderr else ""
        stdout_tail = result.stdout[-1000:] if result.stdout else ""
        raise RuntimeError(
            f"MuseTalk failed (exit {result.returncode}):\n"
            f"STDERR: {stderr_tail}\nSTDOUT: {stdout_tail}"
        )

    produced = sorted(result_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
    # Exclude concat preview files
    produced = [p for p in produced if "_concat" not in p.name]
    if not produced:
        stderr_tail = result.stderr[-1500:] if result.stderr else ""
        stdout_tail = result.stdout[-1500:] if result.stdout else ""
        raise RuntimeError(
            f"MuseTalk produced no output in {result_dir}\n"
            f"STDERR: {stderr_tail}\nSTDOUT: {stdout_tail}"
        )

    shutil.copy2(produced[-1], output_path)
    shutil.rmtree(result_dir, ignore_errors=True)

    logger.info("MuseTalk done in {:.1f}s → {}", elapsed, output_path)
    return elapsed
