"""Shared subprocess runner for model repos."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from loguru import logger

from src.config import settings


def conda_python(env_name: str) -> str:
    """Resolve python executable for a named conda env."""
    base = settings.conda_base
    if base:
        base = Path(base)
        candidates = [
            base / "envs" / env_name / "python",            # Linux
            base / "envs" / env_name / "python.exe",        # Windows
            base / "envs" / env_name / "bin" / "python",    # Linux alt
        ]
        for c in candidates:
            if c.exists():
                return str(c)
    # Fall back to current interpreter (works when already in the env)
    return sys.executable


def run_model(
    cmd: list[str],
    cwd: Path,
    env_name: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    python = conda_python(env_name)
    resolved = [python if c == "PYTHON" else c for c in cmd]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)
    if extra_env:
        env.update(extra_env)
    # Prepend bundled ffmpeg dir so model scripts can call `ffmpeg`
    try:
        import imageio_ffmpeg
        ffmpeg_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
        env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")
    except Exception:
        pass

    logger.debug("Running: {}", " ".join(resolved))
    result = subprocess.run(resolved, cwd=str(cwd), env=env, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error(result.stdout[-1000:])
        logger.error(result.stderr[-1000:])
        raise RuntimeError(f"Subprocess failed (exit {result.returncode}):\n{result.stderr[-1500:]}")
    return result
