"""FFmpeg helpers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from loguru import logger

from src.config import settings


def _resolve_ffmpeg() -> str:
    if shutil.which(settings.ffmpeg_path):
        return settings.ffmpeg_path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return settings.ffmpeg_path


def ffmpeg_available() -> bool:
    return bool(shutil.which(settings.ffmpeg_path)) or _bundled_ffmpeg_available()


def _bundled_ffmpeg_available() -> bool:
    try:
        import imageio_ffmpeg

        return bool(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return False


def run_ffmpeg(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    ffmpeg = _resolve_ffmpeg()
    cmd = [ffmpeg, *args]
    logger.debug("Running ffmpeg: {}", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def extract_audio(video_path: Path, audio_path: Path, sample_rate: int = 16000) -> Path:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            str(audio_path),
        ]
    )
    return audio_path


def mux_audio_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    copy_video: bool = True,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_codec = ["-c:v", "copy"] if copy_video else ["-c:v", "libx264", "-crf", "18"]
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            *video_codec,
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(output_path),
        ]
    )
    return output_path


def get_video_info(video_path: Path) -> dict[str, float | int]:
    ffmpeg = _resolve_ffmpeg()
    result = subprocess.run(
        [
            ffmpeg,
            "-i",
            str(video_path),
            "-hide_banner",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    stderr = result.stderr
    fps = 25.0
    width = height = 0
    for line in stderr.splitlines():
        if "Video:" in line and "fps" in line:
            parts = line.split(",")
            for part in parts:
                part = part.strip()
                if "x" in part and part[0].isdigit():
                    try:
                        w, h = part.split("x")[:2]
                        width, height = int(w), int(h.split()[0])
                    except ValueError:
                        pass
                if " fps" in part:
                    try:
                        fps = float(part.split(" fps")[0].strip())
                    except ValueError:
                        pass
    return {"fps": fps, "width": width, "height": height}
