"""FastAPI dependencies — upload handling and audio extraction."""

from __future__ import annotations

from pathlib import Path

import aiofiles
from fastapi import UploadFile
from loguru import logger

from src.config import settings
from src.utils.upload import safe_upload_path

# MIME types and extensions that indicate a video file submitted as the audio field
_VIDEO_MIMES = frozenset(
    {
        "video/mp4",
        "video/quicktime",
        "video/x-msvideo",
        "video/webm",
        "video/x-matroska",
        "video/mpeg",
        "video/x-ms-wmv",
    }
)
_VIDEO_EXTS = frozenset(
    {".mp4", ".mov", ".avi", ".webm", ".mkv", ".mpeg", ".mpg", ".wmv", ".flv", ".ts"}
)


def is_video_file(upload: UploadFile) -> bool:
    """Return True if the upload is a video (content-type or extension)."""
    if upload.content_type and upload.content_type in _VIDEO_MIMES:
        return True
    if upload.filename:
        return Path(upload.filename).suffix.lower() in _VIDEO_EXTS
    return False


async def save_upload(
    upload: UploadFile,
    dest: Path | None = None,
    *,
    dest_dir: Path | None = None,
    kind: str = "file",
    index: int | None = None,
) -> Path:
    """Save an uploaded file to disk and return the destination path."""
    if dest is None:
        if dest_dir is None:
            raise ValueError("dest or dest_dir required")
        dest = safe_upload_path(dest_dir, upload.filename, kind, index=index)
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await upload.read(1024 * 1024):
            await f.write(chunk)
    return dest


async def save_audio_upload(
    upload: UploadFile,
    dest_dir: Path,
    *,
    index: int | None = None,
) -> tuple[Path, bool]:
    """Save an audio or video upload; extract audio if video was supplied.

    Returns:
        (audio_path, was_extracted) — audio_path is a WAV file ready for
        inference. was_extracted is True when audio had to be stripped from
        a video container.
    """
    saved = await save_upload(upload, dest_dir=dest_dir, kind="audio", index=index)

    if is_video_file(upload):
        logger.info("Audio field contains a video — extracting audio track from {}", saved.name)
        from src.utils.ffmpeg import extract_audio
        audio_path = saved.with_suffix(".wav")
        extract_audio(saved, audio_path)
        # Remove the video file to save space; keep only the extracted WAV
        saved.unlink(missing_ok=True)
        return audio_path, True

    return saved, False
