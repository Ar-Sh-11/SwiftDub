"""Safe upload path helpers — avoid spaces/special chars breaking ffmpeg shell commands."""

from __future__ import annotations

from pathlib import Path

_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}


def safe_upload_path(
    dest_dir: Path,
    original_name: str | None,
    kind: str,
    *,
    index: int | None = None,
) -> Path:
    """Return a filesystem-safe destination path for an uploaded file."""
    ext = Path(original_name or "").suffix.lower()
    if kind == "video":
        ext = ext if ext in _VIDEO_EXT else ".mp4"
    else:
        ext = ext if ext in _AUDIO_EXT else ".wav"

    if index is None:
        stem = kind
    else:
        stem = f"{kind}_{index}"
    return dest_dir / f"{stem}{ext}"
