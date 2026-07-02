"""FastAPI dependency providers."""

from __future__ import annotations

from fastapi import UploadFile
from loguru import logger


async def save_upload(upload: UploadFile, dest_path) -> None:
    """Stream an uploaded file to disk."""
    import shutil
    from pathlib import Path
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    logger.debug("Saved upload → {}", dest)
