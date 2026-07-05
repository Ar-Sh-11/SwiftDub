"""FastAPI dependencies."""

from __future__ import annotations

from pathlib import Path

import aiofiles
from fastapi import UploadFile

from src.config import settings
from src.utils.upload import safe_upload_path


async def save_upload(
    upload: UploadFile,
    dest: Path | None = None,
    *,
    dest_dir: Path | None = None,
    kind: str = "file",
    index: int | None = None,
) -> Path:
    if dest is None:
        if dest_dir is None:
            raise ValueError("dest or dest_dir required")
        dest = safe_upload_path(dest_dir, upload.filename, kind, index=index)
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await upload.read(1024 * 1024):
            await f.write(chunk)
    return dest
