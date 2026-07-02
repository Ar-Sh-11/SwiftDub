"""FastAPI dependencies."""

from __future__ import annotations

from pathlib import Path

import aiofiles
from fastapi import UploadFile

from src.config import settings


async def save_upload(upload: UploadFile, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await upload.read(1024 * 1024):
            await f.write(chunk)
    return dest
