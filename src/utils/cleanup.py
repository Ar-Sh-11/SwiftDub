"""Temp directory cleanup — removes stale job directories on startup and shutdown."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from loguru import logger

from src.config import settings


def cleanup_stale_temp(max_age_hours: float = 24.0) -> int:
    """Remove job temp directories older than `max_age_hours`.

    Returns the number of directories removed.
    """
    removed = 0
    cutoff = time.time() - max_age_hours * 3600
    temp = settings.temp_dir
    if not temp.exists():
        return 0
    for d in temp.iterdir():
        if not d.is_dir():
            continue
        try:
            mtime = d.stat().st_mtime
            if mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        except Exception as exc:
            logger.debug("cleanup skip {}: {}", d, exc)
    if removed:
        logger.info("Cleaned up {} stale temp director(ies) older than {}h", removed, max_age_hours)
    return removed


def cleanup_job_temp(job_id: str) -> None:
    """Remove the temp directory for a specific job (best-effort)."""
    d = settings.temp_dir / job_id
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        logger.debug("Removed temp dir for job {}", job_id)


def cleanup_stale_uploads(max_age_hours: float = 48.0) -> int:
    """Remove upload directories older than `max_age_hours`."""
    removed = 0
    cutoff = time.time() - max_age_hours * 3600
    uploads = settings.uploads_dir
    if not uploads.exists():
        return 0
    for d in uploads.iterdir():
        if not d.is_dir():
            continue
        try:
            if d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        except Exception as exc:
            logger.debug("uploads cleanup skip {}: {}", d, exc)
    if removed:
        logger.info("Cleaned up {} stale upload director(ies) older than {}h", removed, max_age_hours)
    return removed
