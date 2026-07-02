"""Repository pattern for MongoDB job documents."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.db.client import get_db

JOBS = "jobs"


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def create_job(job_id: str, model: str, metadata: dict[str, Any] | None = None) -> dict:
    doc = {
        "_id": job_id,
        "model": model,
        "status": JobStatus.QUEUED,
        "created_at": _now(),
        "updated_at": _now(),
        "output_video": None,
        "error": None,
        "metadata": metadata or {},
    }
    await get_db()[JOBS].insert_one(doc)
    return _clean(doc)


async def update_job(job_id: str, **fields: Any) -> dict | None:
    fields["updated_at"] = _now()
    result = await get_db()[JOBS].find_one_and_update(
        {"_id": job_id},
        {"$set": fields},
        return_document=True,
    )
    return _clean(result) if result else None


async def get_job(job_id: str) -> dict | None:
    doc = await get_db()[JOBS].find_one({"_id": job_id})
    return _clean(doc) if doc else None


async def list_jobs(limit: int = 50) -> list[dict]:
    cursor = get_db()[JOBS].find({}, sort=[("created_at", -1)], limit=limit)
    return [_clean(d) async for d in cursor]


def _clean(doc: dict) -> dict:
    """Rename _id → job_id for clean API responses."""
    if doc and "_id" in doc:
        doc["job_id"] = doc.pop("_id")
    return doc
