"""Job persistence — MongoDB with in-memory fallback when DB is unavailable."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from loguru import logger

from src.config import settings


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ── In-memory fallback store ──────────────────────────────────────────────────
_mem_store: dict[str, dict] = {}
_use_mongo: bool | None = None   # lazily determined


async def _mongo_available() -> bool:
    global _use_mongo
    if _use_mongo is not None:
        return _use_mongo
    if settings.disable_db:
        _use_mongo = False
        return False
    try:
        from src.db.client import get_db
        await asyncio.wait_for(get_db().command("ping"), timeout=2.0)
        _use_mongo = True
    except Exception as exc:
        logger.warning("MongoDB unavailable ({}), using in-memory job store", exc)
        _use_mongo = False
    return _use_mongo


def _db():
    from src.db.client import get_db
    return get_db()


async def create_job(job_id: str, model: str = "latentsync", metadata: dict | None = None) -> dict:
    doc = {
        "job_id": job_id,
        "model": model,
        "status": "queued",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "output_video": None,
        "error": None,
        "metadata": metadata or {},
    }
    if await _mongo_available():
        await _db().jobs.insert_one({**doc, "_id": job_id})
    else:
        _mem_store[job_id] = doc
    return doc


async def update_job(job_id: str, **fields: Any) -> None:
    fields["updated_at"] = datetime.now(timezone.utc)
    # Serialize enum values to strings
    if "status" in fields and hasattr(fields["status"], "value"):
        fields["status"] = fields["status"].value
    if await _mongo_available():
        await _db().jobs.update_one({"_id": job_id}, {"$set": fields})
    else:
        if job_id in _mem_store:
            _mem_store[job_id].update(fields)


async def get_job(job_id: str) -> dict | None:
    if await _mongo_available():
        doc = await _db().jobs.find_one({"_id": job_id})
        if doc:
            doc["job_id"] = doc.pop("_id", job_id)
        return doc
    return _mem_store.get(job_id)


async def list_jobs(limit: int = 50) -> list[dict]:
    if await _mongo_available():
        cursor = _db().jobs.find().sort("created_at", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        for d in docs:
            d["job_id"] = d.pop("_id", d.get("job_id"))
        return docs
    return sorted(_mem_store.values(), key=lambda d: d["created_at"], reverse=True)[:limit]
