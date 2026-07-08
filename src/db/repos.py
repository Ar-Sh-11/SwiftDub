"""Job persistence — MongoDB with in-memory fallback when DB is unavailable."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
from math import ceil
from typing import Any

from loguru import logger

from src.config import settings


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ── In-memory fallback ────────────────────────────────────────────────────────
_mem_store: dict[str, dict] = {}
_use_mongo: bool | None = None


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


async def reset_mongo_flag() -> None:
    """Allow reconnect after transient failure."""
    global _use_mongo
    _use_mongo = None


def _db():
    from src.db.client import get_db
    return get_db()


# ── Indexes ───────────────────────────────────────────────────────────────────

async def ensure_indexes() -> None:
    """Create indexes for fast paginated queries. Safe to call multiple times."""
    if not await _mongo_available():
        return
    from pymongo import ASCENDING, DESCENDING, TEXT
    col = _db().jobs
    try:
        await col.create_index([("created_at", DESCENDING)])
        await col.create_index([("status", ASCENDING)])
        await col.create_index([("metadata.batch_id", ASCENDING)])
        await col.create_index([("metadata.original_video", TEXT)])
        logger.info("MongoDB indexes ensured")
    except Exception as exc:
        logger.warning("Failed to create MongoDB indexes: {}", exc)


# ── CRUD ──────────────────────────────────────────────────────────────────────

async def create_job(job_id: str, model: str = "latentsync", metadata: dict | None = None) -> dict:
    doc: dict[str, Any] = {
        "job_id": job_id,
        "model": model,
        "status": "queued",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "output_video": None,
        "error": None,
        "elapsed_s": None,
        "metadata": metadata or {},
    }
    if await _mongo_available():
        await _db().jobs.insert_one({**doc, "_id": job_id})
    else:
        _mem_store[job_id] = doc
    return doc


async def update_job(job_id: str, **fields: Any) -> None:
    fields["updated_at"] = datetime.now(timezone.utc)
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


async def list_jobs_paginated(
    *,
    page: int = 1,
    per_page: int = 10,
    search: str | None = None,
    status: str | None = None,
    sort_by: str = "created_at",
    order: str = "desc",
    batch_id: str | None = None,
) -> dict[str, Any]:
    """Return paginated, filtered, sorted job list.

    Returns: {jobs, total, page, per_page, pages}
    """
    page = max(1, page)
    per_page = max(1, min(per_page, 100))
    skip = (page - 1) * per_page

    if await _mongo_available():
        return await _paginate_mongo(
            page=page, per_page=per_page, skip=skip,
            search=search, status=status,
            sort_by=sort_by, order=order, batch_id=batch_id,
        )
    return _paginate_mem(
        page=page, per_page=per_page, skip=skip,
        search=search, status=status,
        sort_by=sort_by, order=order, batch_id=batch_id,
    )


async def _paginate_mongo(
    *, page: int, per_page: int, skip: int,
    search: str | None, status: str | None,
    sort_by: str, order: str, batch_id: str | None,
) -> dict[str, Any]:
    from pymongo import ASCENDING, DESCENDING

    _SAFE_SORT = {"created_at", "updated_at", "status", "model", "elapsed_s"}
    sort_field = sort_by if sort_by in _SAFE_SORT else "created_at"
    sort_dir = DESCENDING if order == "desc" else ASCENDING

    query: dict[str, Any] = {}
    if status:
        query["status"] = status
    if batch_id:
        query["metadata.batch_id"] = batch_id
    if search:
        query["$or"] = [
            {"metadata.original_video": {"$regex": search, "$options": "i"}},
            {"_id": {"$regex": search, "$options": "i"}},
        ]

    total = await _db().jobs.count_documents(query)
    cursor = _db().jobs.find(query).sort(sort_field, sort_dir).skip(skip).limit(per_page)
    docs = await cursor.to_list(length=per_page)
    for d in docs:
        d["job_id"] = d.pop("_id", d.get("job_id"))

    return {
        "jobs": docs,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": max(1, ceil(total / per_page)),
    }


def _paginate_mem(
    *, page: int, per_page: int, skip: int,
    search: str | None, status: str | None,
    sort_by: str, order: str, batch_id: str | None,
) -> dict[str, Any]:
    docs = list(_mem_store.values())

    if status:
        docs = [d for d in docs if d.get("status") == status]
    if batch_id:
        docs = [d for d in docs if d.get("metadata", {}).get("batch_id") == batch_id]
    if search:
        lo = search.lower()
        docs = [
            d for d in docs
            if lo in (d.get("metadata", {}).get("original_video") or "").lower()
            or lo in d.get("job_id", "").lower()
        ]

    reverse = order == "desc"
    try:
        docs.sort(key=lambda d: d.get(sort_by) or datetime.min.replace(tzinfo=timezone.utc), reverse=reverse)
    except Exception:
        docs.sort(key=lambda d: d.get("created_at") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    total = len(docs)
    sliced = docs[skip: skip + per_page]
    return {
        "jobs": sliced,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": max(1, ceil(total / per_page)),
    }
