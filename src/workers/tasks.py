"""Celery worker tasks for async dubbing jobs.

Celery prefork workers must NOT reuse Motor clients inherited from the parent
process — they bind to a closed event loop. We reset all async state on fork
(worker_process_init) and use sync pymongo for job status updates in workers.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from celery import Celery
from celery.signals import worker_process_init
from loguru import logger
from pymongo import MongoClient

from src.config import settings

celery_app = Celery(
    "swiftdub",
    broker=settings.celery_broker,
    backend=settings.celery_backend,
)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]
celery_app.conf.task_track_started = True
celery_app.conf.worker_prefetch_multiplier = 1

_worker_loop: asyncio.AbstractEventLoop | None = None


@worker_process_init.connect
def _on_worker_init(**_kwargs) -> None:
    """Reset all async/Motor state after Celery fork — parent state is invalid."""
    global _worker_loop
    _worker_loop = None
    try:
        import src.db.client as _dbc
        import src.db.repos as _repo

        if _dbc._client is not None:
            try:
                _dbc._client.close()
            except Exception:
                pass
        _dbc._client = None
        _repo._use_mongo = None
    except Exception:
        pass
    asyncio.set_event_loop(asyncio.new_event_loop())
    logger.info("Celery worker process initialized — async state reset")


def _get_loop() -> asyncio.AbstractEventLoop:
    global _worker_loop
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_loop)
    return _worker_loop


def _run(coro):
    return _get_loop().run_until_complete(coro)


def _sync_update_job(job_id: str, **fields) -> None:
    """Sync MongoDB update — safe inside Celery prefork workers."""
    fields["updated_at"] = datetime.now(timezone.utc)
    if "status" in fields and hasattr(fields["status"], "value"):
        fields["status"] = fields["status"].value
    try:
        client = MongoClient(settings.mongodb_url, serverSelectionTimeoutMS=5000)
        client[settings.mongodb_db].jobs.update_one({"_id": job_id}, {"$set": fields})
        client.close()
    except Exception as exc:
        logger.error("[{}] Sync DB update failed: {}", job_id, exc)


@celery_app.task(name="swiftdub.dub", bind=True, max_retries=0)
def dub_task(
    self,
    job_id: str,
    video_path: str,
    audio_path: str | None,
    model: str = "latentsync",
    inference_steps: int | None = None,
    guidance_scale: float | None = None,
    seed: int | None = None,
    bbox_shift: int | None = None,
    enable_deepcache: bool | None = None,
):
    from src.core.pipeline import dub_video
    from src.db.repos import JobStatus

    logger.info("[{}] Celery task started", job_id)
    _sync_update_job(job_id, status=JobStatus.PROCESSING)
    try:
        output, elapsed = _run(
            dub_video(
                job_id=job_id,
                video_path=Path(video_path),
                audio_path=Path(audio_path) if audio_path else None,
                model=model,
                inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                enable_deepcache=enable_deepcache,
                bbox_shift=bbox_shift,
            )
        )
        _sync_update_job(
            job_id,
            status=JobStatus.COMPLETED,
            output_video=str(output),
            elapsed_s=round(elapsed, 2),
        )
        logger.info("[{}] Celery task done in {:.1f}s", job_id, elapsed)
        return str(output)
    except Exception as exc:
        logger.exception("[{}] Celery task failed", job_id)
        _sync_update_job(job_id, status=JobStatus.FAILED, error=str(exc)[:1000])
        raise
