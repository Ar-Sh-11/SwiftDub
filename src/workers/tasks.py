"""Celery worker tasks for async dubbing jobs."""

from __future__ import annotations

import asyncio
from pathlib import Path

from celery import Celery
from loguru import logger

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


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


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
    from src.db.repos import JobStatus, update_job

    _run(update_job(job_id, status=JobStatus.PROCESSING))
    try:
        output = _run(
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
        _run(update_job(job_id, status=JobStatus.COMPLETED, output_video=str(output)))
        return str(output)
    except Exception as exc:
        logger.exception("Async dub failed for job {}", job_id)
        _run(update_job(job_id, status=JobStatus.FAILED, error=str(exc)[:1000]))
        raise
