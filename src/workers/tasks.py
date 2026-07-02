"""Celery worker tasks for async inference jobs."""

from __future__ import annotations

import asyncio
import time
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


def _run_async(coro):
    """Run an async coroutine from a sync Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(name="swiftdub.inference", bind=True, max_retries=0)
def run_inference(self, job_id: str, video_path: str, audio_path: str | None, model: str):
    """Run lip-sync inference and update the job document in MongoDB."""
    from src.core.pipeline import run_pipeline
    from src.db.repos import JobStatus, update_job
    from src.metrics.registry import active_jobs, inference_duration, inference_requests

    _run_async(update_job(job_id, status=JobStatus.PROCESSING))
    active_jobs.inc()
    t0 = time.time()

    try:
        output_path = _run_async(
            run_pipeline(
                job_id=job_id,
                video_path=Path(video_path),
                audio_path=Path(audio_path) if audio_path else None,
                model=model,
            )
        )
        elapsed = time.time() - t0
        inference_duration.labels(model=model).observe(elapsed)
        inference_requests.labels(model=model, status="success").inc()
        _run_async(update_job(job_id, status=JobStatus.COMPLETED, output_video=str(output_path)))
        return str(output_path)

    except Exception as exc:
        elapsed = time.time() - t0
        inference_duration.labels(model=model).observe(elapsed)
        inference_requests.labels(model=model, status="error").inc()
        logger.exception("Inference failed for job {}", job_id)
        _run_async(update_job(job_id, status=JobStatus.FAILED, error=str(exc)))
        raise
    finally:
        active_jobs.dec()
