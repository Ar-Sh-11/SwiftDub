"""Inference endpoint — POST /predict."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile
from loguru import logger

from src.api.deps import save_upload
from src.api.schemas import PredictResponse
from src.cache.client import cache_output, get_cached_output
from src.config import ModelName, settings
from src.core.pipeline import run_pipeline
from src.db.repos import JobStatus, create_job, update_job
from src.metrics.registry import active_jobs, inference_duration, inference_requests
from src.workers.tasks import celery_app, run_inference
import time

router = APIRouter(tags=["inference"])


async def _process_sync(job_id: str, video: Path, audio: Path, model: str) -> None:
    """Run inference in-process (sync=True path)."""
    await update_job(job_id, status=JobStatus.PROCESSING)
    active_jobs.inc()
    t0 = time.time()
    try:
        output = await run_pipeline(job_id, video, audio, model)
        inference_duration.labels(model=model).observe(time.time() - t0)
        inference_requests.labels(model=model, status="success").inc()
        await update_job(job_id, status=JobStatus.COMPLETED, output_video=str(output))
    except Exception as exc:
        inference_duration.labels(model=model).observe(time.time() - t0)
        inference_requests.labels(model=model, status="error").inc()
        logger.exception("Inference failed for job {}", job_id)
        await update_job(job_id, status=JobStatus.FAILED, error=str(exc))
    finally:
        active_jobs.dec()


@router.post("/predict", response_model=PredictResponse)
async def predict(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    audio: UploadFile | None = File(None),
    model: ModelName = Form(default=ModelName.WAV2LIP),
    sync: bool = Form(default=True),
) -> PredictResponse:
    job_id = uuid.uuid4().hex
    job_dir = settings.uploads_dir / job_id
    video_path = job_dir / (video.filename or "input.mp4")
    await save_upload(video, video_path)

    audio_path: Path | None = None
    if audio:
        audio_path = job_dir / (audio.filename or "input.wav")
        await save_upload(audio, audio_path)

    # Check Redis cache
    if audio_path:
        cached = await get_cached_output(
            video_path.read_bytes(), audio_path.read_bytes(), model.value
        )
        if cached:
            from src.metrics.registry import cache_hits
            cache_hits.labels(model=model.value).inc()
            logger.info("Cache hit for job {}", job_id)
            doc = await create_job(job_id, model.value)
            await update_job(job_id, status=JobStatus.COMPLETED, output_video=cached)
            return PredictResponse(job_id=job_id, status=JobStatus.COMPLETED, model=model.value, output_video=cached)

    doc = await create_job(job_id, model.value)

    if sync:
        await _process_sync(job_id, video_path, audio_path, model.value)
        updated = await get_job(job_id) if True else doc
        from src.db.repos import get_job
        updated = await get_job(job_id)
        return PredictResponse(
            job_id=job_id,
            status=updated["status"] if updated else JobStatus.FAILED,
            model=model.value,
            output_video=updated.get("output_video") if updated else None,
            error=updated.get("error") if updated else None,
        )

    # Async via Celery
    run_inference.delay(job_id, str(video_path), str(audio_path) if audio_path else None, model.value)
    return PredictResponse(job_id=job_id, status=JobStatus.QUEUED, model=model.value)
