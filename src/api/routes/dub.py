"""Video dubbing endpoints — LatentSync and MuseTalk."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from src.api.deps import save_upload
from src.api.schemas import BatchDubResponse, DubResponse
from src.cache.client import cache_output, get_cached_output
from src.config import ModelName, settings
from src.core.registry import is_ready
from src.core.pipeline import dub_video
from src.db.repos import JobStatus, create_job, update_job
from src.metrics.registry import batch_size_histogram

router = APIRouter(prefix="/dub", tags=["dubbing"])

MAX_BATCH = 10


def _download_url(job_id: str) -> str:
    return f"/jobs/{job_id}/download"


async def _run_sync(
    job_id: str,
    video_path: Path,
    audio_path: Path | None,
    model: str,
    steps: int,
    scale: float,
    seed: int,
    bbox_shift: int,
    enable_deepcache: bool,
) -> DubResponse:
    await update_job(job_id, status=JobStatus.PROCESSING)
    try:
        output = await dub_video(
            job_id=job_id,
            video_path=video_path,
            audio_path=audio_path,
            model=model,
            inference_steps=steps,
            guidance_scale=scale,
            seed=seed,
            enable_deepcache=enable_deepcache,
            bbox_shift=bbox_shift,
        )
        await update_job(job_id, status=JobStatus.COMPLETED, output_video=str(output))
        return DubResponse(
            job_id=job_id,
            status=JobStatus.COMPLETED.value,
            model=model,
            output_video=str(output),
            download_url=_download_url(job_id),
        )
    except Exception as exc:
        await update_job(job_id, status=JobStatus.FAILED, error=str(exc)[:1000])
        raise HTTPException(status_code=500, detail=str(exc)[:500])


@router.post("", response_model=DubResponse, summary="Dub a single video")
async def dub_single(
    video: UploadFile = File(...),
    audio: Optional[UploadFile] = File(None),
    model: ModelName = Form(default=ModelName.LATENTSYNC),
    inference_steps: int = Form(default=20, ge=1, le=50),
    guidance_scale: float = Form(default=1.5, ge=0.5, le=5.0),
    seed: int = Form(default=1247),
    bbox_shift: int = Form(default=0),
    enable_deepcache: bool = Form(default=False),
    sync: bool = Form(default=True),
) -> DubResponse:
    model_str = model.value
    if not is_ready(model_str):
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_str}' weights not found. Run: python scripts/download/models.py --model {model_str}",
        )

    job_id = uuid.uuid4().hex
    job_dir = settings.uploads_dir / job_id
    video_path = job_dir / (video.filename or "input.mp4")
    video_bytes = await video.read()
    video_path.write_bytes(video_bytes)

    audio_path: Path | None = None
    audio_bytes: bytes | None = None
    if audio:
        audio_path = job_dir / (audio.filename or "input.wav")
        audio_bytes = await audio.read()
        audio_path.write_bytes(audio_bytes)

    # Cache lookup (skipped automatically when GPU/RAM > 95%)
    if audio_bytes:
        cached = await get_cached_output(video_bytes, audio_bytes, model_str)
        if cached:
            logger.info("[{}] Cache hit for {}", job_id, model_str)
            await create_job(job_id, model=model_str)
            await update_job(job_id, status=JobStatus.COMPLETED, output_video=cached)
            return DubResponse(
                job_id=job_id,
                status=JobStatus.COMPLETED.value,
                model=model_str,
                output_video=cached,
                download_url=_download_url(job_id),
            )

    await create_job(job_id, model=model_str, metadata={
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "bbox_shift": bbox_shift,
    })
    logger.info("[{}] Dub request model={} sync={}", job_id, model_str, sync)

    if sync:
        resp = await _run_sync(
            job_id, video_path, audio_path, model_str,
            inference_steps, guidance_scale, seed, bbox_shift, enable_deepcache,
        )
        if audio_bytes and resp.output_video:
            await cache_output(video_bytes, audio_bytes, model_str, resp.output_video)
        return resp

    try:
        from src.workers.tasks import dub_task
        dub_task.delay(
            job_id, str(video_path),
            str(audio_path) if audio_path else None,
            model_str, inference_steps, guidance_scale, seed, bbox_shift, enable_deepcache,
        )
    except Exception:
        logger.warning("[{}] Celery unavailable — asyncio background task", job_id)
        asyncio.create_task(
            _run_sync(
                job_id, video_path, audio_path, model_str,
                inference_steps, guidance_scale, seed, bbox_shift, enable_deepcache,
            )
        )
    return DubResponse(job_id=job_id, status=JobStatus.QUEUED.value, model=model_str)


@router.post("/batch", response_model=BatchDubResponse, summary="Dub multiple videos")
async def dub_batch(
    videos: list[UploadFile] = File(...),
    audio: Optional[UploadFile] = File(None),
    model: ModelName = Form(default=ModelName.LATENTSYNC),
    inference_steps: int = Form(default=20, ge=1, le=50),
    guidance_scale: float = Form(default=1.5, ge=0.5, le=5.0),
    seed: int = Form(default=1247),
    bbox_shift: int = Form(default=0),
    sync: bool = Form(default=False),
) -> BatchDubResponse:
    model_str = model.value
    if not is_ready(model_str):
        raise HTTPException(status_code=503, detail=f"Model '{model_str}' not ready")
    if len(videos) > MAX_BATCH:
        raise HTTPException(status_code=422, detail=f"Max {MAX_BATCH} videos per batch")

    batch_size_histogram.observe(len(videos))
    batch_id = uuid.uuid4().hex
    job_ids: list[str] = []

    shared_audio_path: Path | None = None
    if audio:
        batch_dir = settings.uploads_dir / f"batch_{batch_id}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        shared_audio_path = batch_dir / (audio.filename or "shared_audio.wav")
        await save_upload(audio, shared_audio_path)

    for vid in videos:
        job_id = uuid.uuid4().hex
        job_dir = settings.uploads_dir / job_id
        video_path = job_dir / (vid.filename or "input.mp4")
        await save_upload(vid, video_path)

        await create_job(job_id, model=model_str, metadata={
            "batch_id": batch_id,
            "inference_steps": inference_steps,
            "guidance_scale": guidance_scale,
        })
        job_ids.append(job_id)

        asyncio.create_task(
            _run_sync(
                job_id, video_path, shared_audio_path, model_str,
                inference_steps, guidance_scale, seed, bbox_shift, False,
            )
        )

    jobs = [DubResponse(job_id=jid, status=JobStatus.QUEUED.value, model=model_str) for jid in job_ids]
    return BatchDubResponse(batch_id=batch_id, total=len(job_ids), jobs=jobs)
