"""Video dubbing endpoints.

POST /dub          — single video (file upload, sync or async)
POST /dub/batch    — multiple videos at once (up to config limit)
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from loguru import logger

from src.api.deps import save_upload
from src.api.schemas import BatchDubResponse, DubResponse
from src.config import settings
from src.core.latentsync import is_ready
from src.core.pipeline import dub_video
from src.db.repos import JobStatus, create_job, get_job, update_job
from src.metrics.registry import batch_size_histogram

router = APIRouter(prefix="/dub", tags=["dubbing"])

MAX_BATCH = 10  # hard cap — override via MAX_BATCH env if needed


def _download_url(job_id: str) -> str:
    return f"/jobs/{job_id}/download"


async def _run_sync(job_id: str, video_path: Path, audio_path: Path | None,
                    steps: int, scale: float, seed: int) -> DubResponse:
    """Run inference synchronously in the request context."""
    await update_job(job_id, status=JobStatus.PROCESSING)
    try:
        output = await dub_video(
            job_id=job_id,
            video_path=video_path,
            audio_path=audio_path,
            inference_steps=steps,
            guidance_scale=scale,
            seed=seed,
        )
        await update_job(job_id, status=JobStatus.COMPLETED, output_video=str(output))
        return DubResponse(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            output_video=str(output),
            download_url=_download_url(job_id),
        )
    except Exception as exc:
        await update_job(job_id, status=JobStatus.FAILED, error=str(exc)[:1000])
        raise HTTPException(status_code=500, detail=str(exc)[:500])


@router.post("", response_model=DubResponse, summary="Dub a single video")
async def dub_single(
    video: UploadFile = File(..., description="Input video (MP4/AVI/MOV)"),
    audio: Optional[UploadFile] = File(None, description="Dub audio (WAV/MP3); extracted from video if absent"),
    inference_steps: int = Form(default=20, ge=1, le=50),
    guidance_scale: float = Form(default=1.5, ge=0.5, le=5.0),
    seed: int = Form(default=1247),
    sync: bool = Form(default=True, description="Wait for result (true) or return job_id immediately (false)"),
) -> DubResponse:
    if not is_ready():
        raise HTTPException(status_code=503, detail="LatentSync model weights not found")

    job_id = uuid.uuid4().hex
    job_dir = settings.uploads_dir / job_id
    video_path = job_dir / (video.filename or "input.mp4")
    await save_upload(video, video_path)

    audio_path: Path | None = None
    if audio:
        audio_path = job_dir / (audio.filename or "input.wav")
        await save_upload(audio, audio_path)

    await create_job(job_id, metadata={
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
    })
    logger.info("[{}] Dub request received (sync={})", job_id, sync)

    if sync:
        return await _run_sync(job_id, video_path, audio_path, inference_steps, guidance_scale, seed)

    # Async: try Celery first, fall back to asyncio background task
    try:
        from src.workers.tasks import dub_task
        dub_task.delay(job_id, str(video_path), str(audio_path) if audio_path else None,
                       inference_steps, guidance_scale, seed)
    except Exception:
        logger.warning("[{}] Celery unavailable — running as asyncio background task", job_id)
        asyncio.create_task(
            _run_sync(job_id, video_path, audio_path, inference_steps, guidance_scale, seed)
        )
    return DubResponse(job_id=job_id, status=JobStatus.QUEUED)


@router.post("/batch", response_model=BatchDubResponse, summary="Dub multiple videos at once")
async def dub_batch(
    videos: list[UploadFile] = File(..., description="Input videos (max 10)"),
    audio: Optional[UploadFile] = File(None, description="Shared dub audio for all videos"),
    inference_steps: int = Form(default=20, ge=1, le=50),
    guidance_scale: float = Form(default=1.5, ge=0.5, le=5.0),
    seed: int = Form(default=1247),
    sync: bool = Form(default=False, description="Usually false for batches — jobs run concurrently"),
) -> BatchDubResponse:
    if not is_ready():
        raise HTTPException(status_code=503, detail="LatentSync model weights not found")
    if len(videos) > MAX_BATCH:
        raise HTTPException(status_code=422, detail=f"Max {MAX_BATCH} videos per batch")

    batch_size_histogram.observe(len(videos))
    batch_id = uuid.uuid4().hex
    job_ids: list[str] = []

    # Save shared audio once if provided
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
        audio_path = shared_audio_path

        await create_job(job_id, metadata={
            "batch_id": batch_id,
            "inference_steps": inference_steps,
            "guidance_scale": guidance_scale,
        })
        job_ids.append(job_id)

        # Queue as asyncio background task (Celery optional for batch)
        asyncio.create_task(
            _run_sync(job_id, video_path, audio_path, inference_steps, guidance_scale, seed)
        )

    jobs = [DubResponse(job_id=jid, status=JobStatus.QUEUED) for jid in job_ids]
    return BatchDubResponse(batch_id=batch_id, total=len(job_ids), jobs=jobs)
