"""LatentSync dubbing pipeline — audio extraction → inference → log."""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

from loguru import logger

from src.config import settings
from src.core.latentsync import run_latentsync
from src.logging_.inference_log import log_inference_event
from src.metrics.registry import active_jobs, inference_duration, inference_requests
from src.utils.ffmpeg import extract_audio

# Semaphore that limits concurrent GPU inference jobs
_inference_sem: asyncio.Semaphore | None = None


def get_semaphore() -> asyncio.Semaphore:
    global _inference_sem
    if _inference_sem is None:
        _inference_sem = asyncio.Semaphore(settings.max_concurrent_jobs)
    return _inference_sem


async def dub_video(
    job_id: str,
    video_path: Path,
    audio_path: Path | None,
    *,
    inference_steps: int | None = None,
    guidance_scale: float | None = None,
    seed: int | None = None,
    enable_deepcache: bool | None = None,
) -> Path:
    """Full dubbing pipeline for a single video.

    Waits for the concurrency semaphore, extracts audio if needed,
    runs LatentSync, logs the event, and returns the final output path.
    """
    sem = get_semaphore()
    job_tmp = settings.temp_dir / job_id
    job_tmp.mkdir(parents=True, exist_ok=True)

    if audio_path is None:
        audio_path = job_tmp / "extracted_audio.wav"
        logger.info("[{}] Extracting audio from video", job_id)
        extract_audio(video_path, audio_path)

    raw_output = job_tmp / "dubbed.mp4"
    final = settings.outputs_dir / f"{job_id}.mp4"

    t0 = time.time()
    active_jobs.inc()
    try:
        logger.info("[{}] Waiting for GPU slot (max_concurrent={})", job_id, settings.max_concurrent_jobs)
        async with sem:
            logger.info("[{}] Running LatentSync inference", job_id)
            elapsed = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: run_latentsync(
                    video_path, audio_path, raw_output,
                    inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    enable_deepcache=enable_deepcache,
                    temp_dir=job_tmp / "ls_temp",
                )
            )

        # Copy to final outputs dir
        import shutil
        shutil.copy2(raw_output, final)

        wall_time = time.time() - t0
        inference_duration.labels(model="latentsync").observe(elapsed)
        inference_requests.labels(model="latentsync", status="success").inc()

        log_inference_event(
            job_id=job_id,
            status="success",
            elapsed_s=elapsed,
            wall_s=wall_time,
            video_in=str(video_path),
            audio_in=str(audio_path),
            output=str(final),
            steps=inference_steps or settings.latentsync_inference_steps,
            guidance=guidance_scale or settings.latentsync_guidance_scale,
        )
        logger.info("[{}] Done in {:.1f}s → {}", job_id, wall_time, final)
        return final

    except Exception as exc:
        wall_time = time.time() - t0
        inference_requests.labels(model="latentsync", status="error").inc()
        inference_duration.labels(model="latentsync").observe(wall_time)
        log_inference_event(
            job_id=job_id,
            status="error",
            elapsed_s=wall_time,
            wall_s=wall_time,
            video_in=str(video_path),
            audio_in=str(audio_path),
            output=None,
            error=str(exc),
            steps=inference_steps or settings.latentsync_inference_steps,
            guidance=guidance_scale or settings.latentsync_guidance_scale,
        )
        raise
    finally:
        active_jobs.dec()
