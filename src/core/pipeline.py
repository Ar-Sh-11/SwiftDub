"""Video dubbing pipeline — multi-model with GPU semaphore and memory gating."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from loguru import logger

from src.config import ModelName, settings
from src.core import latentsync, musetalk
from src.logging_.inference_log import log_inference_event
from src.metrics.registry import active_jobs, inference_duration, inference_requests, queue_depth
from src.utils.ffmpeg import extract_audio
from src.utils.memory import release_gpu_memory, gpu_free_gb, gpu_memory_pct

_inference_sem: asyncio.Semaphore | None = None
_queue_count: int = 0


def get_semaphore() -> asyncio.Semaphore:
    global _inference_sem
    if _inference_sem is None:
        _inference_sem = asyncio.Semaphore(settings.max_concurrent_jobs)
    return _inference_sem


def _run_inference(
    model: str,
    video_path: Path,
    audio_path: Path,
    raw_output: Path,
    job_tmp: Path,
    *,
    inference_steps: int | None,
    guidance_scale: float | None,
    seed: int | None,
    enable_deepcache: bool | None,
    bbox_shift: int | None,
) -> float:
    if model == ModelName.MUSETALK.value:
        return musetalk.run_musetalk(
            video_path, audio_path, raw_output,
            bbox_shift=bbox_shift,
            temp_dir=job_tmp / "mt_temp",
        )
    return latentsync.run_latentsync(
        video_path, audio_path, raw_output,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        enable_deepcache=enable_deepcache,
        temp_dir=job_tmp / "ls_temp",
    )


async def dub_video(
    job_id: str,
    video_path: Path,
    audio_path: Path | None,
    model: str = ModelName.LATENTSYNC.value,
    *,
    inference_steps: int | None = None,
    guidance_scale: float | None = None,
    seed: int | None = None,
    enable_deepcache: bool | None = None,
    bbox_shift: int | None = None,
) -> Path:
    """Full dubbing pipeline. Returns path to output video."""
    global _queue_count

    model = ModelName(model).value
    job_tmp = settings.temp_dir / job_id
    job_tmp.mkdir(parents=True, exist_ok=True)

    if audio_path is None:
        audio_path = job_tmp / "extracted_audio.wav"
        logger.info("[{}] Extracting audio from video", job_id)
        extract_audio(video_path, audio_path)

    raw_output = job_tmp / f"dubbed_{model}.mp4"
    final = settings.outputs_dir / f"{job_id}_{model}.mp4"

    import time
    t0 = time.time()
    active_jobs.inc()
    try:
        _queue_count += 1
        queue_depth.set(_queue_count)

        logger.info("[{}] Waiting for GPU slot ({}/{})", job_id, model, settings.max_concurrent_jobs)

        # Wait for VRAM headroom before competing for semaphore
        need_gb = 10.0 if model == ModelName.LATENTSYNC.value else 7.0
        for _ in range(60):
            if gpu_free_gb() >= need_gb or gpu_memory_pct() < 75:
                break
            await asyncio.sleep(3)

        async with get_semaphore():
            _queue_count = max(0, _queue_count - 1)
            queue_depth.set(_queue_count)
            logger.info("[{}] Running {} inference", job_id, model)
            elapsed = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _run_inference(
                    model, video_path, audio_path, raw_output, job_tmp,
                    inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    enable_deepcache=enable_deepcache,
                    bbox_shift=bbox_shift,
                ),
            )

        shutil.copy2(raw_output, final)
        wall_time = time.time() - t0
        inference_duration.labels(model=model).observe(elapsed)
        inference_requests.labels(model=model, status="success").inc()

        log_inference_event(
            job_id=job_id,
            model=model,
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
        inference_requests.labels(model=model, status="error").inc()
        inference_duration.labels(model=model).observe(wall_time)
        log_inference_event(
            job_id=job_id,
            model=model,
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
        _queue_count = max(0, _queue_count - 1)
        queue_depth.set(_queue_count)
        active_jobs.dec()
        release_gpu_memory()
