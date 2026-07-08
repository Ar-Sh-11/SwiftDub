"""Video dubbing pipeline — multi-model with dynamic VRAM-aware GPU distribution.

Concurrency model:
  DynamicGPUAllocator.acquire() blocks until a GPU has enough free VRAM to
  accommodate the job (20% overhead always reserved). Multiple small jobs can
  run concurrently on the same GPU if VRAM allows. No fixed semaphore.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

from loguru import logger

from src.config import ModelName, settings
from src.core import latentsync, musetalk
from src.logging_.inference_log import log_inference_event
from src.metrics.registry import active_jobs, inference_duration, inference_requests, queue_depth
from src.utils.ffmpeg import extract_audio
from src.utils.gpu_alloc import get_allocator
from src.utils.memory import release_gpu_memory

_queue_count: int = 0


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
    gpu_id: int | None,
) -> float:
    if model == ModelName.MUSETALK.value:
        return musetalk.run_musetalk(
            video_path,
            audio_path,
            raw_output,
            bbox_shift=bbox_shift,
            temp_dir=job_tmp / "mt_temp",
            gpu_id=gpu_id,
        )
    return latentsync.run_latentsync(
        video_path,
        audio_path,
        raw_output,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        enable_deepcache=enable_deepcache,
        temp_dir=job_tmp / "ls_temp",
        gpu_id=gpu_id,
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
) -> tuple[Path, float]:
    """Full dubbing pipeline.

    Returns (output_video_path, elapsed_wall_seconds).
    """
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

    alloc = get_allocator()
    vram_gb = alloc.vram_for_model(model)

    t0 = time.time()
    active_jobs.inc()
    _queue_count += 1
    queue_depth.set(_queue_count)

    gpu_id: int | None = None
    try:
        # Blocks until a GPU can fit this job (VRAM-aware)
        gpu_id = await alloc.acquire(job_id, vram_gb)
        _queue_count = max(0, _queue_count - 1)
        queue_depth.set(_queue_count)

        logger.info("[{}] Running {} on GPU {} ({:.1f} GB claimed)", job_id, model, gpu_id, vram_gb)

        elapsed = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _run_inference(
                model,
                video_path,
                audio_path,
                raw_output,
                job_tmp,
                inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                enable_deepcache=enable_deepcache,
                bbox_shift=bbox_shift,
                gpu_id=gpu_id,
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
        return final, wall_time

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
            audio_in=str(audio_path) if audio_path else "",
            output=None,
            error=str(exc),
            steps=inference_steps or settings.latentsync_inference_steps,
            guidance=guidance_scale or settings.latentsync_guidance_scale,
        )
        raise
    finally:
        if gpu_id is not None:
            await alloc.release(gpu_id, job_id)
        _queue_count = max(0, _queue_count - 1)
        queue_depth.set(_queue_count)
        active_jobs.dec()
        release_gpu_memory()
