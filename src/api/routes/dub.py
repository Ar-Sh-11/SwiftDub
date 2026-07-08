"""Video dubbing endpoints — LatentSync 1.5 and MuseTalk v1.5.

Features:
  - audio field accepts EITHER audio OR video files (audio is extracted
    from video automatically) — video-to-video dubbing.
  - dub_correct=true: re-sync a badly-dubbed video to its own audio.
  - Batch: flexible pairing + per-pair dub_correct via dub_correct_flags.
  - GPU distribution: VRAM-aware dynamic concurrency.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger

from src.api.deps import save_audio_upload, save_upload
from src.api.schemas import BatchDubResponse, DubResponse
from src.cache.client import cache_output, get_cached_output
from src.config import ModelName, settings
from src.core.batch_queue import enqueue_batch_job
from src.core.registry import assert_model_available
from src.core.pipeline import dub_video
from src.db.repos import JobStatus, create_job, update_job
from src.metrics.registry import batch_size_histogram
from src.utils.ffmpeg import extract_audio
from src.utils.upload import safe_upload_path

router = APIRouter(prefix="/dub", tags=["dubbing"])

MAX_BATCH = 20


def _download_url(job_id: str) -> str:
    return f"/jobs/{job_id}/download"


def _validate_model(model_str: str) -> None:
    try:
        assert_model_available(model_str)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def _resolve_batch_pairs(n_videos: int, n_audios: int) -> list[tuple[int, int | None]]:
    if n_videos < 1:
        raise HTTPException(status_code=422, detail="At least one video required")
    if n_audios == 0:
        return [(i, None) for i in range(n_videos)]
    if n_audios == 1:
        return [(i, 0) for i in range(n_videos)]
    if n_videos == 1:
        return [(0, j) for j in range(n_audios)]
    if n_videos == n_audios:
        return [(i, i) for i in range(n_videos)]
    raise HTTPException(
        status_code=422,
        detail=(
            f"Video/audio count mismatch ({n_videos} videos, {n_audios} audios). "
            "Use 1 shared audio, equal pairs, or 1 video with multiple audios."
        ),
    )


def _parse_dub_flags(flags_json: str | None, n: int, global_flag: bool) -> list[bool]:
    """Parse per-pair dub_correct flags.

    Priority: dub_correct_flags JSON > global dub_correct bool.
    """
    if flags_json:
        try:
            parsed = json.loads(flags_json)
            if isinstance(parsed, list) and len(parsed) == n:
                return [bool(f) for f in parsed]
        except (json.JSONDecodeError, TypeError):
            pass
    return [global_flag] * n


async def _resolve_audio_path(
    job_tmp: Path,
    video_path: Path,
    audio_upload: UploadFile | None,
    audio_dir: Path,
    *,
    dub_correct: bool = False,
    index: int | None = None,
) -> Path | None:
    if dub_correct:
        audio_path = job_tmp / "dub_correct_audio.wav"
        extract_audio(video_path, audio_path)
        logger.info("dub_correct: extracted audio from {}", video_path.name)
        return audio_path
    if audio_upload is None:
        return None
    audio_path, was_extracted = await save_audio_upload(audio_upload, audio_dir, index=index)
    if was_extracted:
        logger.info("Extracted audio track from uploaded video file")
    return audio_path


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
        output, elapsed = await dub_video(
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
        await update_job(
            job_id,
            status=JobStatus.COMPLETED,
            output_video=str(output),
            elapsed_s=round(elapsed, 2),
        )
        return DubResponse(
            job_id=job_id,
            status=JobStatus.COMPLETED.value,
            model=model,
            output_video=str(output),
            download_url=_download_url(job_id),
            elapsed_s=round(elapsed, 2),
        )
    except Exception as exc:
        await update_job(job_id, status=JobStatus.FAILED, error=str(exc)[:1000])
        raise HTTPException(status_code=500, detail=str(exc)[:500])


async def _run_sync_quiet(
    job_id: str,
    video_path: Path,
    audio_path: Path | None,
    model: str,
    steps: int,
    scale: float,
    seed: int,
    bbox_shift: int,
    enable_deepcache: bool,
) -> None:
    """Batch worker — failures are recorded without raising HTTPException."""
    await update_job(job_id, status=JobStatus.PROCESSING)
    try:
        output, elapsed = await dub_video(
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
        await update_job(
            job_id,
            status=JobStatus.COMPLETED,
            output_video=str(output),
            elapsed_s=round(elapsed, 2),
        )
    except Exception as exc:
        logger.error("[{}] Batch job failed: {}", job_id, exc)
        await update_job(job_id, status=JobStatus.FAILED, error=str(exc)[:1000])


@router.post("", response_model=DubResponse, summary="Dub a single video")
async def dub_single(
    video: UploadFile = File(..., description="Input video for lip-sync"),
    audio: Optional[UploadFile] = File(
        None,
        description=(
            "Audio source. Accepts audio files (WAV, MP3, AAC…) OR video files "
            "(the audio track is extracted automatically). Omit to use the video's "
            "own audio track."
        ),
    ),
    model: ModelName = Form(default=ModelName.LATENTSYNC),
    inference_steps: int = Form(default=20, ge=1, le=50),
    guidance_scale: float = Form(default=1.5, ge=0.5, le=5.0),
    seed: int = Form(default=1247),
    bbox_shift: int = Form(default=0),
    enable_deepcache: bool = Form(default=False),
    sync: bool = Form(default=True, description="Wait for result (True) or queue async (False)"),
    dub_correct: bool = Form(
        default=False,
        description=(
            "Re-sync a badly dubbed video to its own audio track. "
            "When True the audio field is ignored."
        ),
    ),
) -> DubResponse:
    model_str = model.value
    _validate_model(model_str)

    job_id = uuid.uuid4().hex
    job_dir = settings.uploads_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job_tmp = settings.temp_dir / job_id
    job_tmp.mkdir(parents=True, exist_ok=True)

    video_path = await save_upload(video, dest_dir=job_dir, kind="video")
    video_bytes = video_path.read_bytes()

    audio_path = await _resolve_audio_path(
        job_tmp, video_path, audio, job_dir, dub_correct=dub_correct
    )

    audio_bytes: bytes | None = audio_path.read_bytes() if audio_path else None
    if audio_bytes:
        cached = await get_cached_output(video_bytes, audio_bytes, model_str)
        if cached:
            logger.info("[{}] Cache hit for {}", job_id, model_str)
            await create_job(job_id, model=model_str)
            await update_job(job_id, status=JobStatus.COMPLETED, output_video=cached, elapsed_s=0.0)
            return DubResponse(
                job_id=job_id,
                status=JobStatus.COMPLETED.value,
                model=model_str,
                output_video=cached,
                download_url=_download_url(job_id),
                elapsed_s=0.0,
            )

    await create_job(
        job_id,
        model=model_str,
        metadata={
            "inference_steps": inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "bbox_shift": bbox_shift,
            "dub_correct": dub_correct,
            "original_video": video.filename,
            "original_audio": audio.filename if audio else None,
        },
    )
    logger.info("[{}] Dub request model={} sync={} dub_correct={}", job_id, model_str, sync, dub_correct)

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
            job_id,
            str(video_path.resolve()),
            str(audio_path.resolve()) if audio_path else None,
            model_str, inference_steps, guidance_scale, seed, bbox_shift, enable_deepcache,
        )
    except Exception:
        logger.warning("[{}] Celery unavailable — asyncio background task", job_id)
        asyncio.create_task(
            _run_sync_quiet(
                job_id, video_path, audio_path, model_str,
                inference_steps, guidance_scale, seed, bbox_shift, enable_deepcache,
            )
        )
    return DubResponse(job_id=job_id, status=JobStatus.QUEUED.value, model=model_str)


@router.post("/batch", response_model=BatchDubResponse, summary="Dub multiple video/audio pairs")
async def dub_batch(
    videos: list[UploadFile] = File(...),
    audios: Optional[list[UploadFile]] = File(
        None,
        description=(
            "Audio sources. Each entry can be an audio file OR a video file "
            "(audio track is extracted). Supports all batch pairing modes."
        ),
    ),
    model: ModelName = Form(default=ModelName.LATENTSYNC),
    inference_steps: int = Form(default=20, ge=1, le=50),
    guidance_scale: float = Form(default=1.5, ge=0.5, le=5.0),
    seed: int = Form(default=1247),
    bbox_shift: int = Form(default=0),
    enable_deepcache: bool = Form(default=False),
    dub_correct: bool = Form(
        default=False,
        description="Apply dub_correct to ALL pairs (overridden by dub_correct_flags per pair).",
    ),
    dub_correct_flags: Optional[str] = Form(
        default=None,
        description=(
            "JSON boolean array — one flag per pair. E.g. '[true, false, true]'. "
            "Length must equal the number of resolved pairs. Overrides global dub_correct."
        ),
    ),
) -> BatchDubResponse:
    """Batch dub with flexible pairing and per-pair dub_correct support.

    Pairing rules (when dub_correct is False for all pairs):
    - N videos + 0 audios  → extract audio from each video
    - N videos + 1 audio   → shared audio for all videos
    - N videos + N audios  → paired 1:1
    - 1 video  + M audios  → same video dubbed with each audio
    """
    model_str = model.value
    _validate_model(model_str)

    audio_files = audios or []
    pairs: list[tuple[int, int | None]] = _resolve_batch_pairs(len(videos), len(audio_files))

    if len(pairs) > MAX_BATCH:
        raise HTTPException(
            status_code=422, detail=f"Max {MAX_BATCH} jobs per batch (got {len(pairs)})"
        )

    # Resolve per-pair dub_correct flags
    dc_flags = _parse_dub_flags(dub_correct_flags, len(pairs), dub_correct)

    batch_size_histogram.observe(len(pairs))
    batch_id = uuid.uuid4().hex
    job_ids: list[str] = []

    batch_dir = settings.uploads_dir / f"batch_{batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    saved_videos: list[Path] = []
    for i, vid in enumerate(videos):
        saved_videos.append(await save_upload(vid, dest_dir=batch_dir, kind="video", index=i))

    saved_audios: list[Path] = []
    for i, aud in enumerate(audio_files):
        aud_path, _ = await save_audio_upload(aud, batch_dir, index=i)
        saved_audios.append(aud_path)

    for pair_idx, (vi, ai) in enumerate(pairs):
        job_id = uuid.uuid4().hex
        job_dir = settings.uploads_dir / job_id
        job_tmp = settings.temp_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job_tmp.mkdir(parents=True, exist_ok=True)

        video_path = job_dir / saved_videos[vi].name
        video_path.write_bytes(saved_videos[vi].read_bytes())

        pair_dc = dc_flags[pair_idx]
        audio_path: Path | None = None
        if pair_dc:
            audio_path = job_tmp / "dub_correct_audio.wav"
            extract_audio(video_path, audio_path)
        elif ai is not None:
            audio_path = job_dir / saved_audios[ai].name
            audio_path.write_bytes(saved_audios[ai].read_bytes())

        await create_job(
            job_id,
            model=model_str,
            metadata={
                "batch_id": batch_id,
                "pair_index": pair_idx,
                "video_index": vi,
                "audio_index": ai,
                "dub_correct": pair_dc,
                "inference_steps": inference_steps,
                "guidance_scale": guidance_scale,
                "original_video": videos[vi].filename,
                "original_audio": (
                    audio_files[ai].filename
                    if ai is not None and ai < len(audio_files)
                    else None
                ),
            },
        )
        job_ids.append(job_id)

        await enqueue_batch_job(
            job_id,
            model_str,
            lambda jid=job_id, vp=video_path, ap=audio_path: _run_sync_quiet(
                jid, vp, ap, model_str,
                inference_steps, guidance_scale, seed, bbox_shift, enable_deepcache,
            ),
        )

    logger.info(
        "Batch {} queued {} jobs (model={} dc_flags={})",
        batch_id, len(job_ids), model_str, dc_flags,
    )
    jobs = [DubResponse(job_id=jid, status=JobStatus.QUEUED.value, model=model_str) for jid in job_ids]
    return BatchDubResponse(batch_id=batch_id, total=len(job_ids), jobs=jobs)
