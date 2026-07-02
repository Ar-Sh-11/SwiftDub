"""Orchestrates face detection → model inference → audio mux."""

from __future__ import annotations

import uuid
from pathlib import Path

from loguru import logger

from src.config import ModelName, settings
from src.core.face import FaceSelector
from src.core.models.registry import get_backend
from src.utils.ffmpeg import extract_audio, mux_audio_video
from src.utils.video import read_frames

_face_selector = FaceSelector()


async def run_pipeline(
    job_id: str,
    video_path: Path,
    audio_path: Path | None,
    model: str,
) -> Path:
    """Full inference pipeline. Returns path to output video."""
    model_name = ModelName(model)
    backend = get_backend(model_name)

    if not backend.is_ready():
        raise RuntimeError(
            f"Model '{model}' weights are missing. "
            "Run: python scripts/download/models.py --model " + model
        )

    job_tmp = settings.temp_dir / job_id
    job_tmp.mkdir(parents=True, exist_ok=True)

    # If no separate audio, extract it from video
    if audio_path is None:
        audio_path = job_tmp / "extracted_audio.wav"
        extract_audio(video_path, audio_path)

    logger.info("[{}] Selecting left-most face", job_id)
    frames, _ = read_frames(video_path, max_frames=120)
    ref_face = _face_selector.select_leftmost(frames)
    face_box = _face_selector.stable_box(frames, ref_face)
    logger.info("[{}] Face box: {}", job_id, face_box)

    raw_output = job_tmp / f"raw_{model}.mp4"
    logger.info("[{}] Running {} inference", job_id, model)
    result = backend.infer(video_path, audio_path, raw_output, face_box=face_box)

    # Final mux: guarantees original audio is untouched
    final = settings.outputs_dir / f"{job_id}_{model}.mp4"
    mux_audio_video(result.output_video, audio_path, final, copy_video=False)
    logger.info("[{}] Done in {:.1f}s → {}", job_id, result.elapsed_s, final)
    return final
