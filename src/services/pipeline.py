"""End-to-end lip-sync pipeline."""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from src.config import LipSyncModel, settings
from src.services.lipsync import get_backend
from src.services.lipsync.base import LipSyncResult
from src.utils.face import FaceSelector
from src.utils.ffmpeg import extract_audio, mux_audio_video
from src.utils.video import read_frames


@dataclass
class PipelineInput:
    video_path: Path
    audio_path: Path | None = None
    model: LipSyncModel = LipSyncModel.WAV2LIP
    job_id: str | None = None


@dataclass
class PipelineOutput:
    job_id: str
    output_video: Path
    model: str
    face_box: tuple[int, int, int, int]
    metadata: dict[str, str | int | float]


class LipSyncPipeline:
    def __init__(self) -> None:
        self._face_selector = FaceSelector()

    def run(self, pipeline_input: PipelineInput) -> PipelineOutput:
        job_id = pipeline_input.job_id or uuid.uuid4().hex
        work_dir = settings.temp_dir / job_id
        work_dir.mkdir(parents=True, exist_ok=True)

        video_path = pipeline_input.video_path
        audio_path = pipeline_input.audio_path
        if audio_path is None:
            audio_path = work_dir / "input_audio.wav"
            extract_audio(video_path, audio_path)

        logger.info("Selecting left-most visible face for job {}", job_id)
        frames, _ = read_frames(video_path, max_frames=120)
        selected_face = self._face_selector.select_leftmost_face(frames)
        face_box = self._face_selector.stable_box(frames, selected_face)

        backend = get_backend(pipeline_input.model)
        if not backend.is_available():
            raise RuntimeError(
                f"Model '{pipeline_input.model.value}' is not installed. "
                "Run scripts/download_models.py first."
            )

        raw_output = work_dir / f"{pipeline_input.model.value}_raw.mp4"
        result: LipSyncResult = backend.run(
            video_path=video_path,
            audio_path=audio_path,
            output_path=raw_output,
            face_box=face_box,
        )

        final_output = settings.outputs_dir / f"{job_id}_{pipeline_input.model.value}.mp4"
        mux_audio_video(result.output_video, audio_path, final_output, copy_video=False)

        return PipelineOutput(
            job_id=job_id,
            output_video=final_output,
            model=result.model,
            face_box=face_box,
            metadata=result.metadata,
        )

    def cleanup_job(self, job_id: str) -> None:
        work_dir = settings.temp_dir / job_id
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
