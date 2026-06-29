"""Prediction routes."""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.config import LipSyncModel, settings
from src.schemas.api import JobStatusResponse, PredictResponse
from src.services.job_manager import JobStatus, job_manager
from src.services.pipeline import LipSyncPipeline, PipelineInput

router = APIRouter(tags=["predict"])
executor = ThreadPoolExecutor(max_workers=1)
pipeline = LipSyncPipeline()


def _save_upload(upload: UploadFile, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return destination


def _process_job(job_id: str, video_path: Path, audio_path: Path | None, model: LipSyncModel) -> None:
    job_manager.update(job_id, status=JobStatus.PROCESSING)
    try:
        output = pipeline.run(
            PipelineInput(
                video_path=video_path,
                audio_path=audio_path,
                model=model,
                job_id=job_id,
            )
        )
        job_manager.update(
            job_id,
            status=JobStatus.COMPLETED,
            output_video=str(output.output_video),
            metadata={
                **output.metadata,
                "face_box": list(output.face_box),
            },
        )
    except Exception as exc:  # noqa: BLE001 - surface pipeline errors to job status
        job_manager.update(job_id, status=JobStatus.FAILED, error=str(exc))


@router.post("/predict", response_model=PredictResponse)
async def predict(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    audio: UploadFile | None = File(None),
    model: LipSyncModel = Form(default=LipSyncModel.WAV2LIP),
    sync: bool = Form(default=True),
) -> PredictResponse:
    record = job_manager.create(model=model.value)
    job_dir = settings.uploads_dir / record.job_id
    video_path = _save_upload(video, job_dir / (video.filename or "input.mp4"))
    audio_path = None
    if audio is not None:
        audio_path = _save_upload(audio, job_dir / (audio.filename or "input.wav"))

    if sync:
        _process_job(record.job_id, video_path, audio_path, model)
        updated = job_manager.get(record.job_id)
        if updated is None:
            raise HTTPException(status_code=500, detail="Job record missing")
        face_box = updated.metadata.get("face_box")
        return PredictResponse(
            job_id=updated.job_id,
            status=updated.status.value,
            model=updated.model,
            output_video=updated.output_video,
            face_box=face_box,
            metadata=updated.metadata,
            error=updated.error,
        )

    background_tasks.add_task(_process_job, record.job_id, video_path, audio_path, model)
    return PredictResponse(
        job_id=record.job_id,
        status=JobStatus.QUEUED.value,
        model=record.model,
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    record = job_manager.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=record.job_id,
        status=record.status.value,
        model=record.model,
        output_video=record.output_video,
        error=record.error,
        metadata=record.metadata,
    )


@router.get("/download/{job_id}")
def download_output(job_id: str) -> FileResponse:
    record = job_manager.get(job_id)
    if record is None or not record.output_video:
        raise HTTPException(status_code=404, detail="Output not found")
    output_path = Path(record.output_video)
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file missing")
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=output_path.name,
    )
