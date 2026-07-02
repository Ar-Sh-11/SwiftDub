"""Job status and download endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.api.schemas import JobResponse
from src.db.repos import get_job, list_jobs

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=list[JobResponse])
async def get_all_jobs(limit: int = 50) -> list[dict]:
    return await list_jobs(limit=limit)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str) -> dict:
    doc = await get_job(job_id)
    if not doc:
        raise HTTPException(404, "Job not found")
    return doc


@router.get("/{job_id}/download")
async def download_output(job_id: str) -> FileResponse:
    doc = await get_job(job_id)
    if not doc or not doc.get("output_video"):
        raise HTTPException(404, "Output not ready")
    path = Path(doc["output_video"])
    if not path.exists():
        raise HTTPException(404, "Output file missing from disk")
    return FileResponse(path, media_type="video/mp4", filename=path.name)
