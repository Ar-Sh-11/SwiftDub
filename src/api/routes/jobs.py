"""Job status, listing, and download endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.api.schemas import JobResponse
from src.db.repos import list_jobs, get_job
from src.config import settings

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _enrich(doc: dict) -> JobResponse:
    jid = doc.get("job_id", "")
    out = doc.get("output_video")
    return JobResponse(
        job_id=jid,
        status=str(doc.get("status", "unknown")),
        model=doc.get("model", "latentsync"),
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at"),
        output_video=out,
        download_url=f"/jobs/{jid}/download" if out else None,
        error=doc.get("error"),
        metadata=doc.get("metadata", {}),
    )


@router.get("", response_model=list[JobResponse])
async def jobs_list(limit: int = 20) -> list[JobResponse]:
    docs = await list_jobs(limit=limit)
    return [_enrich(d) for d in docs]


@router.get("/{job_id}", response_model=JobResponse)
async def job_detail(job_id: str) -> JobResponse:
    doc = await get_job(job_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found")
    return _enrich(doc)


@router.get("/{job_id}/download")
async def job_download(job_id: str) -> FileResponse:
    doc = await get_job(job_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found")
    out = doc.get("output_video")
    if not out or not Path(out).exists():
        raise HTTPException(status_code=404, detail="Output video not ready")
    return FileResponse(out, media_type="video/mp4", filename=f"{job_id}.mp4")
