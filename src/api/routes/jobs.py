"""Job status, listing, and download endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from src.api.schemas import JobResponse, PaginatedJobsResponse
from src.db.repos import list_jobs, list_jobs_paginated, get_job
from src.config import settings

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _enrich(doc: dict) -> JobResponse:
    jid = doc.get("job_id", "")
    meta = doc.get("metadata", {})
    out = doc.get("output_video")
    return JobResponse(
        job_id=jid,
        batch_id=meta.get("batch_id"),
        status=str(doc.get("status", "unknown")),
        model=doc.get("model", "latentsync"),
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at"),
        output_video=out,
        download_url=f"/jobs/{jid}/download" if out else None,
        error=doc.get("error"),
        elapsed_s=doc.get("elapsed_s"),
        dub_correct=bool(meta.get("dub_correct", False)),
        original_video=meta.get("original_video"),
        metadata=meta,
    )


@router.get("", response_model=PaginatedJobsResponse, summary="Paginated job list")
async def jobs_list(
    page: int = Query(default=1, ge=1, description="Page number"),
    per_page: int = Query(default=10, ge=1, le=100, description="Items per page"),
    search: str | None = Query(default=None, description="Search by video name or job ID"),
    status: str | None = Query(default=None, description="Filter by status"),
    sort_by: Literal["created_at", "updated_at", "elapsed_s", "status", "model"] = Query(
        default="created_at"
    ),
    order: Literal["asc", "desc"] = Query(default="desc"),
    batch_id: str | None = Query(default=None, description="Filter by batch ID"),
) -> PaginatedJobsResponse:
    result = await list_jobs_paginated(
        page=page,
        per_page=per_page,
        search=search or None,
        status=status or None,
        sort_by=sort_by,
        order=order,
        batch_id=batch_id or None,
    )
    return PaginatedJobsResponse(
        jobs=[_enrich(d) for d in result["jobs"]],
        total=result["total"],
        page=result["page"],
        per_page=result["per_page"],
        pages=result["pages"],
    )


@router.get("/{job_id}", response_model=JobResponse, summary="Single job detail")
async def job_detail(job_id: str) -> JobResponse:
    doc = await get_job(job_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found")
    return _enrich(doc)


@router.get("/{job_id}/download", summary="Download output video")
async def job_download(job_id: str) -> FileResponse:
    doc = await get_job(job_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found")
    out = doc.get("output_video")
    if not out or not Path(out).exists():
        raise HTTPException(status_code=404, detail="Output video not ready")
    return FileResponse(out, media_type="video/mp4", filename=f"{job_id}.mp4")
