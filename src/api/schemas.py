"""Request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    cuda_available: bool
    ffmpeg_available: bool
    model_ready: bool
    max_concurrent_jobs: int
    active_jobs: int


class DubRequest(BaseModel):
    """For JSON body on /dub/url endpoint."""
    video_url: str
    audio_url: str | None = None
    inference_steps: int = Field(20, ge=1, le=50)
    guidance_scale: float = Field(1.5, ge=0.5, le=5.0)
    seed: int = Field(1247, ge=-1)
    sync: bool = True


class DubResponse(BaseModel):
    job_id: str
    status: str
    output_video: str | None = None
    download_url: str | None = None
    error: str | None = None
    elapsed_s: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchDubResponse(BaseModel):
    batch_id: str
    total: int
    jobs: list[DubResponse]


class JobResponse(BaseModel):
    job_id: str
    status: str
    model: str = "latentsync"
    created_at: datetime | None = None
    updated_at: datetime | None = None
    output_video: str | None = None
    download_url: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
