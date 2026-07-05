"""Request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.config import ModelName


class ModelInfo(BaseModel):
    name: str
    display_name: str
    available: bool
    vram_gb: float
    paper: str


class HealthResponse(BaseModel):
    status: str = "healthy"
    cuda_available: bool
    ffmpeg_available: bool
    models: dict[str, bool]
    max_concurrent_jobs: int
    active_jobs: int
    memory: dict[str, float] = Field(default_factory=dict)
    cache_enabled: bool = True


class DubResponse(BaseModel):
    job_id: str
    status: str
    model: str
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
    model: str = ModelName.LATENTSYNC.value
    created_at: datetime | None = None
    updated_at: datetime | None = None
    output_video: str | None = None
    download_url: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
