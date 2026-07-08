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
    musetalk_disabled: bool = False
    max_concurrent_jobs: int
    active_jobs: int
    memory: dict[str, float] = Field(default_factory=dict)
    cache_enabled: bool = True
    gpu_pool: list[int] = Field(default_factory=list)


class ServiceInfo(BaseModel):
    port: int
    url: str
    local_url: str
    note: str
    reachable_via_proxy: bool
    up: bool = False


class ServicesResponse(BaseModel):
    hint: str
    services: dict[str, ServiceInfo]


class DubResponse(BaseModel):
    job_id: str
    batch_id: str | None = None
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
    batch_id: str | None = None
    status: str
    model: str = ModelName.LATENTSYNC.value
    created_at: datetime | None = None
    updated_at: datetime | None = None
    output_video: str | None = None
    download_url: str | None = None
    error: str | None = None
    elapsed_s: float | None = None
    dub_correct: bool = False
    original_video: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PaginatedJobsResponse(BaseModel):
    jobs: list[JobResponse]
    total: int
    page: int
    per_page: int
    pages: int


# ── GPU metrics ───────────────────────────────────────────────────────────────

class GPUDeviceStatus(BaseModel):
    gpu_id: int
    name: str
    total_gb: float
    overhead_gb: float
    usable_gb: float
    claimed_gb: float
    soft_available_gb: float
    actual_free_gb: float
    hard_available_gb: float
    active_jobs: int
    active_job_ids: list[str] = Field(default_factory=list)
    used_pct: float


class GPUStatusResponse(BaseModel):
    devices: list[GPUDeviceStatus]
    total_active_jobs: int
    overhead_pct: int
