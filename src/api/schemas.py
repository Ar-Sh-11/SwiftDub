"""All Pydantic request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.config import ModelName


class HealthResponse(BaseModel):
    status: str = "healthy"
    cuda_available: bool
    ffmpeg_available: bool
    models: dict[str, bool]


class ModelInfo(BaseModel):
    name: str
    display_name: str
    available: bool
    vram_gb: float
    paper: str


class PredictResponse(BaseModel):
    job_id: str
    status: str
    model: str
    output_video: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobResponse(BaseModel):
    job_id: str
    status: str
    model: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    output_video: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
