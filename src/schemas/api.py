"""API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.config import LipSyncModel


class HealthResponse(BaseModel):
    status: str = "healthy"
    cuda_available: bool
    ffmpeg_available: bool
    models: dict[str, bool]


class PredictResponse(BaseModel):
    job_id: str
    status: str
    model: str
    output_video: str | None = None
    face_box: list[int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    model: str
    output_video: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    name: LipSyncModel
    available: bool
    description: str
