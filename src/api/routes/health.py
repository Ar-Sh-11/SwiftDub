"""Health, models list, and Prometheus metrics."""

from __future__ import annotations

import torch
from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.schemas import HealthResponse, ModelInfo
from src.config import settings
from src.core.registry import list_models, status
from src.metrics.registry import (
    REGISTRY,
    active_jobs,
    gpu_memory_pct_gauge,
    ram_memory_pct_gauge,
)
from src.utils.ffmpeg import ffmpeg_available
from src.utils.memory import memory_pressure, should_use_cache

router = APIRouter(tags=["observability"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    mem = memory_pressure()
    gpu_memory_pct_gauge.set(mem["gpu_pct"])
    ram_memory_pct_gauge.set(mem["ram_pct"])
    return HealthResponse(
        cuda_available=torch.cuda.is_available(),
        ffmpeg_available=ffmpeg_available(),
        models=status(),
        max_concurrent_jobs=settings.max_concurrent_jobs,
        active_jobs=int(active_jobs._value.get()),
        memory=mem,
        cache_enabled=should_use_cache(),
    )


@router.get("/models", response_model=list[ModelInfo])
def models() -> list[ModelInfo]:
    return [ModelInfo(**m) for m in list_models()]


@router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    mem = memory_pressure()
    gpu_memory_pct_gauge.set(mem["gpu_pct"])
    ram_memory_pct_gauge.set(mem["ram_pct"])
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
