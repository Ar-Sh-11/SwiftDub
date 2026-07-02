"""Health, readiness, and Prometheus metrics endpoints."""

from __future__ import annotations

import torch
from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.schemas import HealthResponse
from src.config import settings
from src.core.latentsync import is_ready
from src.metrics.registry import REGISTRY, active_jobs
from src.utils.ffmpeg import ffmpeg_available

router = APIRouter(tags=["observability"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        cuda_available=torch.cuda.is_available(),
        ffmpeg_available=ffmpeg_available(),
        model_ready=is_ready(),
        max_concurrent_jobs=settings.max_concurrent_jobs,
        active_jobs=int(active_jobs._value.get()),
    )


@router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
