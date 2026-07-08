"""Health, models list, and Prometheus metrics."""

from __future__ import annotations

import torch
from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.schemas import HealthResponse, ModelInfo, ServiceInfo, ServicesResponse
from src.config import settings
from src.core.registry import list_models, status
from src.metrics.registry import (
    REGISTRY,
    active_jobs,
    gpu_memory_pct_gauge,
    ram_memory_pct_gauge,
)
from src.utils.ffmpeg import ffmpeg_available
from src.utils.memory import memory_pressure, should_use_cache, gpu_free_gb
from src.utils.services import check_port_open, list_service_urls

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
        musetalk_disabled=settings.is_musetalk_disabled(),
        max_concurrent_jobs=settings.max_concurrent_jobs,
        active_jobs=int(active_jobs._value.get()),
        memory={**mem, "gpu_free_gb": round(gpu_free_gb(), 2)},
        cache_enabled=should_use_cache(),
        gpu_pool=settings.gpu_id_list(),
    )


@router.get("/services", response_model=ServicesResponse, summary="Browser URLs for Grafana/Prometheus/Flower")
def services() -> ServicesResponse:
    raw = list_service_urls()
    hint = "Services run on localhost — open the links below directly in your browser."
    out: dict[str, ServiceInfo] = {}
    for name, meta in raw.items():
        out[name] = ServiceInfo(
            **meta,
            up=check_port_open(meta["port"]),
        )
    return ServicesResponse(hint=hint, services=out)


@router.get("/models", response_model=list[ModelInfo])
def models() -> list[ModelInfo]:
    return [ModelInfo(**m) for m in list_models()]


@router.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    mem = memory_pressure()
    gpu_memory_pct_gauge.set(mem["gpu_pct"])
    ram_memory_pct_gauge.set(mem["ram_pct"])
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
