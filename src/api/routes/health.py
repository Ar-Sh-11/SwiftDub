"""Health, readiness, and model listing endpoints."""

from __future__ import annotations

import torch
from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.schemas import HealthResponse, ModelInfo
from src.config import settings
from src.core.models.registry import status as model_status
from src.metrics.registry import REGISTRY
from src.utils.ffmpeg import ffmpeg_available

router = APIRouter(tags=["observability"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        cuda_available=torch.cuda.is_available(),
        ffmpeg_available=ffmpeg_available(),
        models=model_status(),
    )


@router.get("/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    cfgs = settings.model_configs
    ready = model_status()
    return [
        ModelInfo(
            name=name,
            display_name=cfg.get("display_name", name),
            available=ready.get(name, False),
            vram_gb=cfg.get("vram_gb", 0),
            paper=cfg.get("paper", ""),
        )
        for name, cfg in cfgs.items()
    ]


@router.get("/metrics")
def metrics() -> Response:
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
