"""Health and model info routes."""

from __future__ import annotations

import torch
from fastapi import APIRouter

from src.config import LipSyncModel, settings
from src.schemas.api import HealthResponse, ModelInfo
from src.services.lipsync import BACKENDS
from src.utils.ffmpeg import ffmpeg_available

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        cuda_available=torch.cuda.is_available(),
        ffmpeg_available=ffmpeg_available(),
        models={model.value: backend.is_available() for model, backend in BACKENDS.items()},
    )


@router.get("/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    descriptions = {
        LipSyncModel.WAV2LIP: "Classic GAN lip-sync; fast and lightweight.",
        LipSyncModel.MUSETALK: "Real-time latent inpainting lip-sync (v1.5).",
        LipSyncModel.LATENTSYNC: "Diffusion-based lip-sync with strong temporal consistency.",
    }
    return [
        ModelInfo(
            name=model,
            available=BACKENDS[model].is_available(),
            description=descriptions[model],
        )
        for model in LipSyncModel
    ]
