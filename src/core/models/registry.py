"""Model registry — single source of truth for all backends."""

from __future__ import annotations

from src.config import ModelName
from src.core.models.base import LipSyncBackend
from src.core.models.latentsync import LatentSyncBackend
from src.core.models.musetalk import MuseTalkBackend
from src.core.models.sadtalker import SadTalkerBackend
from src.core.models.video_retalking import VideoRetalkingBackend
from src.core.models.wav2lip import Wav2LipBackend

# Instantiated once; stateless wrappers (no GPU tensors held here)
_REGISTRY: dict[ModelName, LipSyncBackend] = {
    ModelName.WAV2LIP:         Wav2LipBackend(),
    ModelName.VIDEO_RETALKING: VideoRetalkingBackend(),
    ModelName.MUSETALK:        MuseTalkBackend(),
    ModelName.LATENTSYNC:      LatentSyncBackend(),
    ModelName.SADTALKER:       SadTalkerBackend(),
}


def get_backend(name: ModelName) -> LipSyncBackend:
    return _REGISTRY[name]


def status() -> dict[str, bool]:
    return {name.value: backend.is_ready() for name, backend in _REGISTRY.items()}
