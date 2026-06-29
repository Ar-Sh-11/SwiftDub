"""Lip-sync backend registry."""

from __future__ import annotations

from src.config import LipSyncModel
from src.services.lipsync.base import LipSyncBackend
from src.services.lipsync.latentsync import LatentSyncBackend
from src.services.lipsync.musetalk import MuseTalkBackend
from src.services.lipsync.wav2lip import Wav2LipBackend

BACKENDS: dict[LipSyncModel, LipSyncBackend] = {
    LipSyncModel.WAV2LIP: Wav2LipBackend(),
    LipSyncModel.MUSETALK: MuseTalkBackend(),
    LipSyncModel.LATENTSYNC: LatentSyncBackend(),
}


def get_backend(model: LipSyncModel) -> LipSyncBackend:
    return BACKENDS[model]
