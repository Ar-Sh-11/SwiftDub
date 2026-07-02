"""Model registry — LatentSync only in production.
Legacy backends (Wav2Lip, VideoReTalking, MuseTalk, SadTalker) are in legacy/src_models/.
"""
from src.core.latentsync import is_ready


def status() -> dict[str, bool]:
    return {"latentsync": is_ready()}
