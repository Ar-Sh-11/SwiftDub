"""Model registry — LatentSync and MuseTalk."""

from __future__ import annotations

from src.config import ModelName, settings
from src.core import latentsync, musetalk


def status() -> dict[str, bool]:
    return {
        ModelName.LATENTSYNC.value: latentsync.is_ready(),
        ModelName.MUSETALK.value: musetalk.is_ready(),
    }


def is_ready(model: str) -> bool:
    if model == ModelName.LATENTSYNC.value:
        return latentsync.is_ready()
    if model == ModelName.MUSETALK.value:
        return musetalk.is_ready()
    return False


def list_models() -> list[dict]:
    cfgs = settings.model_configs
    ready = status()
    return [
        {
            "name": name,
            "display_name": cfg.get("display_name", name),
            "available": ready.get(name, False),
            "vram_gb": cfg.get("vram_gb", 0),
            "paper": cfg.get("paper", ""),
        }
        for name, cfg in cfgs.items()
    ]
