"""Model registry — LatentSync and MuseTalk."""

from __future__ import annotations

from src.config import ModelName, settings
from src.core import latentsync, musetalk


def is_disabled(model: str) -> bool:
    if model == ModelName.MUSETALK.value and settings.is_musetalk_disabled():
        return True
    return False


def status() -> dict[str, bool]:
    models = {
        ModelName.LATENTSYNC.value: latentsync.is_ready(),
    }
    if not settings.is_musetalk_disabled():
        models[ModelName.MUSETALK.value] = musetalk.is_ready()
    return models


def is_ready(model: str) -> bool:
    if is_disabled(model):
        return False
    if model == ModelName.LATENTSYNC.value:
        return latentsync.is_ready()
    if model == ModelName.MUSETALK.value:
        return musetalk.is_ready()
    return False


def assert_model_available(model: str) -> None:
    """Raise ValueError with a user-facing message if the model cannot be used."""
    if model == ModelName.MUSETALK.value and settings.is_musetalk_disabled():
        raise ValueError(
            "MuseTalk is disabled on this deployment. "
            "Set MUSETALK_DISABLED=false and musetalk.disabled=false in configs/models.yaml to enable."
        )
    if model not in {m.value for m in ModelName}:
        raise ValueError(f"Unknown model '{model}'. Choose: latentsync")
    if not is_ready(model):
        raise ValueError(
            f"Model '{model}' weights not found. Run: python scripts/download/models.py --model {model}"
        )


def list_models() -> list[dict]:
    cfgs = settings.model_configs
    ready = status()
    out: list[dict] = []
    for name, cfg in cfgs.items():
        if name == ModelName.MUSETALK.value and settings.is_musetalk_disabled():
            continue
        out.append({
            "name": name,
            "display_name": cfg.get("display_name", name),
            "available": ready.get(name, False),
            "vram_gb": cfg.get("vram_gb", 0),
            "paper": cfg.get("paper", ""),
        })
    return out
