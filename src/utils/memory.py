"""GPU and system memory helpers — used to gate caching under pressure."""

from __future__ import annotations

from loguru import logger

from src.config import settings


def gpu_memory_pct() -> float:
    """Return fraction of GPU memory in use (0–100)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        free, total = torch.cuda.mem_get_info()
        return (total - free) / total * 100.0
    except Exception:
        return 0.0


def gpu_free_gb() -> float:
    """Return free GPU memory in GB."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        free, _ = torch.cuda.mem_get_info()
        return free / (1024 ** 3)
    except Exception:
        return 0.0


def system_ram_pct() -> float:
    """Return fraction of system RAM in use (0–100)."""
    try:
        import psutil
        return psutil.virtual_memory().percent
    except Exception:
        return 0.0


def memory_pressure() -> dict[str, float]:
    gpu = gpu_memory_pct()
    ram = system_ram_pct()
    return {"gpu_pct": round(gpu, 1), "ram_pct": round(ram, 1)}


def should_use_cache() -> bool:
    """Skip Redis cache when GPU or RAM is above the configured threshold."""
    if settings.disable_cache:
        return False
    mem = memory_pressure()
    threshold = settings.memory_cache_disable_pct
    if mem["gpu_pct"] >= threshold or mem["ram_pct"] >= threshold:
        logger.warning(
            "Cache disabled — memory pressure GPU={:.0f}% RAM={:.0f}% (threshold {:.0f}%)",
            mem["gpu_pct"], mem["ram_pct"], threshold,
        )
        return False
    return True


def release_gpu_memory() -> None:
    """Clear CUDA cache when GPU usage is high."""
    try:
        import gc
        import torch
        if not torch.cuda.is_available():
            return
        if gpu_memory_pct() >= settings.gpu_clear_cache_pct:
            gc.collect()
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared (GPU {:.0f}% used)", gpu_memory_pct())
    except Exception:
        pass
