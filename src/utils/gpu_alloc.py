"""Dynamic VRAM-aware GPU allocator.

Design:
- Each GPU reserves 20% of total VRAM as overhead/safety buffer.
- Remaining 80% is "usable". Incoming jobs claim an estimated VRAM slice.
- Multiple small jobs can run concurrently on the same GPU as long as their
  combined claim fits within the usable budget.
- acquire() blocks (via asyncio.Condition) until a GPU can accommodate the job.
- release() notifies all waiters so the next fitting job can proceed.
- CPU-only mode is fully supported — VRAM is simulated from config defaults.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from src.config import settings

if TYPE_CHECKING:
    pass

OVERHEAD_PCT = 0.20  # 20 % always reserved as safety buffer


@dataclass
class _GPUSlot:
    gpu_id: int
    # job_id → claimed_gb
    claims: dict[str, float] = field(default_factory=dict)

    # ── Hardware queries ──────────────────────────────────────────────────────

    @property
    def total_gb(self) -> float:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(self.gpu_id).total_memory / 1024**3
        except Exception:
            pass
        return getattr(settings, "gpu_vram_default_gb", 24.0)

    @property
    def actual_free_gb(self) -> float:
        """Physical free VRAM from the CUDA driver (or simulated)."""
        try:
            import torch
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info(self.gpu_id)
                return free / 1024**3
        except Exception:
            pass
        # CPU / no-CUDA mode: simulate from soft claims
        return max(0.0, self.total_gb - self.claimed_gb)

    @property
    def gpu_name(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(self.gpu_id).name
        except Exception:
            pass
        return "CPU (simulated)"

    # ── Budget accounting ────────────────────────────────────────────────────

    @property
    def overhead_gb(self) -> float:
        return self.total_gb * OVERHEAD_PCT

    @property
    def usable_gb(self) -> float:
        return self.total_gb * (1 - OVERHEAD_PCT)

    @property
    def claimed_gb(self) -> float:
        return sum(self.claims.values())

    @property
    def soft_available_gb(self) -> float:
        """Budget remaining after soft claims."""
        return max(0.0, self.usable_gb - self.claimed_gb)

    @property
    def hard_available_gb(self) -> float:
        """Physical free minus reserved overhead."""
        return max(0.0, self.actual_free_gb - self.overhead_gb)

    def can_fit(self, vram_gb: float) -> bool:
        return self.soft_available_gb >= vram_gb and self.hard_available_gb >= vram_gb

    @property
    def used_pct(self) -> float:
        total = self.total_gb
        return round((total - self.actual_free_gb) / total * 100.0, 1) if total > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "gpu_id": self.gpu_id,
            "name": self.gpu_name,
            "total_gb": round(self.total_gb, 2),
            "overhead_gb": round(self.overhead_gb, 2),
            "usable_gb": round(self.usable_gb, 2),
            "claimed_gb": round(self.claimed_gb, 2),
            "soft_available_gb": round(self.soft_available_gb, 2),
            "actual_free_gb": round(self.actual_free_gb, 2),
            "hard_available_gb": round(self.hard_available_gb, 2),
            "active_jobs": len(self.claims),
            "active_job_ids": list(self.claims.keys()),
            "used_pct": self.used_pct,
        }


class DynamicGPUAllocator:
    """Thread-safe asyncio-native allocator for distributing inference jobs."""

    def __init__(self) -> None:
        gpu_ids = settings.gpu_id_list()
        self._slots: dict[int, _GPUSlot] = {g: _GPUSlot(g) for g in gpu_ids}
        # Lazily created on first async call so we bind to the running event loop.
        self._condition: asyncio.Condition | None = None

    def _get_condition(self) -> asyncio.Condition:
        if self._condition is None:
            self._condition = asyncio.Condition()
        return self._condition

    def _best_gpu(self, vram_gb: float) -> int | None:
        """Return the GPU with the most soft_available_gb that can fit the job."""
        best_gpu: int | None = None
        best_free = -1.0
        for slot in self._slots.values():
            if slot.can_fit(vram_gb):
                avail = slot.soft_available_gb
                if avail > best_free:
                    best_free = avail
                    best_gpu = slot.gpu_id
        return best_gpu

    async def acquire(self, job_id: str, vram_gb: float) -> int:
        """Block until a GPU can fit the job, then claim the VRAM and return gpu_id."""
        cond = self._get_condition()
        async with cond:
            while True:
                gpu_id = self._best_gpu(vram_gb)
                if gpu_id is not None:
                    self._slots[gpu_id].claims[job_id] = vram_gb
                    logger.debug(
                        "[{}] GPU {} acquired ({:.1f} GB claimed, {:.1f} GB soft-avail remaining)",
                        job_id, gpu_id, vram_gb, self._slots[gpu_id].soft_available_gb,
                    )
                    return gpu_id
                logger.debug(
                    "[{}] No GPU has {:.1f} GB free — waiting (pool: {})",
                    job_id, vram_gb,
                    {g: round(s.soft_available_gb, 1) for g, s in self._slots.items()},
                )
                await cond.wait()

    async def release(self, gpu_id: int, job_id: str) -> None:
        """Free the claimed VRAM and notify waiters."""
        cond = self._get_condition()
        async with cond:
            slot = self._slots.get(gpu_id)
            if slot:
                freed = slot.claims.pop(job_id, 0.0)
                logger.debug(
                    "[{}] GPU {} released {:.1f} GB — {:.1f} GB soft-avail",
                    job_id, gpu_id, freed, slot.soft_available_gb,
                )
            cond.notify_all()

    def status(self) -> list[dict]:
        """Return per-GPU metrics (no lock needed — dict reads are atomic in CPython)."""
        return [slot.as_dict() for slot in self._slots.values()]

    def vram_for_model(self, model: str) -> float:
        """Return the configured VRAM estimate for a given model."""
        if model == "musetalk":
            return float(getattr(settings, "musetalk_vram_gb", 7.0))
        return float(getattr(settings, "latentsync_vram_gb", 10.0))


# Module-level singleton — shared across all async handlers in the same process.
_allocator: DynamicGPUAllocator | None = None


def get_allocator() -> DynamicGPUAllocator:
    global _allocator
    if _allocator is None:
        _allocator = DynamicGPUAllocator()
    return _allocator


# Legacy shim so older callers that instantiate GPUAllocator() still compile.
GPUAllocator = DynamicGPUAllocator
