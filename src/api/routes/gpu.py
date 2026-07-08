"""GPU metrics endpoint — per-device VRAM breakdown and active job info."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import GPUDeviceStatus, GPUStatusResponse
from src.utils.gpu_alloc import OVERHEAD_PCT, get_allocator

router = APIRouter(prefix="/gpu", tags=["observability"])


@router.get("/status", response_model=GPUStatusResponse, summary="Per-GPU VRAM breakdown")
def gpu_status() -> GPUStatusResponse:
    alloc = get_allocator()
    devices = [GPUDeviceStatus(**d) for d in alloc.status()]
    total_active = sum(d.active_jobs for d in devices)
    return GPUStatusResponse(
        devices=devices,
        total_active_jobs=total_active,
        overhead_pct=round(OVERHEAD_PCT * 100),
    )
