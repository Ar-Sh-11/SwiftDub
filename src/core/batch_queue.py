"""Sequential batch job dispatcher — respects GPU semaphore and memory headroom."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger

from src.config import settings
from src.utils.memory import gpu_memory_pct, gpu_free_gb


@dataclass
class QueuedJob:
    job_id: str
    model: str
    runner: Callable[[], Awaitable[Any]]


_queue: asyncio.Queue[QueuedJob] | None = None
_workers: list[asyncio.Task] = []
_started = False


def _effective_workers() -> int:
    """Scale worker count down when VRAM is tight."""
    free = gpu_free_gb()
    if free < 6:
        return 1
    if free < 12:
        return min(1, settings.max_concurrent_jobs)
    return settings.max_concurrent_jobs


async def _wait_for_headroom(model: str) -> None:
    """Block until enough GPU memory is free to start another job."""
    need_gb = 10.0 if model == "latentsync" else 7.0
    for attempt in range(120):
        free = gpu_free_gb()
        used = gpu_memory_pct()
        if free >= need_gb or used < 70:
            return
        logger.info(
            "GPU headroom wait ({:.1f} GB free, {:.0f}% used) — retry {}/120",
            free, used, attempt + 1,
        )
        await asyncio.sleep(5)
    logger.warning("GPU headroom wait timed out — starting job anyway")


async def _worker(worker_id: int) -> None:
    assert _queue is not None
    while True:
        job = await _queue.get()
        try:
            await _wait_for_headroom(job.model)
            await job.runner()
        except Exception as exc:
            logger.error("[{}] Batch queue worker {} failed: {}", job.job_id, worker_id, exc)
        finally:
            _queue.task_done()


def ensure_batch_workers() -> None:
    global _queue, _started, _workers
    if _started:
        return
    _queue = asyncio.Queue()
    n = _effective_workers()
    _workers = [asyncio.create_task(_worker(i)) for i in range(n)]
    _started = True
    logger.info("Batch queue started with {} worker(s)", n)


async def enqueue_batch_job(job_id: str, model: str, runner: Callable[[], Awaitable[Any]]) -> None:
    ensure_batch_workers()
    assert _queue is not None
    await _queue.put(QueuedJob(job_id=job_id, model=model, runner=runner))
