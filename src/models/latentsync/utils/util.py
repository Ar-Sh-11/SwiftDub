"""Minimal LatentSync util helpers for inference."""
from __future__ import annotations

import torch.distributed as dist


def zero_rank_log(logger, message: str) -> None:
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(message)
