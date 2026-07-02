"""Structured JSONL inference logger.

Writes one JSON line per inference event to logs/inference.jsonl.
Each line is a complete record consumable by Grafana Loki, Splunk, or grep.
Loguru is configured to also write human-readable logs to logs/app.log.
"""

from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import settings

_lock = threading.Lock()
_log_file: Path | None = None


def _inference_log_path() -> Path:
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    return settings.logs_dir / "inference.jsonl"


def setup_logging() -> None:
    """Configure loguru sinks: stderr + rotating app.log."""
    settings.logs_dir.mkdir(parents=True, exist_ok=True)

    # Remove default loguru sink
    logger.remove()

    # Pretty stderr
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )

    # Rotating app log (10 MB per file, keep 7 days)
    logger.add(
        str(settings.logs_dir / "app.log"),
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{line} | {message}",
        enqueue=True,
    )

    logger.info("Logging configured → {}", settings.logs_dir)


def log_inference_event(
    *,
    job_id: str,
    status: str,
    elapsed_s: float,
    wall_s: float,
    video_in: str,
    audio_in: str,
    output: str | None,
    steps: int = 20,
    guidance: float = 1.5,
    error: str | None = None,
    **extra: Any,
) -> None:
    """Append one JSONL record to logs/inference.jsonl."""
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "model": "latentsync",
        "status": status,
        "inference_s": round(elapsed_s, 3),
        "wall_s": round(wall_s, 3),
        "steps": steps,
        "guidance_scale": guidance,
        "video_in": video_in,
        "audio_in": audio_in,
        "output": output,
    }
    if error:
        record["error"] = error[:500]
    record.update(extra)

    line = json.dumps(record, ensure_ascii=False)
    path = _inference_log_path()

    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    logger.bind(job_id=job_id).info(
        "inference_event status={} elapsed={:.1f}s", status, elapsed_s
    )
