"""Abstract lip-sync backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class InferenceResult:
    output_video: Path
    model: str
    elapsed_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


class LipSyncBackend(ABC):
    name: str

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True when all model weights are present."""

    @abstractmethod
    def infer(
        self,
        video: Path,
        audio: Path,
        output: Path,
        face_box: tuple[int, int, int, int] | None = None,
    ) -> InferenceResult:
        """Run lip-sync and write output video. face_box = (x1,y1,x2,y2)."""
