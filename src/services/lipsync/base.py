"""Base lip-sync backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LipSyncResult:
    output_video: Path
    model: str
    metadata: dict[str, str | int | float]


class LipSyncBackend(ABC):
    name: str

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        *,
        face_box: tuple[int, int, int, int] | None = None,
    ) -> LipSyncResult:
        raise NotImplementedError
