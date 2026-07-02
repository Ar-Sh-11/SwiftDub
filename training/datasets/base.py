"""Base class for lip-sync training datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Sample:
    video_path: Path
    audio_path: Path
    speaker_id: str = ""
    fps: float = 25.0
    duration_s: float = 0.0


class LipSyncDataset(Dataset, ABC):
    """Each sample yields (video_frames, mel_spectrogram, speaker_id)."""

    # Subclasses set these
    FRAME_SIZE: int = 96         # face crop resolution fed to model
    MEL_BINS: int = 80
    MEL_STEP: int = 16           # mel frames per video frame
    SAMPLE_RATE: int = 16000

    def __init__(self, root: Path, split: str = "train", max_samples: int | None = None):
        self.root = Path(root)
        self.split = split
        self._samples: list[Sample] = []
        self._load_index()
        if max_samples:
            self._samples = self._samples[:max_samples]

    @abstractmethod
    def _load_index(self) -> None:
        """Populate self._samples with Sample objects."""

    def __len__(self) -> int:
        return len(self._samples)

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return {'frames': Tensor[T,3,H,W], 'mel': Tensor[1,80,T*16], 'gt': Tensor[T,3,H,W]}."""
