"""Audio utilities."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_mono(path: Path, sample_rate: int = 16000) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return audio


def save_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)
    return path


def normalize_audio(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_val = np.max(np.abs(audio))
    if max_val < 1e-8:
        return audio
    return audio * (peak / max_val)


def duration_seconds(path: Path) -> float:
    audio = load_mono(path)
    return len(audio) / 16000.0
