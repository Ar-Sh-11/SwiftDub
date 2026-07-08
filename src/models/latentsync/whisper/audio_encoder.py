"""Whisper-based audio-to-feature encoder for LatentSync.

Adapted from ByteDance/LatentSync (Apache 2.0), which was itself adapted from
TMElyralab/MuseTalk.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


def _load_whisper(model_path: str, device: str | None = None):
    """Load the Whisper model from a .pt checkpoint using the bundled loader."""
    from src.models.latentsync.whisper.whisper import load_model

    return load_model(model_path, device=device)


class Audio2Feature:
    """Convert an audio file into per-frame Whisper feature tensors."""

    def __init__(
        self,
        model_path: str = "checkpoints/whisper/tiny.pt",
        device: str | None = None,
        num_frames: int = 16,
        audio_feat_length: list[int] | None = None,
    ) -> None:
        self.model = _load_whisper(model_path, device)
        self.num_frames = num_frames
        self.audio_feat_length = audio_feat_length or [2, 2]
        self.embedding_dim = self.model.dims.n_audio_state

    # ── public API ────────────────────────────────────────────────────────────

    def audio2feat(self, audio_path: str) -> torch.Tensor:
        result = self.model.transcribe(audio_path)
        embeds = []
        for seg in result["segments"]:
            enc = seg["encoder_embeddings"].transpose(0, 2, 1, 3).squeeze(0)
            end_idx = int((seg["end"] - seg["start"]) / 2)
            embeds.append(enc[:end_idx])
        return torch.from_numpy(np.concatenate(embeds, axis=0))

    def feature2chunks(self, feature_array: torch.Tensor, fps: float) -> list[torch.Tensor]:
        chunks: list[torch.Tensor] = []
        multiplier = 50.0 / fps
        i = 0
        while True:
            start_idx = int(i * multiplier)
            chunk, _ = self._get_sliced(feature_array, i, fps)
            chunks.append(chunk)
            i += 1
            if start_idx > len(feature_array):
                break
        return chunks

    def crop_overlap_window(self, feat: torch.Tensor, start: int) -> torch.Tensor:
        slices = [self._get_sliced(feat, start + i, fps=25)[0] for i in range(self.num_frames)]
        return torch.stack(slices)

    # ── internal ──────────────────────────────────────────────────────────────

    def _get_sliced(self, feat: torch.Tensor, vid_idx: int, fps: float = 25):
        length = len(feat)
        center = int(vid_idx * 50 / fps)
        left = center - self.audio_feat_length[0] * 2
        right = center + (self.audio_feat_length[1] + 1) * 2
        selected = []
        indices = []
        for idx in range(left, right):
            idx = max(0, min(length - 1, idx))
            selected.append(feat[idx])
            indices.append(idx)
        selected_t = torch.cat(selected, dim=0).reshape(-1, self.embedding_dim)
        return selected_t, indices
