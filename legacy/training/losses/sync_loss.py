"""SyncNet-based lip-sync contrastive loss.

SyncNet is trained to distinguish in-sync vs out-of-sync audio-visual pairs.
We use its visual/audio encoders frozen to compute a contrastive sync loss
during fine-tuning — the same strategy used in Wav2Lip.

Reference: "Out of Time: Automated Lip Sync in the Wild" (Chung & Zisserman 2016)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SyncNetLoss(nn.Module):
    """Contrastive loss against a frozen SyncNet."""

    def __init__(self, syncnet: nn.Module, lipsync_T: int = 5):
        super().__init__()
        self.syncnet = syncnet
        self.T = lipsync_T
        # Freeze SyncNet
        for p in self.syncnet.parameters():
            p.requires_grad_(False)

    def forward(self, pred_frames: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_frames: [B, T, C, H, W]  predicted lip frames (lower half)
            mel:         [B, 1, 80, T*16]  mel spectrogram window
        Returns:
            scalar sync loss
        """
        B, T, C, H, W = pred_frames.shape
        # SyncNet expects sequences of T=5 frames × 6 channels (stacked)
        face_seq = pred_frames[:, :self.T].reshape(B, self.T * C, H, W)
        audio_seq = mel[:, :, :, :self.T * 16]

        with torch.no_grad():
            v_emb = self.syncnet.forward_video(face_seq)
            a_emb = self.syncnet.forward_audio(audio_seq)

        # Cosine distance — in-sync pairs should be close
        v_norm = F.normalize(v_emb, dim=-1)
        a_norm = F.normalize(a_emb, dim=-1)
        loss = 1.0 - (v_norm * a_norm).sum(dim=-1).mean()
        return loss
