"""Perceptual (VGG) + L1 reconstruction loss for lip-sync fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PerceptualLoss(nn.Module):
    """VGG-16 feature-space loss — penalises structural distortions without pixel rigidity."""

    LAYERS = {"3": 1.0, "8": 1.0, "15": 1.0}   # relu1_2, relu2_2, relu3_3 weights

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.slices = nn.ModuleDict({k: nn.Sequential(*list(vgg.children())[:int(k)+1])
                                      for k in self.LAYERS})
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.l1_loss(pred, target)               # pixel L1
        for k, w in self.LAYERS.items():
            loss = loss + w * F.l1_loss(self.slices[k](pred), self.slices[k](target))
        return loss


class CombinedLoss(nn.Module):
    """Weighted sum of perceptual + sync loss used during Wav2Lip fine-tuning."""

    def __init__(self, syncnet: nn.Module | None = None,
                 w_recon: float = 1.0, w_sync: float = 0.03, w_percep: float = 0.05):
        super().__init__()
        self.recon   = nn.L1Loss()
        self.percep  = PerceptualLoss()
        self.w_recon = w_recon
        self.w_percep = w_percep
        self.w_sync = w_sync
        if syncnet is not None:
            from training.losses.sync_loss import SyncNetLoss
            self.sync_loss: nn.Module | None = SyncNetLoss(syncnet)
        else:
            self.sync_loss = None

    def forward(self, pred, target, mel=None) -> dict[str, torch.Tensor]:
        losses = {
            "recon":  self.w_recon  * self.recon(pred, target),
            "percep": self.w_percep * self.percep(pred, target),
        }
        if self.sync_loss is not None and mel is not None:
            losses["sync"] = self.w_sync * self.sync_loss(pred, mel)
        losses["total"] = sum(losses.values())
        return losses
