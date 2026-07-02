"""SwiftSync — Novel lip-sync architecture (custom, from-scratch).

Architecture:
  Audio path:  WavLM-base → linear projection → cross-attention queries
  Video path:  VideoMAE-v2 patch encoder → spatial tokens
  Decoder:     Cross-attention (audio queries × visual keys) → face-region
               MLP decoder → reconstructed lip patch
  Loss:        L1 + VGG perceptual + SyncNet contrastive + optical-flow smoothness

This is a research prototype designed to replicate and extend proprietary
lip-sync models (HeyGen / Sync.so style) using open-source components only.

Training:  Use LRS3 (438h) + VoxCeleb2 (2442h) for pre-training,
           then HDTF (16h) for domain fine-tuning.

References:
  - WavLM: https://arxiv.org/abs/2110.13900
  - VideoMAE-v2: https://arxiv.org/abs/2303.16727
  - SyncNet: https://arxiv.org/abs/1601.00825
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ── Audio Encoder ─────────────────────────────────────────────────────────────

class AudioEncoder(nn.Module):
    """WavLM-base encoder → fixed-dim projected features.

    Input:  raw waveform [B, T_audio]  (16 kHz)
    Output: audio tokens [B, T_frames, D]
    """

    def __init__(self, out_dim: int = 512, freeze_wavlm: bool = True):
        super().__init__()
        from transformers import WavLMModel
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")
        if freeze_wavlm:
            for p in self.wavlm.parameters():
                p.requires_grad_(False)
        wavlm_dim = self.wavlm.config.hidden_size  # 768
        self.proj = nn.Sequential(
            nn.Linear(wavlm_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # WavLM outputs 1 feature every ~20ms → ~50 features/s at 16 kHz
        hidden = self.wavlm(waveform).last_hidden_state  # [B, T_audio_tokens, 768]
        return self.proj(hidden)                          # [B, T_audio_tokens, D]


# ── Visual Encoder ────────────────────────────────────────────────────────────

class VisualEncoder(nn.Module):
    """Lightweight CNN video patch encoder (VideoMAE-v2 style patching).

    Input:  face crop sequence [B, T, 3, H, W]
    Output: visual tokens [B, T*P, D]  where P = spatial patches per frame
    """

    def __init__(self, patch_size: int = 16, in_dim: int = 3,
                 out_dim: int = 512, img_size: int = 256):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patchify = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, out_dim) * 0.02)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video.shape
        frames = rearrange(video, "b t c h w -> (b t) c h w")
        tokens = self.patchify(frames)                    # [B*T, D, Hp, Wp]
        tokens = rearrange(tokens, "bt d h w -> bt (h w) d")
        tokens = tokens + self.pos_embed
        tokens = self.norm(tokens)
        return rearrange(tokens, "(b t) p d -> b (t p) d", b=B)  # [B, T*P, D]


# ── Cross-Attention Lip Decoder ───────────────────────────────────────────────

class LipDecoder(nn.Module):
    """Audio-driven cross-attention decoder.

    Queries = audio features, Keys/Values = visual tokens
    Output  = reconstructed lip-region patches
    """

    def __init__(self, dim: int = 512, n_heads: int = 8,
                 n_layers: int = 4, patch_size: int = 16,
                 lip_patches: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=dim, nhead=n_heads,
                                       dim_feedforward=dim * 4,
                                       batch_first=True, dropout=0.1)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(dim, patch_size * patch_size * 3)
        self.lip_patches = lip_patches
        self.patch_size = patch_size

    def forward(self, audio_q: torch.Tensor, visual_kv: torch.Tensor) -> torch.Tensor:
        x = audio_q[:, :self.lip_patches]   # trim to lip-region queries
        for layer in self.layers:
            x = layer(x, visual_kv)
        patches = self.head(x)              # [B, P, p*p*3]
        return patches


# ── Full SwiftSync Model ──────────────────────────────────────────────────────

class SwiftSync(nn.Module):
    """Complete SwiftSync model.

    Input:
        waveform:  [B, T_audio]       16kHz mono audio
        video:     [B, T, 3, H, W]    face frames (H=W=256 recommended)
        mask:      [B, T, 3, H, W]    video with lower-half masked (training only)

    Output:
        reconstructed face sequence [B, T, 3, H, W]  (full face, lip region replaced)
    """

    IMG_SIZE = 256
    PATCH_SIZE = 16

    def __init__(self, dim: int = 512):
        super().__init__()
        self.audio_enc  = AudioEncoder(out_dim=dim)
        self.visual_enc = VisualEncoder(patch_size=self.PATCH_SIZE, out_dim=dim,
                                        img_size=self.IMG_SIZE)
        self.decoder    = LipDecoder(dim=dim, patch_size=self.PATCH_SIZE)
        n_patches_h = self.IMG_SIZE // self.PATCH_SIZE   # 16
        # Lip region = lower half = bottom 8 rows × 16 cols = 128 patches
        self.lip_patches = (n_patches_h // 2) * n_patches_h
        self.decoder.lip_patches = self.lip_patches

    def forward(self, waveform: torch.Tensor, video: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T = video.shape[:2]
        audio_tokens  = self.audio_enc(waveform)          # [B, T_a, D]
        visual_tokens = self.visual_enc(mask if mask is not None else video)  # [B, T*P, D]
        lip_patches   = self.decoder(audio_tokens, visual_tokens)             # [B, P_lip, p*p*3]

        # Reconstruct: paste predicted patches into the lower half of the input
        output = video.clone()
        p = self.PATCH_SIZE
        n_col = self.IMG_SIZE // p  # 16
        n_row_half = n_col // 2     # 8 (lower half rows)
        patch_idx = 0
        for row in range(n_row_half, n_col):
            for col in range(n_col):
                if patch_idx >= lip_patches.shape[1]:
                    break
                patch = lip_patches[:, patch_idx].reshape(B, 3, p, p)
                # Apply to all frames (broadcast first frame's audio-conditioned patches)
                for t in range(T):
                    output[:, t, :, row*p:(row+1)*p, col*p:(col+1)*p] = torch.sigmoid(patch)
                patch_idx += 1
        return output


if __name__ == "__main__":
    # Quick shape sanity check
    model = SwiftSync(dim=256)
    wav = torch.randn(2, 16000 * 3)    # 3s audio
    vid = torch.randn(2, 25, 3, 256, 256)
    out = model(wav, vid, mask=vid)
    print("Output shape:", out.shape)  # [2, 25, 3, 256, 256]
