"""Whisper-based audio feature extraction for MuseTalk v1.5.

Adapted from TMElyralab/MuseTalk (Apache 2.0).
"""
from __future__ import annotations

import math

import librosa
import numpy as np
import torch
from einops import rearrange
from transformers import AutoFeatureExtractor


class AudioProcessor:
    """Extract Whisper encoder features aligned to video frame rate."""

    def __init__(self, feature_extractor_path: str = "openai/whisper-tiny") -> None:
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_path)

    def get_audio_feature(self, wav_path: str, weight_dtype=None):
        """Split audio into 30-second segments and extract mel features."""
        audio, sr = librosa.load(wav_path, sr=16000)
        assert sr == 16000, f"Expected 16 kHz, got {sr} Hz"

        seg_len = 30 * sr
        features = []
        for start in range(0, len(audio), seg_len):
            seg = audio[start : start + seg_len]
            feat = self.feature_extractor(seg, return_tensors="pt", sampling_rate=sr).input_features
            if weight_dtype is not None:
                feat = feat.to(dtype=weight_dtype)
            features.append(feat)

        return features, len(audio)

    def get_whisper_chunk(
        self,
        whisper_input_features,
        device: torch.device,
        weight_dtype,
        whisper_model,
        librosa_length: int,
        fps: int = 25,
        audio_padding_length_left: int = 2,
        audio_padding_length_right: int = 2,
    ) -> torch.Tensor:
        """Encode mel features through Whisper encoder and align to video fps."""
        feat_per_frame = 2 * (audio_padding_length_left + audio_padding_length_right + 1)
        whisper_feats = []
        for inp in whisper_input_features:
            inp = inp.to(device).to(weight_dtype)
            hidden = whisper_model.encoder(inp, output_hidden_states=True).hidden_states
            whisper_feats.append(torch.stack(hidden, dim=2))
        whisper_feat = torch.cat(whisper_feats, dim=1)

        sr, audio_fps = 16000, 50
        fps = int(fps)
        multiplier = audio_fps / fps
        num_frames = math.floor((librosa_length / sr) * fps)
        actual_len = math.floor((librosa_length / sr) * audio_fps)
        whisper_feat = whisper_feat[:, :actual_len]

        pad_n = math.ceil(multiplier)
        whisper_feat = torch.cat(
            [
                torch.zeros_like(whisper_feat[:, : pad_n * audio_padding_length_left]),
                whisper_feat,
                torch.zeros_like(whisper_feat[:, : pad_n * 3 * audio_padding_length_right]),
            ],
            dim=1,
        )

        prompts = []
        for fi in range(num_frames):
            ai = math.floor(fi * multiplier)
            clip = whisper_feat[:, ai : ai + feat_per_frame]
            prompts.append(clip)
        prompts_t = torch.cat(prompts, dim=0)
        return rearrange(prompts_t, "b c h w -> b (c h) w")
