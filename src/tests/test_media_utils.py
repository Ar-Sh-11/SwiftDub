"""Tests for media utilities — CPU-only, no GPU required."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest


class TestFFmpegUtils:
    def test_extract_audio_from_video(self, sample_video: Path, tmp_path):
        from src.utils.ffmpeg import extract_audio

        out = tmp_path / "extracted.wav"
        result = extract_audio(sample_video, out)
        assert result.exists(), "extract_audio should produce a WAV file"
        assert result.stat().st_size > 0

    def test_extract_audio_creates_parent_dirs(self, sample_video: Path, tmp_path):
        from src.utils.ffmpeg import extract_audio

        deep = tmp_path / "a" / "b" / "c" / "out.wav"
        extract_audio(sample_video, deep)
        assert deep.exists()

    def test_get_video_info(self, sample_video: Path):
        from src.utils.ffmpeg import get_video_info

        info = get_video_info(sample_video)
        assert "fps" in info
        assert info["fps"] > 0

    def test_ffmpeg_available(self):
        from src.utils.ffmpeg import ffmpeg_available

        # ffmpeg must be available for test environment
        assert ffmpeg_available(), "ffmpeg is required for tests"

    def test_mux_audio_video(self, sample_video: Path, sample_wav: Path, tmp_path):
        from src.utils.ffmpeg import mux_audio_video

        out = tmp_path / "muxed.mp4"
        mux_audio_video(sample_video, sample_wav, out)
        assert out.exists()
        assert out.stat().st_size > 0


class TestVideoUtils:
    def test_read_frames(self, sample_video: Path):
        from src.utils.video import read_frames

        frames, fps = read_frames(sample_video)
        assert len(frames) > 0
        assert fps > 0
        assert frames[0].ndim == 3

    def test_read_frames_max(self, sample_video: Path):
        from src.utils.video import read_frames

        frames, _ = read_frames(sample_video, max_frames=5)
        assert len(frames) <= 5

    def test_write_video(self, sample_video: Path, tmp_path):
        from src.utils.video import read_frames, write_video

        frames, fps = read_frames(sample_video, max_frames=10)
        out = tmp_path / "out.mp4"
        write_video(frames, out, fps)
        assert out.exists()

    def test_crop_frame(self):
        from src.utils.video import crop_frame

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[10:40, 20:60] = 128
        crop = crop_frame(frame, (20, 10, 60, 40))
        assert crop.shape == (30, 40, 3)
        assert np.all(crop == 128)


class TestAudioUtils:
    def test_load_mono(self, sample_wav: Path):
        from src.utils.audio import load_mono

        audio = load_mono(sample_wav)
        assert audio.ndim == 1
        assert len(audio) == 16000  # 1 second at 16 kHz

    def test_duration_seconds(self, sample_wav: Path):
        from src.utils.audio import duration_seconds

        dur = duration_seconds(sample_wav)
        assert 0.9 < dur < 1.1  # ~1 second

    def test_normalize_audio(self):
        from src.utils.audio import normalize_audio

        a = np.array([0.1, 0.5, -0.3, 0.8])
        n = normalize_audio(a, peak=0.95)
        assert abs(np.max(np.abs(n)) - 0.95) < 1e-5

    def test_save_wav(self, tmp_path):
        from src.utils.audio import save_wav, load_mono

        audio = np.sin(np.linspace(0, 2 * np.pi * 440, 16000)).astype(np.float32)
        out = tmp_path / "saved.wav"
        save_wav(out, audio)
        assert out.exists()
        reloaded = load_mono(out)
        assert len(reloaded) > 0
