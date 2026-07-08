"""Shared pytest fixtures for SwiftDub tests."""
from __future__ import annotations

import os
import shutil
import struct
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: live server integration tests")
    config.addinivalue_line("markers", "gpu: requires GPU inference (set RUN_GPU_TESTS=1)")


# ── Disable external services for unit tests ─────────────────────────────────
os.environ.setdefault("DISABLE_DB", "true")
os.environ.setdefault("DISABLE_CACHE", "true")


@pytest.fixture(scope="session")
def tmp_data(tmp_path_factory) -> Path:
    """Session-scoped temp directory."""
    return tmp_path_factory.mktemp("swiftdub_tests")


@pytest.fixture
def sample_wav(tmp_path) -> Path:
    """Generate a 1-second 16 kHz mono WAV file."""
    path = tmp_path / "sample.wav"
    sample_rate = 16000
    num_samples = sample_rate  # 1 second
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        # 440 Hz sine
        t = np.linspace(0, 1, num_samples, endpoint=False)
        samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        wf.writeframes(samples.tobytes())
    return path


@pytest.fixture
def sample_video(tmp_path) -> Path:
    """Generate a minimal silent MP4 via ffmpeg (25 FPS, 1 second, 64×64)."""
    out = tmp_path / "sample.mp4"
    try:
        import subprocess, sys

        # Use a solid color lavfi source — works without GPU
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=blue:size=64x64:rate=25:duration=1",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-t", "1",
            "-c:v", "libx264", "-crf", "28",
            "-c:a", "aac", "-shortest",
            str(out),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0 or not out.exists():
            pytest.skip("ffmpeg not available for video generation")
    except FileNotFoundError:
        pytest.skip("ffmpeg not found")
    return out


@pytest.fixture
def sample_mp4_with_audio(tmp_path) -> Path:
    """An MP4 that has an audio track (used as 'video submitted as audio field')."""
    out = tmp_path / "video_with_audio.mp4"
    try:
        import subprocess

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=red:size=64x64:rate=25:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=2",
            "-t", "2",
            "-c:v", "libx264", "-crf", "28",
            "-c:a", "aac", "-shortest",
            str(out),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0 or not out.exists():
            pytest.skip("ffmpeg not available")
    except FileNotFoundError:
        pytest.skip("ffmpeg not found")
    return out
