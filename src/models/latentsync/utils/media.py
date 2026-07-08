"""Media I/O helpers for LatentSync inference.

Adapted from ByteDance/LatentSync (Apache 2.0).
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import imageio
import numpy as np
import torch
from decord import AudioReader, VideoReader
from loguru import logger


def read_video(video_path: str, change_fps: bool = True, use_decord: bool = True) -> np.ndarray:
    """Read video frames; optionally re-encode to 25 FPS first."""
    if change_fps:
        temp_dir = Path(video_path).parent / "_tmp_fps"
        temp_dir.mkdir(parents=True, exist_ok=True)
        out = temp_dir / "video.mp4"
        cmd = (
            f"ffmpeg -loglevel error -y -nostdin -i {shlex.quote(video_path)} "
            f"-r 25 -crf 18 {shlex.quote(str(out))}"
        )
        subprocess.run(cmd, shell=True, check=False)
        target = str(out)
    else:
        target = video_path

    if use_decord:
        frames = _read_decord(target)
    else:
        frames = _read_cv2(target)

    if change_fps and temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    return frames


def _read_decord(path: str) -> np.ndarray:
    vr = VideoReader(path)
    frames = vr[:].asnumpy()
    vr.seek(0)
    return frames


def _read_cv2(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return np.array([])
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames)


def read_audio(audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
    ar = AudioReader(audio_path, sample_rate=sample_rate, mono=True)
    samples = torch.from_numpy(ar[:].asnumpy()).squeeze(0)
    return samples


def write_video(output_path: str, frames: np.ndarray, fps: int = 25) -> None:
    with imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=["-crf", "13"],
        ffmpeg_log_level="error",
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def mux_audio_video(video_path: str, audio_path: str, output_path: str) -> None:
    cmd = (
        f"ffmpeg -y -loglevel error -nostdin "
        f"-i {shlex.quote(video_path)} "
        f"-i {shlex.quote(audio_path)} "
        f"-c:v libx264 -crf 18 -c:a aac -q:v 0 -q:a 0 "
        f"{shlex.quote(output_path)}"
    )
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed for {output_path}")
