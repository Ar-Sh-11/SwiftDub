"""Video frame utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_frames(video_path: Path, max_frames: int | None = None) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames, float(fps)


def write_video(
    frames: list[np.ndarray],
    output_path: Path,
    fps: float,
    *,
    codec: str = "mp4v",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames to write")

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    for frame in frames:
        writer.write(frame)
    writer.release()
    return output_path


def crop_frame(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return frame[y1:y2, x1:x2].copy()


def composite_face(
    frame: np.ndarray,
    face_patch: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = box
    resized = cv2.resize(face_patch, (x2 - x1, y2 - y1))
    output = frame.copy()
    output[y1:y2, x1:x2] = resized
    return output
