"""Face detection and left-most frontal face selection using OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Face:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    def padded(self, frame_h: int, frame_w: int, pad: float = 0.25) -> "Face":
        px, py = int(self.w * pad), int(self.h * pad)
        return Face(
            max(0, self.x1 - px), max(0, self.y1 - py),
            min(frame_w, self.x2 + px), min(frame_h, self.y2 + py),
        )

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


class FaceSelector:
    """Detect faces and return the left-most fully visible frontal face."""

    _MARGIN = 10   # edge clearance in pixels

    def __init__(self) -> None:
        cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._det = cv2.CascadeClassifier(str(cascade))
        assert not self._det.empty(), "OpenCV face cascade failed to load"

    def detect(self, frame: np.ndarray) -> list[Face]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self._det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        h, w = frame.shape[:2]
        faces = []
        for (x, y, fw, fh) in detections:
            x2, y2 = x + fw, y + fh
            if x > self._MARGIN and y > self._MARGIN and x2 < w - self._MARGIN and y2 < h - self._MARGIN:
                faces.append(Face(int(x), int(y), int(x2), int(y2)))
        return faces

    def select_leftmost(self, frames: list[np.ndarray]) -> Face:
        """Sample frames and return the consistently left-most face."""
        idxs = np.linspace(0, len(frames) - 1, min(15, len(frames)), dtype=int)
        candidates: list[Face] = []
        for i in idxs:
            faces = self.detect(frames[int(i)])
            if faces:
                candidates.append(min(faces, key=lambda f: f.cx))
        if not candidates:
            raise RuntimeError("No face detected in video. Ensure a forward-facing speaker is visible.")
        return min(candidates, key=lambda f: f.cx)

    def stable_box(self, frames: list[np.ndarray], ref: Face, pad: float = 0.25) -> tuple[int, int, int, int]:
        """Average face boxes across sampled frames for temporal stability."""
        h, w = frames[0].shape[:2]
        idxs = np.linspace(0, len(frames) - 1, min(20, len(frames)), dtype=int)
        boxes = []
        for i in idxs:
            faces = self.detect(frames[int(i)])
            if not faces:
                continue
            closest = min(faces, key=lambda f: abs(f.cx - ref.cx))
            p = closest.padded(h, w, pad=pad)
            boxes.append(p.as_tuple())
        if not boxes:
            return ref.padded(h, w, pad=pad).as_tuple()
        arr = np.array(boxes, dtype=np.float32).mean(axis=0).astype(int)
        return tuple(np.clip(arr, [0, 0, 0, 0], [w, h, w, h]).tolist())
