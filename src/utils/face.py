"""Face detection and left-most speaker selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FaceDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    is_frontal: bool

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def padded(self, frame_shape: tuple[int, int, int], pad: float = 0.2) -> tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        pad_x = int(self.width * pad)
        pad_y = int(self.height * pad)
        x1 = max(0, self.x1 - pad_x)
        y1 = max(0, self.y1 - pad_y)
        x2 = min(w, self.x2 + pad_x)
        y2 = min(h, self.y2 + pad_y)
        return x1, y1, x2, y2


class FaceSelector:
    """Detect faces and pick the left-most fully visible frontal face."""

    def __init__(self) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(str(cascade_path))
        if self._detector.empty():
            raise RuntimeError("OpenCV face cascade failed to load")

    def detect_faces(self, frame: np.ndarray) -> list[FaceDetection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        detections = self._detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
        )

        faces: list[FaceDetection] = []
        for x, y, width, height in detections:
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + width), int(y + height)
            score = min(1.0, (width * height) / max(w * h, 1) * 8.0)
            is_frontal = self._is_fully_visible(x1, y1, x2, y2, w, h, score)
            faces.append(
                FaceDetection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=score,
                    is_frontal=is_frontal,
                )
            )
        return faces

    @staticmethod
    def _is_fully_visible(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        frame_w: int,
        frame_h: int,
        score: float,
        edge_margin: int = 8,
    ) -> bool:
        if score < 0.05:
            return False
        if x1 <= edge_margin or y1 <= edge_margin:
            return False
        if x2 >= frame_w - edge_margin or y2 >= frame_h - edge_margin:
            return False
        face_w = x2 - x1
        face_h = y2 - y1
        if face_w < 48 or face_h < 48:
            return False
        aspect = face_w / max(face_h, 1)
        return 0.55 <= aspect <= 1.45

    def select_leftmost_face(self, frames: list[np.ndarray]) -> FaceDetection:
        if not frames:
            raise ValueError("No frames provided for face selection")

        sample_indices = np.linspace(0, len(frames) - 1, num=min(12, len(frames)), dtype=int)
        candidates: list[FaceDetection] = []

        for idx in sample_indices:
            faces = self.detect_faces(frames[int(idx)])
            visible = [f for f in faces if f.is_frontal]
            if visible:
                candidates.append(min(visible, key=lambda f: f.center_x))

        if not candidates:
            for idx in sample_indices:
                faces = self.detect_faces(frames[int(idx)])
                if faces:
                    candidates.append(min(faces, key=lambda f: f.center_x))

        if not candidates:
            raise RuntimeError(
                "No face detected in video. Ensure a visible speaker is present."
            )

        return min(candidates, key=lambda f: f.center_x)

    def stable_box(
        self,
        frames: list[np.ndarray],
        base_face: FaceDetection,
        pad: float = 0.25,
    ) -> tuple[int, int, int, int]:
        boxes = []
        for frame in frames[:: max(1, len(frames) // 20)]:
            faces = self.detect_faces(frame)
            if not faces:
                continue
            closest = min(
                faces,
                key=lambda f: abs(f.center_x - base_face.center_x) + abs(f.area - base_face.area) * 0.001,
            )
            boxes.append(closest.padded(frame.shape, pad=pad))

        if not boxes:
            return base_face.padded(frames[0].shape, pad=pad)

        arr = np.array(boxes, dtype=np.float32)
        x1, y1, x2, y2 = arr.mean(axis=0).astype(int)
        h, w = frames[0].shape[:2]
        return max(0, x1), max(0, y1), min(w, x2), min(h, y2)
