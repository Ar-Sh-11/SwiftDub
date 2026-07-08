"""Unit tests for face-detection edge cases (no GPU)."""

from __future__ import annotations

import numpy as np
import pytest
import torch


class TestFaceCarryForward:
    def test_affine_transform_video_reuses_last_face(self):
        from src.models.latentsync.pipeline import LipsyncPipeline

        class FakeProcessor:
            def __init__(self):
                self.calls = 0

            def affine_transform(self, frame):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("No face detected in frame")
                face = torch.zeros(3, 64, 64)
                return face, [0, 0, 64, 64], np.eye(2, 3)

        pipe = LipsyncPipeline.__new__(LipsyncPipeline)
        pipe.image_processor = FakeProcessor()
        frames = np.zeros((3, 64, 64, 3), dtype=np.uint8)
        faces, boxes, matrices = pipe._affine_transform_video(frames)
        assert faces.shape[0] == 3
        assert len(boxes) == 3
        assert len(matrices) == 3

    def test_probe_faces_counts_hits(self, monkeypatch):
        from src.models.latentsync.utils.image_proc import ImageProcessor

        class FakeDet:
            def __call__(self, image):
                return ([0, 0, 10, 10], np.zeros((106, 2))) if image[0, 0, 0] > 0 else (None, None)

        proc = ImageProcessor.__new__(ImageProcessor)
        proc.face_detector = FakeDet()
        frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
        frames[0, 0, 0, 0] = 1
        hits, sampled = proc.probe_faces(frames, sample=4)
        assert sampled == 4
        assert hits == 1
