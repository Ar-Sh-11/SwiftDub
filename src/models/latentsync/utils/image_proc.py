"""Image / face processing utilities for LatentSync inference.

Adapted from ByteDance/LatentSync (Apache 2.0).
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from einops import rearrange
from torchvision import transforms

from .affine import AlignRestore

# Default mask path relative to this file
_DEFAULT_MASK = Path(__file__).resolve().parent.parent / "assets" / "mask.png"


def load_fixed_mask(resolution: int, mask_image_path: str | None = None) -> torch.Tensor:
    path = mask_image_path or str(_DEFAULT_MASK)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Mask image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4) / 255.0
    mask = rearrange(torch.from_numpy(img), "h w c -> c h w")
    return mask.float()


class FaceDetector:
    """InsightFace-based face + landmark detector (ONNX buffalo_l)."""

    def __init__(self, device: str = "cuda", det_size: tuple = (512, 512)) -> None:
        import insightface
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider"] if "cuda" in device else ["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0 if "cuda" in device else -1, det_size=det_size)

    def __call__(self, image: np.ndarray):
        """Return (bbox, landmark_2d_106) or (None, None) if no face found."""
        faces = self.app.get(image)
        if not faces:
            return None, None
        face = faces[0]
        bbox = face.bbox  # [x1, y1, x2, y2]
        kps = face.landmark_2d_106  # shape (106, 2)
        return bbox, kps


class ImageProcessor:
    """Per-frame face alignment and mask preparation for LatentSync."""

    def __init__(
        self,
        resolution: int = 512,
        device: str = "cpu",
        mask_image: torch.Tensor | None = None,
    ) -> None:
        self.resolution = resolution
        self.resize = transforms.Resize(
            (resolution, resolution),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)
        self.restorer = AlignRestore(resolution=resolution, device=device)

        if mask_image is None:
            self.mask_image = load_fixed_mask(resolution)
        else:
            self.mask_image = mask_image

        self.face_detector: FaceDetector | None = None
        if device != "cpu":
            self.face_detector = FaceDetector(device=device)

    def affine_transform(self, image: np.ndarray):
        if self.face_detector is None:
            raise NotImplementedError("Face detection requires a CUDA device")
        bbox, kps = self.face_detector(image)
        if bbox is None:
            raise RuntimeError("No face detected in frame")

        pt_left_eye = np.mean(kps[[43, 48, 49, 51, 50]], axis=0)
        pt_right_eye = np.mean(kps[101:106], axis=0)
        pt_nose = np.mean(kps[[74, 77, 83, 86]], axis=0)
        landmarks3 = np.round([pt_left_eye, pt_right_eye, pt_nose])

        face, affine_matrix = self.restorer.align_warp_face(
            image.copy(), landmarks3=landmarks3, smooth=True
        )
        box = [0, 0, face.shape[1], face.shape[0]]
        face = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
        face_t = rearrange(torch.from_numpy(face), "h w c -> c h w")
        return face_t, box, affine_matrix

    def preprocess_fixed_mask_image(
        self, image: torch.Tensor, affine_transform: bool = False
    ):
        if affine_transform:
            image, _, _ = self.affine_transform(image.numpy().transpose(1, 2, 0))
        else:
            image = self.resize(image)
        pixel_values = self.normalize(image / 255.0)
        masked_pixel_values = pixel_values * self.mask_image
        return pixel_values, masked_pixel_values, self.mask_image[0:1]

    def prepare_masks_and_masked_images(
        self,
        images: Union[torch.Tensor, np.ndarray],
        affine_transform: bool = False,
    ):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.ndim == 4 and images.shape[-1] == 3:
            images = rearrange(images, "f h w c -> f c h w")

        results = [
            self.preprocess_fixed_mask_image(img, affine_transform=affine_transform)
            for img in images
        ]
        pixel_values, masked_pixel_values, masks = zip(*results)
        return torch.stack(pixel_values), torch.stack(masked_pixel_values), torch.stack(masks)
