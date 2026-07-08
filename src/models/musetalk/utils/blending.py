"""Face blending helpers for MuseTalk v1.5.

Adapted from TMElyralab/MuseTalk (Apache 2.0).
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _get_crop_box(box: list[int], expand: float = 1.5):
    x, y, x1, y1 = box
    xc, yc = (x + x1) // 2, (y + y1) // 2
    s = int(max(x1 - x, y1 - y) // 2 * expand)
    return [xc - s, yc - s, xc + s, yc + s], s


def _face_seg(image: Image.Image, mode: str = "raw", fp=None) -> Image.Image | None:
    seg = fp(image, mode=mode)
    if seg is None:
        return None
    return seg.resize(image.size)


def composite_face(
    image: np.ndarray,
    face: np.ndarray,
    face_box: list[int],
    upper_boundary_ratio: float = 0.5,
    expand: float = 1.5,
    mode: str = "raw",
    fp=None,
) -> np.ndarray:
    """Paste lip-synced face back into original frame with smooth blending."""
    body = Image.fromarray(image[:, :, ::-1])
    face_pil = Image.fromarray(face[:, :, ::-1])

    x, y, x1, y1 = face_box
    crop_box, _ = _get_crop_box(face_box, expand)
    xs, ys, xe, ye = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = _face_seg(face_large, mode=mode, fp=fp)
    if mask_image is None:
        return image
    mask_small = mask_image.crop((x - xs, y - ys, x1 - xs, y1 - ys))
    mask_full = Image.new("L", ori_shape, 0)
    mask_full.paste(mask_small, (x - xs, y - ys, x1 - xs, y1 - ys))

    w, h = mask_full.size
    top_boundary = int(h * upper_boundary_ratio)
    modified_mask = Image.new("L", ori_shape, 0)
    modified_mask.paste(mask_full.crop((0, top_boundary, w, h)), (0, top_boundary))

    blur_k = int(0.05 * ori_shape[0] // 2 * 2) + 1
    mask_arr = cv2.GaussianBlur(np.array(modified_mask), (blur_k, blur_k), 0)
    mask_image = Image.fromarray(mask_arr)

    face_large.paste(face_pil, (x - xs, y - ys, x1 - xs, y1 - ys))
    body.paste(face_large, crop_box[:2], mask_image)
    return np.array(body)[:, :, ::-1]
