"""Face affine warp / restore utilities for LatentSync.

Adapted from ByteDance/LatentSync (Apache 2.0), originally from
https://github.com/guanjz20/StyleSync.
"""
from __future__ import annotations

import cv2
import kornia
import kornia.filters
import kornia.geometry.transform
import kornia.morphology
import numpy as np
import torch
from einops import rearrange


class AlignRestore:
    """Align a face region using 3-point landmarks, then restore it back."""

    def __init__(
        self,
        align_points: int = 3,
        resolution: int = 256,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.upscale_factor = 1
        ratio = resolution / 256 * 2.8
        self.crop_ratio = (ratio, ratio)
        self.face_template = np.array(
            [[19 - 2, 30 - 10], [56 + 2, 30 - 10], [37.5, 45 - 5]]
        )
        self.face_template = self.face_template * ratio
        self.face_size = (int(75 * self.crop_ratio[0]), int(100 * self.crop_ratio[1]))
        self.p_bias = None
        self.device = device
        self.dtype = dtype
        self.fill_value = torch.tensor([127, 127, 127], device=device, dtype=dtype)
        self.mask = torch.ones(
            (1, 1, self.face_size[1], self.face_size[0]), device=device, dtype=dtype
        )

    def align_warp_face(
        self, img: np.ndarray, landmarks3: np.ndarray, smooth: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        affine_matrix, self.p_bias = self._transform_from_points(
            landmarks3, self.face_template, smooth, self.p_bias
        )
        img_t = (
            rearrange(
                torch.from_numpy(img).to(device=self.device, dtype=self.dtype),
                "h w c -> c h w",
            )
            .unsqueeze(0)
        )
        M = torch.from_numpy(affine_matrix).to(device=self.device, dtype=self.dtype).unsqueeze(0)
        cropped = kornia.geometry.transform.warp_affine(
            img_t,
            M,
            (self.face_size[1], self.face_size[0]),
            mode="bilinear",
            padding_mode="fill",
            fill_value=self.fill_value,
        )
        cropped = (
            rearrange(cropped.squeeze(0), "c h w -> h w c").cpu().numpy().astype(np.uint8)
        )
        return cropped, affine_matrix

    def restore_img(
        self, input_img: np.ndarray, face: torch.Tensor, affine_matrix: np.ndarray
    ) -> np.ndarray:
        h, w, _ = input_img.shape
        if isinstance(affine_matrix, np.ndarray):
            affine_matrix = (
                torch.from_numpy(affine_matrix).to(device=self.device, dtype=self.dtype).unsqueeze(0)
            )

        inv_M = kornia.geometry.transform.invert_affine_transform(affine_matrix)
        face = face.to(dtype=self.dtype).unsqueeze(0)

        inv_face = kornia.geometry.transform.warp_affine(
            face, inv_M, (h, w), mode="bilinear", padding_mode="fill", fill_value=self.fill_value
        ).squeeze(0)
        inv_face = (inv_face / 2 + 0.5).clamp(0, 1) * 255

        input_t = rearrange(
            torch.from_numpy(input_img).to(device=self.device, dtype=self.dtype), "h w c -> c h w"
        )
        inv_mask = kornia.geometry.transform.warp_affine(
            self.mask, inv_M, (h, w), padding_mode="zeros"
        )
        inv_mask_erosion = kornia.morphology.erosion(
            inv_mask,
            torch.ones(
                (int(2 * self.upscale_factor), int(2 * self.upscale_factor)),
                device=self.device,
                dtype=self.dtype,
            ),
        )

        total_face_area = torch.sum(inv_mask_erosion.float())
        w_edge = int(total_face_area**0.5) // 20
        erosion_radius = w_edge * 2

        inv_mask_erosion_np = inv_mask_erosion.squeeze().cpu().numpy().astype(np.float32)
        inv_mask_center_np = cv2.erode(
            inv_mask_erosion_np, np.ones((erosion_radius, erosion_radius), np.uint8)
        )
        inv_mask_center = torch.from_numpy(inv_mask_center_np).to(
            device=self.device, dtype=self.dtype
        )[None, None, ...]

        blur_size = w_edge * 2 + 1
        sigma = 0.3 * ((blur_size - 1) * 0.5 - 1) + 0.8
        inv_soft_mask = kornia.filters.gaussian_blur2d(
            inv_mask_center, (blur_size, blur_size), (sigma, sigma)
        ).squeeze(0)

        inv_mask_erosion_t = inv_mask_erosion.squeeze(0).expand_as(inv_face)
        pasted_face = inv_mask_erosion_t * inv_face
        inv_soft_mask_3d = inv_soft_mask.expand_as(inv_face)
        img_back = inv_soft_mask_3d * pasted_face + (1 - inv_soft_mask_3d) * input_t
        img_back = rearrange(img_back, "c h w -> h w c").contiguous().to(dtype=torch.uint8)
        return img_back.cpu().numpy()

    def _transform_from_points(
        self,
        points1: np.ndarray,
        points0: np.ndarray,
        smooth: bool = True,
        p_bias=None,
    ) -> tuple[np.ndarray, object]:
        p2 = torch.tensor(points0, device=self.device, dtype=torch.float32)
        p1 = torch.tensor(points1, device=self.device, dtype=torch.float32)

        c1, c2 = torch.mean(p1, 0), torch.mean(p2, 0)
        p1c, p2c = p1 - c1, p2 - c2
        s1, s2 = torch.std(p1c), torch.std(p2c)
        p1n, p2n = p1c / s1, p2c / s2

        cov = torch.matmul(p1n.T, p2n)
        U, S, V = torch.svd(cov.float())
        R = torch.matmul(V, U.T)
        if torch.det(R.float()) < 0:
            V[:, -1] = -V[:, -1]
            R = torch.matmul(V, U.T)

        sR = (s2 / s1) * R
        T = c2.reshape(2, 1) - (s2 / s1) * torch.matmul(R, c1.reshape(2, 1))
        M = torch.cat((sR, T), dim=1)

        if smooth:
            bias = p2n[2] - p1n[2]
            if p_bias is None:
                p_bias = bias
            else:
                bias = p_bias * 0.2 + bias * 0.8
            p_bias = bias
            M[:, 2] = M[:, 2] + bias

        return M.cpu().numpy(), p_bias
