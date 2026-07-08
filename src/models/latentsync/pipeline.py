"""LatentSync 1.5 lip-sync diffusion pipeline.

Adapted from ByteDance/LatentSync (Apache 2.0), which was adapted from
guoyww/AnimateDiff.

This is the first-party SwiftDub implementation — no vendor/ imports.
"""
from __future__ import annotations

import inspect
import math
import os
import shutil
from typing import Callable, List, Optional, Union

import numpy as np
import soundfile as sf
import torch
import torchvision
import tqdm
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import AutoencoderKL  # noqa: F811
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from einops import rearrange
from loguru import logger
from torchvision import transforms

from .utils.image_proc import ImageProcessor, load_fixed_mask
from .utils.media import mux_audio_video, read_audio, read_video, write_video
from .whisper.audio_encoder import Audio2Feature


class LipsyncPipeline(DiffusionPipeline):
    """DDIM-based diffusion pipeline for lip-sync generation."""

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        unet,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ) -> None:
        super().__init__()
        self.register_modules(vae=vae, audio_encoder=audio_encoder, unet=unet, scheduler=scheduler)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.set_progress_bar_config(desc="Steps")

    # ── VAE helpers ───────────────────────────────────────────────────────────

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        return self.vae.decode(latents).sample

    def _prepare_latents(self, num_frames, num_channels, h, w, dtype, device, generator):
        shape = (1, num_channels, 1, h // self.vae_scale_factor, w // self.vae_scale_factor)
        rand_dev = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_dev, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1) * self.scheduler.init_noise_sigma
        return latents

    def _prepare_mask_latents(self, mask, masked_image, h, w, dtype, device, generator, cfg):
        mask = torch.nn.functional.interpolate(
            mask, size=(h // self.vae_scale_factor, w // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)
        mil = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        mil = (mil - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        mil = rearrange(mil.to(device=device, dtype=dtype), "f c h w -> 1 c f h w")
        mask = rearrange(mask.to(device=device, dtype=dtype), "f c h w -> 1 c f h w")
        if cfg:
            mask = torch.cat([mask] * 2)
            mil = torch.cat([mil] * 2)
        return mask, mil

    def _prepare_image_latents(self, images, device, dtype, generator, cfg):
        images = images.to(device=device, dtype=dtype)
        il = self.vae.encode(images).latent_dist.sample(generator=generator)
        il = (il - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        il = rearrange(il, "f c h w -> 1 c f h w")
        return torch.cat([il] * 2) if cfg else il

    # ── Video helpers ─────────────────────────────────────────────────────────

    def _loop_video(self, whisper_chunks, video_frames):
        n_chunks = len(whisper_chunks)
        if n_chunks > len(video_frames):
            faces, boxes, matrices = self._affine_transform_video(video_frames)
            n_loops = math.ceil(n_chunks / len(video_frames))
            all_frames, all_faces, all_boxes, all_matrices = [], [], [], []
            for i in range(n_loops):
                if i % 2 == 0:
                    all_frames.append(video_frames)
                    all_faces.append(faces)
                    all_boxes += boxes
                    all_matrices += matrices
                else:
                    all_frames.append(video_frames[::-1])
                    all_faces.append(faces.flip(0))
                    all_boxes += boxes[::-1]
                    all_matrices += matrices[::-1]
            video_frames = np.concatenate(all_frames, axis=0)[:n_chunks]
            faces = torch.cat(all_faces, dim=0)[:n_chunks]
            boxes = all_boxes[:n_chunks]
            matrices = all_matrices[:n_chunks]
        else:
            video_frames = video_frames[:n_chunks]
            faces, boxes, matrices = self._affine_transform_video(video_frames)
        return video_frames, faces, boxes, matrices

    def _affine_transform_video(self, frames: np.ndarray):
        faces, boxes, matrices = [], [], []
        logger.info("Affine-transforming {} frames...", len(frames))
        for frame in tqdm.tqdm(frames):
            face, box, matrix = self.image_processor.affine_transform(frame)
            faces.append(face)
            boxes.append(box)
            matrices.append(matrix)
        return torch.stack(faces), boxes, matrices

    def _restore_video(self, faces, video_frames, boxes, matrices):
        video_frames = video_frames[: len(faces)]
        out_frames = []
        logger.info("Restoring {} frames...", len(faces))
        for i, face in enumerate(tqdm.tqdm(faces)):
            x1, y1, x2, y2 = boxes[i]
            h, w = int(y2 - y1), int(x2 - x1)
            face = torchvision.transforms.functional.resize(
                face,
                size=(h, w),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
            out_frames.append(
                self.image_processor.restorer.restore_img(video_frames[i], face, matrices[i])
            )
        return np.stack(out_frames, axis=0)

    # ── Diffusion step helpers ────────────────────────────────────────────────

    @staticmethod
    def _paste_pixels_back(decoded, pixel_values, masks, device, dtype):
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        masks = masks.to(device=device, dtype=dtype)
        return decoded * masks + pixel_values * (1 - masks)

    @staticmethod
    def _to_images(pixel_values: torch.Tensor) -> np.ndarray:
        pv = rearrange(pixel_values, "f c h w -> f h w c")
        pv = (pv / 2 + 0.5).clamp(0, 1)
        return (pv * 255).to(torch.uint8).cpu().numpy()

    def _extra_step_kwargs(self, generator, eta):
        sig = inspect.signature(self.scheduler.step).parameters
        kw = {}
        if "eta" in sig:
            kw["eta"] = eta
        if "generator" in sig:
            kw["generator"] = generator
        return kw

    # ── Main call ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask_image_path: Optional[str] = None,
        temp_dir: str = "temp",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        **kwargs,
    ) -> None:
        is_train = self.unet.training
        self.unet.eval()

        device = self._execution_device
        mask_image = load_fixed_mask(height or 512, mask_image_path)
        self.image_processor = ImageProcessor(height or 512, device=str(device), mask_image=mask_image)
        self.set_progress_bar_config(desc=f"Frames={num_frames}")

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        assert height == width and height % 8 == 0

        do_cfg = guidance_scale > 1.0
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        extra_kwargs = self._extra_step_kwargs(generator, eta)

        whisper_feat = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(whisper_feat, fps=video_fps)
        audio_samples = read_audio(audio_path, audio_sample_rate)
        video_frames = read_video(video_path, use_decord=False)
        video_frames, faces, boxes, matrices = self._loop_video(whisper_chunks, video_frames)

        n_latent_ch = self.vae.config.latent_channels
        all_latents = self._prepare_latents(
            len(whisper_chunks), n_latent_ch, height, width, weight_dtype, device, generator
        )

        synced_frames: list[torch.Tensor] = []
        n_inferences = math.ceil(len(whisper_chunks) / num_frames)

        for i in tqdm.tqdm(range(n_inferences), desc="Inference batches"):
            if self.unet.add_audio_layer:
                audio_emb = torch.stack(whisper_chunks[i * num_frames : (i + 1) * num_frames])
                audio_emb = audio_emb.to(device, dtype=weight_dtype)
                if do_cfg:
                    audio_emb = torch.cat([torch.zeros_like(audio_emb), audio_emb])
            else:
                audio_emb = None

            infer_faces = faces[i * num_frames : (i + 1) * num_frames]
            latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]

            ref_pv, masked_pv, masks = self.image_processor.prepare_masks_and_masked_images(
                infer_faces, affine_transform=False
            )
            mask_latents, masked_img_latents = self._prepare_mask_latents(
                masks, masked_pv, height, width, weight_dtype, device, generator, do_cfg
            )
            ref_latents = self._prepare_image_latents(ref_pv, device, weight_dtype, generator, do_cfg)

            n_warmup = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as pbar:
                for j, t in enumerate(timesteps):
                    unet_in = torch.cat([latents] * 2) if do_cfg else latents
                    unet_in = self.scheduler.scale_model_input(unet_in, t)
                    unet_in = torch.cat([unet_in, mask_latents, masked_img_latents, ref_latents], dim=1)
                    noise_pred = self.unet(unet_in, t, encoder_hidden_states=audio_emb).sample
                    if do_cfg:
                        noise_uncond, noise_audio = noise_pred.chunk(2)
                        noise_pred = noise_uncond + guidance_scale * (noise_audio - noise_uncond)
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_kwargs).prev_sample
                    if j == len(timesteps) - 1 or ((j + 1) > n_warmup and (j + 1) % self.scheduler.order == 0):
                        pbar.update()
                        if callback and j % callback_steps == 0:
                            callback(j, t, latents)

            decoded = self.decode_latents(latents)
            decoded = self._paste_pixels_back(decoded, ref_pv, 1 - masks, device, weight_dtype)
            synced_frames.append(decoded)

        synced = self._restore_video(torch.cat(synced_frames), video_frames, boxes, matrices)

        audio_len = int(synced.shape[0] / video_fps * audio_sample_rate)
        audio_np = audio_samples[:audio_len].cpu().numpy()

        if is_train:
            self.unet.train()

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        tmp_vid = os.path.join(temp_dir, "video.mp4")
        tmp_aud = os.path.join(temp_dir, "audio.wav")
        write_video(tmp_vid, synced, fps=video_fps)
        sf.write(tmp_aud, audio_np, audio_sample_rate)
        mux_audio_video(tmp_vid, tmp_aud, video_out_path)
