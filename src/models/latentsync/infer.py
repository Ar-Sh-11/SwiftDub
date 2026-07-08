"""LatentSync 1.5 inference entry point — SwiftDub first-party script.

Called as a subprocess by src/core/latentsync.py:
    python -m src.models.latentsync.infer [args]

Loads weights from models/weights/latentsync/ only.
No dependency on models/vendor/.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure the SwiftDub root is on sys.path so src.* imports resolve
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _build_unet(config, ckpt_path: str, dtype, device: str = "cpu"):
    """Load UNet3DConditionModel from checkpoint."""
    from src.models.latentsync.models.unet import UNet3DConditionModel
    from omegaconf import OmegaConf
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model), ckpt_path, device="cpu"
    )
    return unet.to(dtype=dtype)


def _ensure_ffmpeg_on_path() -> None:
    """Whisper and media helpers need ffmpeg on PATH (use bundled binary)."""
    try:
        import imageio_ffmpeg
        ffdir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
        os.environ["PATH"] = ffdir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def main() -> None:
    _ensure_ffmpeg_on_path()
    parser = argparse.ArgumentParser(description="SwiftDub LatentSync 1.5 inference")
    parser.add_argument("--unet_config_path", required=True, help="Path to stage2.yaml config")
    parser.add_argument("--inference_ckpt_path", required=True, help="Path to latentsync_unet.pt")
    parser.add_argument("--whisper_model_path", required=True, help="Path to whisper tiny.pt")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--video_out_path", required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--temp_dir", default="temp")
    parser.add_argument("--enable_deepcache", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA device index")
    parser.add_argument("--mask_image_path", default=None)
    args = parser.parse_args()

    import torch
    from omegaconf import OmegaConf
    from accelerate.utils import set_seed
    from diffusers import AutoencoderKL, DDIMScheduler

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")
    if not os.path.exists(args.audio_path):
        raise FileNotFoundError(f"Audio not found: {args.audio_path}")

    # ── Device + precision ───────────────────────────────────────────────────
    device_str = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    is_fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16 else torch.float32
    print(f"[LatentSync] device={device} dtype={dtype}")

    # ── Config ───────────────────────────────────────────────────────────────
    config = OmegaConf.load(args.unet_config_path)

    # ── Audio encoder ────────────────────────────────────────────────────────
    from src.models.latentsync.whisper.audio_encoder import Audio2Feature
    audio_encoder = Audio2Feature(
        model_path=args.whisper_model_path,
        device=device_str,
        num_frames=config.data.num_frames,
        audio_feat_length=list(config.data.audio_feat_length),
    )

    # ── VAE ──────────────────────────────────────────────────────────────────
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    # ── UNet ─────────────────────────────────────────────────────────────────
    unet = _build_unet(config, args.inference_ckpt_path, dtype)

    # ── Scheduler ────────────────────────────────────────────────────────────
    # Resolve scheduler config directory (must contain scheduler_config.json)
    cfg_dir = Path(args.unet_config_path).parent.parent  # configs/
    scheduler = DDIMScheduler.from_pretrained(str(cfg_dir))

    # ── Pipeline ─────────────────────────────────────────────────────────────
    from src.models.latentsync.pipeline import LipsyncPipeline
    pipeline = LipsyncPipeline(vae=vae, audio_encoder=audio_encoder, unet=unet, scheduler=scheduler)
    pipeline = pipeline.to(device_str)

    # ── DeepCache ────────────────────────────────────────────────────────────
    if args.enable_deepcache:
        try:
            from DeepCache import DeepCacheSDHelper
            helper = DeepCacheSDHelper(pipe=pipeline)
            helper.set_params(cache_interval=3, cache_branch_id=0)
            helper.enable()
            print("[LatentSync] DeepCache enabled")
        except ImportError:
            print("[LatentSync] DeepCache not installed — skipping")

    # ── Seed ─────────────────────────────────────────────────────────────────
    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()
    print(f"[LatentSync] seed={torch.initial_seed()}")

    # ── Run ───────────────────────────────────────────────────────────────────
    pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
        num_frames=config.data.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
        mask_image_path=args.mask_image_path,
        temp_dir=args.temp_dir,
    )
    print(f"[LatentSync] Output written: {args.video_out_path}")


if __name__ == "__main__":
    main()
