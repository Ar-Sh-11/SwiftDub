"""MuseTalk v1.5 inference entry point — SwiftDub first-party script.

Called as a subprocess by src/core/musetalk.py:
    python -m src.models.musetalk.infer [args]

Loads weights from models/weights/musetalk/ only.
No dependency on models/vendor/.
"""
from __future__ import annotations

import argparse
import copy
import glob
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Ensure SwiftDub root is importable
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Fallback: add vendor to sys.path so model architecture classes can be found
# Remove when all model classes are migrated to src/models/musetalk/models/
_VENDOR = _ROOT / "models" / "vendor" / "musetalk"


def _ensure_vendor_on_path() -> None:
    if _VENDOR.exists() and str(_VENDOR) not in sys.path:
        sys.path.insert(0, str(_VENDOR))


def _load_all_models(unet_path: str, vae_type: str, unet_config: str, device):
    """Load VAE, UNet, and PositionalEncoding.

    Tries src/models/musetalk/models/ first, then vendor fallback.
    """
    try:
        from src.models.musetalk.models.vae import VAE
        from src.models.musetalk.models.unet import UNet, PositionalEncoding
    except ImportError:
        _ensure_vendor_on_path()
        from musetalk.models.vae import VAE  # type: ignore[import]
        from musetalk.models.unet import UNet, PositionalEncoding  # type: ignore[import]

    # VAE path: stored alongside musetalk weights or in vendor
    vae_path = str(_ROOT / "models" / "weights" / "musetalk" / vae_type)
    if not Path(vae_path).exists():
        # try vendor bundle
        vae_path = str(_VENDOR / "models" / vae_type)

    vae = VAE(model_path=vae_path)
    unet = UNet(unet_config=unet_config, model_path=unet_path, device=device)
    pe = PositionalEncoding(d_model=384)
    return vae, unet, pe


def _get_landmark_and_bbox(img_list: list[str], bbox_shift: int):
    try:
        from src.models.musetalk.utils.preprocess import get_landmark_and_bbox
    except ImportError:
        _ensure_vendor_on_path()
        from musetalk.utils.preprocessing import get_landmark_and_bbox  # type: ignore[import]
    return get_landmark_and_bbox(img_list, bbox_shift)


def _get_face_parsing():
    try:
        from src.models.musetalk.utils.face_parsing import FaceParsing
    except ImportError:
        _ensure_vendor_on_path()
        from musetalk.utils.face_parsing import FaceParsing  # type: ignore[import]
    return FaceParsing


def _datagen(whisper_chunks, vae_latents, batch_size: int, delay: int, device):
    try:
        from src.models.musetalk.utils.datagen import datagen
    except ImportError:
        _ensure_vendor_on_path()
        from musetalk.utils.utils import datagen  # type: ignore[import]
    return datagen(
        whisper_chunks=whisper_chunks,
        vae_encode_latents=vae_latents,
        batch_size=batch_size,
        delay_frame=delay,
        device=device,
    )


def _get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return fps


def _extract_frames(video_path: str, save_dir: str) -> list[str]:
    os.makedirs(save_dir, exist_ok=True)
    cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir}/%08d.png"
    os.system(cmd)
    return sorted(glob.glob(os.path.join(save_dir, "*.png")))


def _read_imgs(img_list: list[str]) -> list[np.ndarray]:
    return [cv2.imread(p) for p in img_list]


def _get_composite(image, face, box, mode, fp):
    """Blend generated face back into original frame."""
    from src.models.musetalk.utils.blending import composite_face
    return composite_face(image, face, box, mode=mode, fp=fp)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="SwiftDub MuseTalk v1.5 inference")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--unet_model_path", required=True)
    parser.add_argument("--unet_config", required=True)
    parser.add_argument("--whisper_dir", required=True)
    parser.add_argument("--vae_type", default="sd-vae")
    parser.add_argument("--bbox_shift", type=int, default=0)
    parser.add_argument("--extra_margin", type=int, default=10)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_float16", action="store_true")
    parser.add_argument("--parsing_mode", default="jaw")
    parser.add_argument("--left_cheek_width", type=int, default=90)
    parser.add_argument("--right_cheek_width", type=int, default=90)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--audio_padding_left", type=int, default=2)
    parser.add_argument("--audio_padding_right", type=int, default=2)
    args = parser.parse_args()

    from transformers import WhisperModel
    from src.models.musetalk.utils.audio import AudioProcessor

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[MuseTalk] device={device}")

    # ── Models ────────────────────────────────────────────────────────────────
    vae, unet, pe = _load_all_models(args.unet_model_path, args.vae_type, args.unet_config, device)
    timesteps = torch.tensor([0], device=device)

    if args.use_float16:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()

    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)

    # ── Whisper ───────────────────────────────────────────────────────────────
    weight_dtype = unet.model.dtype
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    whisper = WhisperModel.from_pretrained(args.whisper_dir).to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    # ── Face parsing ─────────────────────────────────────────────────────────
    FaceParsing = _get_face_parsing()
    fp = FaceParsing(
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width,
    )

    # ── Extract frames ───────────────────────────────────────────────────────
    os.makedirs(args.result_dir, exist_ok=True)
    basename = Path(args.video_path).stem
    audio_basename = Path(args.audio_path).stem
    frame_dir = os.path.join(args.result_dir, basename)
    input_img_list = _extract_frames(args.video_path, frame_dir)
    fps = _get_video_fps(args.video_path) or args.fps

    # ── Audio features ───────────────────────────────────────────────────────
    whisper_feats, librosa_len = audio_processor.get_audio_feature(args.audio_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_feats,
        device,
        weight_dtype,
        whisper,
        librosa_len,
        fps=fps,
        audio_padding_length_left=args.audio_padding_left,
        audio_padding_length_right=args.audio_padding_right,
    )

    # ── Face preprocessing ───────────────────────────────────────────────────
    coord_cache = os.path.join(args.result_dir, f"{basename}.pkl")
    if os.path.exists(coord_cache):
        with open(coord_cache, "rb") as f:
            coord_list = pickle.load(f)
        frame_list = _read_imgs(input_img_list)
    else:
        print("[MuseTalk] Extracting face landmarks (may take a while)...")
        coord_list, frame_list = _get_landmark_and_bbox(input_img_list, args.bbox_shift)
        with open(coord_cache, "wb") as f:
            pickle.dump(coord_list, f)

    _COORD_PLACEHOLDER = (-1, -1, -1, -1)

    # ── VAE encode frames ────────────────────────────────────────────────────
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == _COORD_PLACEHOLDER:
            continue
        x1, y1, x2, y2 = bbox
        y2 = min(y2 + args.extra_margin, frame.shape[0])
        crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
        input_latent_list.append(vae.get_latents_for_unet(crop))

    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    latent_cycle = input_latent_list + input_latent_list[::-1]

    # ── Batch UNet inference ─────────────────────────────────────────────────
    print("[MuseTalk] Running UNet inference...")
    gen = _datagen(whisper_chunks, latent_cycle, args.batch_size, delay_frame=0, device=device)
    video_num = len(whisper_chunks)
    total_batches = int(np.ceil(video_num / args.batch_size))

    res_frame_list = []
    for whisper_batch, latent_batch in tqdm(gen, total=total_batches):
        audio_feat = pe(whisper_batch)
        latent_batch = latent_batch.to(dtype=weight_dtype)
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feat).sample
        for res_frame in vae.decode_latents(pred_latents):
            res_frame_list.append(res_frame)

    # ── Composite frames ──────────────────────────────────────────────────────
    result_img_dir = os.path.join(args.result_dir, f"{basename}_{audio_basename}")
    os.makedirs(result_img_dir, exist_ok=True)

    print("[MuseTalk] Compositing frames...")
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        bbox = coord_list_cycle[i % len(coord_list_cycle)]
        ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
        x1, y1, x2, y2 = bbox
        y2 = min(y2 + args.extra_margin, ori_frame.shape[0])
        try:
            res = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        except Exception:
            continue
        combine = _get_composite(ori_frame, res, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
        cv2.imwrite(f"{result_img_dir}/{str(i).zfill(8)}.png", combine)

    # ── Encode to video ───────────────────────────────────────────────────────
    temp_vid = os.path.join(args.result_dir, f"temp_{basename}_{audio_basename}.mp4")
    cmd_vid = (
        f"ffmpeg -y -v warning -r {fps:.2f} -f image2 "
        f"-i {result_img_dir}/%08d.png "
        f"-vcodec libx264 -vf format=yuv420p -crf 18 {temp_vid}"
    )
    os.system(cmd_vid)

    cmd_mux = f"ffmpeg -y -v warning -i {args.audio_path} -i {temp_vid} {args.output_path}"
    os.system(cmd_mux)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    shutil.rmtree(result_img_dir, ignore_errors=True)
    shutil.rmtree(frame_dir, ignore_errors=True)
    if os.path.exists(temp_vid):
        os.remove(temp_vid)

    print(f"[MuseTalk] Output written: {args.output_path}")


if __name__ == "__main__":
    main()
