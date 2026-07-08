"""Integration tests for lipsync eval input videos."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "data" / "lipsync eval input"


def _ffmpeg_env() -> dict[str, str]:
    env = os.environ.copy()
    try:
        import imageio_ffmpeg
        ffdir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
        env["PATH"] = ffdir + os.pathsep + env.get("PATH", "")
    except Exception:
        pass
    env["PYTHONPATH"] = str(ROOT)
    env["INSIGHTFACE_HOME"] = str(ROOT / "models/weights/latentsync/insightface")
    env["MUSETALK_DISABLED"] = "true"
    return env


def _extract_audio(video: Path, audio: Path) -> None:
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-i", str(video), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio)],
        check=True,
        capture_output=True,
    )


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("name", ["cut1.mp4", "cut2.mp4"])
def test_eval_video_inference(name: str, tmp_path: Path):
    video = EVAL_DIR / name
    if not video.exists():
        pytest.skip(f"{name} not present")
    ckpt = ROOT / "models/weights/latentsync/latentsync_unet.pt"
    if not ckpt.exists():
        pytest.skip("weights not mounted")

    audio = tmp_path / "audio.wav"
    out = tmp_path / "out.mp4"
    _extract_audio(video, audio)

    cmd = [
        sys.executable, "-m", "src.models.latentsync.infer",
        "--unet_config_path", str(ROOT / "configs/unet/stage2.yaml"),
        "--inference_ckpt_path", str(ckpt),
        "--whisper_model_path", str(ROOT / "models/weights/latentsync/whisper/tiny.pt"),
        "--video_path", str(video),
        "--audio_path", str(audio),
        "--video_out_path", str(out),
        "--inference_steps", "5",
        "--guidance_scale", "1.5",
        "--seed", "1247",
        "--temp_dir", str(tmp_path / "ls_temp"),
        "--gpu_id", "0",
    ]
    r = subprocess.run(cmd, cwd=str(ROOT), env=_ffmpeg_env(), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert out.exists() and out.stat().st_size > 1000
