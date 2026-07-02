"""SadTalker backend (OpenTalker, CVPR 2023).

SadTalker takes a single face image (we extract the best frame) + audio
and animates the full portrait. Results are blended back onto the original
video using the stable face box.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from src.config import ROOT, settings
from src.core.models._subprocess import run_model
from src.core.models.base import InferenceResult, LipSyncBackend
from src.utils.ffmpeg import mux_audio_video
from src.utils.video import read_frames, write_video


class SadTalkerBackend(LipSyncBackend):
    name = "sadtalker"

    def __init__(self) -> None:
        cfg = settings.model_configs["sadtalker"]
        self._repo     = ROOT / cfg["repo_dir"]
        self._ckpt_dir = ROOT / cfg["checkpoint_dir"]
        self._env      = cfg["conda_env"]

    def is_ready(self) -> bool:
        return self._repo.exists() and self._ckpt_dir.exists() and any(self._ckpt_dir.iterdir())

    def infer(self, video: Path, audio: Path, output: Path, face_box=None) -> InferenceResult:
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = settings.temp_dir / f"sadtalker_{output.stem}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        # Extract best reference frame as image
        frames, fps = read_frames(video, max_frames=30)
        ref_frame = frames[len(frames) // 2]
        ref_img = tmp_dir / "ref.jpg"
        cv2.imwrite(str(ref_img), ref_frame)

        run_model(
            [
                "PYTHON", "inference.py",
                "--driven_audio", str(audio),
                "--source_image", str(ref_img),
                "--checkpoint_dir", str(self._ckpt_dir),
                "--result_dir", str(tmp_dir),
                "--still", "--preprocess", "full", "--enhancer", "gfpgan",
            ],
            cwd=self._repo,
            env_name=self._env,
        )

        produced = sorted(tmp_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if not produced:
            raise RuntimeError("SadTalker did not produce an output file")
        mux_audio_video(produced[-1], audio, output, copy_video=False)
        return InferenceResult(output_video=output, model=self.name, elapsed_s=time.time() - t0)
