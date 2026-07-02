"""Wav2Lip GAN backend."""

from __future__ import annotations

import time
from pathlib import Path

from src.config import ROOT, settings
from src.core.models._subprocess import run_model
from src.core.models.base import InferenceResult, LipSyncBackend
from src.utils.ffmpeg import mux_audio_video


class Wav2LipBackend(LipSyncBackend):
    name = "wav2lip"

    def __init__(self) -> None:
        cfg = settings.model_configs["wav2lip"]
        self._repo = ROOT / cfg["repo_dir"]
        self._ckpt = ROOT / cfg["checkpoint"]
        self._env  = cfg["conda_env"]

    def is_ready(self) -> bool:
        return self._repo.exists() and self._ckpt.exists()

    def infer(self, video: Path, audio: Path, output: Path, face_box=None) -> InferenceResult:
        output.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        cmd = [
            "PYTHON", "inference.py",
            "--checkpoint_path", str(self._ckpt),
            "--face", str(video),
            "--audio", str(audio),
            "--outfile", str(output),
            "--pads", "0", "10", "0", "0",
        ]
        if face_box:
            x1, y1, x2, y2 = face_box
            cmd += ["--box", str(y1), str(y2), str(x1), str(x2)]

        run_model(cmd, cwd=self._repo, env_name=self._env)

        # If Wav2Lip's internal ffmpeg call failed, remux from temp AVI
        if not output.exists():
            avi = self._repo / "temp" / "result.avi"
            if avi.exists():
                mux_audio_video(avi, audio, output, copy_video=False)

        if not output.exists():
            raise RuntimeError("Wav2Lip did not produce an output file")
        return InferenceResult(output_video=output, model=self.name, elapsed_s=time.time() - t0)
