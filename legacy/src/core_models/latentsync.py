"""LatentSync 1.5 backend (ByteDance)."""

from __future__ import annotations

import time
from pathlib import Path

from src.config import ROOT, settings
from src.core.models._subprocess import run_model
from src.core.models.base import InferenceResult, LipSyncBackend


class LatentSyncBackend(LipSyncBackend):
    name = "latentsync"

    def __init__(self) -> None:
        cfg = settings.model_configs["latentsync"]
        self._repo   = ROOT / cfg["repo_dir"]
        self._ckpt   = ROOT / cfg["checkpoint"]
        self._unet_config = self._repo / cfg["unet_config"]
        self._env    = cfg["conda_env"]
        self._steps  = cfg.get("inference_steps", 20)
        self._scale  = cfg.get("guidance_scale", 1.5)

    def is_ready(self) -> bool:
        return self._repo.exists() and self._ckpt.exists()

    def infer(self, video: Path, audio: Path, output: Path, face_box=None) -> InferenceResult:
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = settings.temp_dir / f"latentsync_{output.stem}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        run_model(
            [
                "PYTHON", "scripts/inference.py",
                "--unet_config_path", str(self._unet_config),
                "--inference_ckpt_path", str(self._ckpt),
                "--video_path", str(video),
                "--audio_path", str(audio),
                "--video_out_path", str(output),
                "--inference_steps", str(self._steps),
                "--guidance_scale", str(self._scale),
                "--temp_dir", str(tmp_dir),
            ],
            cwd=self._repo,
            env_name=self._env,
        )
        if not output.exists():
            raise RuntimeError("LatentSync did not produce an output file")
        return InferenceResult(output_video=output, model=self.name, elapsed_s=time.time() - t0)
