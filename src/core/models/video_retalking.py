"""VideoReTalking backend (SIGGRAPH Asia 2022)."""

from __future__ import annotations

import time
from pathlib import Path

from src.config import ROOT, settings
from src.core.models._subprocess import run_model
from src.core.models.base import InferenceResult, LipSyncBackend


class VideoRetalkingBackend(LipSyncBackend):
    name = "video_retalking"

    def __init__(self) -> None:
        cfg = settings.model_configs["video_retalking"]
        self._repo = ROOT / cfg["repo_dir"]
        self._ckpt_dir = ROOT / cfg["checkpoint_dir"]
        self._env = cfg["conda_env"]

    def is_ready(self) -> bool:
        if not self._repo.exists():
            return False
        required = ["LNet.pth", "ENet.pth", "DNet.pt", "30_net_gen.pth"]
        return all((self._ckpt_dir / f).exists() for f in required)

    def infer(self, video: Path, audio: Path, output: Path, face_box=None) -> InferenceResult:
        output.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        run_model(
            [
                "PYTHON", "inference.py",
                "--face", str(video),
                "--audio", str(audio),
                "--outfile", str(output),
            ],
            cwd=self._repo,
            env_name=self._env,
            extra_env={"CHECKPOINTS_DIR": str(self._ckpt_dir)},
        )
        if not output.exists():
            raise RuntimeError("VideoReTalking did not produce an output file")
        return InferenceResult(output_video=output, model=self.name, elapsed_s=time.time() - t0)
