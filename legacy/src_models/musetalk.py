"""MuseTalk v1.5 backend."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import yaml

from src.config import ROOT, settings
from src.core.models._subprocess import run_model
from src.core.models.base import InferenceResult, LipSyncBackend


class MuseTalkBackend(LipSyncBackend):
    name = "musetalk"

    def __init__(self) -> None:
        cfg = settings.model_configs["musetalk"]
        self._repo   = ROOT / cfg["repo_dir"]
        self._unet   = ROOT / cfg["unet_path"]
        self._config = ROOT / cfg["unet_config"]
        self._env    = cfg["conda_env"]
        self._bbox_shift = cfg.get("bbox_shift", 0)

    def is_ready(self) -> bool:
        return self._repo.exists() and self._unet.exists() and self._config.exists()

    def infer(self, video: Path, audio: Path, output: Path, face_box=None) -> InferenceResult:
        output.parent.mkdir(parents=True, exist_ok=True)
        result_dir = settings.temp_dir / f"musetalk_{output.stem}"
        result_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        # MuseTalk expects a YAML config with video/audio paths
        task_cfg = {"task_0": {"video_path": str(video), "audio_path": str(audio)}}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(task_cfg, f)
            cfg_path = f.name

        try:
            run_model(
                [
                    "PYTHON", "-m", "scripts.inference",
                    "--inference_config", cfg_path,
                    "--result_dir", str(result_dir),
                    "--unet_model_path", str(self._unet),
                    "--unet_config", str(self._config),
                    "--version", "v15",
                    "--bbox_shift", str(self._bbox_shift),
                ],
                cwd=self._repo,
                env_name=self._env,
            )
        finally:
            Path(cfg_path).unlink(missing_ok=True)

        produced = sorted(result_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if not produced:
            raise RuntimeError("MuseTalk did not produce an output file")
        produced[-1].replace(output)
        shutil.rmtree(result_dir, ignore_errors=True)
        return InferenceResult(output_video=output, model=self.name, elapsed_s=time.time() - t0)
