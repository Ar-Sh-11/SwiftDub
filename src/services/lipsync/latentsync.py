"""LatentSync backend wrapper."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

from src.config import settings
from src.services.lipsync.base import LipSyncBackend, LipSyncResult


class LatentSyncBackend(LipSyncBackend):
    name = "latentsync"

    def is_available(self) -> bool:
        return settings.latentsync_repo.exists() and settings.latentsync_checkpoint.exists()

    def _config_path(self) -> Path:
        if settings.latentsync_version == "1.6":
            return settings.latentsync_repo / "configs" / "unet" / "stage2_512.yaml"
        return settings.latentsync_repo / "configs" / "unet" / "stage2.yaml"

    def run(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        *,
        face_box: tuple[int, int, int, int] | None = None,
    ) -> LipSyncResult:
        if not self.is_available():
            raise RuntimeError(
                "LatentSync is not set up. Run: python scripts/download_models.py --model latentsync"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = settings.temp_dir / "latentsync" / output_path.stem
        temp_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "scripts/inference.py",
            "--unet_config_path",
            str(self._config_path().resolve()),
            "--inference_ckpt_path",
            str(settings.latentsync_checkpoint.resolve()),
            "--video_path",
            str(video_path.resolve()),
            "--audio_path",
            str(audio_path.resolve()),
            "--video_out_path",
            str(output_path.resolve()),
            "--inference_steps",
            str(settings.latentsync_inference_steps),
            "--guidance_scale",
            str(settings.latentsync_guidance_scale),
            "--temp_dir",
            str(temp_dir.resolve()),
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(settings.latentsync_repo.resolve())

        logger.info("Running LatentSync inference")
        result = subprocess.run(
            cmd,
            cwd=str(settings.latentsync_repo),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.error(result.stdout)
            logger.error(result.stderr)
            raise RuntimeError(f"LatentSync failed:\n{result.stderr[-2000:]}")

        if not output_path.exists():
            raise RuntimeError("LatentSync did not produce an output video")

        return LipSyncResult(
            output_video=output_path,
            model=self.name,
            metadata={
                "version": settings.latentsync_version,
                "inference_steps": settings.latentsync_inference_steps,
            },
        )
