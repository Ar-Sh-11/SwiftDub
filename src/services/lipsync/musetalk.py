"""MuseTalk backend wrapper."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from loguru import logger

from src.config import settings
from src.services.lipsync.base import LipSyncBackend, LipSyncResult


class MuseTalkBackend(LipSyncBackend):
    name = "musetalk"

    def is_available(self) -> bool:
        return (
            settings.musetalk_repo.exists()
            and settings.musetalk_unet.exists()
            and settings.musetalk_config.exists()
        )

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
                "MuseTalk is not set up. Run: python scripts/download_models.py --model musetalk"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_dir = settings.temp_dir / "musetalk" / output_path.stem
        result_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "task_0": {
                "video_path": str(video_path.resolve()),
                "audio_path": str(audio_path.resolve()),
            }
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
            encoding="utf-8",
        ) as handle:
            yaml.safe_dump(config, handle)
            config_path = handle.name

        try:
            cmd = [
                sys.executable,
                "-m",
                "scripts.inference",
                "--inference_config",
                config_path,
                "--result_dir",
                str(result_dir.resolve()),
                "--unet_model_path",
                str(settings.musetalk_unet.resolve()),
                "--unet_config",
                str(settings.musetalk_config.resolve()),
                "--version",
                "v15",
                "--bbox_shift",
                str(settings.musetalk_bbox_shift),
            ]

            env = os.environ.copy()
            env["PYTHONPATH"] = str(settings.musetalk_repo.resolve())

            logger.info("Running MuseTalk inference")
            result = subprocess.run(
                cmd,
                cwd=str(settings.musetalk_repo),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                logger.error(result.stdout)
                logger.error(result.stderr)
                raise RuntimeError(f"MuseTalk failed:\n{result.stderr[-2000:]}")

            candidates = list(result_dir.rglob("*.mp4"))
            if not candidates:
                raise RuntimeError("MuseTalk did not produce an output video")

            produced = max(candidates, key=lambda p: p.stat().st_mtime)
            produced.replace(output_path)
        finally:
            Path(config_path).unlink(missing_ok=True)

        return LipSyncResult(
            output_video=output_path,
            model=self.name,
            metadata={"version": "v15"},
        )
