"""Wav2Lip backend wrapper."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

from src.config import settings
from src.services.lipsync.base import LipSyncBackend, LipSyncResult


class Wav2LipBackend(LipSyncBackend):
    name = "wav2lip"

    def is_available(self) -> bool:
        return settings.wav2lip_repo.exists() and settings.wav2lip_checkpoint.exists()

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
                "Wav2Lip is not set up. Run: python scripts/download_models.py --model wav2lip"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = settings.temp_dir / "wav2lip"
        temp_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "inference.py",
            "--checkpoint_path",
            str(settings.wav2lip_checkpoint.resolve()),
            "--face",
            str(video_path.resolve()),
            "--audio",
            str(audio_path.resolve()),
            "--outfile",
            str(output_path.resolve()),
            "--pads",
            "0",
            "10",
            "0",
            "0",
        ]

        if face_box is not None:
            y1, y2, x1, x2 = face_box[1], face_box[3], face_box[0], face_box[2]
            cmd.extend(["--box", str(y1), str(y2), str(x1), str(x2)])

        env = os.environ.copy()
        env["PYTHONPATH"] = str(settings.wav2lip_repo.resolve())
        try:
            import imageio_ffmpeg

            ffmpeg_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
            env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")
        except Exception:
            pass

        temp_dir = settings.wav2lip_repo / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Running Wav2Lip inference")
        result = subprocess.run(
            cmd,
            cwd=str(settings.wav2lip_repo),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        temp_avi = temp_dir / "result.avi"
        if not output_path.exists() and temp_avi.exists():
            from src.utils.ffmpeg import mux_audio_video

            logger.info("Wav2Lip ffmpeg mux failed; remuxing with bundled ffmpeg")
            mux_audio_video(temp_avi, audio_path, output_path, copy_video=False)

        if result.returncode != 0 and not output_path.exists():
            logger.error(result.stdout)
            logger.error(result.stderr)
            raise RuntimeError(f"Wav2Lip failed:\n{result.stderr[-2000:]}")

        if not output_path.exists():
            raise RuntimeError("Wav2Lip did not produce an output video")

        return LipSyncResult(
            output_video=output_path,
            model=self.name,
            metadata={"checkpoint": str(settings.wav2lip_checkpoint.name)},
        )
