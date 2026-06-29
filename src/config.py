"""Application configuration."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LipSyncModel(str, Enum):
    WAV2LIP = "wav2lip"
    MUSETALK = "musetalk"
    LATENTSYNC = "latentsync"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "SwiftDub"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    default_model: LipSyncModel = LipSyncModel.WAV2LIP
    max_upload_mb: int = 500
    job_timeout_seconds: int = 3600

    project_root: Path = PROJECT_ROOT
    models_dir: Path = PROJECT_ROOT / "models"
    data_dir: Path = PROJECT_ROOT / "data"
    uploads_dir: Path = PROJECT_ROOT / "data" / "uploads"
    outputs_dir: Path = PROJECT_ROOT / "data" / "outputs"
    samples_dir: Path = PROJECT_ROOT / "data" / "samples"
    temp_dir: Path = PROJECT_ROOT / "data" / "temp"

    wav2lip_repo: Path = PROJECT_ROOT / "models" / "repos" / "wav2lip"
    musetalk_repo: Path = PROJECT_ROOT / "models" / "repos" / "musetalk"
    latentsync_repo: Path = PROJECT_ROOT / "models" / "repos" / "latentsync"

    wav2lip_checkpoint: Path = (
        PROJECT_ROOT / "models" / "weights" / "wav2lip" / "wav2lip_gan.pth"
    )
    musetalk_unet: Path = (
        PROJECT_ROOT / "models" / "weights" / "musetalk" / "musetalkV15" / "unet.pth"
    )
    musetalk_config: Path = (
        PROJECT_ROOT
        / "models"
        / "weights"
        / "musetalk"
        / "musetalkV15"
        / "musetalk.json"
    )
    latentsync_checkpoint: Path = (
        PROJECT_ROOT / "models" / "weights" / "latentsync" / "latentsync_unet.pt"
    )
    latentsync_version: str = "1.5"

    latentsync_inference_steps: int = 20
    latentsync_guidance_scale: float = 1.5
    musetalk_bbox_shift: int = 0

    ffmpeg_path: str = "ffmpeg"


settings = Settings()


def ensure_directories() -> None:
    for path in (
        settings.models_dir,
        settings.data_dir,
        settings.uploads_dir,
        settings.outputs_dir,
        settings.samples_dir,
        settings.temp_dir,
        settings.wav2lip_repo,
        settings.musetalk_repo,
        settings.latentsync_repo,
        settings.wav2lip_checkpoint.parent,
        settings.musetalk_unet.parent,
        settings.latentsync_checkpoint.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)
