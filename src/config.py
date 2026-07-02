"""Central application settings. All values can be overridden via .env or environment variables."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class ModelName(str, Enum):
    WAV2LIP = "wav2lip"
    VIDEO_RETALKING = "video_retalking"
    MUSETALK = "musetalk"
    LATENTSYNC = "latentsync"
    SADTALKER = "sadtalker"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # ── App ──────────────────────────────────────────────────────────────────
    app_name: str = "SwiftDub"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    default_model: ModelName = ModelName.WAV2LIP
    max_upload_mb: int = 500

    # ── Directory layout (all relative to ROOT) ──────────────────────────────
    models_dir: Path = ROOT / "models"
    data_dir: Path = ROOT / "data"
    samples_dir: Path = ROOT / "data" / "samples"
    uploads_dir: Path = ROOT / "data" / "uploads"
    outputs_dir: Path = ROOT / "data" / "outputs"
    temp_dir: Path = ROOT / "data" / "temp"

    # ── External services ────────────────────────────────────────────────────
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db: str = "swiftdub"
    redis_url: str = "redis://localhost:6379/0"
    celery_broker: str = "redis://localhost:6379/1"
    celery_backend: str = "redis://localhost:6379/2"

    # ── Tools ────────────────────────────────────────────────────────────────
    ffmpeg_path: str = "ffmpeg"

    # ── Conda envs ───────────────────────────────────────────────────────────
    conda_base: str = ""          # e.g. /root/miniconda3 or C:/Users/.../Miniconda3
    main_conda_env: str = "swiftdub"
    musetalk_conda_env: str = "swiftdub-musetalk"

    @property
    def model_configs(self) -> dict[str, Any]:
        cfg_path = ROOT / "configs" / "models.yaml"
        return yaml.safe_load(cfg_path.read_text())["models"]

    @property
    def dataset_configs(self) -> dict[str, Any]:
        cfg_path = ROOT / "configs" / "datasets.yaml"
        return yaml.safe_load(cfg_path.read_text())

    def ensure_dirs(self) -> None:
        for d in (self.models_dir, self.data_dir, self.samples_dir,
                  self.uploads_dir, self.outputs_dir, self.temp_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
