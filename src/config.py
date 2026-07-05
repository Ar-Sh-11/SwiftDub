"""SwiftDub application settings."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class ModelName(str, Enum):
    LATENTSYNC = "latentsync"
    MUSETALK = "musetalk"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # ── App ──────────────────────────────────────────────────────────────────
    app_name: str = "SwiftDub"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    max_upload_mb: int = 500
    default_model: ModelName = ModelName.LATENTSYNC

    # ── Concurrency ───────────────────────────────────────────────────────────
    max_concurrent_jobs: int = 2

    # ── Memory / cache ────────────────────────────────────────────────────────
    # Disable Redis result cache when GPU or system RAM exceeds this threshold
    memory_cache_disable_pct: float = 95.0
    # Clear CUDA cache after each job when GPU usage exceeds this
    gpu_clear_cache_pct: float = 85.0

    # ── LatentSync ────────────────────────────────────────────────────────────
    latentsync_vendor: Path = ROOT / "models" / "vendor" / "latentsync"
    latentsync_ckpt: Path = ROOT / "models" / "weights" / "latentsync" / "latentsync_unet.pt"
    latentsync_whisper: Path = ROOT / "models" / "weights" / "latentsync" / "whisper" / "tiny.pt"
    latentsync_unet_config: str = "configs/unet/stage2.yaml"
    latentsync_inference_steps: int = 20
    latentsync_guidance_scale: float = 1.5
    latentsync_seed: int = 1247
    latentsync_enable_deepcache: bool = False

    # ── MuseTalk ──────────────────────────────────────────────────────────────
    musetalk_vendor: Path = ROOT / "models" / "vendor" / "musetalk"
    musetalk_unet: Path = ROOT / "models" / "weights" / "musetalk" / "musetalkV15" / "unet.pth"
    musetalk_unet_config: Path = ROOT / "models" / "weights" / "musetalk" / "musetalkV15" / "musetalk.json"
    musetalk_whisper_dir: Path = ROOT / "models" / "weights" / "musetalk" / "whisper"
    musetalk_dwpose: Path = ROOT / "models" / "weights" / "musetalk" / "dwpose" / "dw-ll_ucoco_384.pth"
    musetalk_bbox_shift: int = 0
    # Set MUSETALK_DISABLED=true (or musetalk.disabled in configs/models.yaml) to hide MuseTalk
    musetalk_disabled: bool = False

    def is_musetalk_disabled(self) -> bool:
        if self.musetalk_disabled:
            return True
        return bool(self.model_configs.get("musetalk", {}).get("disabled", False))

    # ── Directory layout ─────────────────────────────────────────────────────
    data_dir: Path = ROOT / "data"
    samples_dir: Path = ROOT / "data" / "samples"
    uploads_dir: Path = ROOT / "data" / "uploads"
    outputs_dir: Path = ROOT / "data" / "outputs"
    temp_dir: Path = ROOT / "data" / "temp"
    logs_dir: Path = ROOT / "logs"

    # ── External services ────────────────────────────────────────────────────
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db: str = "swiftdub"
    redis_url: str = "redis://localhost:6379/0"
    celery_broker: str = "redis://localhost:6379/1"
    celery_backend: str = "redis://localhost:6379/2"
    disable_cache: bool = False
    disable_db: bool = False

    @property
    def model_configs(self) -> dict[str, Any]:
        cfg_path = ROOT / "configs" / "models.yaml"
        return yaml.safe_load(cfg_path.read_text())

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.samples_dir, self.uploads_dir,
                  self.outputs_dir, self.temp_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def ffmpeg_bin(self) -> str:
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return "ffmpeg"


settings = Settings()
