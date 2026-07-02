"""SwiftDub application settings."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # ── App ──────────────────────────────────────────────────────────────────
    app_name: str = "SwiftDub"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    max_upload_mb: int = 500

    # ── Concurrency ───────────────────────────────────────────────────────────
    # Max simultaneous LatentSync inference jobs (GPU memory gated)
    max_concurrent_jobs: int = 2

    # ── LatentSync model paths ────────────────────────────────────────────────
    latentsync_repo: Path = ROOT / "models" / "repos" / "latentsync"
    latentsync_ckpt: Path = ROOT / "models" / "weights" / "latentsync" / "latentsync_unet.pt"
    latentsync_unet_config: str = "configs/unet/stage2.yaml"   # relative to repo
    latentsync_inference_steps: int = 20
    latentsync_guidance_scale: float = 1.5
    latentsync_seed: int = 1247
    latentsync_enable_deepcache: bool = False

    # ── Directory layout ─────────────────────────────────────────────────────
    data_dir: Path = ROOT / "data"
    samples_dir: Path = ROOT / "data" / "samples"
    uploads_dir: Path = ROOT / "data" / "uploads"
    outputs_dir: Path = ROOT / "data" / "outputs"
    temp_dir: Path = ROOT / "data" / "temp"
    logs_dir: Path = ROOT / "logs"

    # ── External services (graceful fallback when not available) ─────────────
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db: str = "swiftdub"
    redis_url: str = "redis://localhost:6379/0"
    celery_broker: str = "redis://localhost:6379/1"
    celery_backend: str = "redis://localhost:6379/2"

    # Set to true to bypass Redis/MongoDB (dev mode)
    disable_cache: bool = False
    disable_db: bool = False

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.samples_dir, self.uploads_dir,
                  self.outputs_dir, self.temp_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def ffmpeg_bin(self) -> str:
        """Find ffmpeg — prefer imageio_ffmpeg bundled binary."""
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return "ffmpeg"


settings = Settings()
