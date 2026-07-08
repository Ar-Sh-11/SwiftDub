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

    # ── GPU pool ─────────────────────────────────────────────────────────────
    # Comma-separated CUDA device IDs; empty = auto-detect
    gpu_ids: str = ""
    # Default total VRAM per GPU when CUDA is unavailable (CPU-mode simulation)
    gpu_vram_default_gb: float = 23.0
    # Per-model VRAM estimates (GB) used by dynamic allocator — tuned for L4 (23 GB)
    latentsync_vram_gb: float = 10.0
    musetalk_vram_gb: float = 7.0

    # ── Memory / cache ────────────────────────────────────────────────────────
    memory_cache_disable_pct: float = 95.0
    gpu_clear_cache_pct: float = 85.0

    # ── LatentSync ────────────────────────────────────────────────────────────
    # Weight paths (loaded from models/weights/ only — no vendor dependency)
    latentsync_ckpt: Path = ROOT / "models" / "weights" / "latentsync" / "latentsync_unet.pt"
    latentsync_whisper: Path = ROOT / "models" / "weights" / "latentsync" / "whisper" / "tiny.pt"
    # InsightFace buffalo_l ONNX models for face detection
    insightface_root: Path = ROOT / "models" / "weights" / "latentsync" / "insightface"
    # UNet config YAML — shipped with weights or kept in configs/
    latentsync_unet_config: str = "configs/unet/stage2.yaml"
    latentsync_inference_steps: int = 20
    latentsync_guidance_scale: float = 1.5
    latentsync_seed: int = 1247
    latentsync_enable_deepcache: bool = False

    # ── MuseTalk ──────────────────────────────────────────────────────────────
    musetalk_unet: Path = ROOT / "models" / "weights" / "musetalk" / "musetalkV15" / "unet.pth"
    musetalk_unet_config: Path = ROOT / "models" / "weights" / "musetalk" / "musetalkV15" / "musetalk.json"
    musetalk_whisper_dir: Path = ROOT / "models" / "weights" / "musetalk" / "whisper"
    musetalk_dwpose: Path = ROOT / "models" / "weights" / "musetalk" / "dwpose" / "dw-ll_ucoco_384.pth"
    musetalk_bbox_shift: int = 0
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

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def root_dir(self) -> Path:
        return ROOT

    @property
    def latentsync_unet_config_path(self) -> Path:
        """Resolve latentsync UNet config — relative to ROOT or absolute."""
        p = Path(self.latentsync_unet_config)
        return p if p.is_absolute() else ROOT / p

    @property
    def model_configs(self) -> dict[str, Any]:
        cfg_path = ROOT / "configs" / "models.yaml"
        if not cfg_path.exists():
            return {}
        return yaml.safe_load(cfg_path.read_text()) or {}

    @property
    def ffmpeg_bin(self) -> str:
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return "ffmpeg"

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.samples_dir,
            self.uploads_dir,
            self.outputs_dir,
            self.temp_dir,
            self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def gpu_id_list(self) -> list[int]:
        """Parse GPU_IDS env/setting into a list of ints."""
        raw = self.gpu_ids.strip()
        if raw:
            try:
                return [int(x.strip()) for x in raw.split(",") if x.strip()]
            except ValueError:
                pass
        try:
            import torch
            return list(range(torch.cuda.device_count())) or [0]
        except Exception:
            return [0]


settings = Settings()
