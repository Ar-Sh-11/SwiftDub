# SwiftDub

Production-grade open-source lip-sync service. Upload a video and dialogue audio — get back a lip-synced video where the on-screen speaker's lips match the new audio, with identity and background fully preserved.

**Five SOTA models. Full fine-tuning pipeline. Novel SwiftSync architecture. MongoDB + Redis + Prometheus + Grafana.**

---

## Contents

- [Quick Start (Windows laptop)](#quick-start-windows)
- [Quick Start (RunPod GPU)](#quick-start-runpod)
- [API Reference](#api-reference)
- [Models](#models)
- [Datasets](#datasets)
- [Fine-tuning](#fine-tuning)
- [Novel Architecture — SwiftSync](#novel-architecture--swiftsync)
- [Production Stack](#production-stack)
- [RunPod Deployment ($50 budget)](#runpod-deployment)
- [Repository Layout](#repository-layout)

---

## Quick Start (Windows)

### 1. Install Miniconda (Windows 64-bit)

```powershell
# Run as Administrator
Invoke-WebRequest `
  -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" `
  -OutFile "$env:TEMP\miniconda.exe" -UseBasicParsing
Start-Process "$env:TEMP\miniconda.exe" -ArgumentList "/S","/D=C:\Miniconda3" -Wait

# Or use the provided script (handles env creation too)
PowerShell -ExecutionPolicy Bypass -File scripts\setup\install_miniconda_windows.ps1
```

### 2. Create conda environment

```powershell
conda env create -f envs/main.yml
conda activate swiftdub
```

### 3. Download sample data

```powershell
python scripts/download/datasets.py --split benchmark
```

### 4. Download a model

```powershell
# Start with Wav2Lip (no GPU needed on CPU, runs in ~40s)
python scripts/download/models.py --model wav2lip
```

### 5. Run demo

```powershell
python scripts/benchmark/run_benchmark.py --model wav2lip
```

### 6. Start API

```powershell
python -m src.main
# → http://localhost:8000/docs
```

---

## Quick Start (RunPod)

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/SwiftDub.git && cd SwiftDub

# 2. Install Miniconda + envs (Linux, GPU)
bash scripts/setup/install_miniconda_linux.sh --gpu --cuda-ver=cu121

# 3. Activate
conda activate swiftdub

# 4. Download models
python scripts/download/models.py --model all --latentsync-version 1.5

# 5. Download benchmark data
python scripts/download/datasets.py --split benchmark

# 6. Benchmark all models
python scripts/benchmark/run_benchmark.py

# 7. Start API (with full observability stack)
docker compose -f deploy/docker-compose.yml up -d
```

---

## API Reference

### `GET /health`

```json
{
  "status": "healthy",
  "cuda_available": true,
  "ffmpeg_available": true,
  "models": {
    "wav2lip": true,
    "video_retalking": true,
    "musetalk": false,
    "latentsync": true,
    "sadtalker": false
  }
}
```

### `GET /models`

Lists all 5 models with VRAM requirements and availability.

### `POST /predict`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `video` | file | required | Input video (MP4/AVI) |
| `audio` | file | optional | Dub audio WAV/MP3; extracts from video if absent |
| `model` | string | `wav2lip` | One of: `wav2lip`, `video_retalking`, `musetalk`, `latentsync`, `sadtalker` |
| `sync` | bool | `true` | `true` = wait for result; `false` = async job |

**Response:**
```json
{
  "job_id": "abc123",
  "status": "completed",
  "model": "wav2lip",
  "output_video": "/data/outputs/abc123_wav2lip.mp4"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "video=@data/samples/benchmark/sample1_video.mp4" \
  -F "audio=@data/samples/benchmark/sample1_audio.wav" \
  -F "model=latentsync"
```

### `GET /jobs/{job_id}`

Poll async job status.

### `GET /jobs/{job_id}/download`

Download completed output MP4.

### `GET /metrics`

Prometheus metrics endpoint.

---

## Models

| Model | VRAM | Speed (A100) | Best For |
|-------|------|------|----------|
| **Wav2Lip + GAN** | 4 GB | 12s / 10s clip | Fast baseline, CPU-feasible |
| **VideoReTalking** | 6 GB | 45s | High visual quality, face enhancement |
| **MuseTalk v1.5** | 8 GB | 18s | Real-time, identity preservation |
| **LatentSync 1.5** | 8 GB | 120s | Best sync accuracy (diffusion) |
| **SadTalker** | 6 GB | 55s | Animated portrait from still frame |

See [BENCHMARK.md](BENCHMARK.md) for full metric tables.

Install individual models:
```bash
python scripts/download/models.py --model wav2lip
python scripts/download/models.py --model video_retalking
python scripts/download/models.py --model musetalk
python scripts/download/models.py --model latentsync --latentsync-version 1.5
python scripts/download/models.py --model sadtalker
```

---

## Datasets

### Benchmark (no signup, automatic download)

```bash
python scripts/download/datasets.py --split benchmark
# → data/samples/benchmark/sample{1,2,3}_{video,audio}.mp4/wav
```

3 public LatentSync demo clips: English, talking head, frontal.

### Training Datasets

| Dataset | Size | Hours | Resolution | License | Download |
|---------|------|-------|------------|---------|----------|
| **LRS3** | 60 GB | 438h | 224×224 (face) | Research (BBC agreement) | [mmai.io](https://mmai.io/datasets/lip_reading/) |
| **VoxCeleb2** | 145 GB | 2442h | 224×224 (face) | Research (VGG) | [robots.ox.ac.uk](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) |
| **HDTF** | 15 GB | 16h | 720p-1080p | CC-BY 4.0 | `python scripts/download/datasets.py --split hdtf` |
| **CelebV-HQ** | 65 GB | 65h | 512×512 | Non-commercial | `python scripts/download/datasets.py --split celebv` |
| **AVSpeech** | 600 GB | 4700h | Variable | Research | CSV from Google |

**Dataset shapes (PyTorch tensors):**
```
Video frames : [T, 3, H, W]  float32 ∈ [0,1]   (H=W=224 for LRS3/VoxCeleb2)
Audio mel    : [1, 80, T×16] float32 (log-mel, sr=16kHz, hop=160)
```

---

## Fine-tuning

### Fine-tune Wav2Lip on LRS3

```bash
# Requires: data/lrs3/ downloaded
python training/train.py \
  --mode finetune_wav2lip \
  --dataset lrs3 \
  --data-root data/lrs3 \
  --epochs 30 \
  --batch 8 \
  --lr 1e-4
```

Checkpoints saved to `checkpoints/wav2lip_ft/`.

### Fine-tune on HDTF (domain adaptation)

```bash
python scripts/download/datasets.py --split hdtf
python training/train.py \
  --mode finetune_wav2lip \
  --dataset hdtf \
  --data-root data/hdtf \
  --epochs 20 \
  --batch 4 \
  --lr 5e-5
```

### Expected improvement from fine-tuning

| Stage | LSE-D | SSIM |
|-------|-------|------|
| Off-the-shelf Wav2Lip | 6.8 | 0.74 |
| + LRS3 30 epochs | **5.9** | **0.80** |

---

## Novel Architecture — SwiftSync

`training/models/swiftsync.py` implements a custom audio-visual transformer for lip sync:

```
Audio:  WavLM-base (frozen) → linear projection → audio tokens [B, T_a, 512]
Video:  Conv2D patch encoder (VideoMAE-v2 style) → visual tokens [B, T×P, 512]
Decode: Cross-attention (audio queries × visual keys) → lip-region patches
Output: Reconstructed face sequence [B, T, 3, 256, 256]
```

**Training objectives:**
- L1 pixel reconstruction loss
- VGG-16 perceptual loss (relu1_2, relu2_2, relu3_3)
- SyncNet contrastive loss (frozen, same as Wav2Lip training)

**Pre-train from scratch:**
```bash
python training/train.py \
  --mode pretrain_swiftsync \
  --dataset lrs3 \
  --data-root data/lrs3 \
  --epochs 100 \
  --batch 4 \
  --lr 3e-4
```

---

## Production Stack

| Service | Port | Purpose |
|---------|------|---------|
| FastAPI API | 8000 | Inference HTTP service |
| MongoDB | 27017 | Job persistence |
| Redis | 6379 | Cache + Celery broker |
| Celery Worker | — | Async inference jobs |
| Celery Flower | 5555 | Task monitoring UI |
| Prometheus | 9090 | Metrics scraping |
| Grafana | 3000 | Dashboards (admin / swiftdub) |

Start everything:
```bash
docker compose -f deploy/docker-compose.yml up -d
```

Grafana dashboards are auto-provisioned with:
- Request rate per model
- Inference latency p50/p95
- Active jobs gauge
- Cache hit rate

---

## RunPod Deployment

**Recommended pod:** RTX 4090 24GB (~$0.44/h) for Wav2Lip + MuseTalk + LatentSync 1.5  
**Budget breakdown (for $50):**

| Task | Hours | Cost |
|------|-------|------|
| Setup + model download | 1h | $0.44 |
| Benchmark all 5 models | 2h | $0.88 |
| Wav2Lip fine-tune (LRS3, 30ep) | 8h | $3.52 |
| LatentSync inference testing | 4h | $1.76 |
| SwiftSync pre-train (100ep) | 40h | $17.60 |
| Buffer / development | 60h | $26.40 |
| **Total** | **115h** | **$50.60** |

**RunPod template:** Use `runpod/pytorch:2.1.0-py3.10-cuda12.1-devel` base image.

---

## Repository Layout

```
SwiftDub/
├── configs/
│   ├── models.yaml          # All model configs (paths, URLs, params)
│   └── datasets.yaml        # Dataset schemas, metric targets
├── envs/
│   ├── main.yml             # swiftdub conda env
│   └── musetalk.yml         # swiftdub-musetalk conda env (mmcv)
├── src/
│   ├── main.py              # Entrypoint: uvicorn
│   ├── config.py            # Pydantic settings
│   ├── api/                 # FastAPI layer (routes, schemas, deps)
│   ├── core/                # Business logic (pipeline, face, models)
│   │   └── models/          # 5 model wrappers + registry
│   ├── db/                  # MongoDB (motor async)
│   ├── cache/               # Redis (aioredis)
│   ├── workers/             # Celery tasks
│   ├── metrics/             # Prometheus registry
│   └── utils/               # ffmpeg, video, audio helpers
├── training/
│   ├── datasets/            # LRS3, VoxCeleb2, base class
│   ├── losses/              # Perceptual + SyncNet loss
│   ├── models/
│   │   └── swiftsync.py     # Novel WavLM × VideoMAE architecture
│   └── train.py             # Unified training CLI
├── scripts/
│   ├── setup/               # install_miniconda_windows.ps1 / _linux.sh
│   ├── download/
│   │   ├── models.py        # Download all 5 model repos + weights
│   │   └── datasets.py      # Download benchmark + HDTF + CelebV
│   └── benchmark/
│       └── run_benchmark.py # Multi-model benchmark runner
├── deploy/
│   ├── docker-compose.yml   # Full stack: API+Worker+Mongo+Redis+Prometheus+Grafana
│   ├── prometheus.yml
│   └── grafana/
│       ├── dashboards/
│       └── datasources/
├── models/
│   ├── weights/             # Downloaded checkpoints (gitignored)
│   └── repos/               # Cloned upstream repos (gitignored)
├── data/
│   ├── samples/benchmark/   # Public test clips
│   ├── uploads/             # Job input files
│   ├── outputs/             # Finished videos
│   └── temp/                # Per-job scratch
├── BENCHMARK.md
└── README.md
```

---

## Success Criteria Checklist

| Requirement | Implementation |
|-------------|----------------|
| No warbling / deformation | Face-box constrained inference; original frames outside box untouched |
| Convincing lip movement | 5 SOTA models; best: LatentSync (diffusion) |
| Audio unchanged | Output remuxed with original audio via FFmpeg |
| Left-most speaker | OpenCV Haar cascade → stable averaged bounding box |
| English-only | Whisper-based audio encoders (LatentSync, MuseTalk) |
| HTTP API | FastAPI with `/predict`, `/jobs`, `/health`, `/metrics` |
| Async jobs | Celery + Redis broker |
| Observability | Prometheus metrics + Grafana dashboards auto-provisioned |
| Caching | Redis SHA-256 content-addressed cache (1h TTL) |
| Persistence | MongoDB job history via Motor (async) |
| GPU (cloud) | Docker Compose with `nvidia` device reservations |

---

## License

Code in this repository is MIT. Each upstream model has its own license:
- Wav2Lip: non-commercial research
- VideoReTalking: Apache 2.0
- MuseTalk: MIT
- LatentSync: Apache 2.0
- SadTalker: non-commercial research

Review before commercial use.
