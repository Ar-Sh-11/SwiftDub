# SwiftDub

Production video dubbing service with two lip-sync models:

| Model | Type | Best for |
|-------|------|----------|
| **LatentSync 1.5** | Diffusion (DDIM) | Higher quality, CVPR 2025 |
| **MuseTalk v1.5** | Real-time GAN | Faster inference |

Upload a source video and dub audio — get back a lip-synced MP4 where the speaker's mouth matches the new audio, with identity and background preserved.

Includes a web UI with model dropdown, REST API, batch processing, Redis result cache (disabled under memory pressure), structured inference logs, and a Prometheus + Grafana observability stack.

---

## Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Inference Pipeline](#inference-pipeline)
- [Concurrency & GPU](#concurrency--gpu)
- [Inference Parameters](#inference-parameters)
- [Observability](#observability)
- [Docker Compose (Production)](#docker-compose-production)
- [Repository Layout](#repository-layout)
- [Legacy Code](#legacy-code)
- [License](#license)

---

## Architecture

```
Client (Browser / cURL / SDK)
        │
        ▼
┌──────────────────────────────┐
│  SwiftDub API  (port 8000)   │
│  POST /dub       single      │
│  POST /dub/batch multi       │
│  GET  /jobs/{id} status      │
│  GET  /health    health      │
│  GET  /models    model list  │
│  GET  /metrics   prometheus  │
└──────────┬───────────────────┘
           │ model=latentsync | musetalk
           │ sync=true → asyncio.Semaphore (GPU gate)
           │ sync=false → Celery queue (Redis)
           ▼
┌──────────────────────────────┐
│  Model subprocess runners    │
│  • LatentSync 1.5 (diffusion)│
│  • MuseTalk v1.5 (real-time) │
└──────────┬───────────────────┘
           │
     ┌─────┴─────┐
     │ MongoDB   │ job history
     │ Redis     │ Celery broker + result cache*
     │ Prometheus│ /metrics
     │ Grafana   │ dashboards (:3000)
     └───────────┘
     * cache skipped when GPU/RAM ≥ 95%
```

---

## Quick Start

### 1. Install dependencies

```bash
cd SwiftDub
pip install -r requirements.txt
# GPU PyTorch (if not already installed):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. Download model weights

```bash
# Both models (recommended)
python scripts/download/models.py

# LatentSync only
python scripts/download/models.py --latentsync-version 1.5

# MuseTalk only
python scripts/download/models.py --musetalk-only
```

Downloads:
- `models/vendor/latentsync/` — LatentSync inference snapshot (gitignored, **no `.git` remote**)
- `models/weights/latentsync/latentsync_unet.pt` — 5.1 GB U-Net checkpoint
- `models/weights/latentsync/whisper/tiny.pt` — Whisper encoder weights
- `models/vendor/musetalk/` — MuseTalk inference snapshot (gitignored, **no `.git` remote**)
- `models/weights/musetalk/musetalkV15/unet.pth` — MuseTalk v1.5 U-Net

**Important:** Only the SwiftDub repo is tracked in git. Vendor snapshots under `models/vendor/` are downloaded runtime assets — never commit or push from them. SwiftDub-owned fixes live in `scripts/vendor/patches.py`.

Verify:
```bash
ls models/weights/latentsync/latentsync_unet.pt
ls models/weights/musetalk/musetalkV15/unet.pth
curl http://localhost:8000/models   # after server start
```

### 3. Download sample data (optional)

```bash
python scripts/download/datasets.py --split benchmark
# → data/samples/benchmark/sample{1,2,3}_{video,audio}.mp4/wav
```

### 4. Start the server

```bash
# Dev mode — no MongoDB/Redis required (in-memory job store)
DISABLE_DB=true python -m src.main

# Production — start Redis + MongoDB first, then:
python -m src.main
```

| URL | Description |
|-----|-------------|
| http://localhost:8000 | Web UI (drag-and-drop dubbing) |
| http://localhost:8000/docs | Swagger API docs |
| http://localhost:8000/health | Health check |
| http://localhost:8000/metrics | Prometheus metrics |

---

## API Reference

### `POST /dub` — Single video dubbing

```bash
curl -X POST http://localhost:8000/dub \
  -F "video=@sample.mp4" \
  -F "audio=@dub_audio.wav" \
  -F "model=latentsync" \
  -F "inference_steps=20" \
  -F "guidance_scale=1.5" \
  -F "seed=1247" \
  -F "sync=true"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `video` | file | required | Input video (MP4/AVI/MOV) |
| `audio` | file | optional | Dub audio; extracted from video if absent |
| `model` | string | `latentsync` | `latentsync` or `musetalk` |
| `inference_steps` | int | 20 | LatentSync only: DDIM steps (5–50) |
| `guidance_scale` | float | 1.5 | LatentSync only: CFG scale (0.5–5.0) |
| `seed` | int | 1247 | LatentSync only: reproducibility (-1 = random) |
| `sync` | bool | true | Wait for result vs. return job_id immediately |

**Response:**
```json
{
  "job_id": "abc123",
  "status": "completed",
  "output_video": "/app/data/outputs/abc123.mp4",
  "download_url": "/jobs/abc123/download"
}
```

### `POST /dub/batch` — Multi-video dubbing (up to 10)

```bash
curl -X POST http://localhost:8000/dub/batch \
  -F "videos=@vid1.mp4" \
  -F "videos=@vid2.mp4" \
  -F "audio=@dub.wav" \
  -F "inference_steps=20" \
  -F "sync=false"
```

Jobs run concurrently up to `MAX_CONCURRENT_JOBS`, gated by GPU memory headroom. Poll each job via `GET /jobs/{job_id}`.

**Batch pairing modes** (`POST /dub/batch`):

| Videos | Audios | Result |
|--------|--------|--------|
| N | 0 | Extract audio from each video |
| N | 1 | Shared audio for all videos |
| N | N | Paired 1:1 |
| 1 | M | Same video dubbed with each audio |

Uploads are saved as safe filenames (`video.mp4`, `audio_0.mp3`) so spaces in original names never break ffmpeg.

### `GET /jobs` — List recent jobs

### `GET /jobs/{job_id}` — Job status

### `GET /jobs/{job_id}/download` — Download output MP4

### `GET /health`

```json
{
  "status": "healthy",
  "cuda_available": true,
  "ffmpeg_available": true,
  "models": { "latentsync": true, "musetalk": true },
  "max_concurrent_jobs": 2,
  "active_jobs": 0,
  "memory": { "gpu_pct": 12.5, "ram_pct": 45.0 },
  "cache_enabled": true
}
```

### `GET /models` — Available models and readiness

### `GET /metrics` — Prometheus metrics

---

## Inference Pipeline

Implemented in `src/core/pipeline.py`:

```
dub_video(job_id, video_path, audio_path, model)
    │
    ├── extract_audio()          if no separate audio provided
    ├── cache lookup (Redis)     skipped when GPU/RAM ≥ MEMORY_CACHE_DISABLE_PCT
    ├── asyncio.Semaphore        GPU concurrency gate
    │
    └── run_latentsync() | run_musetalk()   subprocess in thread executor
```

Each job runs inference in an isolated subprocess so GPU memory is fully released between jobs.

### Model comparison

| | LatentSync 1.5 | MuseTalk v1.5 |
|--|----------------|---------------|
| Quality | Higher (diffusion) | Good (real-time) |
| Speed (L4, ~30s clip) | ~90–120 s | ~180 s* |
| VRAM per job | ~8–12 GB | ~6–8 GB |
| Extra params | `inference_steps`, `guidance_scale`, `seed` | `bbox_shift` (env default) |

\* MuseTalk is faster on longer clips due to batching; short clips include fixed model-load overhead.

---

## Concurrency & GPU

`MAX_CONCURRENT_JOBS` in `.env` controls simultaneous GPU inference jobs.

| GPU | VRAM | Recommended `MAX_CONCURRENT_JOBS` |
|-----|------|----------------------------------|
| L4 | 23 GB | 1–2 |
| RTX 4090 | 24 GB | 1–2 |
| A100 40 GB | 40 GB | 3 |
| L40S 46 GB | 46 GB | 3–4 |
| H100 80 GB | 80 GB | 6 |

**Memory-aware cache:** Redis result caching is automatically disabled when GPU or system RAM usage exceeds `MEMORY_CACHE_DISABLE_PCT` (default 95%) to avoid OOM under load.

**Typical inference time (L4 GPU):**

| Model | ~30 s clip |
|-------|-----------|
| LatentSync (20 steps) | ~90–120 s |
| MuseTalk v1.5 | ~180 s |

---

## Inference Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `model` | `latentsync` | latentsync, musetalk | Model selection |
| `inference_steps` | 20 | 5–50 | LatentSync: more steps = better quality |
| `guidance_scale` | 1.5 | 0.5–5.0 | LatentSync: higher = stronger lip sync |
| `seed` | 1247 | any | LatentSync reproducibility (-1 = random) |
| `enable_deepcache` | false | bool | LatentSync ~30–40% speedup |
| `bbox_shift` | 0 | int | MuseTalk face crop vertical shift |

Configure defaults in `.env` or override per request.

---

## Observability

### Structured inference logs — `logs/inference.jsonl`

One JSON line per inference event (Grafana Loki / jq compatible):

```json
{
  "ts": "2026-07-03T00:00:00.000Z",
  "job_id": "abc123",
  "model": "latentsync",
  "status": "success",
  "inference_s": 81.7,
  "wall_s": 81.7,
  "steps": 20,
  "guidance_scale": 1.5,
  "video_in": ".../uploads/abc123/source_video.mp4",
  "audio_in": ".../uploads/abc123/dub_audio.wav",
  "output": ".../outputs/abc123.mp4"
}
```

```bash
# Average inference time
cat logs/inference.jsonl | jq -r 'select(.status=="success") | .inference_s' \
  | awk '{s+=$1;n++} END{print s/n"s avg"}'

# Failed jobs
cat logs/inference.jsonl | jq 'select(.status=="error")'
```

### Prometheus metrics — `GET /metrics`

| Metric | Type | Description |
|--------|------|-------------|
| `swiftdub_requests_total{model,status}` | Counter | Total dub requests |
| `swiftdub_inference_duration_seconds{model}` | Histogram | Latency p50/p90/p99 |
| `swiftdub_active_jobs` | Gauge | Current concurrent jobs |
| `swiftdub_batch_size` | Histogram | Batch size distribution |
| `swiftdub_gpu_memory_pct` | Gauge | GPU memory utilization |
| `swiftdub_ram_memory_pct` | Gauge | System RAM utilization |

### Grafana dashboard

Start the observability stack (API must be running on host port 8000 for Prometheus to scrape):

```bash
docker compose -f deploy/docker-compose.yml up -d prometheus grafana flower
python -m src.main   # in another terminal
```

**Lightning / cloud studio:** `localhost` in your laptop browser does **not** reach the studio machine. Use proxy URLs from:

```bash
curl http://localhost:8000/services | python -m json.tool
```

Or open the **Observability** panel at the bottom of the SwiftDub UI — links use the Lightning port proxy automatically.

| Service | Port | Local URL | Notes |
|---------|------|-----------|-------|
| **Grafana** | 3000 | http://localhost:3000 | admin / `swiftdub` |
| **Prometheus** | **9092** | http://localhost:9092 | **not 9090** |
| **Flower (Celery)** | 5556 | http://localhost:5556 | requires `worker` service |
| **API metrics** | 8000 | http://localhost:8000/metrics | raw Prometheus text |

Dashboard: **Dashboards → SwiftDub → SwiftDub — Dubbing Metrics**

### Disable MuseTalk

MuseTalk can be hidden until quality matches LatentSync:

```bash
# .env
MUSETALK_DISABLED=true
```

```yaml
# configs/models.yaml
musetalk:
  disabled: true
```

When disabled: MuseTalk is omitted from `/models`, frontend dropdown, and `POST /dub` returns HTTP 503 with a clear message if `model=musetalk` is requested.

---

## Docker Compose (Production)

```bash
docker compose -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml logs -f api worker
```

| Service | Port | Description |
|---------|------|-------------|
| api | 8000 | FastAPI + web UI |
| worker | — | Celery GPU worker |
| flower | 5555 | Celery task monitor |
| mongodb | 27017 | Job persistence |
| redis | 6379 | Celery broker + result cache |
| prometheus | 9092 | Metrics scraping (host port; 9090/9091 may be in use) |
| grafana | 3000 | Dashboards (admin / `swiftdub`) |

Model weights are mounted from `../models` — download them before starting:
```bash
python scripts/download/models.py
docker compose -f deploy/docker-compose.yml up -d
```

`legacy/` is excluded from the Docker build context via `.dockerignore`.

---

## Repository Layout

```
SwiftDub/
├── configs/
│   └── models.yaml              # LatentSync + MuseTalk config
├── src/
│   ├── main.py                  # Entrypoint: uvicorn
│   ├── config.py                # Pydantic settings (.env)
│   ├── api/                     # FastAPI routes + frontend mount
│   ├── core/
│   │   ├── latentsync.py        # LatentSync subprocess runner
│   │   ├── musetalk.py          # MuseTalk subprocess runner
│   │   ├── registry.py          # Model availability registry
│   │   └── pipeline.py          # Dubbing pipeline + GPU semaphore
│   ├── cache/                   # Redis cache with memory gating
│   ├── db/                      # MongoDB (in-memory fallback)
│   ├── workers/                 # Celery async tasks
│   ├── metrics/                 # Prometheus registry
│   ├── logging_/                # JSONL inference logger
│   └── utils/                   # ffmpeg, video, memory helpers
├── frontend/
│   └── index.html               # Web UI
├── scripts/
│   ├── download/
│   │   ├── models.py            # Download LatentSync + MuseTalk weights
│   │   └── datasets.py          # Download benchmark samples
│   └── setup/
│       └── install_miniconda_linux.sh
├── deploy/
│   ├── docker-compose.yml       # Full production stack
│   ├── prometheus.yml
│   └── grafana/
├── models/
│   ├── vendor/latentsync/       # Downloaded inference snapshot (gitignored, no .git)
│   ├── vendor/musetalk/         # Downloaded inference snapshot (gitignored, no .git)
│   └── weights/                 # Downloaded checkpoints only (gitignored)
├── data/
│   ├── samples/benchmark/       # Test clips
│   ├── uploads/                 # Job inputs (gitignored)
│   └── outputs/                 # Finished videos (gitignored)
├── logs/
│   ├── inference.jsonl          # Structured inference log
│   └── app.log                  # Rotating application log
├── legacy/                      # Old multi-model code (not used at runtime)
├── requirements.txt
├── Dockerfile
└── README.md
```
---

## License

SwiftDub application code: MIT.

LatentSync model: [Apache 2.0](https://github.com/bytedance/LatentSync) (ByteDance).

MuseTalk model: review [upstream license](https://github.com/TMElyralab/MuseTalk) before commercial use.
