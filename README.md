# SwiftDub

Production video dubbing service powered by **LatentSync 1.5** (ByteDance, CVPR 2025). Upload a source video and dub audio — get back a lip-synced MP4 where the speaker's mouth matches the new audio, with identity and background preserved.

Includes a web UI, REST API, batch processing, structured inference logs, and a Prometheus + Grafana observability stack.

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
│  GET  /metrics   prometheus  │
└──────────┬───────────────────┘
           │ sync=true → asyncio.Semaphore (GPU gate)
           │ sync=false → Celery queue (Redis)
           ▼
┌──────────────────────────────┐
│  LatentSync 1.5 subprocess   │
│  models/repos/latentsync     │
│  • DDIM diffusion (20 steps) │
│  • Whisper-tiny audio enc    │
│  • SD-VAE decoder            │
└──────────┬───────────────────┘
           │
     ┌─────┴─────┐
     │ MongoDB   │ job history
     │ Redis     │ Celery broker
     │ Prometheus│ /metrics
     │ Grafana   │ dashboards (:3000)
     └───────────┘
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

### 2. Download LatentSync weights

```bash
python scripts/download/models.py --latentsync-version 1.5
```

Downloads:
- `models/repos/latentsync/` — ByteDance/LatentSync repo
- `models/weights/latentsync/latentsync_unet.pt` — 1.5 GB U-Net checkpoint
- `models/repos/latentsync/checkpoints/whisper/tiny.pt` — Whisper encoder

Verify:
```bash
ls models/repos/latentsync/checkpoints/whisper/tiny.pt
ls models/weights/latentsync/latentsync_unet.pt
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
  -F "inference_steps=20" \
  -F "guidance_scale=1.5" \
  -F "seed=1247" \
  -F "sync=true"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `video` | file | required | Input video (MP4/AVI/MOV) |
| `audio` | file | optional | Dub audio; extracted from video if absent |
| `inference_steps` | int | 20 | DDIM diffusion steps (5–50) |
| `guidance_scale` | float | 1.5 | Classifier-free guidance (0.5–5.0) |
| `seed` | int | 1247 | Reproducibility seed (-1 = random) |
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

Jobs run concurrently up to `MAX_CONCURRENT_JOBS`. Poll each job via `GET /jobs/{job_id}`.

### `GET /jobs` — List recent jobs

### `GET /jobs/{job_id}` — Job status

### `GET /jobs/{job_id}/download` — Download output MP4

### `GET /health`

```json
{
  "status": "healthy",
  "cuda_available": true,
  "ffmpeg_available": true,
  "model_ready": true,
  "max_concurrent_jobs": 2,
  "active_jobs": 0
}
```

### `GET /metrics` — Prometheus metrics

---

## Inference Pipeline

Implemented in `src/core/pipeline.py`:

```
dub_video(job_id, video_path, audio_path)
    │
    ├── extract_audio()          if no separate audio provided
    ├── asyncio.Semaphore        GPU concurrency gate
    │
    └── run_latentsync()         subprocess in thread executor
            └── python scripts/inference.py
                    --unet_config_path configs/unet/stage2.yaml
                    --inference_ckpt_path models/weights/latentsync/latentsync_unet.pt
                    --inference_steps 20
                    --guidance_scale 1.5
```

Each job runs LatentSync in an isolated subprocess so GPU memory is fully released between jobs.

---

## Concurrency & GPU

`MAX_CONCURRENT_JOBS` in `.env` controls simultaneous GPU inference jobs.

| GPU | VRAM | Recommended `MAX_CONCURRENT_JOBS` |
|-----|------|----------------------------------|
| RTX 4090 | 24 GB | 1–2 |
| A100 40 GB | 40 GB | 3 |
| L40S 46 GB | 46 GB | 3–4 |
| H100 80 GB | 80 GB | 6 |

Each LatentSync job uses ~8–12 GB VRAM.

**Typical inference time (L40S, 20 steps):**

| Video length | Inference time |
|-------------|----------------|
| 10 s | ~45 s |
| 30 s | ~90–120 s |
| 60 s | ~180–240 s |

---

## Inference Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `inference_steps` | 20 | 5–50 | More steps = better quality, slower |
| `guidance_scale` | 1.5 | 0.5–5.0 | Higher = stronger lip sync |
| `seed` | 1247 | any | Reproducibility (-1 = random) |
| `enable_deepcache` | false | bool | ~30–40% speedup, minor quality trade-off |

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
| `latentsync_requests_total{model,status}` | Counter | Total dub requests |
| `latentsync_inference_duration_seconds{model}` | Histogram | Latency p50/p90/p99 |
| `latentsync_active_jobs` | Gauge | Current concurrent jobs |
| `latentsync_batch_size` | Histogram | Batch size distribution |

### Grafana dashboard

After starting Docker Compose, open http://localhost:3000 (admin / `swiftdub`). The **SwiftDub — LatentSync Dubbing** dashboard is auto-provisioned.

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
| redis | 6379 | Celery broker |
| prometheus | 9090 | Metrics scraping |
| grafana | 3000 | Dashboards |

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
│   └── models.yaml              # LatentSync model config
├── src/
│   ├── main.py                  # Entrypoint: uvicorn
│   ├── config.py                # Pydantic settings (.env)
│   ├── api/                     # FastAPI routes + frontend mount
│   ├── core/
│   │   ├── latentsync.py        # Subprocess inference runner
│   │   └── pipeline.py          # Dubbing pipeline + GPU semaphore
│   ├── db/                      # MongoDB (in-memory fallback)
│   ├── workers/                 # Celery async tasks
│   ├── metrics/                 # Prometheus registry
│   ├── logging_/                # JSONL inference logger
│   └── utils/                   # ffmpeg, video, audio helpers
├── frontend/
│   └── index.html               # Web UI
├── scripts/
│   ├── download/
│   │   ├── models.py            # Download LatentSync weights
│   │   └── datasets.py          # Download benchmark samples
│   └── setup/
│       └── install_miniconda_linux.sh
├── deploy/
│   ├── docker-compose.yml       # Full production stack
│   ├── prometheus.yml
│   └── grafana/
├── models/
│   ├── repos/latentsync/        # Cloned LatentSync repo (gitignored)
│   └── weights/latentsync/      # Downloaded checkpoint (gitignored)
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

## Legacy Code

The `legacy/` folder preserves the original multi-model SwiftDub codebase (Wav2Lip, VideoReTalking, MuseTalk, SadTalker, training pipeline, benchmarks). It is **not loaded at runtime** and is excluded from Docker builds.

To exclude from git on your next push, uncomment `# legacy/` in `.gitignore`.

---

## License

SwiftDub application code: MIT.

LatentSync model: [Apache 2.0](https://github.com/bytedance/LatentSync) (ByteDance). Review upstream license before commercial use.
