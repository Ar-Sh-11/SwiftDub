# SwiftDub

Production-grade AI video dubbing service powered by **LatentSync 1.5** (diffusion-based lip-sync). Lip-syncs any video to any audio source — including video-to-video dubbing and automated dub correction. MuseTalk support is wired but currently disabled (enable via `MUSETALK_DISABLED=false`).

---

## Features

| Feature                        | Description                                                                              |
| ------------------------------ | ---------------------------------------------------------------------------------------- |
| **LatentSync 1.5**             | DDIM diffusion lip-sync — state-of-the-art quality on NVIDIA L4 / A100                  |
| **Video-to-Video dubbing**     | Upload any video file as the audio source — WAV track is extracted automatically         |
| **Dub Correct**                | Re-sync a badly dubbed video to its own audio in one click                               |
| **Batch processing**           | Up to 20 concurrent jobs with flexible video/audio pairing and per-pair `dub_correct`    |
| **Dynamic GPU allocator**      | VRAM-aware job distribution across all CUDA devices; 20% overhead buffer always reserved |
| **Redis output cache**         | Identical input pairs return instantly without re-running inference                      |
| **Celery async workers**       | Non-blocking job queue with Flower monitoring dashboard                                  |
| **Prometheus + Grafana**       | Full observability — latency, GPU usage, job counts, error rates                         |
| **MongoDB persistence**        | Paginated job history (search by name, filter by status, sort by any field)              |
| **No vendor dependency**       | All inference code lives in `src/models/` — only `models/weights/` needed at runtime     |

---

## Architecture

```
SwiftDub/
├── src/
│   ├── api/               ← FastAPI routes + schemas + deps
│   │   └── routes/        ← dub.py  jobs.py  health.py  gpu.py  models.py
│   ├── core/
│   │   ├── pipeline.py    ← Async dub orchestration + DynamicGPUAllocator
│   │   ├── latentsync.py  ← Subprocess runner for LatentSync
│   │   └── batch_queue.py ← Async batch job scheduler
│   ├── models/
│   │   └── latentsync/    ← First-party LatentSync inference (no vendor copy)
│   │       ├── infer.py   ← Entry point (called by subprocess)
│   │       ├── pipeline.py← LipsyncPipeline (UNet3D + Whisper)
│   │       ├── models/    ← UNet3D, attention, motion modules
│   │       └── whisper/   ← Audio encoder
│   ├── db/                ← MongoDB (Motor async) client + repos
│   ├── cache/             ← Redis output cache
│   ├── utils/
│   │   ├── gpu_alloc.py   ← DynamicGPUAllocator (VRAM-aware)
│   │   └── ffmpeg.py      ← FFmpeg helpers
│   ├── workers/tasks.py   ← Celery task (persistent per-worker event loop)
│   ├── metrics/           ← Prometheus counters + histograms
│   └── tests/             ← Full integration test suite (12 tests)
├── models/
│   └── weights/           ← Model checkpoints (mounted as Docker volume)
│       └── latentsync/
│           ├── latentsync_unet.pt
│           ├── whisper/tiny.pt
│           └── insightface/models/buffalo_l/
├── deploy/
│   ├── docker-compose.yml
│   ├── prometheus.yml
│   └── grafana/
├── scripts/
│   └── download/models.py ← Weight downloader (latentsync + insightface)
├── frontend/index.html    ← Single-file web UI (no build step)
├── Dockerfile
└── requirements.txt
```

---

## Quick Start (local / venv)

### 1. Create virtualenv

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install PyTorch

Choose the CUDA suffix that matches your driver (L4 with CUDA 12.8 → `cu128`; CUDA 12.4 → `cu124`):

```bash
# CUDA 12.8 (L4 / H100 on latest drivers)
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.4 (Docker base image default)
# pip install torch==2.6.0 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

> Docker uses `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` as the base image (torch 2.6 pre-installed). No extra torch install needed inside the container.

### 3. Install application dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
# Edit .env if needed (defaults work for local dev with MongoDB/Redis running)
```

Key variables:

| Variable             | Default                        | Description                                      |
| -------------------- | ------------------------------ | ------------------------------------------------ |
| `MONGODB_URL`        | `mongodb://localhost:27017`    | MongoDB connection string                        |
| `REDIS_URL`          | `redis://localhost:6379/0`     | Redis for output cache                           |
| `CELERY_BROKER`      | `redis://localhost:6379/1`     | Celery broker                                    |
| `CELERY_BACKEND`     | `redis://localhost:6379/2`     | Celery result backend                            |
| `GPU_IDS`            | `0`                            | Comma-separated CUDA device IDs (e.g. `0,1`)     |
| `MUSETALK_DISABLED`  | `true`                         | Set to `false` to enable MuseTalk                |
| `INSIGHTFACE_HOME`   | `models/weights/latentsync/insightface` | InsightFace model cache directory       |
| `TZ`                 | `Asia/Kolkata`                 | Timezone for all log timestamps                  |

### 5. Download model weights (first run only)

```bash
python scripts/download/models.py --model latentsync    # ~5 GB
python scripts/download/models.py --model insightface   # ~300 MB
```

### 6. Start services

```bash
# Start MongoDB and Redis (or use Docker Compose for just infra):
docker run -d --name mongo -p 27017:27017 mongo:7.0
docker run -d --name redis -p 6379:6379 redis:7.2-alpine

# Start FastAPI server
export INSIGHTFACE_HOME=models/weights/latentsync/insightface TZ=Asia/Kolkata MUSETALK_DISABLED=true
python -m src.main

# (Optional) Start Celery worker for async jobs
celery -A src.workers.tasks.celery_app worker --loglevel=info --concurrency=1
```

Open `http://localhost:8000` in your browser.

---

## Docker (production)

### Prerequisites

- Docker Engine ≥ 24 with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- `docker compose` (v2)
- Model weights already downloaded to `models/weights/` (mounted as volume)

### Build and run (clean slate)

```bash
# Download weights first (if not already present)
python scripts/download/models.py --model latentsync
python scripts/download/models.py --model insightface

# Remove all previous containers/orphans
docker compose -f deploy/docker-compose.yml down --remove-orphans -v

# Build from scratch (no cache)
docker compose -f deploy/docker-compose.yml build --no-cache

# Start full stack
docker compose -f deploy/docker-compose.yml up -d

# Follow API logs
docker compose -f deploy/docker-compose.yml logs -f api
```

### Services

| Service       | URL                           | Credentials   |
| ------------- | ----------------------------- | ------------- |
| Web UI + API  | `http://localhost:8000`       | —             |
| Swagger docs  | `http://localhost:8000/docs`  | —             |
| Grafana       | `http://localhost:3000`       | anonymous (Admin role) |
| Prometheus    | `http://localhost:9092`       | —             |
| Flower        | `http://localhost:5556`       | —             |

> **Grafana** is pre-configured with anonymous Admin access (`GF_AUTH_ANONYMOUS_ENABLED=true`). Dashboards are provisioned from `deploy/grafana/`.

### Volume mounts (no weights baked into image)

```yaml
volumes:
  - ../models/weights:/app/models/weights   # model checkpoints
  - ../data:/app/data                        # uploads / outputs
  - ../logs:/app/logs                        # log files
```

---

## API Reference

### `POST /dub` — Single video dubbing

```bash
# Video + audio file
curl -X POST http://localhost:8000/dub \
  -F "video=@speaker.mp4" \
  -F "audio=@new_speech.wav" \
  -F "model=latentsync" \
  -F "inference_steps=20" \
  -F "sync=true"

# Video-to-video: supply another video as audio source (WAV extracted automatically)
curl -X POST http://localhost:8000/dub \
  -F "video=@target.mp4" \
  -F "audio=@source_video.mp4"

# Dub Correct: re-sync a badly dubbed video to its own original audio
curl -X POST http://localhost:8000/dub \
  -F "video=@badly_dubbed.mp4" \
  -F "dub_correct=true" \
  -F "sync=true"

# Async (queue and poll)
curl -X POST http://localhost:8000/dub \
  -F "video=@speaker.mp4" -F "audio=@speech.wav" -F "sync=false"
# Returns: {"job_id": "...", "status": "queued"}
```

| Field              | Type   | Default      | Description                                                              |
| ------------------ | ------ | ------------ | ------------------------------------------------------------------------ |
| `video`            | file   | required     | Input video for lip-sync                                                 |
| `audio`            | file   | optional     | Audio source — WAV/MP3/AAC **or any video** (audio track extracted)      |
| `model`            | string | `latentsync` | `latentsync` (MuseTalk disabled)                                         |
| `dub_correct`      | bool   | `false`      | Re-sync using video's own audio; `audio` field is ignored when true      |
| `inference_steps`  | int    | 20           | DDIM steps (1–50); 5 is fast/test, 20 is production quality             |
| `guidance_scale`   | float  | 1.5          | Classifier-free guidance (0.5–5.0)                                       |
| `seed`             | int    | 1247         | Reproducibility seed                                                     |
| `enable_deepcache` | bool   | `false`      | ~2× speed-up with minor quality trade-off                                |
| `bbox_shift`       | int    | 0            | Vertical crop shift for face ROI (MuseTalk only)                         |
| `sync`             | bool   | `true`       | `true` = wait for result; `false` = return job_id immediately            |

---

### `POST /dub/batch` — Batch dubbing (up to 20 jobs)

```bash
# 2 videos + 2 audios → paired 1:1
curl -X POST http://localhost:8000/dub/batch \
  -F "videos=@vid1.mp4" -F "videos=@vid2.mp4" \
  -F "audios=@audio1.wav" -F "audios=@audio2.wav" \
  -F "model=latentsync"

# Per-pair dub_correct: first job corrects, second dubs with audio
curl -X POST http://localhost:8000/dub/batch \
  -F "videos=@bad_dub.mp4" -F "videos=@clean.mp4" \
  -F "audios=@new_voice.wav" \
  -F "dub_correct_flags=[true,false]"
```

Pairing rules when `dub_correct` is not used:

| Input                    | Result                                   |
| ------------------------ | ---------------------------------------- |
| N videos + 0 audios      | Audio extracted from each video          |
| N videos + 1 audio       | Shared audio applied to all videos       |
| N videos + N audios      | 1:1 pairs                                |
| 1 video + M audios       | Same video dubbed with each audio        |

---

### `GET /jobs` — Paginated job list

```bash
# Page 1, 10 per page
curl "http://localhost:8000/jobs?page=1&per_page=10"

# Filter by status + search by filename
curl "http://localhost:8000/jobs?status=completed&search=speaker.mp4&sort_by=elapsed_s&order=desc"

# Filter by batch ID
curl "http://localhost:8000/jobs?batch_id=<batch_id>"
```

Query params: `page`, `per_page`, `search`, `status` (`queued|processing|completed|failed`), `sort_by` (`created_at|elapsed_s|status|model`), `order` (`asc|desc`), `batch_id`.

---

### `GET /jobs/{job_id}` — Job detail

```bash
curl http://localhost:8000/jobs/<job_id>
```

---

### `GET /jobs/{job_id}/download` — Download output video

```bash
curl -O http://localhost:8000/jobs/<job_id>/download
```

---

### `GET /gpu/status` — GPU VRAM breakdown

```bash
curl http://localhost:8000/gpu/status
```

Returns per-device VRAM usage, overhead allocation, usable budget, active job count.

---

### `GET /health` — Health check

```bash
curl http://localhost:8000/health
```

---

### `GET /models` — Available models

```bash
curl http://localhost:8000/models
```

---

### `GET /metrics` — Prometheus metrics

```bash
curl http://localhost:8000/metrics
```

---

### `GET /services` — Service URL map

```bash
curl http://localhost:8000/services
```

Returns URLs for Grafana, Prometheus, Flower.

---

## Dynamic GPU Distribution

`DynamicGPUAllocator` (`src/utils/gpu_alloc.py`) manages VRAM-aware concurrency:

- **20% overhead reserved** — never touches the last 20% of each GPU's VRAM
- **Concurrent small jobs** — multiple small jobs fit on one GPU if VRAM allows
- **Queue-and-wait** — jobs block asynchronously until VRAM is available
- **Multi-GPU** — set `GPU_IDS=0,1,2` in `.env`; jobs go to the least-loaded device

```bash
# Check live GPU allocation
curl http://localhost:8000/gpu/status
```

---

## Testing

```bash
# Start infrastructure first
docker run -d --name mongo -p 27017:27017 mongo:7.0
docker run -d --name redis -p 6379:6379 redis:7.2-alpine
export INSIGHTFACE_HOME=models/weights/latentsync/insightface TZ=Asia/Kolkata MUSETALK_DISABLED=true
python -m src.main &
celery -A src.workers.tasks.celery_app worker --concurrency=1 &

# Non-GPU tests (API, health, pagination)
python -m pytest src/tests/test_stack_local.py -m "not gpu" -v

# All tests including GPU inference (requires L4 / ≥16 GB VRAM)
RUN_GPU_TESTS=1 python -m pytest src/tests/test_stack_local.py -v

# Unit tests (no server / GPU needed)
python -m pytest src/tests/test_api.py src/tests/test_pipeline_unit.py -v
```

---

## Practical Edge Cases

| Scenario                          | How to handle                                                                       |
| --------------------------------- | ----------------------------------------------------------------------------------- |
| Audio longer than video           | Pipeline automatically loops/trims the video to match audio length                  |
| Video file as audio source        | `audio` field accepts video files; WAV is extracted automatically via FFmpeg        |
| Badly dubbed video                | Use `dub_correct=true` — extracts original audio and re-runs lip-sync              |
| Same video, multiple voice tracks | Batch: 1 video + N audio files                                                      |
| No audio track in video           | Provide an explicit `audio` file; omitting audio when video has no track will error |
| Face not detected                 | Inference fails with `RuntimeError`; ensure face is clearly visible, ≥240p          |
| DeepCache artifacts               | Disable `enable_deepcache` for higher quality at cost of ~2× inference time         |
| Multi-GPU server                  | Set `GPU_IDS=0,1,2` in `.env`; jobs auto-distribute by available VRAM              |

---

## Observability

Prometheus scrapes `http://api:8000/metrics` every 10 s. Grafana dashboards are provisioned automatically from `deploy/grafana/`.

Custom metrics:

| Metric                           | Type      | Description                         |
| -------------------------------- | --------- | ----------------------------------- |
| `swiftdub_jobs_total`            | Counter   | Total jobs by status + model        |
| `swiftdub_inference_duration_seconds` | Histogram | End-to-end inference latency   |
| `swiftdub_batch_size`            | Histogram | Jobs per batch request              |
| `swiftdub_active_jobs`           | Gauge     | Currently running jobs              |

---

## License

SwiftDub application code: MIT.  
LatentSync model code adapted from [ByteDance/LatentSync](https://github.com/bytedance/LatentSync) (Apache 2.0).  
MuseTalk model code adapted from [TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk) (Apache 2.0).
