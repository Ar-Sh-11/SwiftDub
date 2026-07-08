# SwiftDub

Production-grade AI video dubbing service powered by **LatentSync 1.5** (diffusion) and **MuseTalk v1.5** (GAN). Lip-syncs any video to any audio source — including video-to-video dubbing and automated dub correction.

---

## Features


| Feature                      | Description                                                                  |
| ---------------------------- | ---------------------------------------------------------------------------- |
| **LatentSync 1.5**           | DDIM diffusion-based lip sync — highest quality                              |
| **MuseTalk v1.5**            | Real-time GAN-based lip sync — fastest                                       |
| **Video-to-Video dubbing**   | Upload any video file as the audio source — audio is extracted automatically |
| **Dub Correct**              | Re-sync a badly dubbed video to its own audio track in one click             |
| **Batch processing**         | Up to 10 concurrent jobs with flexible video/audio pairing                   |
| **Dynamic GPU distribution** | Jobs are spread across all available CUDA devices automatically              |
| **Redis result cache**       | Skip inference for identical input pairs                                     |
| **Celery workers**           | Async background processing with Flower monitoring                           |
| **Prometheus + Grafana**     | Full observability stack                                                     |
| **MongoDB persistence**      | Job history with in-memory fallback                                          |


---



## Architecture

```
models/
  weights/          ← Only this is needed at runtime (docker volume)
    latentsync/
      latentsync_unet.pt
      whisper/tiny.pt
    musetalk/
      musetalkV15/unet.pth
      whisper/
      dwpose/
      face-parse-bisent/

src/
  models/
    latentsync/     ← First-party LatentSync inference pipeline
      infer.py      ← Subprocess entry point
      pipeline.py   ← LipsyncPipeline (extracted from ByteDance/LatentSync)
      whisper/      ← Audio2Feature encoder
      utils/        ← Image processing, affine transforms, media I/O
    musetalk/       ← First-party MuseTalk inference pipeline
      infer.py      ← Subprocess entry point
      utils/        ← Audio processor, face blending
  core/
    latentsync.py   ← Subprocess runner (calls src.models.latentsync.infer)
    musetalk.py     ← Subprocess runner (calls src.models.musetalk.infer)
    pipeline.py     ← Async dub orchestration + GPU semaphore
  api/              ← FastAPI routes, schemas, deps
  utils/
    gpu_alloc.py    ← Dynamic multi-GPU allocator
    ffmpeg.py       ← ffmpeg helpers
    audio.py        ← Audio I/O
    video.py        ← Frame I/O
  tests/            ← CPU-compatible test suite
```

> **No vendor dependency at runtime.** `models/vendor/` is only needed during initial setup to copy model architecture files. After running `scripts/vendor/migrate_models.py`, you can delete the vendor directory — only `models/weights/` is required in production.

---



## Quick Start



### 1. Install dependencies

```bash
pip install -r requirements.txt
```



### 2. Download weights

```bash
python scripts/download/models.py --model latentsync   # ~5 GB
python scripts/download/models.py --model musetalk     # ~5 GB (optional)
```



### 3. Run the server

```bash
# CPU mode (no inference — for development/testing)
DISABLE_DB=true python -m src.main

# GPU mode
python -m src.main
```

Visit `http://localhost:8000` to open the web UI.

---



## Docker (production)

```bash
# Start full stack (API + worker + MongoDB + Redis + Grafana + Prometheus)
cd deploy
docker compose up -d

# Volumes: weights must be downloaded locally first
# models/weights/ is mounted as a volume — no weights in the image
```

---



## API Reference



### `POST /dub` — Single video dubbing

```bash
# Basic: video + audio
curl -X POST http://localhost:8000/dub \
  -F "video=@speaker.mp4" \
  -F "audio=@new_speech.wav" \
  -F "model=latentsync" \
  -F "inference_steps=20" \
  -F "guidance_scale=1.5"

# Video-to-video: supply another video as the audio source
curl -X POST http://localhost:8000/dub \
  -F "video=@target_speaker.mp4" \
  -F "audio=@source_video.mp4"       # audio extracted automatically

# Dub correct: re-sync a badly dubbed video to its own audio
curl -X POST http://localhost:8000/dub \
  -F "video=@bad_dub.mp4" \
  -F "dub_correct=true"
```

**Parameters:**


| Field              | Type   | Default      | Description                                                              |
| ------------------ | ------ | ------------ | ------------------------------------------------------------------------ |
| `video`            | file   | required     | Input video for lip-sync                                                 |
| `audio`            | file   | optional     | Audio source — WAV/MP3/AAC **or any video file** (audio track extracted) |
| `model`            | string | `latentsync` | `latentsync` or `musetalk`                                               |
| `dub_correct`      | bool   | `false`      | Re-sync using video's own audio; ignores `audio` field                   |
| `inference_steps`  | int    | 20           | DDIM steps (1–50); more = higher quality                                 |
| `guidance_scale`   | float  | 1.5          | Classifier-free guidance (0.5–5.0)                                       |
| `seed`             | int    | 1247         | Random seed; -1 for random                                               |
| `enable_deepcache` | bool   | false        | ~2× speedup with slight quality trade-off                                |
| `bbox_shift`       | int    | 0            | MuseTalk bbox vertical shift                                             |
| `sync`             | bool   | true         | Wait for result (true) or queue async (false)                            |


---



### `POST /dub/batch` — Batch dubbing (up to 10 jobs)

```bash
# Pairing modes:
# N videos + N audios → 1:1 pairs
# N videos + 1 audio  → shared audio for all
# 1 video + M audios  → same video with each audio
# N videos + 0 audios → extract audio from each video
# dub_correct=true    → re-sync each video to its own audio

curl -X POST http://localhost:8000/dub/batch \
  -F "videos=@vid1.mp4" -F "videos=@vid2.mp4" \
  -F "audios=@audio1.wav" -F "audios=@audio2.wav" \
  -F "model=latentsync"
```

---



### `GET /jobs` — List recent jobs

```bash
curl http://localhost:8000/jobs?limit=20
```



### `GET /jobs/{id}/download` — Download output video

```bash
curl -O http://localhost:8000/jobs/{job_id}/download
```

---



## GPU Distribution

When multiple GPUs are available, SwiftDub automatically distributes jobs across them using a least-loaded allocator.

```env
# .env
GPU_IDS=0,1,2              # Explicit GPU pool; leave empty for auto-detect
GPU_MAX_JOBS_PER_DEVICE=1  # Max simultaneous jobs per GPU
MAX_CONCURRENT_JOBS=3      # Global job concurrency
```

The `GPUAllocator` (`src/utils/gpu_alloc.py`) selects the GPU with the fewest active jobs for each new inference request. Each subprocess receives `--gpu_id N` so CUDA is initialized on the correct device.

---



## Testing

```bash
# CPU-compatible tests (no GPU or weights required)
python -m pytest src/tests/ -v

# Specific test files
python -m pytest src/tests/test_api.py -v
python -m pytest src/tests/test_pipeline_unit.py -v
python -m pytest src/tests/test_media_utils.py -v   # requires ffmpeg
```

---



## Practical Edge Cases


| Scenario                        | How to handle                                                                     |
| ------------------------------- | --------------------------------------------------------------------------------- |
| Audio longer than video         | Pipeline automatically loops/pings the video to match audio length                |
| Video submitted as audio source | `audio` field transparently accepts video files; WAV is extracted via ffmpeg      |
| Badly dubbed video              | Use `dub_correct=true` — extracts original audio and re-runs lip-sync             |
| Same video for multiple voices  | Batch with 1 video + N audio files                                                |
| No audio in video               | Use an explicit `audio` file; omitting `audio` when video has no track will error |
| Multi-GPU server                | Set `GPU_IDS=0,1,2` in `.env`; jobs auto-distribute                               |
| Face not detected               | Inference fails with `RuntimeError`; check video resolution and face visibility   |
| DeepCache artifacts             | Disable `enable_deepcache` for higher quality at the cost of ~2× inference time   |


---



## Vendor Independence

The `models/vendor/` directory contains LatentSync and MuseTalk repository snapshots used during initial development. All inference-relevant code has been extracted into `src/models/`.

**Runtime dependency:** only `models/weights/` (the `.pt`/`.pth` checkpoint files).

**Migration helper** (run once, then delete vendor):

```bash
python scripts/vendor/migrate_models.py   # copies model architecture files to src/
```

After migration, `models/vendor/` can be removed:

```bash
rm -rf models/vendor/
```

The inference runners fall back to vendor if `src/models/*/models/unet.py` is not present, so the transition is gradual.

---



## Observability


| Service    | URL                          | Purpose              |
| ---------- | ---------------------------- | -------------------- |
| API        | `http://localhost:8000`      | Web UI + REST        |
| Swagger    | `http://localhost:8000/docs` | Interactive API docs |
| Grafana    | `http://localhost:3000`      | Dashboards           |
| Prometheus | `http://localhost:9090`      | Metrics scraping     |
| Flower     | `http://localhost:5555`      | Celery task monitor  |


---



## License

SwiftDub application code: MIT.  
LatentSync model code adapted from [ByteDance/LatentSync](https://github.com/bytedance/LatentSync) (Apache 2.0).  
MuseTalk model code adapted from [TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk) (Apache 2.0).