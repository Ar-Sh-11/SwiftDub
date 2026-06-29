# SwiftDub

Open-source HTTP lip-sync API for English dialogue videos. SwiftDub accepts a video plus isolated speech audio and returns a lip-synced video while preserving identity, background, and the original audio stream.

Supported backends:

| Model | VRAM (approx.) | Notes |
|-------|----------------|-------|
| **Wav2Lip** | 4 GB | Fast baseline, good sync |
| **MuseTalk v1.5** | 8 GB | Real-time latent inpainting |
| **LatentSync 1.5** | 8 GB | Diffusion-based, strong quality |

## Quick start (local)

### 1. Environment

**Windows (PowerShell):**

```powershell
.\setup_env.ps1
.\swiftdub\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
bash setup_env.ps1   # or manual venv + pip install -r requirements.txt
source swiftdub/bin/activate
```

Install PyTorch with CUDA if you have a GPU. FFmpeg must be on `PATH`.

### 2. Download sample data

Public talking-head samples from the [LatentSync demo assets](https://github.com/bytedance/LatentSync/tree/main/assets):

```bash
python scripts/download_sample_data.py
```

This saves `data/samples/source_video.mp4` and `data/samples/dub_audio.wav`.

### 3. Download models

```bash
# Start with Wav2Lip (smallest / fastest)
python scripts/download_models.py --model wav2lip

# All three backends
python scripts/download_models.py --model all --latentsync-version 1.5
```

Weights and cloned repos live under `models/weights/` and `models/repos/`.

### 4. Run pipeline demo

```bash
python scripts/run_pipeline_demo.py --model wav2lip
python scripts/run_pipeline_demo.py --all-models
```

Output videos are written to `data/outputs/`.

### 5. Start the API

```bash
python -m src.main
```

Open `http://localhost:8000/docs` for interactive Swagger UI.

## API

### `GET /health`

Returns CUDA/FFmpeg status and which models are installed.

### `GET /models`

Lists available lip-sync backends.

### `POST /predict`

Multipart form upload:

| Field | Type | Description |
|-------|------|-------------|
| `video` | file | Input video with visible speaker |
| `audio` | file (optional) | Dub dialogue WAV/MP3; uses video audio if omitted |
| `model` | string | `wav2lip`, `musetalk`, or `latentsync` |
| `sync` | bool | `true` (default) waits for result; `false` runs async |

**Example (curl):**

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "video=@data/samples/source_video.mp4" \
  -F "audio=@data/samples/dub_audio.wav" \
  -F "model=wav2lip"
```

### `GET /jobs/{job_id}`

Poll async job status.

### `GET /download/{job_id}`

Download completed output MP4.

## Pipeline

```
Video + Audio
     │
     ▼
Face detection (MediaPipe)
     │
     ▼
Left-most fully visible face selection
     │
     ▼
Lip-sync model (Wav2Lip / MuseTalk / LatentSync)
     │
     ▼
Mux original audio (unchanged)
     │
     ▼
Output MP4
```

## Colab GPU demo (hosted testing)

1. Upload this repo to Google Drive or push to GitHub.
2. Open [`notebooks/SwiftDub_Colab.ipynb`](notebooks/SwiftDub_Colab.ipynb) in Colab.
3. Set runtime to **GPU**.
4. Run all cells — downloads models, runs demo, exposes API via ngrok.
5. Optional: add `NGROK_AUTHTOKEN` in Colab secrets for a stable public URL.

## Docker

```bash
docker build -t swiftdub .
docker run --gpus all -p 8000:8000 swiftdub
```

## Project layout

```
SwiftDub/
├── src/
│   ├── main.py              # FastAPI entrypoint
│   ├── config.py
│   ├── routes/              # /health, /predict
│   ├── services/            # pipeline + model wrappers
│   ├── schemas/
│   └── utils/               # face, ffmpeg, video
├── scripts/
│   ├── download_models.py
│   ├── download_sample_data.py
│   └── run_pipeline_demo.py
├── models/
│   ├── weights/             # downloaded checkpoints
│   └── repos/               # cloned upstream repos
├── data/
│   ├── samples/
│   ├── uploads/
│   └── outputs/
└── notebooks/
    └── SwiftDub_Colab.ipynb
```

## Success criteria mapping

| Requirement | Implementation |
|-------------|----------------|
| Convincing lip movement | Pluggable Wav2Lip / MuseTalk / LatentSync backends |
| No warping elsewhere | Face-box constrained inference; original audio remuxed |
| Audio unchanged | Output video re-muxed with input audio stream |
| Left-most speaker | MediaPipe detection + left-most frontal face heuristic |
| English dialogue | Whisper-based models (LatentSync/MuseTalk); Wav2Lip mel features |

## License

Upstream models have their own licenses (Wav2Lip, MuseTalk, LatentSync). Review before commercial use.
