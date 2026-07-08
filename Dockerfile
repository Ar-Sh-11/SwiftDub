# SwiftDub — Production Docker image (LatentSync 1.5)
# Base: PyTorch 2.6 + CUDA 12.4 (L4 / A100 / L40S / H100 compatible)
# transformers>=5.x requires torch>=2.4; this base satisfies that requirement.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Prevent apt/tzdata interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata

# System deps (tzdata must be non-interactive — set ENV above before apt install)
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get update && apt-get install -y --no-install-recommends \
    tzdata git ffmpeg libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps (torch already in base image — do not reinstall)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source + configs
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY .env.example ./.env

RUN chmod +x scripts/docker/entrypoint.sh \
    && mkdir -p data/uploads data/outputs data/temp data/samples logs models/weights

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV INSIGHTFACE_HOME=/app/models/weights/latentsync/insightface

EXPOSE 8000

ENTRYPOINT ["/app/scripts/docker/entrypoint.sh"]
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
