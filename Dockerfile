# SwiftDub — Production Docker image (LatentSync 1.5)
# Base: PyTorch 2.1 + CUDA 12.1 (works on L40S / A100 / H100)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY configs/ ./configs/
COPY .env .

# Directories (volumes will override data/ and models/ at runtime)
RUN mkdir -p data/uploads data/outputs data/temp data/samples logs

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
