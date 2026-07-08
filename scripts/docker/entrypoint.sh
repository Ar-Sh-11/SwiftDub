#!/bin/bash
set -euo pipefail

cd /app

echo "==> SwiftDub entrypoint — checking model weights..."

if [ ! -f "models/weights/latentsync/latentsync_unet.pt" ] || \
   [ ! -f "models/weights/latentsync/whisper/tiny.pt" ]; then
  echo "==> Weights missing — downloading LatentSync (first run only)..."
  python scripts/download/models.py --model latentsync
else
  echo "==> LatentSync weights found (mounted volume or cached)"
fi

# InsightFace buffalo_l for face detection
if [ ! -d "models/weights/latentsync/insightface/models/buffalo_l" ]; then
  echo "==> InsightFace models missing — downloading..."
  python scripts/download/models.py --model insightface
fi

export INSIGHTFACE_HOME="${INSIGHTFACE_HOME:-/app/models/weights/latentsync/insightface}"
mkdir -p data/uploads data/outputs data/temp logs

echo "==> Starting: $*"
exec "$@"
