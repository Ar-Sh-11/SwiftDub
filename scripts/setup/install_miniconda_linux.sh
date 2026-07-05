#!/usr/bin/env bash
# Install Miniconda on Linux (RunPod / Ubuntu), create SwiftDub conda environments.
# Usage:  bash scripts/setup/install_miniconda_linux.sh [--gpu] [--cuda-ver cu121]

set -euo pipefail

INSTALL_DIR="${HOME}/miniconda3"
GPU=false
CUDA_VER="cu121"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

for arg in "$@"; do
  case $arg in
    --gpu)        GPU=true ;;
    --cuda-ver=*) CUDA_VER="${arg#*=}" ;;
    --skip-conda) SKIP_CONDA=true ;;
  esac
done

info()  { echo -e "\033[0;36m==> $*\033[0m"; }
ok()    { echo -e "\033[0;32m[OK] $*\033[0m"; }

# ── 1. Install Miniconda ──────────────────────────────────────────────────────
if [[ -z "${SKIP_CONDA:-}" ]]; then
  INSTALLER="/tmp/miniconda_installer.sh"
  info "Downloading Miniconda (Linux x86_64)"
  curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
       -o "$INSTALLER"
  bash "$INSTALLER" -b -u -p "$INSTALL_DIR"
  rm -f "$INSTALLER"
  ok "Miniconda installed at $INSTALL_DIR"
fi

export PATH="$INSTALL_DIR/bin:$PATH"
conda init bash 2>/dev/null || true
source "$INSTALL_DIR/etc/profile.d/conda.sh"

# ── 2. Create main env ───────────────────────────────────────────────────────
info "Creating conda env 'swiftdub'"
conda env create -f "$REPO_ROOT/envs/main.yml" --force
ok "swiftdub env created"

if [[ "$GPU" == "true" ]]; then
  info "Installing CUDA PyTorch ($CUDA_VER) into swiftdub"
  conda run -n swiftdub pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
      --index-url "https://download.pytorch.org/whl/$CUDA_VER" --force-reinstall
  ok "GPU PyTorch installed"
fi

# ── 3. Create MuseTalk env ───────────────────────────────────────────────────
info "Creating conda env 'swiftdub-musetalk'"
conda env create -f "$REPO_ROOT/envs/musetalk.yml" --force

if [[ "$GPU" == "true" ]]; then
  conda run -n swiftdub-musetalk pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
      --index-url "https://download.pytorch.org/whl/$CUDA_VER" --force-reinstall
fi
conda run -n swiftdub-musetalk mim install "mmcv==2.1.0" "mmdet==3.2.0" "mmpose==1.3.1"
ok "swiftdub-musetalk env created"

info "
Setup complete.

Activate env:
  conda activate swiftdub

Next:
  python scripts/download/models.py --model latentsync
  python scripts/download/datasets.py --split benchmark
  python -m src.main
"
