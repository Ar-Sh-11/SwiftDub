#!/usr/bin/env pwsh
# SwiftDub Windows environment setup

$ErrorActionPreference = "Stop"
$EnvName = "swiftdub"

Write-Host "=========================================="
Write-Host "        SwiftDub Environment Setup"
Write-Host "=========================================="

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python is not installed or not on PATH."
}

if (-not (Test-Path $EnvName)) {
    python -m venv $EnvName
}

& "$EnvName\Scripts\Activate.ps1"
python -m pip install --upgrade pip setuptools wheel

$hasGpu = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($hasGpu) {
    Write-Host "Installing CUDA PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
} else {
    Write-Host "Installing CPU PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

pip install --prefer-binary -r requirements.txt

python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

Write-Host ""
Write-Host "Next steps:"
Write-Host "  python scripts/download_sample_data.py"
Write-Host "  python scripts/download_models.py --model wav2lip"
Write-Host "  python scripts/run_pipeline_demo.py --model wav2lip"
Write-Host "  python -m src.main"
