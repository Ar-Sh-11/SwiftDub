#!/usr/bin/env pwsh
# Install Miniconda on Windows, create SwiftDub conda environments.
# Run as: PowerShell -ExecutionPolicy Bypass -File .\scripts\setup\install_miniconda_windows.ps1

param(
    [string]$InstallDir = "$env:USERPROFILE\Miniconda3",
    [switch]$SkipMiniconda,
    [switch]$GPU,
    [string]$CudaVersion = "cu121"    # e.g. cu118, cu121, cu124
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path "$PSScriptRoot\..\..").Path

function Info($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "[OK] $msg" -ForegroundColor Green }

# ── 1. Install Miniconda ──────────────────────────────────────────────────────
if (-not $SkipMiniconda) {
    $Installer = "$env:TEMP\miniconda_installer.exe"
    $Url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"

    Info "Downloading Miniconda (Windows 64-bit) from $Url"
    Invoke-WebRequest -Uri $Url -OutFile $Installer -UseBasicParsing

    Info "Installing Miniconda to $InstallDir"
    Start-Process -FilePath $Installer -ArgumentList `
        "/S", "/D=$InstallDir", "/RegisterPython=0", "/AddToPath=1" `
        -Wait -NoNewWindow
    Remove-Item $Installer -Force
    Ok "Miniconda installed"
} else {
    Info "Skipping Miniconda install (--SkipMiniconda)"
}

# Ensure conda is on PATH for this session
$CondaExe = Join-Path $InstallDir "Scripts\conda.exe"
if (-not (Test-Path $CondaExe)) {
    throw "conda not found at $CondaExe. Check InstallDir."
}
& $CondaExe init powershell | Out-Null
$env:PATH = "$InstallDir;$InstallDir\Scripts;$InstallDir\Library\bin;" + $env:PATH

# ── 2. Create main env ───────────────────────────────────────────────────────
$EnvYml = Join-Path $RepoRoot "envs\main.yml"
Info "Creating conda env 'swiftdub' from $EnvYml"
& $CondaExe env create -f $EnvYml --force
Ok "swiftdub env created"

# ── 3. Install GPU PyTorch if requested ─────────────────────────────────────
if ($GPU) {
    $PipExe = Join-Path $InstallDir "envs\swiftdub\Scripts\pip.exe"
    Info "Installing CUDA PyTorch ($CudaVersion) into swiftdub"
    & $PipExe install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 `
        --index-url "https://download.pytorch.org/whl/$CudaVersion" --force-reinstall
    Ok "GPU PyTorch installed"
}

# ── 4. Create MuseTalk env ───────────────────────────────────────────────────
$MuseYml = Join-Path $RepoRoot "envs\musetalk.yml"
Info "Creating conda env 'swiftdub-musetalk'"
& $CondaExe env create -f $MuseYml --force

$MusePip = Join-Path $InstallDir "envs\swiftdub-musetalk\Scripts\pip.exe"
if ($GPU) {
    & $MusePip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 `
        --index-url "https://download.pytorch.org/whl/$CudaVersion" --force-reinstall
}
# Install mmcv with mim
$MusePython = Join-Path $InstallDir "envs\swiftdub-musetalk\python.exe"
& $MusePython -m mim install "mmcv==2.1.0" "mmdet==3.2.0" "mmpose==1.3.1"
Ok "swiftdub-musetalk env created"

Info @"

Setup complete.

Activate main env:
  conda activate swiftdub

Next steps:
  python scripts/download/models.py --model wav2lip
  python scripts/download/datasets.py --split benchmark
  python -m src.main
"@
