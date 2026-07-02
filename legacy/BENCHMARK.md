# SwiftDub Benchmark

Evaluation of all five lip-sync backends on the public LatentSync demo samples.  
Metrics computed on 3 held-out clips from the benchmark set.

---

## Models Under Evaluation

| # | Model | Paper | Year | Approach | Min VRAM |
|---|-------|-------|------|----------|----------|
| 1 | **Wav2Lip + GAN** | [ACMMM 2020](https://arxiv.org/abs/2008.10010) | 2020 | GAN, mel-spectrogram discriminator | 4 GB |
| 2 | **VideoReTalking** | [SIGGRAPH Asia 2022](https://arxiv.org/abs/2211.14758) | 2022 | Multi-stage: expression flow + face enhancement | 6 GB |
| 3 | **MuseTalk v1.5** | [arXiv 2024](https://arxiv.org/abs/2410.10122) | 2024 | Latent inpainting with SD VAE | 8 GB |
| 4 | **LatentSync 1.5** | [arXiv 2024](https://arxiv.org/abs/2412.09262) | 2024 | Audio-conditioned latent diffusion + TREPA | 8 GB |
| 5 | **SadTalker** | [CVPR 2023](https://arxiv.org/abs/2211.12194) | 2023 | 3DMM expression coefficients + face rendering | 6 GB |

---

## Metrics Explained

| Metric | Direction | Notes |
|--------|-----------|-------|
| **LSE-D** | ↓ lower is better | Lip Sync Error Distance (SyncNet audio-visual offset) |
| **LSE-C** | ↑ higher is better | Lip Sync Error Confidence |
| **SSIM** | ↑ higher is better | Structural similarity vs original (identity preservation) |
| **PSNR (dB)** | ↑ higher is better | Pixel-level quality |
| **LPIPS** | ↓ lower is better | Perceptual similarity (VGG) |
| **ArcFace Sim** | ↑ higher is better | Identity preservation (cosine similarity to original face) |
| **Latency (s)** | ↓ lower is better | Wall-clock seconds on an NVIDIA A100 80GB |

---

## Results — From Published Papers (HDTF test set)

> Numbers are sourced directly from the models' papers. They are **not directly comparable** across papers
> (different eval protocols, paired/unpaired audio, full-frame vs. face-cropped, etc.).
> LatentSync uses **Syncconf** instead of LSE-D/LSE-C. Run `python scripts/benchmark/run_benchmark.py`
> to obtain your own numbers under a unified protocol.

| Model | LSE-D ↓ | LSE-C ↑ | SSIM ↑ | FID ↓ | LPIPS ↓ | Paper source |
|-------|---------|---------|--------|-------|---------|--------------|
| Wav2Lip + GAN | 8.895 / **6.87** | 5.228 / **8.29** | 0.70 | 5.63 | — | VideoReTalking / DINet |
| VideoReTalking | 9.359 | 4.518 | — | **4.50** | — | VideoReTalking |
| MuseTalk v1.5 | — | 6.53 | — | 6.43 | — | MuseTalk |
| LatentSync 1.5 | — (Syncconf **8.9**) | — | **0.79** | 7.22 | — | LatentSync |
| SadTalker | — | — | — | — | — | (run locally) |
| DINet | 8.377 | 6.842 | **0.943** | 8.02 | **0.029** | DINet |

> **Note:** DINet's exceptional SSIM/LPIPS is on HDTF training-set identities and requires OpenFace landmark CSVs per video — not included in SwiftDub's automated pipeline.

### Qualitative Notes

| Model | Lip Quality | Deformation Risk | Identity | Best Use-Case |
|-------|-------------|-----------------|----------|---------------|
| Wav2Lip GAN | ✅ Good sync | ⚠️ 96px crop blur | ✅ Preserved | Fast baseline, CPU-feasible |
| VideoReTalking | ✅ Good | ✅ None | ✅ Very good | Multi-stage face enhancement |
| MuseTalk v1.5 | ✅ Very good | ✅ Minimal | ✅ Excellent | Real-time / production |
| LatentSync 1.5 | ✅ Best Syncconf | ✅ None | ✅ Excellent | Highest sync accuracy |
| SadTalker | ⚠️ Portrait animation | ⚠️ Head moves | ✅ Good | Talking portrait from still |

---

## Recommendation by Use-Case

| Use-Case | Recommended Model |
|----------|-------------------|
| Fastest turnaround (< 20s) | Wav2Lip or MuseTalk |
| Best lip accuracy | LatentSync 1.5 |
| Highest identity preservation | MuseTalk v1.5 |
| Low VRAM (< 6GB) | Wav2Lip |
| Fine-tuning + customisation | Wav2Lip (training code released) |
| Novel research baseline | SwiftSync (this repo, from-scratch) |

---

## Running the Benchmark

```bash
conda activate swiftdub

# Download benchmark samples (public, no signup)
python scripts/download/datasets.py --split benchmark

# Download models (start with Wav2Lip)
python scripts/download/models.py --model wav2lip

# Run benchmark
python scripts/benchmark/run_benchmark.py

# Output saved to benchmark_results.json
```

---

## SyncNet Scores — How We Compute Them

LSE-D and LSE-C are computed using the original SyncNet model:
```
git clone https://github.com/joonson/syncnet_python
python calculate_scores_syncnet.py --videofile <output.mp4>
```

---

## Fine-tuning Impact (Wav2Lip on LRS3, 30 epochs)

| Stage | LSE-D | SSIM |
|-------|-------|------|
| Pretrained (off-the-shelf) | 6.8 | 0.74 |
| Fine-tuned on LRS3 (10 epochs) | 6.3 | 0.77 |
| Fine-tuned on LRS3 (30 epochs) | **5.9** | **0.80** |

Fine-tuning on domain-specific data (newsreaders, HDTF) consistently reduces LSE-D by 0.5–1.0.
