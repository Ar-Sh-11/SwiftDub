#!/usr/bin/env python3
"""SwiftSync training / fine-tuning launcher.

Usage (fine-tune Wav2Lip on LRS3):
  python training/train.py --mode finetune_wav2lip
         --data-root data/lrs3 --dataset lrs3
         --epochs 30 --batch 8 --lr 1e-4

Usage (pretrain SwiftSync from scratch):
  python training/train.py --mode pretrain_swiftsync
         --data-root data/lrs3 --dataset lrs3
         --epochs 100 --batch 4 --lr 3e-4

Usage (domain-finetune SwiftSync on HDTF):
  python training/train.py --mode finetune_swiftsync
         --checkpoint checkpoints/swiftsync_pretrain.pt
         --data-root data/hdtf --dataset hdtf
         --epochs 20 --lr 5e-5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


def get_dataset(name: str, root: Path, split: str, max_samples: int | None):
    if name == "lrs3":
        from training.datasets.lrs3 import LRS3Dataset
        return LRS3Dataset(root, split, max_samples)
    if name == "voxceleb2":
        from training.datasets.voxceleb2 import VoxCeleb2Dataset
        return VoxCeleb2Dataset(root, split, max_samples)
    raise ValueError(f"Unknown dataset: {name}")


def finetune_wav2lip(args: argparse.Namespace) -> None:
    """Fine-tune Wav2Lip on a custom dataset using perceptual + sync loss."""
    import sys
    sys.path.insert(0, str(ROOT / "models" / "repos" / "wav2lip"))

    from models import Wav2Lip  # type: ignore (Wav2Lip repo must be cloned)
    from training.losses.perceptual import CombinedLoss

    dataset = get_dataset(args.dataset, Path(args.data_root), "train", args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Wav2Lip().to(device)
    loss_fn = CombinedLoss().to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    ckpt_dir = Path("checkpoints/wav2lip_ft")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            mel    = batch["mel"].to(device)
            frames = batch["frames"].to(device)
            gt     = batch["gt"].to(device)

            pred = model(mel, frames)
            losses = loss_fn(pred, gt, mel)
            opt.zero_grad()
            losses["total"].backward()
            opt.step()
            total_loss += losses["total"].item()

        avg = total_loss / max(len(loader), 1)
        logger.info("Epoch {} | loss {:.4f}", epoch, avg)

        if epoch % 5 == 0:
            ckpt = ckpt_dir / f"wav2lip_ft_epoch{epoch:03d}.pth"
            torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt)
            logger.info("Saved {}", ckpt)


def pretrain_swiftsync(args: argparse.Namespace) -> None:
    """Train SwiftSync from scratch."""
    from training.models.swiftsync import SwiftSync
    from training.losses.perceptual import CombinedLoss

    dataset = get_dataset(args.dataset, Path(args.data_root), "train", args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SwiftSync().to(device)
    loss_fn = CombinedLoss().to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    ckpt_dir = Path("checkpoints/swiftsync")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            wav    = batch.get("waveform", torch.zeros(args.batch, 16000))
            video  = batch["gt"].to(device)
            mask   = batch["frames"].to(device)
            wav    = wav.to(device)

            pred   = model(wav, video, mask)
            losses = loss_fn(pred, video)
            opt.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += losses["total"].item()

        scheduler.step()
        avg = total_loss / max(len(loader), 1)
        logger.info("Epoch {} | loss {:.4f} | lr {:.2e}", epoch, avg, scheduler.get_last_lr()[0])

        if epoch % 10 == 0:
            ckpt = ckpt_dir / f"swiftsync_epoch{epoch:03d}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt)


MODES = {
    "finetune_wav2lip": finetune_wav2lip,
    "pretrain_swiftsync": pretrain_swiftsync,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=list(MODES), required=True)
    p.add_argument("--dataset", choices=["lrs3", "voxceleb2", "hdtf"], default="lrs3")
    p.add_argument("--data-root", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()

    logger.info("Mode: {} | Dataset: {} | Epochs: {} | Batch: {} | LR: {}", 
                args.mode, args.dataset, args.epochs, args.batch, args.lr)
    MODES[args.mode](args)


if __name__ == "__main__":
    main()
