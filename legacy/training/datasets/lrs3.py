"""LRS3 dataset loader (Oxford TED Talks, 438h).

Directory structure after download:
  data/lrs3/
    trainval/
      <speaker_id>/
        <utterance_id>.mp4
        <utterance_id>.txt
    test/
      ...

Dataset shape per sample:
  frames: [T, 3, 224, 224]  float32 in [0,1]
  mel:    [80, T*16]         float32 log-mel spectrogram
"""

from __future__ import annotations

from pathlib import Path

import cv2
import librosa
import numpy as np
import torch

from training.datasets.base import LipSyncDataset, Sample


class LRS3Dataset(LipSyncDataset):
    FRAME_SIZE = 224

    def _load_index(self) -> None:
        split_dir = self.root / ("trainval" if self.split == "train" else "test")
        if not split_dir.exists():
            raise FileNotFoundError(f"LRS3 {self.split} dir not found: {split_dir}")
        for mp4 in sorted(split_dir.rglob("*.mp4")):
            speaker = mp4.parent.name
            self._samples.append(
                Sample(video_path=mp4, audio_path=mp4, speaker_id=speaker)
            )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self._samples[idx]
        frames = self._load_frames(s.video_path)
        mel = self._load_mel(s.audio_path, n_frames=len(frames))
        masked = frames.clone()
        masked[:, :, self.FRAME_SIZE // 2:, :] = 0.0   # mask lower half (mouth region)
        return {"frames": masked, "mel": mel, "gt": frames}

    def _load_frames(self, path: Path) -> torch.Tensor:
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            f = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (self.FRAME_SIZE, self.FRAME_SIZE))
            frames.append(f)
        cap.release()
        arr = np.stack(frames).astype(np.float32) / 255.0   # [T,H,W,3]
        return torch.from_numpy(arr).permute(0, 3, 1, 2)    # [T,3,H,W]

    def _load_mel(self, path: Path, n_frames: int) -> torch.Tensor:
        wav, _ = librosa.load(str(path), sr=self.SAMPLE_RATE, mono=True)
        mel = librosa.feature.melspectrogram(y=wav, sr=self.SAMPLE_RATE,
                                              n_mels=self.MEL_BINS, hop_length=160)
        mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        target_len = n_frames * self.MEL_STEP
        if mel.shape[1] < target_len:
            mel = np.pad(mel, ((0, 0), (0, target_len - mel.shape[1])))
        else:
            mel = mel[:, :target_len]
        return torch.from_numpy(mel).unsqueeze(0)   # [1, 80, T*16]
