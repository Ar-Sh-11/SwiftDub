"""VoxCeleb2 dataset loader (2442h, celebrity talking heads).

Directory structure after download:
  data/voxceleb2/
    dev/
      mp4/
        <speaker_id>/
          <video_id>/
            <utterance_id>.mp4
    test/
      mp4/
        ...

Dataset shape per sample:
  frames: [T, 3, 224, 224]  float32 in [0,1]
  mel:    [80, T*16]
"""

from __future__ import annotations

from pathlib import Path

import cv2
import librosa
import numpy as np
import torch

from training.datasets.base import LipSyncDataset, Sample


class VoxCeleb2Dataset(LipSyncDataset):
    FRAME_SIZE = 224

    def _load_index(self) -> None:
        split_name = "dev" if self.split == "train" else "test"
        split_dir = self.root / split_name / "mp4"
        if not split_dir.exists():
            raise FileNotFoundError(f"VoxCeleb2 dir not found: {split_dir}")
        for mp4 in sorted(split_dir.rglob("*.mp4")):
            speaker = mp4.parts[-3]
            self._samples.append(Sample(video_path=mp4, audio_path=mp4, speaker_id=speaker))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self._samples[idx]
        frames = self._load_frames(s.video_path)
        mel = self._load_mel(s.audio_path, len(frames))
        masked = frames.clone()
        masked[:, :, self.FRAME_SIZE // 2:, :] = 0.0
        return {"frames": masked, "mel": mel, "gt": frames, "speaker": s.speaker_id}

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
        if not frames:
            frames = [np.zeros((self.FRAME_SIZE, self.FRAME_SIZE, 3), dtype=np.uint8)]
        arr = np.stack(frames).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(0, 3, 1, 2)

    def _load_mel(self, path: Path, n_frames: int) -> torch.Tensor:
        wav, _ = librosa.load(str(path), sr=self.SAMPLE_RATE, mono=True)
        mel = librosa.feature.melspectrogram(y=wav, sr=self.SAMPLE_RATE,
                                              n_mels=self.MEL_BINS, hop_length=160)
        mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        target = n_frames * self.MEL_STEP
        mel = mel[:, :target] if mel.shape[1] >= target else np.pad(mel, ((0,0),(0, target - mel.shape[1])))
        return torch.from_numpy(mel).unsqueeze(0)
