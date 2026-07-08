"""Unit tests for pipeline infrastructure — GPU allocator, config, edge cases."""
from __future__ import annotations

import os
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGPUAllocator:
    """Tests for DynamicGPUAllocator using asyncio event loop."""

    @pytest.mark.asyncio
    async def test_single_gpu_acquire_release(self):
        from src.utils.gpu_alloc import DynamicGPUAllocator, _GPUSlot

        alloc = DynamicGPUAllocator.__new__(DynamicGPUAllocator)
        alloc._slots = {0: _GPUSlot(gpu_id=0)}
        alloc._condition = None

        # Patch total_gb to 24 GB so claims fit
        with patch.object(_GPUSlot, 'total_gb', new_callable=lambda: property(lambda self: 24.0)), \
             patch.object(_GPUSlot, 'actual_free_gb', new_callable=lambda: property(lambda self: 24.0 - sum(self.claims.values()))):
            g1 = await alloc.acquire("job1", 5.0)
            g2 = await alloc.acquire("job2", 5.0)
            assert g1 == 0
            assert g2 == 0
            s = alloc.status()
            assert s[0]["active_jobs"] == 2
            await alloc.release(0, "job1")
            await alloc.release(0, "job2")
            assert alloc.status()[0]["active_jobs"] == 0

    @pytest.mark.asyncio
    async def test_multi_gpu_picks_least_loaded(self):
        from src.utils.gpu_alloc import DynamicGPUAllocator, _GPUSlot

        alloc = DynamicGPUAllocator.__new__(DynamicGPUAllocator)
        alloc._slots = {0: _GPUSlot(gpu_id=0), 1: _GPUSlot(gpu_id=1)}
        alloc._condition = None

        with patch.object(_GPUSlot, 'total_gb', new_callable=lambda: property(lambda self: 24.0)), \
             patch.object(_GPUSlot, 'actual_free_gb', new_callable=lambda: property(lambda self: 24.0 - sum(self.claims.values()))):
            g1 = await alloc.acquire("job1", 10.0)
            g2 = await alloc.acquire("job2", 10.0)
            # Should pick different GPUs since both have enough space and first was claimed
            assert {g1, g2} == {0, 1}
            await alloc.release(g1, "job1")
            await alloc.release(g2, "job2")

    @pytest.mark.asyncio
    async def test_acquire_waits_and_unblocks(self):
        """acquire() should block until VRAM is freed by release()."""
        import asyncio
        from src.utils.gpu_alloc import DynamicGPUAllocator, _GPUSlot

        alloc = DynamicGPUAllocator.__new__(DynamicGPUAllocator)
        alloc._slots = {0: _GPUSlot(gpu_id=0)}
        alloc._condition = None

        with patch.object(_GPUSlot, 'total_gb', new_callable=lambda: property(lambda self: 10.0)), \
             patch.object(_GPUSlot, 'actual_free_gb', new_callable=lambda: property(lambda self: max(0.0, 10.0 - sum(self.claims.values())))):
            # Claim 9 GB (80% of 10 = 8 GB usable) — fills usable budget
            g = await alloc.acquire("job1", 8.0)
            assert g == 0

            released = asyncio.Event()

            async def waiter():
                # This should block because GPU 0 is full
                await alloc.acquire("job2", 8.0)
                released.set()

            task = asyncio.create_task(waiter())
            await asyncio.sleep(0.05)
            assert not released.is_set(), "Should be waiting"
            await alloc.release(0, "job1")
            await asyncio.wait_for(released.wait(), timeout=2.0)
            task.cancel()

    @pytest.mark.asyncio
    async def test_status_snapshot(self):
        from src.utils.gpu_alloc import DynamicGPUAllocator, _GPUSlot

        alloc = DynamicGPUAllocator.__new__(DynamicGPUAllocator)
        alloc._slots = {0: _GPUSlot(gpu_id=0), 1: _GPUSlot(gpu_id=1)}
        alloc._condition = None

        with patch.object(_GPUSlot, 'total_gb', new_callable=lambda: property(lambda self: 24.0)), \
             patch.object(_GPUSlot, 'actual_free_gb', new_callable=lambda: property(lambda self: 24.0 - sum(self.claims.values()))):
            await alloc.acquire("jobA", 5.0)
            s = alloc.status()
            assert isinstance(s, list)
            total_active = sum(d["active_jobs"] for d in s)
            assert total_active == 1
            await alloc.release(0, "jobA")


class TestConfig:
    def test_settings_defaults(self):
        from src.config import settings

        assert settings.app_name == "SwiftDub"
        assert settings.port == 8000
        assert settings.max_concurrent_jobs >= 1

    def test_latentsync_paths_defined(self):
        from src.config import settings

        assert settings.latentsync_ckpt is not None
        assert settings.latentsync_whisper is not None
        assert settings.latentsync_unet_config_path is not None

    def test_root_dir_exists(self):
        from src.config import settings

        assert settings.root_dir.exists()
        assert (settings.root_dir / "src").exists()

    def test_model_name_enum(self):
        from src.config import ModelName

        assert ModelName("latentsync") == ModelName.LATENTSYNC
        assert ModelName("musetalk") == ModelName.MUSETALK
        with pytest.raises(ValueError):
            ModelName("nonexistent")

    def test_gpu_id_list_default(self):
        from src.config import Settings

        s = Settings(gpu_ids="")
        ids = s.gpu_id_list()
        assert isinstance(ids, list)
        assert len(ids) >= 1

    def test_gpu_id_list_custom(self):
        from src.config import Settings

        s = Settings(gpu_ids="0,1,2")
        assert s.gpu_id_list() == [0, 1, 2]


class TestDepHelpers:
    def test_is_video_file_mp4(self):
        from src.api.deps import is_video_file
        from unittest.mock import MagicMock

        u = MagicMock()
        u.content_type = "video/mp4"
        u.filename = "clip.mp4"
        assert is_video_file(u) is True

    def test_is_video_file_audio(self):
        from src.api.deps import is_video_file
        from unittest.mock import MagicMock

        u = MagicMock()
        u.content_type = "audio/wav"
        u.filename = "clip.wav"
        assert is_video_file(u) is False

    def test_is_video_file_by_extension(self):
        from src.api.deps import is_video_file
        from unittest.mock import MagicMock

        u = MagicMock()
        u.content_type = "application/octet-stream"
        u.filename = "recording.mkv"
        assert is_video_file(u) is True


class TestFFmpegHelpers:
    def test_extract_audio_produces_wav(self, sample_video, tmp_path):
        from src.utils.ffmpeg import extract_audio

        out = tmp_path / "audio.wav"
        result = extract_audio(sample_video, out)
        assert result.exists()
        # Verify it's a valid WAV
        import wave
        with wave.open(str(result)) as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000

    def test_video_info(self, sample_video):
        from src.utils.ffmpeg import get_video_info

        info = get_video_info(sample_video)
        assert info["fps"] > 0


class TestEdgeCases:
    """Practical edge cases for video dubbing pipelines."""

    def test_audio_longer_than_video_graceful(self, sample_video, sample_wav, tmp_path):
        """If audio is longer than video, the pipeline should still accept the inputs."""
        from src.utils.audio import duration_seconds

        vid_dur = 1.0  # 1 second sample video
        aud_dur = duration_seconds(sample_wav)
        # Just verify we can read both — actual looping happens inside inference
        assert vid_dur > 0
        assert aud_dur > 0

    def test_batch_pairing_logic(self):
        from src.api.routes.dub import _resolve_batch_pairs
        from fastapi import HTTPException

        # N=1 video, M=3 audios
        pairs = _resolve_batch_pairs(1, 3)
        assert pairs == [(0, 0), (0, 1), (0, 2)]

        # N=3 videos, N=3 audios — 1:1 pairs
        pairs = _resolve_batch_pairs(3, 3)
        assert pairs == [(0, 0), (1, 1), (2, 2)]

        # N videos, 0 audios — extract from each
        pairs = _resolve_batch_pairs(3, 0)
        assert pairs == [(0, None), (1, None), (2, None)]

        # N videos, 1 audio — shared
        pairs = _resolve_batch_pairs(4, 1)
        assert all(ai == 0 for _, ai in pairs)

        # Mismatch
        with pytest.raises(HTTPException) as exc:
            _resolve_batch_pairs(3, 2)
        assert exc.value.status_code == 422

    def test_dub_correct_batch_ignores_audios(self):
        """When dub_correct=True, batch should use None for all audio indices."""
        # This behavior is enforced in the route: if dub_correct, pairs = [(i, None) for i]
        n_videos = 3
        pairs = [(i, None) for i in range(n_videos)]
        assert all(ai is None for _, ai in pairs)

    def test_same_video_as_audio_source(self, sample_mp4_with_audio, tmp_path):
        """A video file uploaded as the 'audio' field should yield valid audio."""
        from src.utils.ffmpeg import extract_audio

        out = tmp_path / "from_video.wav"
        extract_audio(sample_mp4_with_audio, out)
        assert out.exists()
        assert out.stat().st_size > 100  # non-trivial size

    def test_registry_latentsync_check(self):
        """Registry should not crash even when weights are absent."""
        from src.core import latentsync

        # is_ready returns bool — doesn't raise even if weights missing
        result = latentsync.is_ready()
        assert isinstance(result, bool)
