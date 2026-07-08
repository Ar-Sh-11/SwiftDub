"""FastAPI endpoint tests — CPU-compatible, no model inference executed."""
from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Prevent real inference in tests
os.environ.setdefault("DISABLE_DB", "true")
os.environ.setdefault("DISABLE_CACHE", "true")


@pytest.fixture(scope="module")
def client():
    from src.api.app import create_app

    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestHealthEndpoints:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_schema(self, client):
        d = client.get("/health").json()
        assert "status" in d
        assert "cuda_available" in d
        assert "ffmpeg_available" in d
        assert "models" in d

    def test_models_endpoint(self, client):
        r = client.get("/models")
        assert r.status_code == 200
        models = r.json()
        assert isinstance(models, list)
        assert any(m["name"] == "latentsync" for m in models)

    def test_services_endpoint(self, client):
        r = client.get("/services")
        assert r.status_code == 200
        assert "services" in r.json()

    def test_metrics_endpoint(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "swiftdub" in r.text or "process" in r.text

    def test_root_returns_html(self, client):
        r = client.get("/")
        # Returns HTML or 200 redirect
        assert r.status_code in (200, 301, 302)
        if r.status_code == 200:
            assert "SwiftDub" in r.text


class TestJobsEndpoints:
    def test_list_jobs_empty(self, client):
        r = client.get("/jobs")
        assert r.status_code == 200
        data = r.json()
        # Paginated response
        assert "jobs" in data
        assert "total" in data
        assert "page" in data
        assert "pages" in data
        assert isinstance(data["jobs"], list)

    def test_job_not_found(self, client):
        r = client.get("/jobs/nonexistent_job_id_12345")
        assert r.status_code == 404

    def test_download_not_found(self, client):
        r = client.get("/jobs/nonexistent_job_id/download")
        assert r.status_code == 404


class TestDubEndpointValidation:
    """Test request validation without running inference."""

    def _minimal_video_bytes(self) -> bytes:
        """Return minimal MP4 bytes (not a real video, just for upload testing)."""
        return b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41"

    def test_dub_requires_video(self, client):
        r = client.post("/dub", data={"model": "latentsync"})
        assert r.status_code == 422

    def test_dub_invalid_model(self, client):
        video_bytes = self._minimal_video_bytes()
        r = client.post(
            "/dub",
            data={"model": "nonexistent_model"},
            files={"video": ("test.mp4", video_bytes, "video/mp4")},
        )
        # Either 422 (validation) or 503 (model not available)
        assert r.status_code in (422, 503)

    def test_dub_steps_validation(self, client):
        video_bytes = self._minimal_video_bytes()
        # inference_steps=0 is below minimum
        r = client.post(
            "/dub",
            data={"model": "latentsync", "inference_steps": "0"},
            files={"video": ("test.mp4", video_bytes, "video/mp4")},
        )
        assert r.status_code == 422

    def test_batch_requires_video(self, client):
        r = client.post("/dub/batch", data={"model": "latentsync"})
        assert r.status_code == 422


class TestDubCorrectFlag:
    """dub_correct=true should be accepted and recognized."""

    def _video_bytes(self):
        return b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41"

    def test_dub_correct_field_accepted(self, client, monkeypatch):
        """dub_correct=true should not raise 422 — behavior depends on model availability."""
        # Mock pipeline to avoid actual inference
        import src.core.pipeline as pl

        async def mock_dub(*args, **kwargs):
            raise RuntimeError("Model not ready (test)")

        monkeypatch.setattr(pl, "dub_video", mock_dub)

        r = client.post(
            "/dub",
            data={"model": "latentsync", "dub_correct": "true", "sync": "true"},
            files={"video": ("test.mp4", self._video_bytes(), "video/mp4")},
        )
        # Should not be 422 (validation error); 500 is OK (model not ready in test)
        assert r.status_code != 422, f"dub_correct flag should be accepted, got 422: {r.text}"


class TestVideoAsAudioInput:
    """Audio field should accept video files transparently."""

    def test_is_video_file_detection(self):
        from unittest.mock import MagicMock
        from src.api.deps import is_video_file

        # Video MIME
        upload = MagicMock()
        upload.content_type = "video/mp4"
        upload.filename = "clip.mp4"
        assert is_video_file(upload) is True

        # Audio MIME
        upload.content_type = "audio/mpeg"
        upload.filename = "audio.mp3"
        assert is_video_file(upload) is False

        # Video extension, no MIME
        upload.content_type = None
        upload.filename = "clip.mkv"
        assert is_video_file(upload) is True

        # Ambiguous: content-type not set, known audio extension
        upload.filename = "sound.wav"
        assert is_video_file(upload) is False
