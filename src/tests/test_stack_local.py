"""Local stack integration tests — run against a live server at BASE_URL.

Usage:
  # Start server first: python -m src.main
  BASE_URL=http://localhost:8000 pytest src/tests/test_stack_local.py -v -s

Requires: MongoDB + Redis optional (falls back to in-memory if unavailable).
GPU tests marked with @pytest.mark.gpu are skipped unless RUN_GPU_TESTS=1.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
ROOT = Path(__file__).resolve().parents[2]
SAMPLE_VIDEO = ROOT / "data" / "lipsync eval input" / "cut1.mp4"
SAMPLE_AUDIO = ROOT / "data" / "samples" / "dub_audio.wav"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client() -> httpx.Client:
    with httpx.Client(base_url=BASE_URL, timeout=120.0) as c:
        yield c


def test_health(client: httpx.Client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["musetalk_disabled"] is True
    assert "latentsync" in data["models"]


def test_gpu_status(client: httpx.Client):
    r = client.get("/gpu/status")
    assert r.status_code == 200
    data = r.json()
    assert "devices" in data
    assert data["overhead_pct"] == 20
    if data["devices"]:
        dev = data["devices"][0]
        assert dev["total_gb"] > 0
        assert dev["overhead_gb"] > 0


def test_models_list(client: httpx.Client):
    r = client.get("/models")
    assert r.status_code == 200
    models = r.json()
    names = [m["name"] for m in models]
    assert "latentsync" in names
    assert "musetalk" not in names  # disabled


def test_jobs_paginated(client: httpx.Client):
    r = client.get("/jobs", params={"page": 1, "per_page": 5})
    assert r.status_code == 200
    data = r.json()
    assert "jobs" in data and "total" in data and "pages" in data


def test_services_endpoint(client: httpx.Client):
    r = client.get("/services")
    assert r.status_code == 200
    svcs = r.json()["services"]
    assert "grafana" in svcs
    assert "prometheus" in svcs


def test_metrics_endpoint(client: httpx.Client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"swiftdub" in r.content or b"python" in r.content


def test_frontend_served(client: httpx.Client):
    r = client.get("/")
    assert r.status_code == 200
    assert "SwiftDub" in r.text
    assert "Grafana Dashboard" in r.text


def test_dub_validation_no_video(client: httpx.Client):
    r = client.post("/dub", data={"model": "latentsync"})
    assert r.status_code == 422


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("RUN_GPU_TESTS") != "1", reason="Set RUN_GPU_TESTS=1")
def test_dub_sync_latentsync(client: httpx.Client):
    if not SAMPLE_VIDEO.exists():
        pytest.skip("Sample video not found")
    with SAMPLE_VIDEO.open("rb") as vf:
        files = {"video": ("cut1.mp4", vf, "video/mp4")}
        data = {"model": "latentsync", "inference_steps": "5", "sync": "true"}
        if SAMPLE_AUDIO.exists():
            with SAMPLE_AUDIO.open("rb") as af:
                files["audio"] = ("dub_audio.wav", af, "audio/wav")
                r = client.post("/dub", files=files, data=data, timeout=600.0)
        else:
            r = client.post("/dub", files=files, data=data, timeout=600.0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "completed"
    assert body.get("download_url")
    assert body.get("elapsed_s") is not None


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("RUN_GPU_TESTS") != "1", reason="Set RUN_GPU_TESTS=1")
def test_dub_correct(client: httpx.Client):
    if not SAMPLE_VIDEO.exists():
        pytest.skip("Sample video not found")
    with SAMPLE_VIDEO.open("rb") as vf:
        r = client.post(
            "/dub",
            files={"video": ("cut1.mp4", vf, "video/mp4")},
            data={"model": "latentsync", "dub_correct": "true", "inference_steps": "5", "sync": "true"},
            timeout=600.0,
        )
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "completed"


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("RUN_GPU_TESTS") != "1", reason="Set RUN_GPU_TESTS=1")
def test_dub_async_celery(client: httpx.Client):
    """Async dub via Celery worker — requires celery worker running."""
    if not SAMPLE_VIDEO.exists():
        pytest.skip("Sample video not found")
    import time

    with SAMPLE_VIDEO.open("rb") as vf:
        r = client.post(
            "/dub",
            files={"video": ("cut1.mp4", vf, "video/mp4")},
            data={"model": "latentsync", "inference_steps": "5", "sync": "false"},
            timeout=30.0,
        )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    assert r.json()["status"] == "queued"

    for _ in range(120):
        time.sleep(3)
        jr = client.get(f"/jobs/{job_id}")
        if jr.status_code != 200:
            continue
        st = jr.json()["status"]
        if st in ("completed", "failed"):
            assert st == "completed", jr.json().get("error")
            assert jr.json().get("elapsed_s") is not None
            return
    pytest.fail("Async job timed out")


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("RUN_GPU_TESTS") != "1", reason="Set RUN_GPU_TESTS=1")
def test_dub_batch(client: httpx.Client):
    if not SAMPLE_VIDEO.exists():
        pytest.skip("Sample video not found")
    import time

    with SAMPLE_VIDEO.open("rb") as v1, SAMPLE_VIDEO.open("rb") as v2:
        r = client.post(
            "/dub/batch",
            files=[
                ("videos", ("a.mp4", v1, "video/mp4")),
                ("videos", ("b.mp4", v2, "video/mp4")),
            ],
            data={
                "model": "latentsync",
                "inference_steps": "5",
                "dub_correct_flags": "[true, false]",
            },
            timeout=60.0,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 2
    job_ids = [j["job_id"] for j in body["jobs"]]

    for _ in range(180):
        time.sleep(5)
        done = 0
        for jid in job_ids:
            jr = client.get(f"/jobs/{jid}")
            if jr.status_code == 200 and jr.json()["status"] in ("completed", "failed"):
                if jr.json()["status"] == "failed":
                    pytest.fail(jr.json().get("error"))
                done += 1
        if done == len(job_ids):
            return
    pytest.fail("Batch jobs timed out")
