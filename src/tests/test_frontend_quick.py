"""Quick frontend+backend integration test (< 3 min).

Simulates what the UI does: bootstrap calls, paginated jobs, quick dub.
Run with local server: python -m src.main
"""

from __future__ import annotations

import os
import time
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
    with httpx.Client(base_url=BASE_URL, timeout=180.0) as c:
        yield c


def test_frontend_bootstrap(client: httpx.Client):
    """All calls the UI makes on page load."""
    health = client.get("/health").json()
    assert health["status"] == "healthy"
    assert health["musetalk_disabled"] is True

    gpu = client.get("/gpu/status").json()
    assert "devices" in gpu and gpu["overhead_pct"] == 20

    jobs = client.get("/jobs", params={"per_page": 5, "sort_by": "created_at", "order": "desc"}).json()
    assert "jobs" in jobs and "pages" in jobs

    models = client.get("/models").json()
    names = [m["name"] if isinstance(m, dict) else m for m in models]
    assert "latentsync" in names
    assert "musetalk" not in names

    html = client.get("/").text
    assert "SwiftDub" in html and "Grafana Dashboard" in html
    assert "loadModels" in html       # models loaded dynamically
    assert "showSection" in html      # section switching function present


def test_jobs_table_filters(client: httpx.Client):
    r = client.get("/jobs", params={"page": 1, "per_page": 5, "status": "completed", "sort_by": "elapsed_s"})
    assert r.status_code == 200
    data = r.json()
    assert data["per_page"] == 5


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("RUN_GPU_TESTS") != "1", reason="Set RUN_GPU_TESTS=1")
def test_quick_dub_sync_5_steps(client: httpx.Client):
    """Short video + 5 steps — matches frontend default."""
    if not SAMPLE_VIDEO.exists():
        pytest.skip("sample video missing")
    with SAMPLE_VIDEO.open("rb") as vf:
        files = {"video": ("cut1.mp4", vf, "video/mp4")}
        data = {"model": "latentsync", "inference_steps": "5", "sync": "true"}
        if SAMPLE_AUDIO.exists():
            with SAMPLE_AUDIO.open("rb") as af:
                files["audio"] = ("dub_audio.wav", af, "audio/wav")
                r = client.post("/dub", files=files, data=data)
        else:
            r = client.post("/dub", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "completed"
    assert body.get("download_url")


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("RUN_GPU_TESTS") != "1", reason="Set RUN_GPU_TESTS=1")
def test_quick_dub_async(client: httpx.Client):
    """Async path the UI uses when sync toggle is off."""
    if not SAMPLE_VIDEO.exists():
        pytest.skip("sample video missing")
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

    deadline = time.time() + 120
    while time.time() < deadline:
        jr = client.get(f"/jobs/{job_id}")
        if jr.status_code == 200 and jr.json()["status"] in ("completed", "failed"):
            assert jr.json()["status"] == "completed", jr.json().get("error")
            return
        time.sleep(2)
    pytest.fail("async job timed out")
