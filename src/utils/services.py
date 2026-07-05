"""Service URL helpers for local localhost links."""

from __future__ import annotations

import os
import socket


def port_url(port: int, path: str = "") -> str:
    host = os.environ.get("SWIFTDUB_PUBLIC_HOST", "localhost")
    return f"http://{host}:{port}{path}"


def list_service_urls() -> dict[str, dict]:
    """Return browser-reachable URLs for each stack component."""
    services = {
        "api": {"port": 8000, "path": "/", "note": "SwiftDub UI + REST API"},
        "grafana": {
            "port": 3000,
            "path": "/d/swiftdub-main/swiftdub-e28094-dubbing-metrics?",
            "note": "SwiftDub dashboard (anonymous access enabled)",
        },
        "prometheus": {"port": 9092, "path": "/graph", "note": "Prometheus Graph UI"},
        "celery": {"port": 5556, "path": "/", "note": "Celery Flower monitor"},
    }
    result: dict[str, dict] = {}
    for name, meta in services.items():
        url = port_url(meta["port"], meta["path"])
        result[name] = {
            "port": meta["port"],
            "url": url,
            "local_url": url,
            "note": meta["note"],
            "reachable_via_proxy": False,
        }
    return result


def check_port_open(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False
