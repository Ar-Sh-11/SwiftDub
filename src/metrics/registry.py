"""Prometheus metrics definitions."""

from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

REGISTRY = CollectorRegistry(auto_describe=True)

inference_requests = Counter(
    "swiftdub_inference_requests_total",
    "Total inference requests",
    ["model", "status"],
    registry=REGISTRY,
)

inference_duration = Histogram(
    "swiftdub_inference_duration_seconds",
    "Inference latency in seconds",
    ["model"],
    buckets=[5, 10, 30, 60, 120, 300, 600],
    registry=REGISTRY,
)

active_jobs = Gauge(
    "swiftdub_active_jobs",
    "Number of currently processing jobs",
    registry=REGISTRY,
)

cache_hits = Counter(
    "swiftdub_cache_hits_total",
    "Redis cache hits",
    ["model"],
    registry=REGISTRY,
)
