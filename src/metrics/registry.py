"""Prometheus metrics for LatentSync dubbing service."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

inference_requests = Counter(
    "latentsync_requests_total",
    "Total dubbing requests",
    ["model", "status"],
    registry=REGISTRY,
)

inference_duration = Histogram(
    "latentsync_inference_duration_seconds",
    "LatentSync inference latency in seconds",
    ["model"],
    buckets=[10, 30, 60, 90, 120, 180, 300, 600],
    registry=REGISTRY,
)

active_jobs = Gauge(
    "latentsync_active_jobs",
    "Number of LatentSync jobs currently running",
    registry=REGISTRY,
)

queue_depth = Gauge(
    "latentsync_queue_depth",
    "Jobs waiting for a GPU slot",
    registry=REGISTRY,
)

batch_size_histogram = Histogram(
    "latentsync_batch_size",
    "Number of videos in a batch request",
    buckets=[1, 2, 3, 5, 10, 20],
    registry=REGISTRY,
)

video_duration_seconds = Histogram(
    "latentsync_video_duration_seconds",
    "Duration of input videos processed",
    buckets=[5, 10, 30, 60, 120, 300],
    registry=REGISTRY,
)
