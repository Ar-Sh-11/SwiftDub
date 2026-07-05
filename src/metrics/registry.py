"""Prometheus metrics for SwiftDub dubbing service."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

inference_requests = Counter(
    "swiftdub_requests_total",
    "Total dubbing requests",
    ["model", "status"],
    registry=REGISTRY,
)

inference_duration = Histogram(
    "swiftdub_inference_duration_seconds",
    "Inference latency in seconds",
    ["model"],
    buckets=[5, 10, 30, 60, 90, 120, 180, 300, 600],
    registry=REGISTRY,
)

active_jobs = Gauge(
    "swiftdub_active_jobs",
    "Jobs currently running on GPU",
    registry=REGISTRY,
)

queue_depth = Gauge(
    "swiftdub_queue_depth",
    "Jobs waiting for a GPU slot",
    registry=REGISTRY,
)

batch_size_histogram = Histogram(
    "swiftdub_batch_size",
    "Videos per batch request",
    buckets=[1, 2, 3, 5, 10, 20],
    registry=REGISTRY,
)

gpu_memory_pct_gauge = Gauge(
    "swiftdub_gpu_memory_pct",
    "GPU memory utilisation percent",
    registry=REGISTRY,
)

ram_memory_pct_gauge = Gauge(
    "swiftdub_ram_memory_pct",
    "System RAM utilisation percent",
    registry=REGISTRY,
)
