"""LatentSync 1.5 inference pipeline — first-party SwiftDub implementation.

Extracted from the ByteDance LatentSync repository (Apache 2.0) and organized
as first-class SwiftDub source code. Training code and data pipeline are excluded.

Usage (subprocess entry):
    python -m src.models.latentsync.infer --help
"""
from .pipeline import LipsyncPipeline

__all__ = ["LipsyncPipeline"]
