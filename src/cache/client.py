"""Redis async client via redis.asyncio (Python 3.11 compatible).

Uses content-addressing: SHA-256 of (video_bytes, audio_bytes, model)
so identical inputs always hit the same cache key.
"""

from __future__ import annotations

import hashlib
from redis.asyncio import Redis

from src.config import settings

_redis: Redis | None = None


async def get_redis() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


async def close() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None


def _cache_key(video_hash: str, audio_hash: str, model: str) -> str:
    return f"swiftdub:output:{model}:{video_hash}:{audio_hash}"


async def get_cached_output(video_bytes: bytes, audio_bytes: bytes, model: str) -> str | None:
    key = _cache_key(
        hashlib.sha256(video_bytes).hexdigest()[:16],
        hashlib.sha256(audio_bytes).hexdigest()[:16],
        model,
    )
    return await (await get_redis()).get(key)


async def cache_output(
    video_bytes: bytes,
    audio_bytes: bytes,
    model: str,
    output_path: str,
    ttl: int = 3600,
) -> None:
    key = _cache_key(
        hashlib.sha256(video_bytes).hexdigest()[:16],
        hashlib.sha256(audio_bytes).hexdigest()[:16],
        model,
    )
    await (await get_redis()).set(key, output_path, ex=ttl)
