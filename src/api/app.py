"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routes import health, infer, jobs
from src.cache.client import close as close_redis
from src.config import settings
from src.db.client import close as close_mongo


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    logger.info("{} starting on {}:{}", settings.app_name, settings.host, settings.port)
    yield
    await close_mongo()
    await close_redis()
    logger.info("{} stopped", settings.app_name)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="Open-source lip-sync API — Wav2Lip | VideoReTalking | MuseTalk | LatentSync | SadTalker",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(health.router)
    app.include_router(infer.router)
    app.include_router(jobs.router)
    return app
