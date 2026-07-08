"""FastAPI application factory — SwiftDub."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from src.api.routes import dub, gpu, health, jobs
from src.config import settings
from src.logging_.inference_log import setup_logging

ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings.ensure_dirs()
    logger.info("{} starting — host={}:{}", settings.app_name, settings.host, settings.port)
    logger.info("LatentSync ckpt: {}", settings.latentsync_ckpt)
    logger.info("GPU pool: {}", settings.gpu_id_list())
    logger.info("Max concurrent jobs: {}", settings.max_concurrent_jobs)
    if settings.is_musetalk_disabled():
        logger.info("MuseTalk DISABLED — LatentSync only")

    # Warm up MongoDB and ensure indexes
    if not settings.disable_db:
        try:
            from src.db.client import get_db
            await get_db().command("ping")
            logger.info("MongoDB connected")
            from src.db.repos import ensure_indexes
            await ensure_indexes()
        except Exception as exc:
            logger.warning("MongoDB unavailable — using in-memory job store ({})", exc)

    # Clean up stale temp/upload dirs from previous runs
    try:
        from src.utils.cleanup import cleanup_stale_temp, cleanup_stale_uploads
        await asyncio.get_event_loop().run_in_executor(None, cleanup_stale_temp)
        await asyncio.get_event_loop().run_in_executor(None, cleanup_stale_uploads)
    except Exception as exc:
        logger.debug("Startup cleanup error: {}", exc)

    yield

    # Graceful shutdown
    try:
        from src.cache.client import close as close_redis
        await close_redis()
    except Exception:
        pass

    logger.info("{} stopped", settings.app_name)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description=(
            "SwiftDub — production video dubbing with LatentSync 1.5 and MuseTalk v1.5. "
            "Upload a video + audio and receive a lip-synced MP4."
        ),
        version="2.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(dub.router)
    app.include_router(jobs.router)
    app.include_router(gpu.router)

    # Serve output videos at /outputs/<filename>
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=str(settings.outputs_dir)), name="outputs")

    # Sample files for frontend demo loading
    if settings.samples_dir.exists():
        app.mount("/samples", StaticFiles(directory=str(settings.samples_dir)), name="samples")

    # Serve frontend
    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def frontend():
        html = FRONTEND_DIR / "index.html"
        if html.exists():
            return HTMLResponse(html.read_text())
        return HTMLResponse("<h1>SwiftDub</h1><p>Visit <a href='/docs'>/docs</a></p>")

    return app
