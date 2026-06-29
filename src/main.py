"""SwiftDub FastAPI application."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.config import ensure_directories, settings
from src.routes.health import router as health_router
from src.routes.predict import router as predict_router


def create_app() -> FastAPI:
    ensure_directories()

    app = FastAPI(
        title=settings.app_name,
        description="Open-source lip-sync HTTP API (Wav2Lip, MuseTalk, LatentSync)",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(predict_router)
    return app


app = create_app()


def main() -> None:
    logger.info("Starting {} on {}:{}", settings.app_name, settings.host, settings.port)
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
