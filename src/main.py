"""Entrypoint: uvicorn server for SwiftDub."""

import uvicorn

from src.api.app import create_app
from src.config import settings

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )
