"""Compose the FastAPI service after validating configuration, with no import-time I/O."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version
from typing import Literal

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from met_agent.config import OptionalServices, Settings, load_settings
from met_agent.middleware import ErrorBoundaryMiddleware, RequestContextMiddleware
from met_agent.observability.logging import configure_logging
from met_agent.routes import router

VERSION = version("met-collection-agent-api")
logger = structlog.get_logger(__name__)


class HealthResponse(BaseModel):
    """Liveness and deployment metadata; configuration flags are not dependency probes."""

    status: Literal["ok"] = "ok"
    version: str
    git_sha: str
    optional_services: OptionalServices


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build an isolated application; fail before accepting requests on invalid settings."""
    config = settings if settings is not None else load_settings()
    configure_logging(config.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("service_started", version=VERSION, git_sha=config.git_sha)
        try:
            yield
        finally:
            if getattr(app.state, "runtime", None) is not None:
                app.state.runtime.close()
            logger.info("service_stopped")

    app = FastAPI(
        title="Met Collection Agent API",
        description="Independent Open Access collection assistant. Not affiliated with The Met.",
        version=VERSION,
        lifespan=lifespan,
    )
    app.state.settings = config
    app.include_router(router)
    app.add_middleware(ErrorBoundaryMiddleware)
    # CORS wraps request handling so controlled 500 responses carry CORS headers too.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
        allow_credentials=False,
    )
    app.add_middleware(RequestContextMiddleware)

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    async def health() -> HealthResponse:
        """Return liveness without contacting optional services or exposing configuration."""
        return HealthResponse(
            version=VERSION,
            git_sha=config.git_sha,
            optional_services=config.optional_services,
        )

    return app
