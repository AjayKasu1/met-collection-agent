"""Compose the FastAPI service after validating configuration, with no import-time I/O."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version
from typing import Literal

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import JSONResponse

from met_agent.config import OptionalServices, Settings, load_settings
from met_agent.llm.providers import configured_models
from met_agent.middleware import (
    ErrorBoundaryMiddleware,
    OriginAuthenticationMiddleware,
    RequestContextMiddleware,
)
from met_agent.observability.logging import configure_logging
from met_agent.routes import router
from met_agent.runtime import Runtime

VERSION = version("met-collection-agent-api")
logger = structlog.get_logger(__name__)


class HealthResponse(BaseModel):
    """Liveness and deployment metadata; configuration flags are not dependency probes."""

    status: Literal["ok"] = "ok"
    version: str
    git_sha: str
    optional_services: OptionalServices


class ReadinessResponse(BaseModel):
    """Required dependency state for deployment and alerting probes."""

    status: Literal["ok", "unavailable"]
    checks: dict[str, bool]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build an isolated application; fail before accepting requests on invalid settings."""
    config = settings if settings is not None else load_settings()
    configure_logging(config.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("service_started", version=VERSION, git_sha=config.git_sha)
        logger.info(
            "model_fallbacks_configured",
            active={
                alias: model
                for alias, model in configured_models(config).items()
                if alias not in {"lite", "main"}
            },
        )
        try:
            if config.startup_warmup:
                app.state.runtime = Runtime(config)
                try:
                    await asyncio.to_thread(app.state.runtime.warmup)
                except Exception as error:
                    logger.error("startup_warmup_failed", error_type=type(error).__name__)
                    raise RuntimeError("Retrieval startup warmup failed") from None
                logger.info("startup_warmup_completed")
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
    app.add_middleware(
        OriginAuthenticationMiddleware,
        required=config.edge_auth_required,
        token=(config.edge_origin_token.get_secret_value() if config.edge_origin_token else None),
    )
    app.add_middleware(ErrorBoundaryMiddleware)
    # CORS wraps request handling so controlled 500 responses carry CORS headers too.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID", "X-Origin-Auth"],
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

    @app.get(
        "/ready",
        response_model=ReadinessResponse,
        responses={503: {"model": ReadinessResponse}},
        tags=["operations"],
    )
    async def readiness(request: Request) -> ReadinessResponse | JSONResponse:
        """Verify durable dependencies while retaining a cheap liveness endpoint."""
        if getattr(request.app.state, "runtime", None) is None:
            request.app.state.runtime = Runtime(config)
        checks = await asyncio.to_thread(request.app.state.runtime.readiness)
        response = ReadinessResponse(
            status="ok" if all(checks.values()) else "unavailable",
            checks=checks,
        )
        if response.status == "unavailable":
            return JSONResponse(status_code=503, content=response.model_dump(mode="json"))
        return response

    return app
