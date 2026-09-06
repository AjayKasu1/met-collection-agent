"""Preserve request context across ASGI streaming without buffering response bodies."""

import hmac
import re
from time import perf_counter
from uuid import uuid4

import structlog
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", re.ASCII)
logger = structlog.get_logger(__name__)


class OriginAuthenticationMiddleware:
    """Reject direct origin traffic when the edge authentication boundary is enabled."""

    def __init__(self, app: ASGIApp, *, required: bool, token: str | None) -> None:
        self.app = app
        self.required = required
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") == "/health" or not self.required:
            await self.app(scope, receive, send)
            return

        supplied = Headers(scope=scope).getlist("x-origin-auth")
        if (
            self.token is None
            or len(supplied) != 1
            or not hmac.compare_digest(supplied[0].encode(), self.token.encode())
        ):
            logger.warning("origin_authentication_rejected", path=scope.get("path", ""))
            response = JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"},
                headers={"Cache-Control": "no-store"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


class RequestContextMiddleware:
    """Attach a bounded request ID to every response, including CORS preflights."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        supplied = Headers(scope=scope).getlist("x-request-id")
        request_id = (
            supplied[0]
            if len(supplied) == 1 and _REQUEST_ID.fullmatch(supplied[0])
            else uuid4().hex
        )
        scope.setdefault("state", {})["request_id"] = request_id
        context = structlog.contextvars.bind_contextvars(request_id=request_id)
        started = perf_counter()
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            route = scope.get("route")
            logger.info(
                "request_completed",
                method=scope["method"],
                route=getattr(route, "path", "<unmatched>"),
                status_code=status_code,
                latency_ms=round((perf_counter() - started) * 1000, 3),
            )
            structlog.contextvars.reset_contextvars(**context)


class ErrorBoundaryMiddleware:
    """Return safe errors inside CORS; never replace a stream that has already started."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        response_started = False

        async def track_start(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, track_start)
        except Exception as error:
            # Exception messages can contain provider keys or user content.
            logger.error("request_failed", error_type=type(error).__name__)
            if response_started:
                raise RuntimeError("Response stream failed; consult the request ID") from None
            response = JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "request_id": scope["state"]["request_id"],
                },
            )
            await response(scope, receive, send)
