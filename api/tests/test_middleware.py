"""Check the ASGI boundary forwards streams incrementally and preserves protocol errors."""

import asyncio
from typing import cast

import pytest
from starlette.types import Message, Receive, Scope, Send

from met_agent.middleware import (
    ErrorBoundaryMiddleware,
    OriginAuthenticationMiddleware,
    RequestContextMiddleware,
)


def test_stream_chunks_reach_the_client_before_the_application_finishes() -> None:
    messages: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        messages.append(message)

    async def streaming_app(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"first", "more_body": True})
        assert messages[-1]["body"] == b"first"
        await send({"type": "http.response.body", "body": b"last", "more_body": False})

    scope: Scope = {"type": "http", "method": "POST", "headers": []}
    middleware = RequestContextMiddleware(ErrorBoundaryMiddleware(streaming_app))
    asyncio.run(middleware(scope, receive, send))
    assert len(messages) == 3
    assert (b"x-request-id", scope["state"]["request_id"].encode()) in messages[0]["headers"]


def test_started_stream_is_not_replaced_by_a_second_response() -> None:
    messages: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b""}

    async def send(message: Message) -> None:
        messages.append(message)

    async def failing_stream(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise ValueError("unit-test-provider-secret")

    scope: Scope = {"type": "http", "method": "POST", "headers": []}
    middleware = RequestContextMiddleware(ErrorBoundaryMiddleware(failing_stream))
    with pytest.raises(RuntimeError, match="Response stream failed") as error:
        asyncio.run(middleware(scope, receive, send))
    assert "unit-test-provider-secret" not in str(error.value)
    assert len(messages) == 1


def test_non_http_scopes_pass_through_unchanged() -> None:
    received: list[Scope] = []

    async def receive() -> Message:
        return {"type": "lifespan.startup"}

    async def send(message: Message) -> None:
        return None

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        received.append(scope)

    scope: Scope = {"type": "lifespan"}
    middleware = RequestContextMiddleware(ErrorBoundaryMiddleware(downstream))
    asyncio.run(middleware(scope, receive, send))
    assert received == [scope]
    assert "state" not in scope


def test_origin_authentication_is_fail_closed_and_health_remains_public() -> None:
    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def status(path: str, headers: list[tuple[bytes, bytes]]) -> int:
        sent: list[Message] = []

        async def send(message: Message) -> None:
            sent.append(message)

        scope: Scope = {"type": "http", "method": "GET", "path": path, "headers": headers}
        middleware = OriginAuthenticationMiddleware(
            downstream, required=True, token="unit-test-origin-secret"
        )
        await middleware(scope, receive, send)
        first = sent[0]
        assert first["type"] == "http.response.start"
        return cast(int, first["status"])

    assert asyncio.run(status("/health", [])) == 204
    assert asyncio.run(status("/chat", [])) == 401
    assert asyncio.run(status("/chat", [(b"x-origin-auth", b"wrong")])) == 401
    assert asyncio.run(status("/chat", [(b"x-origin-auth", b"unit-test-origin-secret")])) == 204
