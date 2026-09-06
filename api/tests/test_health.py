"""Verify public API contracts, request isolation, CORS, and safe operational errors."""

import asyncio
import json
from collections.abc import Iterator
from io import StringIO
from pathlib import Path

import httpx
import pytest
import structlog
from fastapi import FastAPI
from fastapi.testclient import TestClient

from met_agent.config import ConfigurationError, Settings
from met_agent.main import VERSION, create_app
from met_agent.observability.logging import configure_logging


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    """Build the API using isolated test configuration."""
    return create_app(settings)


@pytest.fixture
def logs(app: FastAPI) -> StringIO:
    """Capture the actual production JSON renderer for assertions."""
    output = StringIO()
    configure_logging("INFO", stream=output)
    return output


@pytest.fixture
def client(app: FastAPI, logs: StringIO) -> Iterator[TestClient]:
    """Exercise lifespan hooks as well as HTTP requests."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_is_typed_and_does_not_expose_configuration(
    client: TestClient, settings: Settings
) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": VERSION,
        "git_sha": "unknown",
        "optional_services": settings.optional_services.model_dump(),
    }
    assert settings.require_api_key("gemini").get_secret_value() not in response.text
    assert settings.llm_model not in response.text
    assert str(settings.qdrant_url) not in response.text


def test_health_reports_supplied_build_revision(
    valid_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from met_agent.config import load_settings

    monkeypatch.setenv("GIT_SHA", "abc1234")
    with TestClient(create_app(load_settings(env_file=None))) as client:
        assert client.get("/health").json()["git_sha"] == "abc1234"


def test_readiness_reports_required_dependencies(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from met_agent import main as module

    class ReadyRuntime:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings

        def readiness(self) -> dict[str, bool]:
            return {"audit_store": True, "qdrant": True}

        def close(self) -> None:
            return None

    monkeypatch.setattr(module, "Runtime", ReadyRuntime)
    with TestClient(create_app(settings)) as client:
        response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "checks": {"audit_store": True, "qdrant": True},
    }


def test_readiness_fails_when_a_required_dependency_is_down(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from met_agent import main as module

    class UnreadyRuntime:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings

        def readiness(self) -> dict[str, bool]:
            return {"audit_store": True, "qdrant": False}

        def close(self) -> None:
            return None

    monkeypatch.setattr(module, "Runtime", UnreadyRuntime)
    with TestClient(create_app(settings)) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "checks": {"audit_store": True, "qdrant": False},
    }


def test_docs_and_openapi_are_available(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "/health" in schema["paths"]
    assert schema["info"]["version"] == VERSION


@pytest.mark.parametrize("path", ["/health", "/docs", "/not-found"])
def test_request_id_is_returned_on_every_response(client: TestClient, path: str) -> None:
    response = client.get(path, headers={"X-Request-ID": "client-request:123"})
    assert response.headers["x-request-id"] == "client-request:123"


@pytest.mark.parametrize("value", ["", "too long " * 30, "bad id", "x" * 129])
def test_untrusted_request_ids_are_replaced(client: TestClient, value: str) -> None:
    response = client.get("/health", headers={"X-Request-ID": value})
    request_id = response.headers["x-request-id"]
    assert request_id != value
    assert len(request_id) == 32
    int(request_id, 16)


def test_missing_and_duplicate_request_ids_are_regenerated(client: TestClient) -> None:
    first = client.get("/health").headers["x-request-id"]
    second = client.get("/health").headers["x-request-id"]
    duplicate = client.get(
        "/health", headers=[("X-Request-ID", "first"), ("X-Request-ID", "second")]
    ).headers["x-request-id"]
    assert first != second
    assert duplicate not in {"first", "second"}


def test_configured_origin_receives_cors_headers(client: TestClient) -> None:
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert response.headers["access-control-expose-headers"] == "X-Request-ID"
    assert "access-control-allow-credentials" not in response.headers


def test_unconfigured_origin_does_not_receive_cors_access(client: TestClient) -> None:
    response = client.get("/health", headers={"Origin": "https://untrusted.example"})
    assert "access-control-allow-origin" not in response.headers


def test_preflight_includes_cors_and_request_id(client: TestClient) -> None:
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-request-id",
            "X-Request-ID": "preflight-request",
        },
    )
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "preflight-request"
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_request_log_uses_route_template_and_omits_user_content(
    client: TestClient, logs: StringIO
) -> None:
    client.get(
        "/health?api_key=unit-test-query-secret",
        headers={
            "X-Request-ID": "logged-request",
            "Authorization": "Bearer unit-test-header-secret",
        },
    )
    client.get("/unknown-private-path")
    entries = [json.loads(line) for line in logs.getvalue().splitlines()]
    requests = [entry for entry in entries if entry["event"] == "request_completed"]
    assert requests[0]["request_id"] == "logged-request"
    assert requests[0]["route"] == "/health"
    assert requests[0]["status_code"] == 200
    assert requests[0]["latency_ms"] >= 0
    assert requests[0]["timestamp"].endswith("Z")
    assert requests[1]["route"] == "<unmatched>"
    for private in ("unit-test-query-secret", "unit-test-header-secret", "unknown-private-path"):
        assert private not in logs.getvalue()


def test_failures_are_sanitized_and_keep_cors_and_request_id(
    app: FastAPI, client: TestClient, logs: StringIO
) -> None:
    @app.get("/failure")
    async def failing_endpoint() -> None:
        raise ValueError("unit-test-provider-secret")

    response = client.get(
        "/failure", headers={"X-Request-ID": "failed-request", "Origin": "http://localhost:3000"}
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error", "request_id": "failed-request"}
    assert response.headers["x-request-id"] == "failed-request"
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "unit-test-provider-secret" not in response.text + logs.getvalue()
    failures = [
        json.loads(line) for line in logs.getvalue().splitlines() if '"request_failed"' in line
    ]
    assert failures[0]["error_type"] == "ValueError"
    assert failures[0]["request_id"] == "failed-request"


def test_concurrent_requests_keep_separate_context(app: FastAPI, logs: StringIO) -> None:
    @app.get("/context")
    async def context_endpoint() -> dict[str, str]:
        await asyncio.sleep(0)
        return {"request_id": str(structlog.contextvars.get_contextvars()["request_id"])}

    async def exercise() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            responses = await asyncio.gather(
                *(
                    client.get("/context", headers={"X-Request-ID": f"parallel-{i}"})
                    for i in range(12)
                )
            )
        for i, response in enumerate(responses):
            assert response.json()["request_id"] == f"parallel-{i}"
            assert response.headers["x-request-id"] == f"parallel-{i}"
        assert "request_id" not in structlog.contextvars.get_contextvars()

    asyncio.run(exercise())


def test_lifespan_logs_start_and_stop(app: FastAPI, logs: StringIO) -> None:
    with TestClient(app):
        pass
    events = [json.loads(line)["event"] for line in logs.getvalue().splitlines()]
    assert events == ["service_started", "model_fallbacks_configured", "service_stopped"]
    assert json.loads(logs.getvalue().splitlines()[1])["active"] == {}


def test_factory_fails_before_serving_with_missing_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigurationError, match="LLM_MODEL: missing"):
        create_app()
