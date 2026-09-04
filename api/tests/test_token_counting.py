"""Verify provider token counting and separate gateway/provider credentials without live calls."""

import json

import httpx
import pytest
from pydantic import HttpUrl, SecretStr

from met_agent.config import Settings
from met_agent.ingestion.http import SourceError
from met_agent.ingestion.token_counting import GeminiTokenCounter
from met_agent.retrieval.embeddings import ModelUnavailableError


@pytest.mark.parametrize("gateway", [False, True])
def test_token_counting_uses_the_configured_provider_path(
    settings: Settings, gateway: bool
) -> None:
    configured = settings.model_copy(
        update={
            "embedding_model": "gemini-embedding-001",
            "use_ai_gateway": gateway,
            "cf_ai_gateway_url": HttpUrl("https://gateway.ai.cloudflare.com/v1/account/gateway"),
            "cf_ai_gateway_token": SecretStr("unit-gateway"),
        }
    )

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models/gemini-embedding-001:countTokens")
        assert (
            request.headers["x-goog-api-key"]
            == settings.require_api_key("gemini").get_secret_value()
        )
        assert ("cf-aig-authorization" in request.headers) == gateway
        body = json.loads(request.content)
        assert len(body["contents"]) == 2
        return httpx.Response(200, json={"totalTokens": 16})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        counter = GeminiTokenCounter(configured, client)
        assert counter.count([]) == 0
        assert counter.count(["Bronze", "Marble"]) == 16


@pytest.mark.parametrize("status", [401, 429, 500, 404, 200])
def test_token_count_failures_do_not_leak_provider_bodies(settings: Settings, status: int) -> None:
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(status, text="private-provider-data")
        )
    ) as client:
        with pytest.raises((SourceError, ModelUnavailableError)) as error:
            GeminiTokenCounter(settings, client).count(["Vessel"])
        assert "private-provider-data" not in str(error.value)


def test_token_count_transport_failures_are_sanitized(settings: Settings) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private-provider-data")

    with (
        httpx.Client(transport=httpx.MockTransport(fail)) as client,
        pytest.raises(SourceError, match="transport"),
    ):
        GeminiTokenCounter(settings, client).count(["Vessel"])
