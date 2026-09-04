"""Exercise provider response contracts and explicit credential routing without network calls."""

import logging
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import HttpUrl, SecretStr

from met_agent.config import ConfigurationError, Settings
from met_agent.llm.router import create_embedding_router, embedding_route
from met_agent.retrieval.embeddings import (
    BM25Embedder,
    EmbeddingError,
    EmbeddingRateLimitError,
    GeminiEmbedder,
    ModelUnavailableError,
)


class ResponseTransport:
    def __init__(self, response: object) -> None:
        self.response = response
        self.purpose = ""
        self.dimensions = 0

    def embedding(self, *, model: str, input: list[str], task_type: str, dimensions: int) -> object:
        self.purpose = task_type
        self.dimensions = dimensions
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_response_order_and_retrieval_purpose() -> None:
    transport = ResponseTransport(
        {
            "data": [
                {"index": 1, "embedding": [0, 1]},
                {"index": 0, "embedding": [1, 0]},
            ]
        }
    )
    embedder = GeminiEmbedder(transport, output_dimensionality=2)
    assert embedder.embed(["first", "second"]) == [[1, 0], [0, 1]]
    assert transport.purpose == "RETRIEVAL_DOCUMENT"
    assert embedder.embed([]) == []
    transport.response = {"data": [{"index": 0, "embedding": [1, 1]}]}
    assert embedder.embed(["question"], purpose="query")[0] == pytest.approx([2**-0.5, 2**-0.5])
    assert transport.purpose == "RETRIEVAL_QUERY"


def test_requested_dimensions_reach_native_gemini_and_vectors_are_normalized() -> None:
    from litellm.llms.vertex_ai.gemini_embeddings.batch_embed_content_transformation import (
        transform_openai_input_gemini_content,
    )

    transport = ResponseTransport({"data": [{"index": 0, "embedding": [2.0] * 768}]})
    vector = GeminiEmbedder(transport, output_dimensionality=768).embed(["Vessel"])[0]
    assert transport.dimensions == 768 and len(vector) == 768
    assert sum(value**2 for value in vector) == pytest.approx(1.0)
    body = transform_openai_input_gemini_content(
        input=["Vessel"],
        model="gemini-embedding-001",
        optional_params={"dimensions": transport.dimensions, "task_type": transport.purpose},
    )
    assert body["requests"][0]["outputDimensionality"] == 768
    with pytest.raises(EmbeddingError, match="shape"):
        GeminiEmbedder(transport, output_dimensionality=3072).embed(["Vessel"])
    with pytest.raises(ValueError, match="dimensionality"):
        GeminiEmbedder(transport, output_dimensionality=0)


@pytest.mark.parametrize(
    "data",
    [
        [],
        [{"index": 1, "embedding": [1]}],
        [{"index": 0, "embedding": []}],
        [{"index": 0, "embedding": [0, 0]}],
        [{"index": 0, "embedding": [float("nan")]}],
        [{"index": 0, "embedding": [float("inf")]}],
        [{"index": 0, "embedding": [1]}, {"index": 0, "embedding": [1]}],
    ],
)
def test_malformed_vectors_fail_before_indexing(data: list[dict[str, object]]) -> None:
    with pytest.raises(EmbeddingError, match="validation"):
        GeminiEmbedder(ResponseTransport({"data": data})).embed(["text"])


def test_dimension_mismatch_and_secret_bearing_errors_are_sanitized() -> None:
    with pytest.raises(EmbeddingError):
        GeminiEmbedder(
            ResponseTransport(
                {
                    "data": [
                        {"index": 0, "embedding": [1]},
                        {"index": 1, "embedding": [1, 2]},
                    ]
                }
            )
        ).embed(["first", "second"])
    with pytest.raises(EmbeddingError) as caught:
        GeminiEmbedder(ResponseTransport(RuntimeError("private-provider-key"))).embed(["text"])
    assert "private-provider-key" not in str(caught.value)
    error = type("NotFoundError", (RuntimeError,), {})("private-provider-key")
    with pytest.raises(ModelUnavailableError, match="EMBEDDING_MODEL"):
        GeminiEmbedder(ResponseTransport(error)).embed(["text"])


@pytest.mark.parametrize(
    "response_body",
    [
        {
            "error": {
                "details": [
                    {"retryDelay": "120s"},
                    {"violations": [{"quotaId": "EmbedContentRequestsPerDay-FreeTier"}]},
                ]
            }
        },
        {"details": [{"retryDelay": "120s"}, {"violations": [{"quotaId": "per_day"}]}]},
    ],
)
def test_provider_quota_details_are_exposed_only_as_scheduling_fields(
    response_body: dict[str, object],
) -> None:
    class RateLimitError(RuntimeError):
        def __init__(self) -> None:
            super().__init__("private-provider-key")
            self.response = httpx.Response(429, json=response_body)

    with pytest.raises(EmbeddingRateLimitError) as caught:
        GeminiEmbedder(ResponseTransport(RateLimitError())).embed(["Vessel"])
    assert caught.value.daily and caught.value.retry_after == 120
    assert "private-provider-key" not in str(caught.value)


@pytest.mark.parametrize(
    "body",
    [
        "not JSON",
        '{"error":{"details":[null,{"retryDelay":"invalid"},{"retryDelay":"nan"}]}}',
        '{"error":{"details":[{"violations":[null]},{"retryDelay":"5s"}]}}',
        '{"error":{"details":"invalid"}}',
    ],
)
def test_unavailable_quota_details_use_conservative_minute_backoff(body: str) -> None:
    class RateLimitError(RuntimeError):
        def __init__(self) -> None:
            super().__init__("private-provider-key")
            self.body = body

    with pytest.raises(EmbeddingRateLimitError) as caught:
        GeminiEmbedder(ResponseTransport(RateLimitError())).embed(["Vessel"])
    assert not caught.value.daily and caught.value.retry_after == 61


def test_direct_route_and_gateway_use_distinct_authentication(settings: Settings) -> None:
    direct = settings.model_copy(update={"embedding_model": "models/test-embedding"})
    route = embedding_route(direct)["litellm_params"]
    assert route["model"] == "gemini/test-embedding"
    assert "api_base" not in route and "extra_headers" not in route
    gateway = direct.model_copy(
        update={
            "use_ai_gateway": True,
            "cf_ai_gateway_url": HttpUrl("https://gateway.ai.cloudflare.com/v1/account/gateway"),
            "cf_ai_gateway_token": SecretStr("test-gateway-token"),
        }
    )
    route = embedding_route(gateway)["litellm_params"]
    assert route["api_base"].endswith("/google-ai-studio/v1beta")
    assert route["extra_headers"] == {"cf-aig-authorization": "Bearer test-gateway-token"}
    assert route["api_key"] == settings.gemini_api_key.get_secret_value()
    with pytest.raises(ValueError, match="incomplete"):
        embedding_route(direct.model_copy(update={"use_ai_gateway": True}))


def test_router_retries_transient_errors_without_embedding_fallback(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import litellm.router

    captured: dict[str, Any] = {}

    def build(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(litellm.router, "Router", build)
    create_embedding_router(settings)
    assert captured["fallbacks"] == []
    assert captured["max_fallbacks"] == 0
    assert captured["retry_policy"].RateLimitErrorRetries == settings.llm_max_retries
    assert captured["retry_policy"].AuthenticationErrorRetries == 0
    logging.getLogger("LiteLLM").error("private-provider-token")
    captured_logs = capsys.readouterr()
    assert "private-provider-token" not in captured_logs.out + captured_logs.err


@pytest.mark.parametrize(
    "suffix", ["", "/", "/google-ai-studio", "/google-ai-studio/v1beta", "/google-ai-studio/v1"]
)
def test_gateway_accepts_only_supported_native_bases(settings: Settings, suffix: str) -> None:
    configured = settings.model_copy(
        update={
            "use_ai_gateway": True,
            "cf_ai_gateway_url": HttpUrl(
                "https://gateway.ai.cloudflare.com/v1/account/gateway" + suffix
            ),
            "cf_ai_gateway_token": SecretStr("test-gateway-token"),
        }
    )
    base = embedding_route(configured)["litellm_params"]["api_base"]
    assert base.count("google-ai-studio") == 1
    assert base.endswith("/v1" if suffix.endswith("/v1") else "/v1beta")


@pytest.mark.parametrize(
    "url",
    [
        "https://api.cloudflare.com/client/v4/accounts/account/ai/v1/chat/completions",
        "https://gateway.ai.cloudflare.com/v1/account/gateway/compat",
        "https://gateway.ai.cloudflare.com/v1/account/gateway/google-ai-studio/v1beta/models/model",
        "https://gateway.ai.cloudflare.com/v1/account",
        "https://gateway.ai.cloudflare.com/v1//gateway",
        "http://gateway.ai.cloudflare.com/v1/account/gateway",
        "https://gateway.ai.cloudflare.com/v1/account/gateway?key=private-input",
        "https://gateway.ai.cloudflare.com/v1/account/gateway#private-input",
        "https://user:private-input@gateway.ai.cloudflare.com/v1/account/gateway",
    ],
)
def test_gateway_rejects_misrouted_requests_before_sending_credentials(
    settings: Settings, url: str
) -> None:
    configured = settings.model_copy(
        update={
            "use_ai_gateway": True,
            "cf_ai_gateway_url": HttpUrl(url),
            "cf_ai_gateway_token": SecretStr("test-gateway-token"),
        }
    )
    with pytest.raises(ConfigurationError, match="CF_AI_GATEWAY_URL") as error:
        embedding_route(configured)
    assert "private-input" not in str(error.value)


def test_bm25_converts_numpy_vectors_without_dense_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fastembed
    import numpy as np

    class FakeBM25:
        def __init__(self, model_name: str, *, cache_dir: str, threads: int) -> None:
            assert model_name == "Qdrant/bm25" and threads == 1

        def embed(self, documents: Sequence[str]) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(indices=np.array([12]), values=np.array([1.5])) for _ in documents
            ]

    monkeypatch.setattr(fastembed, "SparseTextEmbedding", FakeBM25)
    vectors = BM25Embedder(tmp_path).embed(["bronze vessel"])
    assert vectors[0].indices == [12] and vectors[0].values == [1.5]
