"""Exercise provider response contracts and explicit credential routing without network calls."""

import logging
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import HttpUrl, SecretStr

from met_agent.config import Settings
from met_agent.llm.router import create_embedding_router, embedding_route
from met_agent.retrieval.embeddings import (
    BM25Embedder,
    EmbeddingError,
    GeminiEmbedder,
    ModelUnavailableError,
)


class ResponseTransport:
    def __init__(self, response: object) -> None:
        self.response = response
        self.purpose = ""

    def embedding(self, *, model: str, input: list[str], task_type: str) -> object:
        self.purpose = task_type
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
    embedder = GeminiEmbedder(transport)
    assert embedder.embed(["first", "second"]) == [[1, 0], [0, 1]]
    assert transport.purpose == "RETRIEVAL_DOCUMENT"
    assert embedder.embed([]) == []
    transport.response = {"data": [{"index": 0, "embedding": [1, 1]}]}
    assert embedder.embed(["question"], purpose="query") == [[1, 1]]
    assert transport.purpose == "RETRIEVAL_QUERY"


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
