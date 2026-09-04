"""Count embedding inputs through the configured native Gemini endpoint before admission."""

from collections.abc import Sequence
from typing import Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError

from met_agent.config import Settings
from met_agent.ingestion.http import SourceError
from met_agent.llm.router import embedding_route
from met_agent.retrieval.embeddings import ModelUnavailableError


class TokenCounter(Protocol):
    def count(self, texts: Sequence[str]) -> int: ...


class _TokenCount(BaseModel):
    totalTokens: int = Field(gt=0, strict=True)


class GeminiTokenCounter:
    """Keep token counting on the same gateway/provider path and never log response bodies."""

    def __init__(self, settings: Settings, client: httpx.Client) -> None:
        route = embedding_route(settings)["litellm_params"]
        base = route.get("api_base", "https://generativelanguage.googleapis.com/v1beta")
        model = str(route["model"]).removeprefix("gemini/")
        self._url = f"{base}/models/{model}:countTokens"
        self._headers = {**route.get("extra_headers", {}), "x-goog-api-key": route["api_key"]}
        self._client = client

    def count(self, texts: Sequence[str]) -> int:
        if not texts:
            return 0
        try:
            response = self._client.post(
                self._url,
                headers=self._headers,
                json={"contents": [{"parts": [{"text": text}]} for text in texts]},
            )
        except httpx.HTTPError:
            raise SourceError("Gemini token-count request failed in transport") from None
        if response.status_code == 404:
            raise ModelUnavailableError("Embedding model was not found during token counting")
        if response.status_code != 200:
            raise SourceError(f"Gemini token-count request returned HTTP {response.status_code}")
        try:
            return _TokenCount.model_validate_json(response.content).totalTokens
        except ValidationError:
            raise SourceError("Gemini token-count response failed validation") from None
