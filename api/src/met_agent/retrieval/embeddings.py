"""Validate dense provider responses and build BM25 sparse vectors without loading torch."""

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError
from qdrant_client import models

from met_agent.retrieval.schema import EmbeddingIdentity


class EmbeddingError(RuntimeError):
    """A sanitized provider failure, safe to print without leaking keys or request data."""


class ModelUnavailableError(EmbeddingError):
    """Stop ingestion when the configured provider model is unavailable."""


class EmbeddingRateLimitError(EmbeddingError):
    """Expose only safe scheduling information from a provider quota rejection."""

    def __init__(self, *, retry_after: float = 61, daily: bool = False) -> None:
        super().__init__("Embedding rate limit reached; quota scheduling must wait")
        self.retry_after = retry_after
        self.daily = daily


def _rate_limit(error: Exception) -> EmbeddingRateLimitError:
    response = getattr(error, "response", None)
    data = getattr(error, "body", None)
    try:
        if response is not None:
            data = response.json()
        elif isinstance(data, str):
            data = json.loads(data)
    except (ValueError, AttributeError):
        data = None
    seconds = 61.0
    daily = False
    if isinstance(data, dict):
        envelope = data.get("error", data)
        details = envelope.get("details", []) if isinstance(envelope, dict) else []
        for detail in details if isinstance(details, list) else []:
            if not isinstance(detail, dict):
                continue
            delay = detail.get("retryDelay")
            if isinstance(delay, str):
                try:
                    candidate = float(delay.removesuffix("s"))
                    if math.isfinite(candidate) and candidate >= 0:
                        seconds = max(seconds, candidate)
                except ValueError:
                    pass
            violations = detail.get("violations", [])
            for violation in violations if isinstance(violations, list) else []:
                if isinstance(violation, dict):
                    quota_id = str(violation.get("quotaId", "")).lower()
                    daily |= "perday" in quota_id or "per_day" in quota_id
    return EmbeddingRateLimitError(retry_after=seconds, daily=daily)


class EmbeddingTransport(Protocol):
    """Narrow typed seam around LiteLLM's provider response."""

    def embedding(
        self, *, model: str, input: list[str], task_type: str, dimensions: int
    ) -> object: ...


class DenseEmbedder(Protocol):
    """The same embedding contract is used in deterministic tests and real ingestion."""

    def embed(
        self, texts: Sequence[str], *, purpose: Literal["document", "query"] = "document"
    ) -> list[list[float]]: ...


class SparseEmbedder(Protocol):
    """Sparse encoders return Qdrant's typed index/value arrays."""

    def embed(self, texts: Sequence[str]) -> list[models.SparseVector]: ...


class IdentifiedEmbedder(DenseEmbedder, Protocol):
    identity: EmbeddingIdentity


class _Vector(BaseModel):
    index: int = Field(ge=0)
    embedding: list[float] = Field(min_length=1)


class _Response(BaseModel):
    data: list[_Vector]


class GeminiEmbedder:
    """Use a configured Router and reject malformed, reordered, or non-finite vectors."""

    def __init__(
        self,
        router: EmbeddingTransport,
        *,
        output_dimensionality: int = 768,
        model_name: str = "gemini-embedding-001",
    ) -> None:
        if output_dimensionality <= 0 or output_dimensionality > 3072:
            raise ValueError("Embedding output dimensionality must be between 1 and 3072")
        self.router = router
        self.output_dimensionality = output_dimensionality
        self.identity = EmbeddingIdentity(
            provider="gemini", model=model_name, dimensions=output_dimensionality
        )

    def embed(
        self, texts: Sequence[str], *, purpose: Literal["document", "query"] = "document"
    ) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self.router.embedding(
                model="embedding",
                input=list(texts),
                task_type="RETRIEVAL_DOCUMENT" if purpose == "document" else "RETRIEVAL_QUERY",
                # LiteLLM maps dimensions to Gemini's native outputDimensionality field.
                dimensions=self.output_dimensionality,
            )
        except Exception as error:
            if (
                type(error).__name__ == "RateLimitError"
                or getattr(error, "status_code", None) == 429
            ):
                raise _rate_limit(error) from None
            if (
                type(error).__name__ == "NotFoundError"
                or getattr(error, "status_code", None) == 404
            ):
                raise ModelUnavailableError(
                    "Embedding model was not found; check EMBEDDING_MODEL before retrying"
                ) from None
            raise EmbeddingError(f"Embedding request failed ({type(error).__name__})") from None
        try:
            payload = _Response.model_validate(response, from_attributes=True)
            ordered = sorted(payload.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(texts))):
                raise ValueError("Embedding response indices do not match the input batch")
            vectors = [item.embedding for item in ordered]
            if {len(vector) for vector in vectors} != {self.output_dimensionality} or any(
                not all(math.isfinite(value) for value in vector) or not any(vector)
                for vector in vectors
            ):
                raise ValueError("Embedding response contains invalid vector values")
            # Gemini Embedding 1 requires normalization when requesting reduced dimensions.
            normalized = []
            for vector in vectors:
                norm = math.hypot(*vector)
                if not math.isfinite(norm) or norm == 0:
                    raise ValueError("Embedding norm is invalid")
                normalized.append([value / norm for value in vector])
            return normalized
        except (ValidationError, ValueError):
            raise EmbeddingError(
                "Embedding response failed shape and finite-value validation"
            ) from None


class BM25Embedder:
    """Lazily load FastEmbed's fixed BM25 model and keep its cache in the data directory."""

    def __init__(self, cache_dir: Path) -> None:
        from fastembed import SparseTextEmbedding

        self.model = SparseTextEmbedding("Qdrant/bm25", cache_dir=str(cache_dir), threads=1)

    def embed(self, texts: Sequence[str]) -> list[models.SparseVector]:
        return [
            models.SparseVector(indices=vector.indices.tolist(), values=vector.values.tolist())
            for vector in self.model.embed(list(texts))
        ]
