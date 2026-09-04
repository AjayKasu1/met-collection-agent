"""Validate dense provider responses and build BM25 sparse vectors without loading torch."""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError
from qdrant_client import models


class EmbeddingError(RuntimeError):
    """A sanitized provider failure, safe to print without leaking keys or request data."""


class ModelUnavailableError(EmbeddingError):
    """Stop ingestion when the configured provider model is unavailable."""


class EmbeddingTransport(Protocol):
    """Narrow typed seam around LiteLLM's provider response."""

    def embedding(self, *, model: str, input: list[str], task_type: str) -> object: ...


class DenseEmbedder(Protocol):
    """The same embedding contract is used in deterministic tests and real ingestion."""

    def embed(
        self, texts: Sequence[str], *, purpose: Literal["document", "query"] = "document"
    ) -> list[list[float]]: ...


class SparseEmbedder(Protocol):
    """Sparse encoders return Qdrant's typed index/value arrays."""

    def embed(self, texts: Sequence[str]) -> list[models.SparseVector]: ...


class _Vector(BaseModel):
    index: int = Field(ge=0)
    embedding: list[float] = Field(min_length=1)


class _Response(BaseModel):
    data: list[_Vector]


class GeminiEmbedder:
    """Use a configured Router and reject malformed, reordered, or non-finite vectors."""

    def __init__(self, router: EmbeddingTransport) -> None:
        self.router = router

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
            )
        except Exception as error:
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
            if len({len(vector) for vector in vectors}) != 1 or any(
                not all(math.isfinite(value) for value in vector) or not any(vector)
                for vector in vectors
            ):
                raise ValueError("Embedding response contains invalid vector values")
            return vectors
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
