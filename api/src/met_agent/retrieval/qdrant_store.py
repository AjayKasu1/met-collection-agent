"""Own the hybrid collection schema and idempotent, acknowledged batch writes."""

import math
import time
from collections.abc import Sequence
from typing import Literal

import structlog
from qdrant_client import QdrantClient, models

from met_agent.ingestion.models import IndexDocument
from met_agent.retrieval.embeddings import DenseEmbedder, SparseEmbedder

logger = structlog.get_logger(__name__)

SCHEMA_VERSION: Literal[2] = 2
PAYLOAD_INDEXES = {
    "department": models.PayloadSchemaType.KEYWORD,
    "object_id": models.PayloadSchemaType.INTEGER,
    "is_highlight": models.PayloadSchemaType.BOOL,
    "gallery_number": models.PayloadSchemaType.KEYWORD,
    "object_begin_date": models.PayloadSchemaType.INTEGER,
    "object_end_date": models.PayloadSchemaType.INTEGER,
    "source_url": models.PayloadSchemaType.KEYWORD,
}


class IndexCompatibilityError(RuntimeError):
    """Refuse to mix models or schemas in an existing index."""


class HybridStore:
    """Share schema and vector validation across collection and visitor ingestion."""

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        embedding_model: str,
        embedding_dimensions: int = 768,
    ) -> None:
        if embedding_dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive")
        self.client = client
        self.collection = collection
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions

    def ensure_collection(self, dimensions: int) -> None:
        """Create a disk-backed quantized index, or verify an existing compatible one."""
        if dimensions != self.embedding_dimensions:
            raise IndexCompatibilityError("Embedding dimensions differ from the configured index")
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "embedding_model": self.embedding_model,
            "sparse_model": "Qdrant/bm25",
            "dimensions": self.embedding_dimensions,
        }
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            sparse = info.config.params.sparse_vectors or {}
            dense = vectors.get("dense") if isinstance(vectors, dict) else None
            if (
                dense is None
                or dense.size != dimensions
                or dense.distance != models.Distance.COSINE
                or "sparse" not in sparse
                or sparse["sparse"].modifier != models.Modifier.IDF
                or info.config.metadata != metadata
            ):
                raise IndexCompatibilityError(
                    "Existing collection uses a different embedding model or schema"
                )
        else:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "dense": models.VectorParams(
                        size=dimensions,
                        distance=models.Distance.COSINE,
                        on_disk=True,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=True),
                        modifier=models.Modifier.IDF,
                    )
                },
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        always_ram=True,
                    )
                ),
                on_disk_payload=True,
                metadata=metadata,
            )
        indexes = self.client.get_collection(self.collection).payload_schema
        for name, schema in PAYLOAD_INDEXES.items():
            if name not in indexes:
                self.client.create_payload_index(
                    self.collection, name, field_schema=schema, wait=True
                )

    def ingest(
        self,
        documents: Sequence[IndexDocument],
        dense: DenseEmbedder,
        sparse: SparseEmbedder,
        *,
        batch_size: int,
        batch_delay_seconds: float = 0,
    ) -> int:
        """Upsert deterministic IDs; a failed batch never produces partially paired vectors."""
        if batch_size <= 0:
            raise ValueError("Embedding batch size must be positive")
        if not math.isfinite(batch_delay_seconds) or batch_delay_seconds < 0:
            raise ValueError("Embedding batch delay must be finite and nonnegative")
        if documents and self.client.collection_exists(self.collection):
            self.ensure_collection(self.embedding_dimensions)
        count = 0
        dimensions: int | None = None
        for offset in range(0, len(documents), batch_size):
            if offset and batch_delay_seconds:
                time.sleep(batch_delay_seconds)
            batch = documents[offset : offset + batch_size]
            texts = [document.text for document in batch]
            dense_vectors = dense.embed(texts)
            sparse_vectors = sparse.embed(texts)
            if len(dense_vectors) != len(batch) or len(sparse_vectors) != len(batch):
                raise IndexCompatibilityError("Embedding count does not match the batch")
            sizes = {len(vector) for vector in dense_vectors}
            if len(sizes) != 1 or 0 in sizes:
                raise IndexCompatibilityError("Dense vectors have inconsistent dimensions")
            size = next(iter(sizes))
            if dimensions is None:
                self.ensure_collection(size)
                dimensions = size
            elif dimensions != size:
                raise IndexCompatibilityError("Embedding dimensions changed between batches")
            points = [
                models.PointStruct(
                    id=document.point_id,
                    payload=document.payload,
                    vector={"dense": dense_vector, "sparse": sparse_vector},
                )
                for document, dense_vector, sparse_vector in zip(
                    batch, dense_vectors, sparse_vectors, strict=True
                )
            ]
            self.client.upsert(self.collection, points=points, wait=True)
            count += len(points)
            logger.info("embedding_batch_indexed", count=count, total=len(documents))
        return count

    def remove_stale_page_chunks(self, source_url: str, retained_ids: Sequence[str]) -> None:
        """Remove obsolete chunks only after every replacement chunk was acknowledged."""
        if not retained_ids:
            raise ValueError("Cannot replace a page with no chunks")
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_url", match=models.MatchValue(value=source_url)
                        )
                    ],
                    must_not=[models.HasIdCondition(has_id=list(retained_ids))],
                )
            ),
            wait=True,
        )
