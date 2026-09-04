"""Own the hybrid collection schema and idempotent, acknowledged batch writes."""

import math
import time
from collections.abc import Mapping, Sequence
from typing import Literal

import structlog
from qdrant_client import QdrantClient, models

from met_agent.ingestion.models import IndexDocument
from met_agent.retrieval.embeddings import DenseEmbedder, IdentifiedEmbedder, SparseEmbedder
from met_agent.retrieval.schema import EmbeddingIdentity, EmbeddingProvider, ImageVectorSpec

logger = structlog.get_logger(__name__)

SCHEMA_VERSION: Literal[3] = 3
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
        *,
        embedding_provider: EmbeddingProvider = "gemini",
        image: ImageVectorSpec | None = None,
    ) -> None:
        if embedding_dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive")
        self.client = client
        self.collection = collection
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.embedding_provider = embedding_provider
        self.image = image

    @property
    def identity(self) -> EmbeddingIdentity:
        return EmbeddingIdentity(
            provider=self.embedding_provider,
            model=self.embedding_model,
            dimensions=self.embedding_dimensions,
        )

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "sparse_model": "Qdrant/bm25",
            "dimensions": self.embedding_dimensions,
            "image": self.image.model_dump() if self.image else None,
        }

    def validate_collection(self) -> None:
        """Read-only compatibility gate used before querying, writing, or exporting."""
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        sparse = info.config.params.sparse_vectors or {}
        dense = vectors.get("dense") if isinstance(vectors, dict) else None
        image = vectors.get("image") if isinstance(vectors, dict) else None
        names = {"dense", "image"} if self.image else {"dense"}
        if (
            dense is None
            or dense.size != self.embedding_dimensions
            or dense.distance != models.Distance.COSINE
            or not isinstance(vectors, dict)
            or set(vectors) != names
            or "sparse" not in sparse
            or sparse["sparse"].modifier != models.Modifier.IDF
            or info.config.metadata != self.metadata
            or (
                self.image is not None
                and (
                    image is None
                    or image.size != self.image.dimensions
                    or image.distance != models.Distance.COSINE
                )
            )
        ):
            raise IndexCompatibilityError(
                "Existing collection uses a different embedding provider, model, or schema"
            )

    def embed_query(self, query: str, embedder: IdentifiedEmbedder) -> list[float]:
        """Reject incompatible query models before embedding or searching an index."""
        if embedder.identity != self.identity:
            raise IndexCompatibilityError("Query embedding model differs from the index")
        self.validate_collection()
        vectors = embedder.embed([query], purpose="query")
        if len(vectors) != 1 or len(vectors[0]) != self.embedding_dimensions:
            raise IndexCompatibilityError("Query embedding dimensions differ from the index")
        return vectors[0]

    def ensure_collection(self, dimensions: int) -> None:
        """Create a disk-backed quantized index, or verify an existing compatible one."""
        if dimensions != self.embedding_dimensions:
            raise IndexCompatibilityError("Embedding dimensions differ from the configured index")
        if self.client.collection_exists(self.collection):
            self.validate_collection()
        else:
            vectors = {
                "dense": models.VectorParams(
                    size=dimensions,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                )
            }
            if self.image:
                vectors["image"] = models.VectorParams(
                    size=self.image.dimensions,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                )
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=vectors,
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
                metadata=self.metadata,
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
        images: Mapping[int, list[float]] | None = None,
    ) -> int:
        """Upsert deterministic IDs; a failed batch never produces partially paired vectors."""
        if batch_size <= 0:
            raise ValueError("Embedding batch size must be positive")
        if not math.isfinite(batch_delay_seconds) or batch_delay_seconds < 0:
            raise ValueError("Embedding batch delay must be finite and nonnegative")
        if (images is not None) != (self.image is not None):
            raise IndexCompatibilityError("Image vectors require matching collection provenance")
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
            if images is not None and self.image is not None:
                for point in points:
                    vector = images.get(int(point.id))
                    if vector is not None:
                        if (
                            len(vector) != self.image.dimensions
                            or not all(math.isfinite(value) for value in vector)
                            or not math.isclose(math.hypot(*vector), 1.0, abs_tol=1e-4)
                        ):
                            raise IndexCompatibilityError("Image vector differs from its schema")
                        if not isinstance(point.vector, dict):
                            raise IndexCompatibilityError("Expected named vectors")
                        point.vector["image"] = vector
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
