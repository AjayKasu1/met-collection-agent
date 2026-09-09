"""Share lazy local model instances and serialize CPU inference across requests."""

import time
from threading import Lock

import structlog
from qdrant_client import QdrantClient, models

from met_agent.config import Settings
from met_agent.retrieval.embeddings import BM25Embedder, IdentifiedEmbedder
from met_agent.retrieval.hybrid import HybridRetriever, ScoreBreakdown, SearchFilters
from met_agent.retrieval.providers import create_embedder
from met_agent.retrieval.qdrant_store import HybridStore
from met_agent.retrieval.rerank import LocalReranker
from met_agent.retrieval.schema import ImageVectorSpec


class SearchService:
    """Do no model I/O on construction or health checks; validate the index before inference."""

    def __init__(self, client: QdrantClient, settings: Settings) -> None:
        self.client, self.settings = client, settings
        self._lock = Lock()
        self._dense: IdentifiedEmbedder | None = None
        self._sparse: BM25Embedder | None = None
        self._reranker: LocalReranker | None = None
        self.warmed = False

    def _models(self) -> tuple[IdentifiedEmbedder, BM25Embedder, LocalReranker]:
        """Called under the inference lock so concurrent requests share one model set."""
        cache = self.settings.model_cache_dir or self.settings.data_dir / "models"
        if self._dense is None:
            self._dense = create_embedder(self.settings)
        if self._sparse is None:
            self._sparse = BM25Embedder(cache, local_files_only=self.settings.models_offline)
        if self._reranker is None:
            self._reranker = LocalReranker(
                cache,
                threads=self.settings.embedding_threads,
                local_files_only=self.settings.models_offline,
            )
        return self._dense, self._sparse, self._reranker

    def warmup(self) -> None:
        """Exercise local inference once; never spend remote embedding quota on a probe."""
        with self._lock:
            if self.warmed:
                return
            started = time.monotonic()
            dense, sparse, reranker = self._models()
            loaded = time.monotonic()
            query = "Museum collection and visitor information"
            if self.settings.embedding_provider == "local":
                dense.embed([query], purpose="query")
            sparse.embed([query])
            reranker.score(query, ["Museum visitor information and collection records."])
            self.warmed = True
            structlog.get_logger(__name__).info(
                "retrieval_warmed",
                model_load_ms=(loaded - started) * 1000,
                inference_ms=(time.monotonic() - loaded) * 1000,
                provider=self.settings.embedding_provider,
            )

    def store(self, collection: str) -> HybridStore:
        info = self.client.get_collection(collection)
        raw = (info.config.metadata or {}).get("image")
        image = ImageVectorSpec.model_validate(raw) if raw is not None else None
        store = HybridStore(
            self.client,
            collection,
            self.settings.embedding_model,
            self.settings.embedding_dimensions,
            embedding_provider=self.settings.embedding_provider,
            image=image,
        )
        store.validate_collection()
        return store

    def search(
        self, collection: str, query: str, *, filters: SearchFilters | None = None, k: int = 8
    ) -> list[tuple[models.ScoredPoint, ScoreBreakdown]]:
        queued = time.monotonic()
        with self._lock:
            acquired = time.monotonic()
            store = self.store(collection)
            validated = time.monotonic()
            dense, sparse, reranker = self._models()
            loaded = time.monotonic()
            result = HybridRetriever(store, dense, sparse, reranker).search(
                query, filters=filters, k=k
            )
            structlog.get_logger(__name__).info(
                "retrieval_request_timing",
                queue_wait_ms=(acquired - queued) * 1000,
                index_validation_ms=(validated - acquired) * 1000,
                model_load_ms=(loaded - validated) * 1000,
                search_ms=(time.monotonic() - loaded) * 1000,
                total_ms=(time.monotonic() - queued) * 1000,
            )
            return result

    def gallery(
        self, collection: str, gallery_number: str, *, k: int = 5
    ) -> list[tuple[models.ScoredPoint, ScoreBreakdown]]:
        """Use the indexed gallery keyword without loading local inference models."""
        filters = SearchFilters(gallery_number=gallery_number)
        if not 1 <= k <= 40:
            raise ValueError("Gallery lookup requires k from 1 to 40")
        started = time.monotonic()
        with self._lock:
            self.store(collection)
            records, _ = self.client.scroll(
                collection,
                scroll_filter=filters.qdrant(),
                limit=k,
                with_payload=True,
                with_vectors=False,
            )
        results: list[tuple[models.ScoredPoint, ScoreBreakdown]] = []
        for rank, record in enumerate(records, 1):
            payload = record.payload or {}
            if not payload.get("text"):
                raise ValueError("Retrieved gallery records lack their indexed text")
            reciprocal_rank = 1 / (60 + rank)
            results.append(
                (
                    models.ScoredPoint(
                        id=record.id,
                        version=0,
                        score=reciprocal_rank,
                        payload=payload,
                    ),
                    ScoreBreakdown(rrf=reciprocal_rank, rerank=1.0),
                )
            )
        structlog.get_logger(__name__).info(
            "retrieval_timing",
            mode="gallery_filter",
            exact_filter_ms=(time.monotonic() - started) * 1000,
            candidates=len(results),
        )
        return results
