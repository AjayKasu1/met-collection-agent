"""Share lazy local model instances and serialize CPU inference across requests."""

from threading import Lock

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
        with self._lock:
            store = self.store(collection)
            if self._dense is None:
                self._dense = create_embedder(self.settings)
            if self._sparse is None:
                self._sparse = BM25Embedder(self.settings.data_dir / "models")
            if self._reranker is None:
                self._reranker = LocalReranker(
                    self.settings.data_dir / "models", threads=self.settings.embedding_threads
                )
            return HybridRetriever(store, self._dense, self._sparse, self._reranker).search(
                query, filters=filters, k=k
            )
