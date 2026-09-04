"""Fuse dense and BM25 candidates, preserve score evidence, and rerank bounded results."""

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from qdrant_client import models

from met_agent.retrieval.embeddings import IdentifiedEmbedder, SparseEmbedder
from met_agent.retrieval.qdrant_store import HybridStore, IndexCompatibilityError


class SearchFilters(BaseModel):
    """Apply date overlap and gallery observations without implying current display status."""

    model_config = ConfigDict(extra="forbid", strict=True)
    department: str | None = Field(default=None, min_length=1, max_length=200)
    date_from: int | None = Field(default=None, ge=-100000, le=3000)
    date_to: int | None = Field(default=None, ge=-100000, le=3000)
    on_view: bool | None = None

    @model_validator(mode="after")
    def ordered_dates(self) -> "SearchFilters":
        if (
            self.date_from is not None
            and self.date_to is not None
            and self.date_from > self.date_to
        ):
            raise ValueError("date_from must not exceed date_to")
        return self

    def qdrant(self) -> models.Filter:
        must: list[models.Condition] = []
        must_not: list[models.Condition] = []
        if self.department:
            must.append(
                models.FieldCondition(
                    key="department", match=models.MatchValue(value=self.department)
                )
            )
        if self.date_from is not None:
            must.append(
                models.FieldCondition(key="object_end_date", range=models.Range(gte=self.date_from))
            )
        if self.date_to is not None:
            must.append(
                models.FieldCondition(key="object_begin_date", range=models.Range(lte=self.date_to))
            )
        absent: list[models.Condition] = [
            models.FieldCondition(key="gallery_number", match=models.MatchValue(value="")),
            models.IsEmptyCondition(is_empty=models.PayloadField(key="gallery_number")),
        ]
        if self.on_view is True:
            must_not.extend(absent)
        elif self.on_view is False:
            must.append(models.Filter(should=absent))
        return models.Filter(must=must, must_not=must_not)


class ScoreBreakdown(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    dense: float | None = None
    sparse: float | None = None
    rrf: float = Field(ge=0)
    rerank: float


class RetrievedObject(BaseModel):
    object_id: int
    title: str
    source_url: str
    text: str
    record: dict[str, JsonValue]
    scores: ScoreBreakdown


class RetrievedChunk(BaseModel):
    point_id: str
    source_url: str
    page_title: str
    section_heading: str
    fetched_at: str
    text: str
    scores: ScoreBreakdown


class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class HybridRetriever:
    """Use 40 candidates per vector, reciprocal rank fusion, then cross-encoder scores."""

    def __init__(
        self,
        store: HybridStore,
        dense: IdentifiedEmbedder,
        sparse: SparseEmbedder,
        reranker: Reranker,
    ) -> None:
        self.store, self.dense, self.sparse, self.reranker = store, dense, sparse, reranker

    def search(
        self, query: str, *, filters: SearchFilters | None = None, k: int = 8
    ) -> list[tuple[models.ScoredPoint, ScoreBreakdown]]:
        if not query.strip() or not 1 <= k <= 40:
            raise ValueError("Search requires a query and k from 1 to 40")
        vector = self.store.embed_query(query, self.dense)
        sparse_vectors = self.sparse.embed([query])
        if len(sparse_vectors) != 1:
            raise IndexCompatibilityError("Sparse query vector count differs from input")
        condition = filters.qdrant() if filters else None
        dense = self.store.client.query_points(
            self.store.collection,
            query=vector,
            using="dense",
            query_filter=condition,
            limit=40,
            with_payload=True,
        ).points
        sparse = self.store.client.query_points(
            self.store.collection,
            query=sparse_vectors[0],
            using="sparse",
            query_filter=condition,
            limit=40,
            with_payload=True,
        ).points
        points: dict[int | str, models.ScoredPoint] = {}
        scores: dict[int | str, dict[str, float]] = {}
        for kind, matches in (("dense", dense), ("sparse", sparse)):
            for rank, point in enumerate(matches, 1):
                point_id = point.id if isinstance(point.id, int) else str(point.id)
                points[point_id] = point
                score = scores.setdefault(point_id, {"rrf": 0.0})
                score[kind] = point.score
                score["rrf"] += 1 / (60 + rank)
        ordered = sorted(points, key=lambda point_id: (-scores[point_id]["rrf"], str(point_id)))
        if not ordered:
            return []
        texts = [str((points[point_id].payload or {}).get("text", "")) for point_id in ordered]
        if any(not text for text in texts):
            raise IndexCompatibilityError("Retrieved records lack their indexed text")
        reranked = self.reranker.score(query, texts)
        if len(reranked) != len(ordered):
            raise IndexCompatibilityError("Reranker score count differs from candidates")
        results = [
            (points[point_id], ScoreBreakdown(**scores[point_id], rerank=score))
            for point_id, score in zip(ordered, reranked, strict=True)
        ]
        return sorted(results, key=lambda item: (-item[1].rerank, -item[1].rrf, str(item[0].id)))[
            :k
        ]
