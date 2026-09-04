"""Verify fusion, rerank order, model gates, and record-date/display filters offline."""

from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from met_agent.ingestion.models import IndexDocument
from met_agent.retrieval import rerank
from met_agent.retrieval.embeddings import EmbeddingError
from met_agent.retrieval.hybrid import HybridRetriever, SearchFilters
from met_agent.retrieval.qdrant_store import HybridStore, IndexCompatibilityError
from met_agent.retrieval.schema import EmbeddingIdentity

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")


class Dense:
    identity = EmbeddingIdentity(provider="local", model="fixture", dimensions=2)

    def embed(
        self, texts: Sequence[str], *, purpose: Literal["document", "query"] = "document"
    ) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class Sparse:
    def embed(self, texts: Sequence[str]) -> list[models.SparseVector]:
        return [models.SparseVector(indices=[1], values=[1.0]) for _ in texts]


class Rank:
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        return [float(text.rsplit(" ", 1)[-1]) for text in documents]


def test_fusion_reranking_filters_and_model_identity() -> None:
    with closing(QdrantClient(location=":memory:")) as client:
        store = HybridStore(client, "test", "fixture", 2, embedding_provider="local")
        docs = [
            IndexDocument(
                point_id=i,
                text=f"Object {i}",
                payload={
                    "text": f"Object {i}",
                    "department": "Paintings" if i < 3 else "Sculpture",
                    "object_begin_date": 1800 + i,
                    "object_end_date": 1900 + i,
                    "gallery_number": "131" if i == 1 else "",
                },
            )
            for i in (1, 2, 3)
        ]
        store.ingest(docs, Dense(), Sparse(), batch_size=3)
        retrieval = HybridRetriever(store, Dense(), Sparse(), Rank())
        result = retrieval.search("object", k=2)
        assert [point.id for point, _ in result] == [3, 2]
        assert all(
            score.dense is not None and score.sparse is not None and score.rrf > 1 / 61
            for _, score in result
        )
        assert [
            point.id for point, _ in retrieval.search("object", filters=SearchFilters(on_view=True))
        ] == [1]
        assert [
            point.id
            for point, _ in retrieval.search(
                "object",
                filters=SearchFilters(
                    on_view=False, department="Paintings", date_from=1850, date_to=1855
                ),
            )
        ] == [2]
        assert retrieval.search("object", filters=SearchFilters(date_from=2000)) == []
        with pytest.raises(ValueError):
            retrieval.search("", k=1)
        with pytest.raises(ValueError):
            retrieval.search("object", k=41)
        with pytest.raises(ValidationError):
            SearchFilters(date_from=2000, date_to=1000)
        bad = Dense()
        bad.identity = Dense.identity.model_copy(update={"model": "other"})
        with pytest.raises(IndexCompatibilityError):
            HybridRetriever(store, bad, Sparse(), Rank()).search("object")


def test_reranker_uses_pinned_assets_and_rejects_malformed_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    state = SimpleNamespace(scores=[0.4], fail=False)

    def download(*args: Any, **kwargs: Any) -> str:
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError("uncached")
        return str(tmp_path)

    class CrossEncoder:
        @staticmethod
        def list_supported_models() -> list[dict[str, str]]:
            return [{"model": rerank.MODEL}]

        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["specific_model_path"] == str(tmp_path)

        def rerank(self, *args: Any, **kwargs: Any) -> list[float]:
            if state.fail:
                raise RuntimeError("private-input")
            return [float(value) for value in state.scores]

    monkeypatch.setattr(rerank, "snapshot_download", download)
    monkeypatch.setattr(rerank, "TextCrossEncoder", CrossEncoder)
    model = rerank.LocalReranker(tmp_path)
    assert all(c["revision"] == rerank.REVISION and c["token"] is False for c in calls)
    assert model.score("vase", []) == []
    assert model.score("vase", ["Vessel"]) == [0.4]
    for scores in ([], [float("nan")]):
        state.scores = scores
        with pytest.raises(EmbeddingError):
            model.score("vase", ["Vessel"])
    state.fail = True
    with pytest.raises(EmbeddingError, match="reranking failed"):
        model.score("vase", ["Vessel"])


def test_rerank_receives_only_top_twenty_fused_candidates() -> None:
    class BoundedRank:
        def score(self, query: str, documents: Sequence[str]) -> list[float]:
            assert len(documents) == 20
            return [1.0] * len(documents)

    with closing(QdrantClient(location=":memory:")) as client:
        store = HybridStore(client, "bounded", "fixture", 2, embedding_provider="local")
        store.ingest(
            [
                IndexDocument(point_id=i, text=f"Object {i}", payload={"text": f"Object {i}"})
                for i in range(1, 61)
            ],
            Dense(),
            Sparse(),
            batch_size=60,
        )
        results = HybridRetriever(store, Dense(), Sparse(), BoundedRank()).search("object", k=40)
        assert len(results) == 20


def test_small_reranker_registration_is_idempotent_and_batches_are_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registered: list[dict[str, Any]] = []

    class Encoder:
        @staticmethod
        def list_supported_models() -> list[dict[str, Any]]:
            return registered

        @staticmethod
        def add_custom_model(**kwargs: Any) -> None:
            registered.append(kwargs)

        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["model_name"] == rerank.MODEL
            assert kwargs["specific_model_path"] == str(tmp_path)

        def rerank(self, query: str, documents: list[str], *, batch_size: int) -> list[float]:
            assert batch_size == 2
            return [1.0 for _ in documents]

    monkeypatch.setattr(rerank, "TextCrossEncoder", Encoder)
    monkeypatch.setattr(rerank, "snapshot_download", lambda *args, **kwargs: str(tmp_path))
    for _ in range(2):
        assert rerank.LocalReranker(tmp_path).score("Query", ["A", "B"]) == [1.0, 1.0]
    assert len(registered) == 1
    assert registered[0]["sources"].hf == rerank.MODEL
