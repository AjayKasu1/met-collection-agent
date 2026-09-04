"""Exercise joins, recovery, and model gates with embedded Qdrant and synthetic vectors."""

from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Literal

import pytest
from qdrant_client import QdrantClient, models

from met_agent.ingestion.checkpoints import IngestionCheckpoint
from met_agent.ingestion.http import SourceError
from met_agent.ingestion.image_vectors import verify_image_index
from met_agent.ingestion.local_run import run_local_ingestion
from met_agent.ingestion.models import IndexDocument
from met_agent.retrieval.qdrant_store import HybridStore, IndexCompatibilityError
from met_agent.retrieval.schema import EmbeddingIdentity, ImageVectorSpec

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")
SOURCE = ImageVectorSpec(
    dataset="metmuseum/test", revision="a" * 40, model="test-image", dimensions=2
)


class Dense:
    def __init__(self, fail_at: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.fail_at = fail_at
        self.identity = EmbeddingIdentity(provider="local", model="test-local", dimensions=4)

    def embed(
        self, texts: Sequence[str], *, purpose: Literal["document", "query"] = "document"
    ) -> list[list[float]]:
        self.calls.append(list(texts))
        if len(self.calls) == self.fail_at:
            raise KeyboardInterrupt()
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class Sparse:
    def embed(self, texts: Sequence[str]) -> list[models.SparseVector]:
        return [models.SparseVector(indices=[1], values=[1.0]) for _ in texts]


def test_local_resume_skips_completed_batches_and_has_no_quota(tmp_path: Path) -> None:
    docs = [
        IndexDocument(point_id=i, text=f"Vessel {i}", payload={"object_id": i}) for i in (1, 2, 3)
    ]
    images = {1: [1.0, 0.0], 2: [0.0, 1.0]}
    path = tmp_path / "checkpoint.json"
    with closing(QdrantClient(location=":memory:")) as client:
        store = HybridStore(
            client, "local", "test-local", 4, embedding_provider="local", image=SOURCE
        )
        with pytest.raises(KeyboardInterrupt):
            run_local_ingestion(
                store,
                docs,
                Dense(fail_at=2),
                Sparse(),
                path,
                destination="test",
                batch_size=1,
                images=images,
                image_sha256="a" * 64,
            )
        checkpoint = IngestionCheckpoint.model_validate_json(path.read_bytes())
        assert (
            checkpoint.next_offset == 1
            and checkpoint.quota is None
            and checkpoint.status == "failed"
        )
        dense = Dense()
        assert (
            run_local_ingestion(
                store,
                docs,
                dense,
                Sparse(),
                path,
                destination="test",
                batch_size=1,
                images=images,
                image_sha256="a" * 64,
            )
            == 3
        )
        assert dense.calls == [["Vessel 2"], ["Vessel 3"]]
        assert verify_image_index(client, "local", {1, 2, 3}, images) == 2
        assert store.embed_query("Vase", dense) == [1.0, 0.0, 0.0, 0.0]
        for update in ({"provider": "gemini"}, {"model": "other"}, {"dimensions": 8}):
            dense.identity = store.identity.model_copy(update=update)
            with pytest.raises(IndexCompatibilityError, match="Query"):
                store.embed_query("Vase", dense)
        with pytest.raises(SourceError, match="changed"):
            run_local_ingestion(
                store,
                docs,
                Dense(),
                Sparse(),
                path,
                destination="test",
                images=images,
                image_sha256="b" * 64,
            )
        with pytest.raises(SourceError, match="nonempty"):
            run_local_ingestion(
                store,
                docs,
                Dense(),
                Sparse(),
                tmp_path / "other.json",
                destination="test",
                images=images,
            )
        client.delete_vectors("local", vectors=["image"], points=[1], wait=True)
        with pytest.raises(SourceError, match="differs"):
            verify_image_index(client, "local", {1, 2, 3}, images)


def test_length_grouping_resumes_by_completed_id_instead_of_input_offset(tmp_path: Path) -> None:
    docs = [
        IndexDocument(point_id=i, text=text, payload={})
        for i, text in (
            (1, "long description of a painted vessel"),
            (2, "cup"),
            (3, "bronze bowl"),
        )
    ]
    path = tmp_path / "checkpoint.json"
    with closing(QdrantClient(location=":memory:")) as client:
        store = HybridStore(client, "local", "test-local", 4, embedding_provider="local")
        with pytest.raises(KeyboardInterrupt):
            run_local_ingestion(
                store, docs, Dense(fail_at=2), Sparse(), path, destination="test", batch_size=1
            )
        saved = IngestionCheckpoint.model_validate_json(path.read_bytes())
        assert saved.completed_ids == [2]
        dense = Dense()
        run_local_ingestion(store, docs, dense, Sparse(), path, destination="test", batch_size=1)
        assert dense.calls == [["bronze bowl"], ["long description of a painted vessel"]]
