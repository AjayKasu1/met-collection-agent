"""Verify hybrid schemas, repeatable writes, and page replacement on a real local Qdrant server."""

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient, models

from met_agent.ingestion.models import IndexDocument
from met_agent.retrieval.qdrant_store import PAYLOAD_INDEXES, HybridStore, IndexCompatibilityError


class TestDense:
    __test__ = False

    def embed(
        self, texts: Sequence[str], *, purpose: Literal["document", "query"] = "document"
    ) -> list[list[float]]:
        return [[1.0, len(text) / 100, 0.5, 0.25] for text in texts]


class TestSparse:
    __test__ = False

    def embed(self, texts: Sequence[str]) -> list[models.SparseVector]:
        return [models.SparseVector(indices=[1, 2], values=[0.5, 1.0]) for _ in texts]


@pytest.fixture
def local_store() -> Iterator[HybridStore]:
    client = QdrantClient(url="http://127.0.0.1:6333", timeout=60, check_compatibility=False)
    try:
        client.info()
    except Exception:
        client.close()
        pytest.skip("Local Qdrant is unavailable at 127.0.0.1:6333")
    name = "test_ingestion_" + uuid4().hex
    store = HybridStore(client, name, "test-embedding-v1", 4)
    try:
        yield store
    finally:
        if client.collection_exists(name):
            client.delete_collection(name)
        client.close()


@pytest.mark.integration
def test_twenty_object_hybrid_ingest_is_idempotent(local_store: HybridStore) -> None:
    documents = [
        IndexDocument(
            point_id=i,
            text=f"Bronze vessel {i}",
            payload={
                "object_id": i,
                "department": "Greek and Roman Art",
                "raw_fields": {"Title": f"Vessel {i}"},
            },
        )
        for i in range(1, 21)
    ]
    for _ in range(2):
        assert local_store.ingest(documents, TestDense(), TestSparse(), batch_size=7) == 20
    client = local_store.client
    info = client.get_collection(local_store.collection)
    assert client.count(local_store.collection, exact=True).count == 20
    assert isinstance(info.config.params.vectors, dict)
    assert info.config.params.vectors["dense"].on_disk
    assert info.config.metadata and info.config.metadata["dimensions"] == 4
    assert info.config.metadata["schema_version"] == 3
    sparse = info.config.params.sparse_vectors
    assert sparse and sparse["sparse"].modifier == models.Modifier.IDF
    assert sparse["sparse"].index and sparse["sparse"].index.on_disk
    assert isinstance(info.config.quantization_config, models.ScalarQuantization)
    assert info.config.quantization_config.scalar.type == models.ScalarType.INT8
    assert set(PAYLOAD_INDEXES) <= set(info.payload_schema)
    points = client.retrieve(local_store.collection, ids=[1], with_vectors=True)
    assert points[0].payload and points[0].payload["raw_fields"] == {"Title": "Vessel 1"}
    assert isinstance(points[0].vector, dict) and set(points[0].vector) == {"dense", "sparse"}
    with pytest.raises(IndexCompatibilityError):
        HybridStore(client, local_store.collection, "different-model", 4).ensure_collection(4)
    with pytest.raises(IndexCompatibilityError):
        local_store.ensure_collection(3)


@pytest.mark.integration
def test_visitor_replacement_only_deletes_obsolete_chunks_on_same_page(
    local_store: HybridStore,
) -> None:
    first, stale, other = [str(uuid4()) for _ in range(3)]
    documents = [
        IndexDocument(point_id=point_id, text="Visitor information", payload={"source_url": url})
        for point_id, url in [
            (first, "https://museum/visit"),
            (stale, "https://museum/visit"),
            (other, "https://museum/access"),
        ]
    ]
    local_store.ingest(documents, TestDense(), TestSparse(), batch_size=3)
    with pytest.raises(ValueError):
        local_store.remove_stale_page_chunks("https://museum/visit", [])
    local_store.remove_stale_page_chunks("https://museum/visit", [first])
    points, _ = local_store.client.scroll(local_store.collection)
    assert {point.id for point in points} == {first, other}


def test_invalid_batch_size_is_rejected_without_network() -> None:
    client = QdrantClient(location=":memory:")
    try:
        with pytest.raises(ValueError):
            HybridStore(client, "test", "test-model", 4).ingest(
                [], TestDense(), TestSparse(), batch_size=0
            )
    finally:
        client.close()


def test_batch_pacing_waits_only_between_batches_and_validates_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[float] = []
    monkeypatch.setattr("met_agent.retrieval.qdrant_store.time.sleep", waits.append)
    client = QdrantClient(location=":memory:")
    store = HybridStore(client, "paced", "test-model", 4)
    monkeypatch.setattr(store, "ensure_collection", lambda _: None)
    monkeypatch.setattr(client, "upsert", lambda *a, **kw: None)
    documents = [IndexDocument(point_id=i, text="Vessel", payload={}) for i in range(1, 6)]
    try:
        assert (
            store.ingest(documents, TestDense(), TestSparse(), batch_size=2, batch_delay_seconds=30)
            == 5
        )
        assert waits == [30, 30]
        for invalid in (-1, float("inf"), float("nan")):
            with pytest.raises(ValueError, match="finite and nonnegative"):
                store.ingest(
                    documents, TestDense(), TestSparse(), batch_size=2, batch_delay_seconds=invalid
                )
    finally:
        client.close()


@pytest.mark.integration
@pytest.mark.parametrize("with_images", [False, True])
def test_snapshot_round_trip_without_embedding_and_refuses_overwrite(
    local_store: HybridStore, tmp_path: Path, with_images: bool
) -> None:
    import httpx

    from met_agent.ingestion.artifacts import (
        export_bundle,
        restore_bundle,
        validate_files,
        verify_index,
    )
    from met_agent.ingestion.http import SourceError
    from met_agent.ingestion.models import CollectionObject
    from met_agent.ingestion.storage import save_objects

    objects = [
        CollectionObject(
            object_id=i,
            title=f"Vessel {i}",
            source_url=f"https://www.metmuseum.org/art/collection/search/{i}",
            raw_fields={"Title": f"Vessel {i}"},
        )
        for i in range(1, 21)
    ]
    save_objects(tmp_path / "objects.parquet", objects)
    documents = [obj.document() for obj in objects]
    images = None
    if with_images:
        import polars as pl

        from met_agent.ingestion.image_vectors import join_image_shards, load_images
        from met_agent.retrieval.schema import ImageVectorSpec

        source = ImageVectorSpec(
            dataset="metmuseum/test", revision="a" * 40, model="test-image", dimensions=2
        )
        local_store.image = source
        shard = tmp_path / "source.parquet"
        pl.DataFrame(
            {
                "objectID": [1, 2],
                "embedding": [[1.0, 0.0], [0.0, 1.0]],
                "model": ["test-image"] * 2,
                "dim": [2, 2],
            }
        ).write_parquet(shard)
        join_image_shards([shard], {obj.object_id for obj in objects}, source, tmp_path)
        _, images = load_images(tmp_path, {obj.object_id for obj in objects})
    local_store.ingest(documents, TestDense(), TestSparse(), batch_size=7, images=images)
    restored = "test_restored_" + uuid4().hex
    client = local_store.client
    with httpx.Client(base_url="http://127.0.0.1:6333/", timeout=30) as http:
        manifest = export_bundle(
            client,
            http,
            tmp_path,
            tmp_path / "bundle",
            {"collection": local_store.collection},
            local_store.embedding_model,
            4,
        )
        assert manifest.indexes[0].points == 20
        assert manifest.indexes[0].dimensions == 4
        assert manifest.indexes[0].schema_version == 3
        assert manifest.indexes[0].image == local_store.image
        assert manifest.indexes[0].image_points == (2 if with_images else 0)
        try:
            with pytest.raises(IndexCompatibilityError, match="EMBEDDING_PROVIDER"):
                restore_bundle(
                    client,
                    http,
                    tmp_path / "bundle",
                    {"collection": restored},
                    local_store.embedding_model,
                    4,
                    embedding_provider="local",
                )
            restore_bundle(
                client,
                http,
                tmp_path / "bundle",
                {"collection": restored},
                local_store.embedding_model,
                4,
            )
            assert client.count(restored, exact=True).count == 20
            with pytest.raises(IndexCompatibilityError, match="overwrite"):
                restore_bundle(
                    client,
                    http,
                    tmp_path / "bundle",
                    {"collection": restored},
                    local_store.embedding_model,
                    4,
                )
            with pytest.raises(IndexCompatibilityError, match="EMBEDDING_MODEL"):
                restore_bundle(
                    client,
                    http,
                    tmp_path / "bundle",
                    {"collection": "unused"},
                    "different-model",
                    4,
                )
            with pytest.raises(IndexCompatibilityError, match="payloads"):
                verify_index(
                    client,
                    restored,
                    [documents[0].model_copy(update={"payload": {"private": "unexpected"}})],
                )
            with pytest.raises(IndexCompatibilityError, match="empty or duplicate"):
                verify_index(client, restored, [])
            (tmp_path / "bundle" / "collection.snapshot").write_bytes(b"corrupt")
            with pytest.raises(SourceError, match="integrity"):
                validate_files(tmp_path / "bundle", manifest)
        finally:
            if client.collection_exists(restored):
                client.delete_collection(restored)


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")
def test_embedded_store_pairs_vectors_and_preserves_model_identity() -> None:
    client = QdrantClient(location=":memory:")
    store = HybridStore(client, "offline", "test-model", 4)
    first, stale = str(uuid4()), str(uuid4())
    documents = [
        IndexDocument(
            point_id=point_id, text="Bronze vessel", payload={"source_url": "https://museum/visit"}
        )
        for point_id in (first, stale)
    ]
    try:
        for _ in range(2):
            assert store.ingest(documents, TestDense(), TestSparse(), batch_size=1) == 2
        assert client.count("offline", exact=True).count == 2
        points = client.retrieve("offline", ids=[first], with_vectors=True)
        assert isinstance(points[0].vector, dict) and set(points[0].vector) == {"dense", "sparse"}
        with pytest.raises(IndexCompatibilityError):
            HybridStore(client, "offline", "different-model", 4).ensure_collection(4)
        store.remove_stale_page_chunks("https://museum/visit", [first])
        assert client.count("offline", exact=True).count == 1
        client.update_collection(
            "offline",
            metadata={
                "schema_version": 2,
                "embedding_model": "test-model",
                "sparse_model": "Qdrant/bm25",
                "dimensions": 768,
            },
        )
        with pytest.raises(IndexCompatibilityError, match="schema"):
            store.ensure_collection(4)
    finally:
        client.close()


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")
@pytest.mark.parametrize("failure", ["count", "empty_vector", "dimension_drift"])
def test_bad_embedding_batches_never_write_unpaired_points(failure: str) -> None:
    class BadDense(TestDense):
        def __init__(self) -> None:
            self.calls = 0

        def embed(
            self, texts: Sequence[str], *, purpose: Literal["document", "query"] = "document"
        ) -> list[list[float]]:
            self.calls += 1
            if failure == "count":
                return []
            if failure == "empty_vector":
                return [[]]
            return [[1.0] * (self.calls + 2)]

    client = QdrantClient(location=":memory:")
    store = HybridStore(client, "offline", "test-model", 3)
    documents = [IndexDocument(point_id=i, text="Vessel", payload={}) for i in (1, 2)]
    try:
        with pytest.raises(IndexCompatibilityError):
            store.ingest(documents, BadDense(), TestSparse(), batch_size=1)
        if failure == "dimension_drift":
            assert client.count("offline", exact=True).count == 1
        else:
            assert not client.collection_exists("offline")
    finally:
        client.close()
