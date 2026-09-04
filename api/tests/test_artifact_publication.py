"""Test publication allowlists, pinned downloads, and integrity checks with an offline Hub."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from met_agent.ingestion import artifacts
from met_agent.ingestion.artifacts import Artifact, IndexSpec, Manifest
from met_agent.ingestion.http import SourceError
from met_agent.ingestion.models import CollectionObject
from met_agent.ingestion.storage import save_objects, sha256_file


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    obj = CollectionObject(
        object_id=1,
        title="Vessel",
        source_url="https://www.metmuseum.org/art/collection/search/1",
        raw_fields={},
    )
    save_objects(source / "objects.parquet", [obj])
    (source / "collection.snapshot").write_bytes(b"snapshot-test-data")
    (source / "README.md").write_text("Synthetic dataset card")
    manifest = Manifest(
        created_at=datetime.now(UTC),
        qdrant_version="1.19.0",
        files={
            name: Artifact(
                sha256=sha256_file(source / name), size_bytes=(source / name).stat().st_size
            )
            for name in ("objects.parquet", "collection.snapshot", "README.md")
        },
        indexes=[
            IndexSpec(kind="collection", points=1, dimensions=4, embedding_model="test-model")
        ],
    )
    (source / "manifest.json").write_text(manifest.model_dump_json())
    (source / "private-input.txt").write_text("must never upload")
    return source


def test_publish_exact_allowlist_and_pin_every_download(
    bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str, str | bool]] = []
    uploaded: list[str] = []

    class FakeHub:
        def __init__(self, *, token: str | bool) -> None:
            self.token = token

        def repo_info(self, repo: str, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(sha="a" * 40)

        def create_commit(self, **kwargs: Any) -> SimpleNamespace:
            assert kwargs["parent_commit"] == "a" * 40
            uploaded.extend(op.path_in_repo for op in kwargs["operations"])
            return SimpleNamespace(oid="b" * 40)

    def download(
        repo: str, *, filename: str, revision: str, token: str | bool, **kwargs: Any
    ) -> str:
        calls.append((filename, revision, token))
        return str(bundle / filename)

    monkeypatch.setattr(artifacts, "HfApi", FakeHub)
    monkeypatch.setattr(artifacts, "hf_hub_download", download)
    assert artifacts.publish_bundle(bundle, "account/dataset", "unit-token") == "b" * 40
    assert set(uploaded) == {"manifest.json", "objects.parquet", "collection.snapshot", "README.md"}
    target = tmp_path / "download"
    manifest = artifacts.download_bundle("account/dataset", "main", target)
    assert manifest.indexes[0].points == 1
    assert all(revision == "a" * 40 and token is False for _, revision, token in calls)
    assert (target / "revision.txt").read_text().strip() == "a" * 40


def test_manifest_rejects_extra_paths_wrong_counts_and_versions(bundle: Path) -> None:
    manifest = Manifest.model_validate_json((bundle / "manifest.json").read_bytes())
    data = manifest.model_dump()
    data["files"]["../private-input.txt"] = data["files"]["objects.parquet"]
    with pytest.raises(ValidationError):
        Manifest.model_validate(data)
    updates: tuple[dict[str, Any], ...] = (
        {"format_version": 999},
        {"created_at": datetime(2026, 1, 1)},
        {"indexes": []},
        {"files": {}},
    )
    for update in updates:
        with pytest.raises(ValidationError):
            Manifest.model_validate({**manifest.model_dump(), **update})
    bad = manifest.model_copy(
        update={"indexes": [manifest.indexes[0].model_copy(update={"points": 2})]}
    )
    with pytest.raises(SourceError, match="count"):
        artifacts.validate_files(bundle, bad)
    (bundle / "objects.parquet").write_bytes(b"corrupted")
    with pytest.raises(SourceError, match="integrity"):
        artifacts.publish_bundle(bundle, "account/dataset", "unit-token")


def test_restore_preflight_rejects_version_model_and_existing_targets(
    bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import Mock

    import httpx
    from qdrant_client import QdrantClient

    from met_agent.retrieval.qdrant_store import IndexCompatibilityError

    client = Mock(spec=QdrantClient)
    client.info.return_value = SimpleNamespace(version="1.18.0")
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: pytest.fail("Restore contacted server before validation")
        )
    ) as http:
        with pytest.raises(IndexCompatibilityError, match="version"):
            artifacts.restore_bundle(
                client, http, bundle, {"collection": "target"}, "test-model", 4
            )
        client.info.return_value = SimpleNamespace(version="1.19.0")
        with pytest.raises(IndexCompatibilityError, match="EMBEDDING_DIMENSIONS"):
            artifacts.restore_bundle(
                client, http, bundle, {"collection": "target"}, "test-model", 768
            )
        with pytest.raises(IndexCompatibilityError, match="EMBEDDING_MODEL"):
            artifacts.restore_bundle(
                client, http, bundle, {"collection": "target"}, "wrong-model", 4
            )
        client.collection_exists.return_value = True
        with pytest.raises(IndexCompatibilityError, match="overwrite"):
            artifacts.restore_bundle(
                client, http, bundle, {"collection": "target"}, "test-model", 4
            )


def test_visitor_bundle_requires_both_source_and_snapshot(bundle: Path) -> None:
    from met_agent.ingestion.models import VisitorChunk
    from met_agent.ingestion.storage import save_chunks

    chunk = VisitorChunk(
        point_id="00000000-0000-0000-0000-000000000001",
        source_url="https://www.metmuseum.org/visit",
        page_title="Hours",
        section_heading="Hours",
        fetched_at=datetime.now(UTC),
        chunk_index=0,
        text="Closed Wednesday",
    )
    save_chunks(bundle / "visitor_chunks.jsonl", [chunk])
    assert artifacts.load_documents(bundle, "visitor") == [chunk.document()]
    manifest = Manifest.model_validate_json((bundle / "manifest.json").read_bytes())
    data = manifest.model_dump()
    data["indexes"].append(
        IndexSpec(kind="visitor", points=1, dimensions=4, embedding_model="test-model").model_dump()
    )
    with pytest.raises(ValidationError, match="file set"):
        Manifest.model_validate(data)
