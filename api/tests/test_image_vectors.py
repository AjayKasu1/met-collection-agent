"""Validate published image provenance, coverage, values, and the image tool contract."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from met_agent.ingestion import image_vectors
from met_agent.ingestion.http import SourceError
from met_agent.ingestion.image_vectors import ImageManifest, join_image_shards, load_images
from met_agent.retrieval.schema import ImageVectorSpec
from met_agent.tools.schemas import FIND_SIMILAR_OBJECTS_TOOL, FindSimilarObjectsArguments

SOURCE = ImageVectorSpec(
    dataset="metmuseum/test", revision="a" * 40, model="test-image", dimensions=2
)


def shard(path: Path, **updates: object) -> Path:
    pl.DataFrame(
        {
            "objectID": [1, 2, 99],
            "embedding": [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            "model": ["test-image"] * 3,
            "dim": [2] * 3,
            **updates,
        }
    ).write_parquet(path)
    return path


def test_published_image_join_preserves_values_and_declares_absences(tmp_path: Path) -> None:
    path = shard(tmp_path / "source.parquet")
    manifest = join_image_shards([path], {1, 2, 3}, SOURCE, tmp_path)
    assert manifest.included_ids == [1, 2] and manifest.missing_ids == [3]
    restored, vectors = load_images(tmp_path, {1, 2, 3})
    assert restored == manifest and vectors == {1: [1.0, 0.0], 2: [0.0, 1.0]}
    with pytest.raises(SourceError, match="selected"):
        load_images(tmp_path, {1})
    with pytest.raises(ValidationError):
        ImageManifest.model_validate({**manifest.model_dump(), "missing_ids": [1, 3]})
    (tmp_path / "image_vectors.parquet").write_bytes(b"corrupt")
    with pytest.raises(SourceError, match="artifact"):
        load_images(tmp_path, {1, 2, 3})


@pytest.mark.parametrize(
    "updates",
    [
        {"objectID": [1, 1, 99]},
        {"model": ["other"] * 3},
        {"dim": [3] * 3},
        {"embedding": [[0.0, 0.0]] * 3},
        {"embedding": [[float("nan"), 1.0]] * 3},
    ],
)
def test_malformed_published_vectors_are_rejected(
    tmp_path: Path, updates: dict[str, object]
) -> None:
    with pytest.raises(SourceError):
        join_image_shards(
            [shard(tmp_path / "source.parquet", **updates)], {1, 2, 3}, SOURCE, tmp_path
        )


def test_similar_objects_contract_is_bounded_and_image_only() -> None:
    assert FindSimilarObjectsArguments(object_id=1).k == 5
    assert "image" in str(FIND_SIMILAR_OBJECTS_TOOL)
    for data in (
        {"object_id": 0},
        {"object_id": 1, "k": 51},
        {"object_id": "1"},
        {"object_id": 1, "unexpected": True},
    ):
        with pytest.raises(ValidationError):
            FindSimilarObjectsArguments.model_validate(data)


def test_image_download_is_pinned_and_never_discovers_private_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard(tmp_path / "source.parquet")
    head = SimpleNamespace(sha=SOURCE.revision)
    downloads: list[dict[str, Any]] = []

    def api(*, token: bool) -> SimpleNamespace:
        assert token is False
        return SimpleNamespace(repo_info=lambda *a, **kw: head)

    def download(repo: str, **kwargs: Any) -> str:
        assert repo == SOURCE.dataset
        downloads.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(image_vectors, "HfApi", api)
    monkeypatch.setattr(image_vectors, "snapshot_download", download)
    manifest = image_vectors.prepare_images(tmp_path / "joined", {1, 2, 3}, SOURCE)
    assert manifest.missing_ids == [3]
    assert downloads[0]["revision"] == SOURCE.revision
    assert downloads[0]["token"] is False
    assert downloads[0]["allow_patterns"] == ["*.parquet"]
    head.sha = "b" * 40
    with pytest.raises(SourceError, match="pinned"):
        image_vectors.prepare_images(tmp_path / "rejected", {1}, SOURCE)
    assert len(downloads) == 1
