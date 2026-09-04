"""Exercise command composition with synthetic sources and no real credentials or model calls."""

import csv
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from met_agent.config import ConfigurationError, Settings
from met_agent.ingestion import commands
from met_agent.ingestion.http import SourceError
from met_agent.ingestion.models import CollectionObject, IndexDocument
from met_agent.ingestion.storage import load_objects, save_objects
from met_agent.retrieval.embeddings import EmbeddingError


@pytest.fixture
def configured_commands(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Settings:
    configured = settings.model_copy(update={"data_dir": tmp_path, "ingest_max_objects": 2})
    monkeypatch.setattr(commands, "load_settings", lambda: configured)
    return configured


@pytest.fixture
def stub_index(monkeypatch: pytest.MonkeyPatch) -> list[IndexDocument]:
    indexed: list[IndexDocument] = []

    class Store:
        def __init__(self, *args: Any) -> None:
            pass

        def ingest(self, documents: Sequence[IndexDocument], *args: Any, **kwargs: Any) -> int:
            indexed.extend(documents)
            return len(documents)

        def remove_stale_page_chunks(self, url: str, ids: list[str]) -> None:
            assert ids

    monkeypatch.setattr(commands, "HybridStore", Store)
    monkeypatch.setattr(commands, "qdrant_client", lambda _: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(commands, "create_embedding_router", lambda _: None)
    monkeypatch.setattr(commands, "GeminiEmbedder", lambda _: None)
    monkeypatch.setattr(commands, "BM25Embedder", lambda _: None)
    return indexed


def test_prepare_and_reuse_sample_without_repeating_source_calls(
    configured_commands: Settings, stub_index: list[IndexDocument], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = configured_commands.data_dir
    source = root / "source.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["Object ID", "Title", "Is Public Domain", "Link Resource"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "Object ID": i,
                    "Title": f"Vessel {i}",
                    "Is Public Domain": "True",
                    "Link Resource": f"https://www.metmuseum.org/art/collection/search/{i}",
                }
                for i in (1, 2, 3)
            ]
        )
    monkeypatch.setattr(commands, "download_csv", lambda *a, **kw: source)
    monkeypatch.setattr(commands, "image_inventory", lambda *a, **kw: {1, 2, 3})
    assert commands.ingest_collection(["--prepare-only", "--limit", "2"]) == 0
    assert [obj.object_id for obj in load_objects(root / "objects.parquet")] == [1, 2]
    assert stub_index == []
    assert (
        commands.ingest_collection(["--reuse-prepared", "--limit", "1", "--collection", "pilot"])
        == 0
    )
    assert [doc.point_id for doc in stub_index] == [1]
    with pytest.raises(ValueError, match="positive"):
        commands.ingest_collection(["--limit", "0"])
    monkeypatch.setattr(commands, "load_objects", lambda _: [])
    with pytest.raises(SourceError, match="empty"):
        commands.ingest_collection(["--reuse-prepared"])


def test_live_enrichment_is_explicit(
    configured_commands: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    obj = CollectionObject(
        object_id=1, title="Vessel", source_url="https://www.metmuseum.org/item/1", raw_fields={}
    )
    monkeypatch.setattr(commands, "download_csv", lambda *a, **kw: Path("unused"))
    monkeypatch.setattr(commands, "image_inventory", lambda *a, **kw: {1})
    monkeypatch.setattr(commands, "select_objects", lambda *a, **kw: [obj])
    monkeypatch.setattr(
        commands, "enrich_object", lambda *a: obj.model_copy(update={"gallery_number": "202"})
    )
    assert commands.ingest_collection(["--enrich-live", "--prepare-only"]) == 0
    assert load_objects(configured_commands.data_dir / "objects.parquet")[0].gallery_number == "202"


def test_visitor_prepare_and_index_preserves_provenance(
    configured_commands: Settings, stub_index: list[IndexDocument], monkeypatch: pytest.MonkeyPatch
) -> None:
    class Crawler:
        def __init__(self, *args: Any) -> None:
            pass

        def fetch(self, url: str) -> tuple[str, str]:
            return url, "<title>Visit</title><main><h1>Hours</h1><p>Closed Wednesday.</p></main>"

    monkeypatch.setattr(commands, "VisitorCrawler", Crawler)
    sources = configured_commands.data_dir / "sources.yaml"
    sources.write_text("pages:\n  - label: Hours\n    url: https://www.metmuseum.org/visit\n")
    assert commands.ingest_visitor_info(["--sources", str(sources)]) == 0
    assert stub_index[0].payload["source_url"] == "https://www.metmuseum.org/visit"
    assert (configured_commands.data_dir / "visitor_chunks.jsonl").exists()
    assert commands.ingest_visitor_info(["--sources", str(sources), "--prepare-only"]) == 0
    monkeypatch.setattr(commands, "chunk_markdown", lambda *a, **kw: [])
    with pytest.raises(SourceError, match="No visitor chunks"):
        commands.ingest_visitor_info(["--sources", str(sources)])


def test_verify_writes_report_and_returns_nonzero_for_upstream_failure(
    configured_commands: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = configured_commands.data_dir
    save_objects(
        root / "objects.parquet",
        [
            CollectionObject(
                object_id=1,
                title="Vessel",
                source_url="https://www.metmuseum.org/item/1",
                raw_fields={},
            )
        ],
    )
    golden = root / "golden.jsonl"
    golden.write_text('{"id":"lookup","question":"Which title?","expected_object_ids":[1]}\n')
    client_factory = httpx.Client

    def client(**kwargs: Any) -> httpx.Client:
        return client_factory(transport=httpx.MockTransport(lambda _: httpx.Response(403)))

    monkeypatch.setattr(httpx, "Client", client)
    assert commands.verify(["--golden", str(golden)]) == 1
    assert "unknown" in capsys.readouterr().out
    assert (root / "verify_golden.md").exists()


@pytest.mark.parametrize(
    "error",
    [
        ConfigurationError("Missing key"),
        SourceError("HTTP 403"),
        EmbeddingError("Rate limit"),
        ValueError("private-input"),
        OSError("private-input"),
        RuntimeError("private-input"),
    ],
)
def test_command_errors_never_print_untrusted_exception_bodies(
    error: Exception, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail() -> int:
        raise error

    assert commands.run_command(fail) == 1
    assert "private-input" not in capsys.readouterr().out
    assert commands.run_command(lambda: 0) == 0
