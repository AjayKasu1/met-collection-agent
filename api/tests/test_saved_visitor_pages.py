"""Verify capture provenance and local-file boundaries without crawling or provider requests."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from met_agent.ingestion import saved_pages
from met_agent.ingestion.http import SourceError
from met_agent.ingestion.saved_pages import load_saved_pages


@pytest.fixture
def saved_source(tmp_path: Path) -> dict[str, Any]:
    (tmp_path / "hours.html").write_text(
        "<title>Visit</title><main><h1>Hours</h1>Closed Wednesday.</main>"
    )
    return {
        "label": "Hours",
        "url": "https://www.metmuseum.org/plan-your-visit",
        "html_file": "hours.html",
        "fetched_at": "2026-09-03T18:00:00-04:00",
    }


def save_manifest(path: Path, pages: list[dict[str, Any]]) -> None:
    path.write_text(yaml.safe_dump({"pages": pages}))


def test_saved_pages_preserve_capture_time_and_source_url(
    tmp_path: Path, saved_source: dict[str, Any]
) -> None:
    manifest = tmp_path / "sources.yaml"
    save_manifest(manifest, [saved_source])
    page = load_saved_pages(tmp_path)[0]
    assert page.fetched_at == datetime(2026, 9, 3, 22, tzinfo=UTC)
    assert page.url == saved_source["url"]
    assert "Closed Wednesday" in page.html
    alternate = tmp_path / "alternate.yaml"
    manifest.rename(alternate)
    assert load_saved_pages(tmp_path, manifest=alternate) == [page]


@pytest.mark.parametrize(
    "change",
    [
        {"fetched_at": "2026-09-03T18:00:00"},
        {"fetched_at": None},
        {"html_file": None},
        {"unknown": "private-input"},
    ],
)
def test_manifest_rejects_missing_or_ambiguous_provenance(
    tmp_path: Path, saved_source: dict[str, Any], change: dict[str, Any]
) -> None:
    save_manifest(tmp_path / "sources.yaml", [{**saved_source, **change}])
    with pytest.raises(SourceError, match="manifest") as error:
        load_saved_pages(tmp_path)
    assert "private-input" not in str(error.value)


def test_missing_duplicate_and_invalid_manifests_fail_without_live_fallback(
    tmp_path: Path, saved_source: dict[str, Any]
) -> None:
    with pytest.raises(SourceError, match="directory is missing"):
        load_saved_pages(tmp_path / "missing")
    with pytest.raises(SourceError, match="manifest"):
        load_saved_pages(tmp_path)
    manifest = tmp_path / "sources.yaml"
    manifest.write_text("pages: [")
    with pytest.raises(SourceError, match="manifest"):
        load_saved_pages(tmp_path)
    save_manifest(manifest, [])
    with pytest.raises(SourceError, match="manifest"):
        load_saved_pages(tmp_path)
    save_manifest(manifest, [saved_source, saved_source])
    with pytest.raises(SourceError, match="duplicate"):
        load_saved_pages(tmp_path)


@pytest.mark.parametrize(
    "filename", ["../outside.html", "/outside.html", "hours.txt", "escape.html"]
)
def test_saved_files_cannot_escape_the_directory_or_import_unrelated_files(
    tmp_path: Path, saved_source: dict[str, Any], filename: str
) -> None:
    (tmp_path / "escape.html").symlink_to(tmp_path.parent / "outside.html")
    save_manifest(tmp_path / "sources.yaml", [{**saved_source, "html_file": filename}])
    with pytest.raises(SourceError, match="inside --html-dir"):
        load_saved_pages(tmp_path)


def test_invalid_url_missing_file_encoding_and_size_are_rejected(
    tmp_path: Path, saved_source: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "sources.yaml"
    save_manifest(manifest, [{**saved_source, "url": "https://outside.example/visit"}])
    with pytest.raises(SourceError, match=r"metmuseum\.org"):
        load_saved_pages(tmp_path)
    save_manifest(manifest, [{**saved_source, "html_file": "missing.html"}])
    with pytest.raises(SourceError, match="missing or unreadable"):
        load_saved_pages(tmp_path)
    save_manifest(manifest, [saved_source])
    (tmp_path / "hours.html").write_bytes(b"\xff")
    with pytest.raises(SourceError, match="UTF-8"):
        load_saved_pages(tmp_path)
    (tmp_path / "hours.html").write_bytes(b"long HTML")
    monkeypatch.setattr(saved_pages, "MAX_HTML_BYTES", 2)
    with pytest.raises(SourceError, match="20 MiB"):
        load_saved_pages(tmp_path)
