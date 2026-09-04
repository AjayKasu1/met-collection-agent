"""Verify capture provenance and local-file boundaries without crawling or provider requests."""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from met_agent.ingestion import saved_pages
from met_agent.ingestion.http import SourceError
from met_agent.ingestion.saved_pages import load_saved_pages


@pytest.mark.parametrize("canonical", [True, False])
def test_browser_save_discovery_records_source_and_mtime(tmp_path: Path, canonical: bool) -> None:
    url = "https://www.metmuseum.org/plan-your-visit"
    source = (
        f'<link rel="canonical" href="{url}">'
        if canonical
        else (f"<!-- saved from url=(0041){url} -->")
    )
    page = tmp_path / "Visit.html"
    page.write_text(source + "<title>Visit</title><main>Closed Wednesday.</main>")
    os.utime(page, (1000, 1000))
    assets = tmp_path / "Visit_files"
    assets.mkdir()
    (assets / "tracker.html").write_text("Unrelated advertising frame")
    loaded = load_saved_pages(tmp_path)
    assert len(loaded) == 1 and loaded[0].url == url
    assert loaded[0].fetched_at == datetime.fromtimestamp(1000, UTC)
    assert loaded[0].provenance.endswith("file_mtime")
    assert (tmp_path / "sources.yaml").exists()
    os.utime(page, (2000, 2000))
    assert load_saved_pages(tmp_path)[0].fetched_at == loaded[0].fetched_at


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
