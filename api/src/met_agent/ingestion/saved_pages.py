"""Read captured visitor HTML with explicit source URLs and capture times, without HTTP access."""

import re
from datetime import UTC, datetime
from pathlib import Path

import yaml
from bs4 import BeautifulSoup, Comment
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from met_agent.ingestion.http import SourceError
from met_agent.ingestion.visitors import validate_visitor_url

MAX_HTML_BYTES = 20 * 1024 * 1024


class VisitorPageContent(BaseModel):
    """Use the same attributed input for live and locally captured visitor pages."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str = Field(min_length=1)
    url: str
    fetched_at: AwareDatetime
    provenance: str = "explicit_manifest"
    html: str

    @field_validator("url")
    @classmethod
    def check_url(cls, value: str) -> str:
        return validate_visitor_url(value)


class _SavedPage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str = Field(min_length=1)
    url: str
    html_file: Path
    fetched_at: AwareDatetime
    provenance: str = "explicit_manifest"


class _SavedManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    pages: list[_SavedPage] = Field(min_length=1)


def load_saved_pages(directory: Path, *, manifest: Path | None = None) -> list[VisitorPageContent]:
    """Load a manifest or derive provenance from top-level browser saves."""
    root = directory.resolve()
    if not root.is_dir():
        raise SourceError("Saved HTML directory is missing; create --html-dir and its sources.yaml")
    manifest_path = manifest or root / "sources.yaml"
    if manifest is None and not manifest_path.exists():
        discover_manifest(root, manifest_path)
    try:
        sources = _SavedManifest.model_validate(
            yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        )
    except (OSError, UnicodeError, yaml.YAMLError, ValueError):
        raise SourceError(
            "Saved HTML manifest must contain pages with label, url, html_file, "
            "and a timezone-aware fetched_at capture time"
        ) from None
    if len({page.url for page in sources.pages}) != len(sources.pages):
        raise SourceError("Saved HTML manifest contains duplicate source URLs")
    result = []
    for page in sources.pages:
        validate_visitor_url(page.url)
        path = (root / page.html_file).resolve()
        if (
            page.html_file.is_absolute()
            or not path.is_relative_to(root)
            or path.suffix.lower() not in {".html", ".htm"}
        ):
            raise SourceError("Saved HTML files must be .html or .htm files inside --html-dir")
        try:
            with path.open("rb") as handle:
                raw = handle.read(MAX_HTML_BYTES + 1)
        except OSError:
            raise SourceError("Saved HTML file is missing or unreadable") from None
        if len(raw) > MAX_HTML_BYTES:
            raise SourceError("Saved HTML file exceeds the 20 MiB input limit")
        try:
            html = raw.decode("utf-8-sig")
        except UnicodeError:
            raise SourceError("Saved HTML must use UTF-8 encoding") from None
        result.append(
            VisitorPageContent(
                label=page.label,
                url=page.url,
                fetched_at=page.fetched_at,
                provenance=page.provenance,
                html=html,
            )
        )
    return result


def discover_manifest(root: Path, output: Path) -> None:
    """Persist canonical or Chrome saved-from URLs and UTC mtimes for reproducible offline input."""
    pages = []
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in {".html", ".htm"}:
            continue
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            raise SourceError("Saved HTML files must stay inside --html-dir")
        if path.stat().st_size > MAX_HTML_BYTES:
            raise SourceError("Saved HTML file exceeds the 20 MiB input limit")
        try:
            soup = BeautifulSoup(path.read_text(encoding="utf-8-sig"), "html.parser")
        except (UnicodeError, OSError):
            raise SourceError("Saved HTML must be readable UTF-8") from None
        canonical = {
            str(link["href"]).strip() for link in soup.select('link[rel~="canonical"][href]')
        }
        urls = canonical
        provenance = "canonical_link_and_file_mtime"
        if not urls:
            provenance = "chrome_saved_from_and_file_mtime"
            urls = {
                match.group(1)
                for comment in soup.find_all(string=lambda value: isinstance(value, Comment))
                if (
                    match := re.fullmatch(
                        r"\s*saved from url=\(\d+\)(https://\S+)\s*", str(comment)
                    )
                )
            }
        if len(urls) != 1:
            raise SourceError(
                "Saved HTML needs a manifest: missing or ambiguous canonical/source URL"
            )
        url = validate_visitor_url(urls.pop())
        pages.append(
            {
                "label": soup.title.get_text(" ", strip=True) if soup.title else path.stem,
                "url": url,
                "html_file": path.name,
                "fetched_at": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
                "provenance": provenance,
            }
        )
    if not pages:
        raise SourceError("Saved HTML manifest requires at least one top-level HTML page")
    if len({page["url"] for page in pages}) != len(pages):
        raise SourceError("Saved HTML manifest contains duplicate source URLs")
    from met_agent.ingestion.storage import atomic_write

    atomic_write(output, yaml.safe_dump({"pages": pages}, sort_keys=False).encode())
