"""Read captured visitor HTML with explicit source URLs and capture times, without HTTP access."""

from pathlib import Path

import yaml
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


class _SavedManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    pages: list[_SavedPage] = Field(min_length=1)


def load_saved_pages(directory: Path, *, manifest: Path | None = None) -> list[VisitorPageContent]:
    """Require a manifest and contained HTML files; never infer source freshness from file times."""
    root = directory.resolve()
    if not root.is_dir():
        raise SourceError("Saved HTML directory is missing; create --html-dir and its sources.yaml")
    manifest_path = manifest or root / "sources.yaml"
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
                html=html,
            )
        )
    return result
