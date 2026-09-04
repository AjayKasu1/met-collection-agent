"""Validate collection records and visitor chunks before they enter a retrieval index."""

from datetime import datetime
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


class IndexDocument(BaseModel):
    """The provider-independent contract between source preparation and vector storage."""

    point_id: int | str
    text: str
    payload: dict[str, JsonValue]


class CollectionObject(BaseModel):
    """Normalized factual fields plus the untouched source record and optional live metadata."""

    model_config = ConfigDict(frozen=True)
    object_id: Annotated[int, Field(gt=0)]
    title: str
    artist_display_name: str = ""
    accession_number: str = ""
    culture: str = ""
    period: str = ""
    object_date: str = ""
    medium: str = ""
    classification: str = ""
    department: str = ""
    credit_line: str = ""
    gallery_number: str = ""
    tags: str = ""
    curatorial_description: str = ""
    object_begin_date: int | None = None
    object_end_date: int | None = None
    is_highlight: bool = False
    is_public_domain: Literal[True] = True
    has_image: bool = False
    primary_image: str = ""
    primary_image_small: str = ""
    source_url: str
    raw_fields: dict[str, str]
    live_fields: dict[str, JsonValue] = Field(default_factory=dict)
    fetched_at: datetime | None = None

    @field_validator("source_url")
    @classmethod
    def validate_source(cls, value: str) -> str:
        """Keep object citations on the museum's own domain."""
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {"www.metmuseum.org", "metmuseum.org"}
            or parsed.username is not None
        ):
            raise ValueError("Object source must be a public Met URL")
        return value

    def document(self) -> IndexDocument:
        """Render only supplied record fields, without generating interpretive text."""
        labels = {
            "Object ID": str(self.object_id),
            "Title": self.title,
            "Artist": self.artist_display_name,
            "Accession number": self.accession_number,
            "Culture": self.culture,
            "Period": self.period,
            "Date": self.object_date,
            "Medium": self.medium,
            "Classification": self.classification,
            "Department": self.department,
            "Credit line": self.credit_line,
            "Gallery number (at ingestion)": self.gallery_number,
            "Tags": self.tags,
            "Met curatorial description": self.curatorial_description,
        }
        text = "\n".join(f"{label}: {value}" for label, value in labels.items() if value)
        payload = self.model_dump(mode="json")
        payload["text"] = text
        payload["source_kind"] = "collection"
        return IndexDocument(point_id=self.object_id, text=text, payload=payload)


class VisitorChunk(BaseModel):
    """A source-attributed text window with a stable point identity and fetch timestamp."""

    point_id: str
    source_url: str
    page_title: str
    section_heading: str
    fetched_at: datetime
    chunk_index: int
    provenance: str = "explicit_manifest"
    text: str

    def document(self) -> IndexDocument:
        """Retain provenance alongside the exact chunk text used for embedding."""
        payload = self.model_dump(mode="json")
        payload["source_kind"] = "visitor_info"
        return IndexDocument(point_id=self.point_id, text=self.text, payload=payload)
