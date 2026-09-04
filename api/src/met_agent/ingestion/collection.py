"""Select public-domain CSV records, prioritize highlights and images, and retain raw facts."""

import csv
import heapq
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError

from met_agent.ingestion.http import SourceClient, SourceError
from met_agent.ingestion.models import CollectionObject
from met_agent.ingestion.storage import write_json

CSV_URL = "https://media.githubusercontent.com/media/metmuseum/openaccess/master/MetObjects.csv"
_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
_FIELDS = {
    "title": "Title",
    "artist_display_name": "Artist Display Name",
    "accession_number": "Object Number",
    "culture": "Culture",
    "period": "Period",
    "object_date": "Object Date",
    "medium": "Medium",
    "classification": "Classification",
    "department": "Department",
    "credit_line": "Credit Line",
    "gallery_number": "Gallery Number",
    "tags": "Tags",
    "curatorial_description": "Curatorial Description",
}


class ImageInventory(BaseModel):
    """Validate the API's complete image-bearing ID list before using it for ranking."""

    total: int = Field(ge=0)
    objectIDs: list[int] | None = None


def image_inventory(
    source: SourceClient, api_base: str, cache: Path, *, refresh: bool = False
) -> set[int]:
    """Reuse a public API inventory, with count checks to catch truncated or invalid responses."""
    if cache.exists() and not refresh:
        inventory = ImageInventory.model_validate_json(cache.read_bytes())
    else:
        data = source.json(api_base.rstrip("/") + "/search", params={"hasImages": "true", "q": "*"})
        inventory = ImageInventory.model_validate(data)
    ids = set(inventory.objectIDs or [])
    if not ids or len(ids) != inventory.total or any(item <= 0 for item in ids):
        raise SourceError("Image inventory was empty, truncated, or invalid")
    write_json(cache, inventory.model_dump())
    return ids


def _year(value: str) -> int | None:
    try:
        return int(value) if value.strip() else None
    except ValueError:
        raise SourceError("CSV contains an invalid object year") from None


def object_from_row(row: Mapping[str, str], image_ids: set[int]) -> CollectionObject:
    """Map source columns explicitly; an absent description remains absent."""
    if row.get("Is Public Domain", "").strip().lower() != "true":
        raise SourceError("Only explicitly public-domain rows may be prepared")
    object_id = int(row["Object ID"])
    fields = {name: row.get(column, "").strip() for name, column in _FIELDS.items()}
    if not fields["curatorial_description"]:
        fields["curatorial_description"] = row.get("Description", "").strip()
    return CollectionObject.model_validate(
        {
            **fields,
            "object_id": object_id,
            "object_begin_date": _year(row.get("Object Begin Date", "")),
            "object_end_date": _year(row.get("Object End Date", "")),
            "is_highlight": row.get("Is Highlight", "").strip().lower() == "true",
            "has_image": object_id in image_ids,
            "source_url": row["Link Resource"].strip(),
            "raw_fields": dict(row),
        }
    )


def select_objects(path: Path, *, limit: int, image_ids: set[int]) -> list[CollectionObject]:
    """Scan the CSV once using O(limit) retained records, with deterministic tie breaking."""
    if limit <= 0:
        raise ValueError("Object limit must be positive")

    def eligible() -> Iterator[CollectionObject]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"Object ID", "Is Public Domain", "Link Resource", "Title"}
            if not required.issubset(reader.fieldnames or []):
                raise SourceError("CSV is missing required collection columns")
            for row in reader:
                if row.get("Is Public Domain", "").strip().lower() != "true":
                    continue
                if not row.get("Link Resource", "").strip():
                    continue
                if any(key is None or value is None for key, value in row.items()):
                    raise SourceError("CSV contains a malformed row")
                yield object_from_row(row, image_ids)

    selected = heapq.nsmallest(
        limit, eligible(), key=lambda obj: (not obj.is_highlight, not obj.has_image, obj.object_id)
    )
    if not selected:
        raise SourceError("No eligible public-domain objects were found")
    if len({obj.object_id for obj in selected}) != len(selected):
        raise SourceError("CSV contains duplicate selected object IDs")
    return selected


def enrich_object(obj: CollectionObject, source: SourceClient, api_base: str) -> CollectionObject:
    """Refresh images and gallery fields from live metadata without inventing missing text."""
    data = _JSON_OBJECT.validate_python(
        source.json(f"{api_base.rstrip('/')}/objects/{obj.object_id}")
    )
    if data.get("objectID") != obj.object_id:
        raise SourceError("Live API returned a different object ID")
    if data.get("isPublicDomain") is not True:
        raise SourceError(f"Object {obj.object_id} is no longer marked public domain")

    def text(name: str) -> str:
        value = data.get(name, "")
        return value.strip() if isinstance(value, str) else ""

    updates: dict[str, object] = {
        "live_fields": data,
        "fetched_at": datetime.now(UTC),
        "primary_image": text("primaryImage"),
        "primary_image_small": text("primaryImageSmall"),
        "has_image": bool(text("primaryImage") or text("primaryImageSmall")),
        "gallery_number": text("GalleryNumber"),
    }
    if text("objectURL"):
        updates["source_url"] = text("objectURL")
    if text("curatorialDescription"):
        updates["curatorial_description"] = text("curatorialDescription")
    try:
        return CollectionObject.model_validate({**obj.model_dump(), **updates})
    except ValidationError:
        raise SourceError("Live object metadata failed validation") from None
