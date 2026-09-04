"""Select public-domain CSV records, prioritize highlights and images, and retain raw facts."""

import csv
import heapq
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError

from met_agent.ingestion.http import SourceClient, SourceError
from met_agent.ingestion.models import CollectionObject
from met_agent.ingestion.storage import load_objects, sha256_file, write_json

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


class SelectionReport(BaseModel):
    """Record golden coverage and source eligibility exceptions alongside a prepared artifact."""

    required_ids: list[int]
    included_ids: list[int]
    excluded_ids: dict[int, Literal["not_public_domain", "missing_source_url"]]
    artifact_sha256: str = ""


class CollectionSelection(BaseModel):
    """Keep selected records and their eligibility audit together."""

    objects: list[CollectionObject]
    report: SelectionReport


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


def select_objects(
    path: Path, *, limit: int, image_ids: set[int], must_include_ids: frozenset[int] = frozenset()
) -> CollectionSelection:
    """Reserve eligible golden records before filling a bounded, deterministic sample."""
    if limit <= 0:
        raise ValueError("Object limit must be positive")
    if any(object_id <= 0 for object_id in must_include_ids):
        raise ValueError("Required object IDs must be positive")
    seen_required: set[int] = set()
    exclusions: dict[int, Literal["not_public_domain", "missing_source_url"]] = {}

    def eligible() -> Iterator[CollectionObject]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"Object ID", "Is Public Domain", "Link Resource", "Title"}
            if not required.issubset(reader.fieldnames or []):
                raise SourceError("CSV is missing required collection columns")
            for row in reader:
                if any(key is None or value is None for key, value in row.items()):
                    raise SourceError("CSV contains a malformed row")
                object_id = int(row["Object ID"])
                required_id = object_id in must_include_ids
                if required_id:
                    if object_id in seen_required:
                        raise SourceError("CSV contains duplicate required object IDs")
                    seen_required.add(object_id)
                if row.get("Is Public Domain", "").strip().lower() != "true":
                    if required_id:
                        exclusions[object_id] = "not_public_domain"
                    continue
                if not row.get("Link Resource", "").strip():
                    if required_id:
                        exclusions[object_id] = "missing_source_url"
                    continue
                yield object_from_row(row, image_ids)

    selected = heapq.nsmallest(
        limit,
        eligible(),
        key=lambda obj: (
            obj.object_id not in must_include_ids,
            not obj.is_highlight,
            not obj.has_image,
            obj.object_id,
        ),
    )
    missing = must_include_ids - seen_required
    if missing:
        raise SourceError(f"Required object IDs are absent from the CSV: {sorted(missing)}")
    required_eligible = must_include_ids - exclusions.keys()
    if len(required_eligible) > limit:
        raise SourceError(
            f"Object limit must accommodate {len(required_eligible)} eligible golden IDs"
        )
    if not selected:
        raise SourceError("No eligible public-domain objects were found")
    if len({obj.object_id for obj in selected}) != len(selected):
        raise SourceError("CSV contains duplicate selected object IDs")
    return CollectionSelection(
        objects=selected,
        report=SelectionReport(
            required_ids=sorted(must_include_ids),
            included_ids=sorted(required_eligible),
            excluded_ids=exclusions,
        ),
    )


def reuse_selection(
    output: Path, *, limit: int, must_include_ids: frozenset[int]
) -> CollectionSelection:
    """Refuse stale preparation or truncation that would silently drop required golden records."""
    report_path = output / "selection.json"
    if not report_path.exists():
        raise SourceError("Prepared selection report is missing; rerun without --reuse-prepared")
    report = SelectionReport.model_validate_json(report_path.read_bytes())
    artifact = output / "objects.parquet"
    if report.artifact_sha256 != sha256_file(artifact):
        raise SourceError("Prepared artifact does not match its selection report; prepare it again")
    if set(report.required_ids) != must_include_ids:
        raise SourceError("Golden IDs changed after preparation; prepare the collection again")
    if (
        set(report.included_ids) & report.excluded_ids.keys()
        or set(report.included_ids) | report.excluded_ids.keys() != must_include_ids
    ):
        raise SourceError("Prepared selection report has inconsistent golden coverage")
    objects = load_objects(artifact)[:limit]
    if not set(report.included_ids).issubset(obj.object_id for obj in objects):
        raise SourceError("Prepared sample or limit omits required golden IDs; prepare it again")
    return CollectionSelection(objects=objects, report=report)


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
