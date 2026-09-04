"""Atomically persist source artifacts and preserve nested raw records in portable Parquet."""

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from met_agent.ingestion.models import CollectionObject, VisitorChunk


def atomic_write(path: Path, content: bytes) -> None:
    """Publish a complete artifact, leaving the previous version intact on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".writing-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    """Hash files incrementally so snapshot verification does not require extra RAM."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_objects(path: Path, objects: Sequence[CollectionObject]) -> None:
    """Expose useful scalar columns while preserving the complete record as JSON."""
    if not objects:
        raise ValueError("Cannot publish an empty collection artifact")
    rows = [
        {
            "object_id": obj.object_id,
            "title": obj.title,
            "artist": obj.artist_display_name,
            "department": obj.department,
            "gallery_number": obj.gallery_number,
            "is_highlight": obj.is_highlight,
            "has_image": obj.has_image,
            "object_begin_date": obj.object_begin_date,
            "object_end_date": obj.object_end_date,
            "source_url": obj.source_url,
            "document": obj.document().text,
            "payload_json": obj.model_dump_json(),
        }
        for obj in objects
    ]
    frame = pl.DataFrame(
        rows,
        schema_overrides={
            "object_id": pl.Int64,
            "object_begin_date": pl.Int64,
            "object_end_date": pl.Int64,
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".parquet-", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.write_parquet(temporary, compression="zstd")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_objects(path: Path) -> list[CollectionObject]:
    """Validate every stored record before reuse or publication."""
    payloads = pl.read_parquet(path, columns=["payload_json"]).get_column("payload_json")
    return [CollectionObject.model_validate_json(str(value)) for value in payloads]


def save_chunks(path: Path, chunks: Sequence[VisitorChunk]) -> None:
    """Save UTF-8 JSONL without overwriting the last good file on an interrupted write."""
    atomic_write(path, ("\n".join(chunk.model_dump_json() for chunk in chunks) + "\n").encode())


def write_json(path: Path, data: object) -> None:
    """Write deterministic readable metadata alongside downloaded artifacts."""
    atomic_write(
        path, (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    )
