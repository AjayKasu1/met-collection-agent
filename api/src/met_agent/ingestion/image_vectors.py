"""Join the Met's published image vectors by Object ID without running an image model."""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl
from huggingface_hub import HfApi, snapshot_download
from pydantic import BaseModel, ConfigDict, Field, model_validator
from qdrant_client import QdrantClient

from met_agent.ingestion.http import SourceError
from met_agent.ingestion.storage import atomic_write, sha256_file
from met_agent.retrieval.schema import ImageVectorSpec

SIGLIP2 = ImageVectorSpec(
    dataset="metmuseum/openaccess-embeddings-siglip2",
    revision="45fe67456ef1cffbe5ba801d5cd341fcf31cd806",
    model="google/siglip2-so400m-patch14-384",
    dimensions=1152,
)


class ImageManifest(BaseModel):
    """Keep source lineage, selected coverage, and the joined artifact's integrity together."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source: ImageVectorSpec
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    selected_ids: list[int]
    included_ids: list[int]
    missing_ids: list[int]
    source_files: dict[str, str]

    @model_validator(mode="after")
    def validate_coverage(self) -> "ImageManifest":
        selected, included, missing = map(
            set, (self.selected_ids, self.included_ids, self.missing_ids)
        )
        if (
            selected != included | missing
            or included & missing
            or any(
                len(ids) != len(set(ids))
                for ids in (self.selected_ids, self.included_ids, self.missing_ids)
            )
        ):
            raise ValueError("Image coverage contains duplicate or inconsistent IDs")
        return self


def validate_image_vector(vector: Sequence[float], dimensions: int) -> None:
    if len(vector) != dimensions or not all(math.isfinite(value) for value in vector):
        raise SourceError("Published image vector has invalid dimensions or nonfinite values")
    if not math.isclose(math.hypot(*vector), 1.0, abs_tol=1e-4):
        raise SourceError("Published image vector is not unit-normalized")


def join_image_shards(
    files: Sequence[Path], selected_ids: set[int], source: ImageVectorSpec, output: Path
) -> ImageManifest:
    """Keep source values unchanged; reject duplicates and model drift instead of guessing."""
    if not files or not selected_ids:
        raise SourceError("Image preparation requires source shards and selected object IDs")
    frames = []
    hashes = {}
    for path in files:
        hashes[path.name] = sha256_file(path)
        frame = pl.read_parquet(path).filter(pl.col("objectID").is_in(sorted(selected_ids)))
        if frame.is_empty():
            continue
        for row in frame.iter_rows(named=True):
            if row["model"] != source.model or row["dim"] != source.dimensions:
                raise SourceError("Published image rows disagree with the configured source model")
            validate_image_vector(row["embedding"], source.dimensions)
        frames.append(frame.select("objectID", "embedding"))
    if not frames:
        raise SourceError("No selected objects have a published image vector")
    joined = pl.concat(frames).sort("objectID")
    ids = joined["objectID"].to_list()
    if len(set(ids)) != len(ids):
        raise SourceError("Published image source contains duplicate selected Object IDs")
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "image_vectors.parquet.partial"
    try:
        joined.write_parquet(temporary)
        temporary.replace(output / "image_vectors.parquet")
    finally:
        temporary.unlink(missing_ok=True)
    manifest = ImageManifest(
        source=source,
        artifact_sha256=sha256_file(output / "image_vectors.parquet"),
        selected_ids=sorted(selected_ids),
        included_ids=ids,
        missing_ids=sorted(selected_ids - set(ids)),
        source_files=hashes,
    )
    atomic_write(output / "image_vectors.json", manifest.model_dump_json(indent=2).encode())
    return manifest


def prepare_images(
    output: Path, selected_ids: set[int], source: ImageVectorSpec = SIGLIP2
) -> ImageManifest:
    """Download only published Parquet vectors from one immutable, reviewed Hub revision."""
    info = HfApi(token=False).repo_info(
        source.dataset, repo_type="dataset", revision=source.revision
    )
    if info.sha != source.revision:
        raise SourceError("Image dataset did not resolve to the pinned revision")
    directory = Path(
        snapshot_download(
            source.dataset,
            repo_type="dataset",
            revision=source.revision,
            token=False,
            cache_dir=str(output / "sources" / "hf"),
            allow_patterns=["*.parquet"],
            max_workers=3,
        )
    )
    return join_image_shards(sorted(directory.rglob("*.parquet")), selected_ids, source, output)


def load_images(
    directory: Path, selected_ids: set[int]
) -> tuple[ImageManifest, dict[int, list[float]]]:
    manifest = ImageManifest.model_validate_json((directory / "image_vectors.json").read_bytes())
    path = directory / "image_vectors.parquet"
    if manifest.artifact_sha256 != sha256_file(path) or set(manifest.selected_ids) != selected_ids:
        raise SourceError("Image artifact or selected object IDs differ from their manifest")
    frame = pl.read_parquet(path)
    vectors = {row["objectID"]: row["embedding"] for row in frame.iter_rows(named=True)}
    if len(vectors) != len(frame) or set(vectors) != set(manifest.included_ids):
        raise SourceError("Image artifact IDs differ from declared coverage")
    for vector in vectors.values():
        validate_image_vector(vector, manifest.source.dimensions)
    return manifest, vectors


def verify_image_index(
    client: QdrantClient,
    collection: str,
    selected_ids: set[int],
    expected: Mapping[int, list[float]],
) -> int:
    """Verify every published vector and every declared absence after Qdrant's cosine storage."""
    ids = sorted(selected_ids)
    found = 0
    for offset in range(0, len(ids), 128):
        batch = ids[offset : offset + 128]
        points = client.retrieve(collection, ids=batch, with_payload=False, with_vectors=["image"])
        if {point.id for point in points} != set(batch):
            raise SourceError("Image verification found missing collection points")
        for point in points:
            vector = point.vector.get("image") if isinstance(point.vector, dict) else None
            reference = expected.get(int(point.id))
            if reference is None:
                if vector is not None:
                    raise SourceError("Unexpected image vector without published provenance")
            elif (
                not isinstance(vector, list)
                or len(vector) != len(reference)
                or any(
                    not isinstance(value, int | float)
                    or not math.isclose(value, original, abs_tol=1e-5)
                    for value, original in zip(vector, reference, strict=True)
                )
            ):
                raise SourceError("Stored image vector differs from the published source")
            else:
                found += 1
    return found
