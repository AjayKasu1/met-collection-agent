"""Package verified indexes and restore pinned, checksummed snapshots without embedding calls."""

import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import quote

import httpx
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from pydantic import BaseModel, ConfigDict, Field, model_validator
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import PointId

from met_agent.ingestion.http import SourceError
from met_agent.ingestion.image_vectors import load_images, verify_image_index
from met_agent.ingestion.models import IndexDocument, VisitorChunk
from met_agent.ingestion.storage import atomic_write, load_objects, sha256_file
from met_agent.retrieval.qdrant_store import SCHEMA_VERSION, HybridStore, IndexCompatibilityError
from met_agent.retrieval.schema import EmbeddingProvider, ImageVectorSpec

ArtifactName = Literal[
    "objects.parquet",
    "collection.snapshot",
    "visitor_chunks.jsonl",
    "visitor.snapshot",
    "image_vectors.parquet",
    "image_vectors.json",
    "README.md",
]
IndexKind = Literal["collection", "visitor"]


class Artifact(BaseModel):
    """A content digest and size checked before any restore request."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    size_bytes: Annotated[int, Field(gt=0)]


class IndexSpec(BaseModel):
    """The model identity and geometry needed to safely reuse an embedding space."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: IndexKind
    points: Annotated[int, Field(gt=0)]
    dimensions: Annotated[int, Field(gt=0)]
    embedding_model: str
    embedding_provider: EmbeddingProvider = "gemini"
    image: ImageVectorSpec | None = None
    image_points: Annotated[int, Field(ge=0)] = 0
    sparse_model: Literal["Qdrant/bm25"] = "Qdrant/bm25"
    schema_version: Literal[3] = SCHEMA_VERSION


class Manifest(BaseModel):
    """An exact file allowlist prevents publication or download of unrelated private inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    format_version: Literal[3] = 3
    created_at: datetime
    qdrant_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]
    files: dict[ArtifactName, Artifact]
    indexes: list[IndexSpec]

    @model_validator(mode="after")
    def validate_bundle(self) -> Self:
        kinds = [index.kind for index in self.indexes]
        if sorted(kinds) not in [["collection"], ["collection", "visitor"]]:
            raise ValueError(
                "Bundle requires exactly one collection index and optional visitor index"
            )
        expected = {"objects.parquet", "collection.snapshot", "README.md"}
        if "visitor" in kinds:
            expected |= {"visitor_chunks.jsonl", "visitor.snapshot"}
        for index in self.indexes:
            if index.image is not None:
                if index.kind != "collection" or not 0 < index.image_points <= index.points:
                    raise ValueError("Image coverage is inconsistent with its collection")
                expected |= {"image_vectors.parquet", "image_vectors.json"}
            elif index.image_points:
                raise ValueError("Image coverage requires source provenance")
        if set(self.files) != expected or self.created_at.tzinfo is None:
            raise ValueError("Manifest file set or timestamp is invalid")
        return self


def load_documents(directory: Path, kind: IndexKind) -> list[IndexDocument]:
    """Read typed source artifacts before comparing them with index payloads."""
    if kind == "collection":
        return [obj.document() for obj in load_objects(directory / "objects.parquet")]
    return [
        VisitorChunk.model_validate_json(line).document()
        for line in (directory / "visitor_chunks.jsonl").read_text().splitlines()
        if line.strip()
    ]


def verify_index(client: QdrantClient, collection: str, documents: Sequence[IndexDocument]) -> None:
    """Require exact IDs and public payloads, not merely an equal point count."""
    expected = {document.point_id: document.payload for document in documents}
    if not expected or len(expected) != len(documents):
        raise IndexCompatibilityError("Source artifact has empty or duplicate document IDs")
    seen: set[int | str] = set()
    offset: PointId | None = None
    while True:
        points, offset = client.scroll(collection, limit=256, offset=offset, with_vectors=False)
        for point in points:
            point_id = point.id if isinstance(point.id, int) else str(point.id)
            if point_id not in expected or point.payload != expected[point_id]:
                raise IndexCompatibilityError(
                    "Index payloads differ from the public source artifact"
                )
            seen.add(point_id)
        if offset is None:
            break
    if seen != set(expected):
        raise IndexCompatibilityError("Index is missing source records")


def _snapshot_path(kind: IndexKind) -> ArtifactName:
    return "collection.snapshot" if kind == "collection" else "visitor.snapshot"


def _source_path(kind: IndexKind) -> ArtifactName:
    return "objects.parquet" if kind == "collection" else "visitor_chunks.jsonl"


def export_bundle(
    client: QdrantClient,
    http: httpx.Client,
    data_dir: Path,
    output: Path,
    collections: dict[IndexKind, str],
    embedding_model: str,
    embedding_dimensions: int = 768,
    *,
    embedding_provider: EmbeddingProvider = "gemini",
) -> Manifest:
    """Export single-node snapshots after checking source identity and model compatibility.

    Call with ingestion writers stopped. The HTTP client must target the same Qdrant
    server, with an explicit base URL and API-key header when configured.
    """
    output.mkdir(parents=True, exist_ok=True)
    indexes = []
    files: dict[ArtifactName, Artifact] = {}
    for kind, collection in collections.items():
        info = client.get_collection(collection)
        vectors = info.config.params.vectors
        if not isinstance(vectors, dict) or "dense" not in vectors:
            raise IndexCompatibilityError("Collection has no named dense vector")
        dimensions = vectors["dense"].size
        raw_image = (info.config.metadata or {}).get("image")
        image = ImageVectorSpec.model_validate(raw_image) if raw_image is not None else None
        HybridStore(
            client,
            collection,
            embedding_model,
            embedding_dimensions,
            embedding_provider=embedding_provider,
            image=image,
        ).validate_collection()
        cluster = client.collection_cluster_info(collection)
        if cluster.remote_shards:
            raise IndexCompatibilityError("Export requires a single-node collection snapshot")
        documents = load_documents(data_dir, kind)
        verify_index(client, collection, documents)
        image_points = 0
        if image is not None:
            selected = {int(doc.point_id) for doc in documents}
            image_manifest, image_vectors = load_images(data_dir, selected)
            if image_manifest.source != image:
                raise IndexCompatibilityError("Image source differs from collection metadata")
            image_points = verify_image_index(client, collection, selected, image_vectors)
            for image_name in ("image_vectors.parquet", "image_vectors.json"):
                shutil.copyfile(data_dir / image_name, output / image_name)
                files[image_name] = Artifact(
                    sha256=sha256_file(output / image_name),
                    size_bytes=(output / image_name).stat().st_size,
                )
        source_name = _source_path(kind)
        shutil.copyfile(data_dir / source_name, output / source_name)
        snapshot = client.create_snapshot(collection, wait=True)
        if snapshot is None:
            raise SourceError("Qdrant did not acknowledge snapshot creation")
        snapshot_name = _snapshot_path(kind)
        temporary = output / (snapshot_name + ".partial")
        try:
            endpoint = (
                f"collections/{quote(collection, safe='')}/snapshots/"
                f"{quote(snapshot.name, safe='')}"
            )
            with http.stream("GET", endpoint) as response:
                if response.status_code != 200:
                    raise SourceError(f"Snapshot download returned HTTP {response.status_code}")
                with temporary.open("wb") as handle:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        handle.write(chunk)
            if snapshot.checksum and sha256_file(temporary) != snapshot.checksum:
                raise SourceError("Downloaded snapshot checksum differs from Qdrant")
            temporary.replace(output / snapshot_name)
        finally:
            temporary.unlink(missing_ok=True)
            client.delete_snapshot(collection, snapshot.name, wait=True)
        for name in (source_name, snapshot_name):
            path = output / name
            files[name] = Artifact(sha256=sha256_file(path), size_bytes=path.stat().st_size)
        indexes.append(
            IndexSpec(
                kind=kind,
                points=len(documents),
                dimensions=dimensions,
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                image=image,
                image_points=image_points,
            )
        )
    atomic_write(output / "README.md", dataset_card(indexes).encode())
    files["README.md"] = Artifact(
        sha256=sha256_file(output / "README.md"),
        size_bytes=(output / "README.md").stat().st_size,
    )
    manifest = Manifest(
        created_at=datetime.now(UTC),
        qdrant_version=client.info().version,
        files=files,
        indexes=indexes,
    )
    atomic_write(output / "manifest.json", manifest.model_dump_json(indent=2).encode())
    return manifest


def dataset_card(indexes: Sequence[IndexSpec]) -> str:
    """Publish attribution and actual modifications with every snapshot revision."""
    lines = [
        "---",
        "license: cc0-1.0",
        "pretty_name: Independent Met collection search index",
        "---",
        "",
        "# Collection search index",
        "",
        "Collection metadata: The Metropolitan Museum of Art, "
        "[metmuseum/openaccess](https://github.com/metmuseum/openaccess), CC0. "
        "The [Met's Hugging Face dataset](https://huggingface.co/datasets/metmuseum/openaccess) "
        "provides the associated attribution and usage guidance.",
        "",
        "This is a modified derivative: selected public-domain records, normalized fields, "
        "assembled retrieval text, locally or remotely generated text vectors, and a Qdrant index. "
        "Raw collection fields and original source links are retained.",
        "",
        "This project is not affiliated with or endorsed by The Metropolitan Museum of Art. "
        "No Museum logos or trademarks are used as project branding.",
        "",
    ]
    for index in indexes:
        lines.append(
            f"- {index.kind}: {index.points} records; text provider `{index.embedding_provider}`, "
            f"model `{index.embedding_model}`, {index.dimensions} dimensions."
        )
        if index.image:
            source = index.image
            lines.append(
                f"- Image embeddings: The Metropolitan Museum of Art, "
                f"[{source.dataset}](https://huggingface.co/datasets/{source.dataset}), CC0; "
                f"revision `{source.revision}`, model `{source.model}`, "
                f"{source.dimensions} dimensions. "
                f"{index.image_points} records have a published image vector. Vectors were joined "
                "by Object ID; no image embeddings were computed by this project. "
                "Objects missing a published vector remain without one."
            )
    lines.extend(
        [
            "",
            "The manifest records file hashes, model identities, source revisions, and coverage. "
            "See the [source repository](https://github.com/AjayKasu1/met-collection-agent) for "
            "the pinned runtime and restore instructions.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_files(directory: Path, manifest: Manifest) -> None:
    """Check all artifacts before uploading or modifying any index."""
    for name, artifact in manifest.files.items():
        path = directory / name
        if path.stat().st_size != artifact.size_bytes or sha256_file(path) != artifact.sha256:
            raise SourceError(f"Artifact integrity check failed: {name}")
    for index in manifest.indexes:
        if len(load_documents(directory, index.kind)) != index.points:
            raise SourceError("Source artifact count differs from manifest")
        if index.image is not None:
            image_manifest, vectors = load_images(
                directory, {int(doc.point_id) for doc in load_documents(directory, index.kind)}
            )
            if image_manifest.source != index.image or len(vectors) != index.image_points:
                raise SourceError("Published image provenance or coverage differs from manifest")


def publish_bundle(directory: Path, repo: str, token: str) -> str:
    """Upload only manifest-declared files in one commit to an existing HF dataset."""
    manifest = Manifest.model_validate_json((directory / "manifest.json").read_bytes())
    validate_files(directory, manifest)
    api = HfApi(token=token)
    head = api.repo_info(repo, repo_type="dataset").sha
    result = api.create_commit(
        repo_id=repo,
        repo_type="dataset",
        parent_commit=head,
        operations=[
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=directory / name)
            for name in ["manifest.json", *sorted(manifest.files)]
        ],
        commit_message="Publish verified hybrid collection index",
    )
    return str(result.oid)


def download_bundle(repo: str, revision: str, output: Path, token: str | bool = False) -> Manifest:
    """Pin every download to one immutable Hub commit, even if a branch moves mid-download."""
    head = HfApi(token=token).repo_info(repo, repo_type="dataset", revision=revision).sha
    if not head:
        raise SourceError("Dataset revision did not resolve to a commit")
    output.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo, filename="manifest.json", repo_type="dataset", revision=head, token=token
    )
    shutil.copyfile(str(path), output / "manifest.json")
    manifest = Manifest.model_validate_json((output / "manifest.json").read_bytes())
    for artifact in manifest.files:
        path = hf_hub_download(
            repo, filename=artifact, repo_type="dataset", revision=head, token=token
        )
        shutil.copyfile(str(path), output / artifact)
    validate_files(output, manifest)
    atomic_write(output / "revision.txt", (head + "\n").encode())
    return manifest


def restore_bundle(
    client: QdrantClient,
    http: httpx.Client,
    directory: Path,
    collections: dict[IndexKind, str],
    embedding_model: str,
    embedding_dimensions: int = 768,
    *,
    embedding_provider: EmbeddingProvider = "gemini",
) -> Manifest:
    """Restore into absent collections only, after validating the entire bundle."""
    manifest = Manifest.model_validate_json((directory / "manifest.json").read_bytes())
    validate_files(directory, manifest)
    source_version = tuple(map(int, manifest.qdrant_version.split(".")))
    target_version = tuple(map(int, client.info().version.split(".")))
    if target_version[:2] != source_version[:2] or target_version < source_version:
        raise IndexCompatibilityError(
            "Restore requires the same Qdrant minor version and an equal or newer patch"
        )
    targets = [collections[index.kind] for index in manifest.indexes]
    if len(set(targets)) != len(targets):
        raise IndexCompatibilityError("Each index requires a distinct target collection")
    for index in manifest.indexes:
        if index.embedding_provider != embedding_provider:
            raise IndexCompatibilityError("EMBEDDING_PROVIDER differs from the published index")
        if index.embedding_model != embedding_model:
            raise IndexCompatibilityError("EMBEDDING_MODEL differs from the published index")
        if index.dimensions != embedding_dimensions:
            raise IndexCompatibilityError("EMBEDDING_DIMENSIONS differs from the published index")
        if client.collection_exists(collections[index.kind]):
            raise IndexCompatibilityError(
                "Seed refuses to overwrite an existing collection; choose an unused name"
            )
    for index in manifest.indexes:
        collection = collections[index.kind]
        with (directory / _snapshot_path(index.kind)).open("rb") as handle:
            response = http.post(
                f"collections/{quote(collection, safe='')}/snapshots/upload",
                params={"priority": "snapshot", "wait": "true"},
                files={"snapshot": handle},
            )
        if response.status_code != 200:
            raise SourceError(f"Snapshot restore returned HTTP {response.status_code}")
        HybridStore(
            client,
            collection,
            embedding_model,
            embedding_dimensions,
            embedding_provider=embedding_provider,
            image=index.image,
        ).validate_collection()
        verify_index(client, collection, load_documents(directory, index.kind))
        if index.image is not None:
            selected = {int(doc.point_id) for doc in load_documents(directory, index.kind)}
            _, vectors = load_images(directory, selected)
            verify_image_index(client, collection, selected, vectors)
    return manifest
