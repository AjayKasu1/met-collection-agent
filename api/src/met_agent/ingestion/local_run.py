"""Checkpoint local inference after acknowledged writes without any API quota scheduling."""

import hashlib
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import structlog

from met_agent.ingestion.artifacts import verify_index
from met_agent.ingestion.checkpoints import (
    IngestionCheckpoint,
    RunIdentity,
    checkpoint_lock,
    save_checkpoint,
)
from met_agent.ingestion.http import SourceError
from met_agent.ingestion.image_vectors import verify_image_index
from met_agent.ingestion.models import IndexDocument
from met_agent.ingestion.resumable import input_digest
from met_agent.retrieval.embeddings import DenseEmbedder, SparseEmbedder
from met_agent.retrieval.qdrant_store import HybridStore

logger = structlog.get_logger(__name__)


def run_local_ingestion(
    store: HybridStore,
    documents: Sequence[IndexDocument],
    dense: DenseEmbedder,
    sparse: SparseEmbedder,
    checkpoint_path: Path,
    *,
    destination: str,
    batch_size: int = 32,
    images: Mapping[int, list[float]] | None = None,
    image_sha256: str | None = None,
) -> int:
    """Resume the exact input/model/image artifact and verify all previously completed records."""
    if store.embedding_provider != "local" or batch_size < 1:
        raise SourceError("Local ingestion requires a local provider and positive batch size")
    if not documents or len({doc.point_id for doc in documents}) != len(documents):
        raise SourceError("Local ingestion requires nonempty documents with unique IDs")
    identity = RunIdentity(
        input_sha256=input_digest(documents),
        destination_sha256=hashlib.sha256(destination.encode()).hexdigest(),
        collection=store.collection,
        embedding_provider="local",
        embedding_model=store.embedding_model,
        dimensions=store.embedding_dimensions,
        image_sha256=image_sha256,
        total=len(documents),
    )
    with checkpoint_lock(checkpoint_path):
        exists = store.client.collection_exists(store.collection)
        if checkpoint_path.exists():
            checkpoint = IngestionCheckpoint.model_validate_json(checkpoint_path.read_bytes())
            if checkpoint.identity != identity or checkpoint.quota is not None:
                raise SourceError(
                    "Local checkpoint inputs, destination, model, or image artifact changed"
                )
            if checkpoint.next_offset and not exists:
                raise SourceError("Local checkpoint target collection is missing")
        else:
            if exists and store.client.count(store.collection, exact=True).count:
                raise SourceError("A new local checkpoint cannot adopt a nonempty collection")
            checkpoint = IngestionCheckpoint(identity=identity, updated_at=time.time())
        store.ensure_collection(store.embedding_dimensions)
        # Earlier local checkpoints used a contiguous prefix. Preserve that acknowledged set.
        if checkpoint.completed_ids is None:
            checkpoint.completed_ids = [doc.point_id for doc in documents[: checkpoint.next_offset]]
        by_id = {doc.point_id: doc for doc in documents}
        if not set(checkpoint.completed_ids) <= set(by_id):
            raise SourceError("Local checkpoint contains unknown Object IDs")
        for offset in range(0, checkpoint.next_offset, 128):
            expected = {
                point_id: by_id[point_id].payload
                for point_id in checkpoint.completed_ids[offset : offset + 128]
            }
            points = store.client.retrieve(store.collection, ids=list(expected), with_payload=True)
            if {point.id: point.payload for point in points} != expected:
                raise SourceError("Checkpointed local records are missing or changed")
        checkpoint.status = "running"
        save_checkpoint(checkpoint_path, checkpoint, now=time.time())
        started = time.monotonic()
        prior_elapsed = checkpoint.elapsed_seconds
        completed = set(checkpoint.completed_ids)
        # Similar UTF-8 lengths reduce E5's padded sequence work without changing input text.
        remaining = sorted(
            (doc for doc in documents if doc.point_id not in completed),
            key=lambda doc: (len(doc.text.encode()), str(doc.point_id)),
        )
        logger.info(
            "local_ingestion_started",
            count=checkpoint.next_offset,
            total=len(documents),
            model=store.embedding_model,
        )
        try:
            for offset in range(0, len(remaining), batch_size):
                batch = remaining[offset : offset + batch_size]
                batch_started = time.monotonic()
                store.ingest(batch, dense, sparse, batch_size=batch_size, images=images)
                checkpoint.next_offset += len(batch)
                checkpoint.completed_ids.extend(doc.point_id for doc in batch)
                checkpoint.elapsed_seconds = prior_elapsed + time.monotonic() - started
                save_checkpoint(checkpoint_path, checkpoint, now=time.time())
                logger.info(
                    "local_ingestion_progress",
                    count=checkpoint.next_offset,
                    total=len(documents),
                    elapsed_seconds=round(checkpoint.elapsed_seconds, 2),
                    batch_seconds=round(time.monotonic() - batch_started, 2),
                )
            verify_index(store.client, store.collection, documents)
            if images is not None:
                verify_image_index(
                    store.client, store.collection, {int(doc.point_id) for doc in documents}, images
                )
            checkpoint.status = "complete"
        except BaseException:
            checkpoint.status = "failed"
            raise
        finally:
            checkpoint.elapsed_seconds = prior_elapsed + time.monotonic() - started
            save_checkpoint(checkpoint_path, checkpoint, now=time.time())
        logger.info(
            "local_ingestion_completed",
            count=checkpoint.next_offset,
            elapsed_seconds=round(checkpoint.elapsed_seconds, 2),
        )
    return checkpoint.next_offset
