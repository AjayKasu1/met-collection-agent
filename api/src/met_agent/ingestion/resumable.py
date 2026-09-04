"""Run bounded, quota-aware ingestion with an exclusive checkpoint and acknowledged progress."""

import hashlib
import math
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import structlog

from met_agent.ingestion.artifacts import verify_index
from met_agent.ingestion.checkpoints import (
    IngestionCheckpoint,
    RunIdentity,
    checkpoint_lock,
    load_checkpoint,
    save_checkpoint,
)
from met_agent.ingestion.http import SourceError
from met_agent.ingestion.models import IndexDocument
from met_agent.ingestion.quota import QuotaBudget, QuotaLimits, next_reset
from met_agent.ingestion.token_counting import TokenCounter
from met_agent.retrieval.embeddings import DenseEmbedder, EmbeddingRateLimitError, SparseEmbedder
from met_agent.retrieval.qdrant_store import HybridStore

logger = structlog.get_logger(__name__)


def input_digest(documents: Sequence[IndexDocument]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document.model_dump_json().encode())
        digest.update(b"\n")
    return digest.hexdigest()


class ResumableIngestor:
    """Reserve every attempt before sending, and skip all previously checkpointed documents."""

    def __init__(
        self,
        store: HybridStore,
        dense: DenseEmbedder,
        sparse: SparseEmbedder,
        counter: TokenCounter,
        limits: QuotaLimits,
        checkpoint: Path,
        *,
        batch_size: int = 100,
        max_retries: int = 2,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 1 <= batch_size <= 100 or max_retries < 0:
            raise ValueError("Batch size must be 1-100 and retries must be nonnegative")
        self.store, self.dense, self.sparse, self.counter = store, dense, sparse, counter
        self.limits, self.path = limits, checkpoint
        self.batch_size = min(batch_size, limits.requests_per_minute, limits.requests_per_day)
        self.max_retries, self.clock, self.sleep = max_retries, clock, sleep

    def _save(self, checkpoint: IngestionCheckpoint) -> None:
        save_checkpoint(self.path, checkpoint, now=self.clock())

    def _wait(self, checkpoint: IngestionCheckpoint, seconds: float, reason: str) -> None:
        checkpoint.status = "waiting"
        self._save(checkpoint)
        until = self.clock() + seconds
        logger.info(
            "ingestion_quota_wait",
            reason=reason,
            seconds=round(seconds, 2),
            resume_at=datetime.fromtimestamp(until, UTC).isoformat(),
            count=checkpoint.next_offset,
        )
        while (remaining := until - self.clock()) > 0:
            self.sleep(min(60.0, remaining))
        checkpoint.status = "running"

    def _verify_prefix(self, documents: Sequence[IndexDocument], offset: int) -> None:
        for start in range(0, offset, 100):
            expected = {
                doc.point_id: doc.payload for doc in documents[start : min(start + 100, offset)]
            }
            points = self.store.client.retrieve(
                self.store.collection,
                ids=list(expected),
                with_payload=True,
                with_vectors=False,
            )
            actual = {
                point.id if isinstance(point.id, int) else str(point.id): point.payload
                for point in points
            }
            if actual != expected:
                raise SourceError(
                    "Checkpointed Qdrant records are missing or changed; resume refused"
                )

    def _batch(
        self,
        documents: Sequence[IndexDocument],
        offset: int,
        available: int,
    ) -> tuple[Sequence[IndexDocument], int, int]:
        batch = documents[offset : offset + min(self.batch_size, available)]
        for document in batch:
            # UTF-8 byte length is a conservative cheap guard for ordinary text inputs.
            if len(document.text.encode()) > 2048 and self.counter.count([document.text]) > 2048:
                raise SourceError(
                    f"Object {document.point_id} exceeds Gemini's 2048-token input limit"
                )
        lower, upper, size = 1, len(batch), len(batch)
        best: tuple[Sequence[IndexDocument], int, int] | None = None
        while lower <= upper:
            candidate = batch[:size]
            tokens = self.counter.count([document.text for document in candidate])
            reserved = math.ceil(tokens * 1.1)
            if 0 < reserved <= self.limits.tokens_per_minute:
                best = candidate, tokens, reserved
                lower = size + 1
            else:
                upper = size - 1
            size = (lower + upper) // 2
        if best is None:
            raise SourceError("One document exceeds the configured token budget")
        return best

    def run(
        self,
        documents: Sequence[IndexDocument],
        *,
        destination: str,
        initial_daily_requests: int = 0,
    ) -> int:
        if not documents or len({doc.point_id for doc in documents}) != len(documents):
            raise SourceError("Ingestion requires nonempty documents with unique IDs")
        identity = RunIdentity(
            input_sha256=input_digest(documents),
            destination_sha256=hashlib.sha256(destination.encode()).hexdigest(),
            collection=self.store.collection,
            embedding_model=self.store.embedding_model,
            embedding_provider=self.store.embedding_provider,
            dimensions=self.store.embedding_dimensions,
            total=len(documents),
        )
        with checkpoint_lock(self.path):
            fresh = not self.path.exists()
            checkpoint = load_checkpoint(
                self.path,
                identity,
                now=self.clock(),
                initial_daily_requests=initial_daily_requests,
            )
            exists = self.store.client.collection_exists(self.store.collection)
            if checkpoint.next_offset and not exists:
                raise SourceError("Checkpoint target collection is missing; resume refused")
            if (
                fresh
                and exists
                and self.store.client.count(self.store.collection, exact=True).count
            ):
                raise SourceError("A new checkpoint cannot adopt a nonempty collection")
            if checkpoint.quota is None:
                raise SourceError("Gemini checkpoints require a persisted quota budget")
            budget = QuotaBudget(self.limits, checkpoint.quota)
            self.store.ensure_collection(self.store.embedding_dimensions)
            self._verify_prefix(documents, checkpoint.next_offset)
            checkpoint.status = "running"
            self._save(checkpoint)
            logger.info(
                "ingestion_resumed" if not fresh else "ingestion_started",
                count=checkpoint.next_offset,
                total=len(documents),
                dimensions=identity.dimensions,
                batch_size=self.batch_size,
                quotas=self.limits.model_dump(),
            )
            try:
                while checkpoint.next_offset < len(documents):
                    available = budget.remaining_today(self.clock())
                    if not available:
                        self._wait(checkpoint, next_reset(self.clock()) + 1 - self.clock(), "daily")
                        continue
                    batch, tokens, reserved = self._batch(
                        documents, checkpoint.next_offset, available
                    )
                    retries = 0
                    while True:
                        delay = budget.delay(self.clock(), requests=len(batch), tokens=reserved)
                        if delay:
                            self._wait(checkpoint, delay, "requests_or_tokens")
                            continue
                        budget.reserve(self.clock(), requests=len(batch), tokens=reserved)
                        self._save(checkpoint)
                        try:
                            self.store.ingest(batch, self.dense, self.sparse, batch_size=len(batch))
                            break
                        except EmbeddingRateLimitError as error:
                            budget.defer(self.clock(), seconds=error.retry_after, daily=error.daily)
                            self._save(checkpoint)
                            if not error.daily:
                                retries += 1
                                if retries > self.max_retries:
                                    raise
                            logger.warning(
                                "embedding_quota_rejected", daily=error.daily, attempt=retries
                            )
                    checkpoint.next_offset += len(batch)
                    checkpoint.completed_tokens += tokens
                    self._save(checkpoint)
                    eta = budget.estimate_finish(
                        self.clock(),
                        remaining=len(documents) - checkpoint.next_offset,
                        tokens_per_object=checkpoint.completed_tokens / checkpoint.next_offset,
                    )
                    logger.info(
                        "ingestion_progress",
                        count=checkpoint.next_offset,
                        total=len(documents),
                        batch_tokens=tokens,
                        reserved_tokens=reserved,
                        requests_today=budget.state.used_today,
                        estimated_finish_at=datetime.fromtimestamp(eta, UTC).isoformat(),
                    )
                verify_index(self.store.client, self.store.collection, documents)
                checkpoint.status = "complete"
                self._save(checkpoint)
            except BaseException:
                checkpoint.status = "failed"
                self._save(checkpoint)
                raise
        logger.info("ingestion_completed", count=checkpoint.next_offset, total=len(documents))
        return checkpoint.next_offset
