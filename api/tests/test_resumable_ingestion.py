"""Exercise interruption, quota waits, and resume verification with real embedded Qdrant storage."""

from collections.abc import Iterator, Sequence
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Literal

import pytest
from qdrant_client import QdrantClient, models

from met_agent.ingestion.checkpoints import IngestionCheckpoint
from met_agent.ingestion.http import SourceError
from met_agent.ingestion.models import IndexDocument
from met_agent.ingestion.quota import PACIFIC, QuotaLimits
from met_agent.ingestion.resumable import ResumableIngestor
from met_agent.retrieval.embeddings import EmbeddingRateLimitError
from met_agent.retrieval.qdrant_store import HybridStore

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 4, 23, 59, tzinfo=PACIFIC).timestamp()
        self.waits: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds


class Dense:
    def __init__(self, fail_at: int = 0, error: BaseException | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_at = fail_at
        self.error = error or KeyboardInterrupt()

    def embed(
        self, texts: Sequence[str], *, purpose: Literal["document", "query"] = "document"
    ) -> list[list[float]]:
        self.calls.append(list(texts))
        if len(self.calls) == self.fail_at:
            raise self.error
        return [[1.0, 0.5, 0.25, 0.125] for _ in texts]


class Sparse:
    def embed(self, texts: Sequence[str]) -> list[models.SparseVector]:
        return [models.SparseVector(indices=[1], values=[1.0]) for _ in texts]


class Counter:
    def count(self, texts: Sequence[str]) -> int:
        return sum(10 if len(text) <= 2048 else 3000 for text in texts)


@pytest.fixture
def store() -> Iterator[HybridStore]:
    with closing(QdrantClient(location=":memory:")) as client:
        yield HybridStore(client, "resumable", "test-model", 4)


def documents(count: int) -> list[IndexDocument]:
    return [
        IndexDocument(point_id=i, text=f"Vessel {i}", payload={"object_id": i})
        for i in range(1, count + 1)
    ]


def runner(store: HybridStore, dense: Dense, clock: Clock, path: Path) -> ResumableIngestor:
    # Keep the real constructor typed in each test while sharing deterministic service seams.
    return ResumableIngestor(
        store, dense, Sparse(), Counter(), QuotaLimits(), path, clock=clock.time, sleep=clock.sleep
    )


def test_interruption_resumes_after_last_acknowledged_batch_without_reembedding(
    store: HybridStore, tmp_path: Path
) -> None:
    path, clock, dense = tmp_path / "checkpoint.json", Clock(), Dense(fail_at=2)
    docs = documents(200)
    with pytest.raises(KeyboardInterrupt):
        runner(store, dense, clock, path).run(docs, destination="https://qdrant.example")
    saved = IngestionCheckpoint.model_validate_json(path.read_bytes())
    assert saved.next_offset == 100 and saved.status == "failed"
    assert store.client.count(store.collection, exact=True).count == 100
    resumed = Dense()
    assert (
        runner(store, resumed, clock, path).run(docs, destination="https://qdrant.example") == 200
    )
    assert resumed.calls == [[doc.text for doc in docs[100:]]]
    assert store.client.count(store.collection, exact=True).count == 200
    assert IngestionCheckpoint.model_validate_json(path.read_bytes()).status == "complete"
    completed = Dense()
    assert (
        runner(store, completed, clock, path).run(docs, destination="https://qdrant.example") == 200
    )
    assert not completed.calls


def test_token_budget_splits_batches_and_waits_without_a_fixed_delay(
    store: HybridStore, tmp_path: Path
) -> None:
    clock, dense = Clock(), Dense()
    job = ResumableIngestor(
        store,
        dense,
        Sparse(),
        Counter(),
        QuotaLimits(tokens_per_minute=25),
        tmp_path / "checkpoint.json",
        batch_size=100,
        clock=clock.time,
        sleep=clock.sleep,
    )
    assert job.run(documents(5), destination="test") == 5
    assert [len(call) for call in dense.calls] == [2, 2, 1]
    assert sum(clock.waits) >= 120


def test_daily_quota_waits_for_reset_and_uses_remaining_daily_capacity(
    store: HybridStore, tmp_path: Path
) -> None:
    clock, dense = Clock(), Dense()
    job = ResumableIngestor(
        store,
        dense,
        Sparse(),
        Counter(),
        QuotaLimits(requests_per_day=3),
        tmp_path / "checkpoint.json",
        clock=clock.time,
        sleep=clock.sleep,
    )
    assert job.run(documents(5), destination="test", initial_daily_requests=1) == 5
    assert [len(call) for call in dense.calls] == [2, 3]
    assert sum(clock.waits) >= 61


def test_provider_daily_quota_rejection_is_persisted_and_retried_after_reset(
    store: HybridStore, tmp_path: Path
) -> None:
    clock, dense = Clock(), Dense(fail_at=1, error=EmbeddingRateLimitError(daily=True))
    assert (
        runner(store, dense, clock, tmp_path / "checkpoint.json").run(
            documents(2), destination="test"
        )
        == 2
    )
    assert len(dense.calls) == 2 and sum(clock.waits) >= 61


def test_resume_refuses_changed_inputs_missing_or_modified_acknowledged_records(
    store: HybridStore, tmp_path: Path
) -> None:
    path, clock = tmp_path / "checkpoint.json", Clock()
    job = runner(store, Dense(), clock, path)
    docs = documents(2)
    job.run(docs, destination="test")
    with pytest.raises(SourceError, match="changed"):
        job.run(documents(3), destination="test")
    store.client.set_payload(store.collection, payload={"changed": True}, points=[1], wait=True)
    with pytest.raises(SourceError, match="missing or changed"):
        job.run(docs, destination="test")
    store.client.delete_collection(store.collection)
    with pytest.raises(SourceError, match="target collection is missing"):
        job.run(docs, destination="test")


def test_new_checkpoint_refuses_existing_data_and_invalid_inputs(
    store: HybridStore, tmp_path: Path
) -> None:
    path, clock = tmp_path / "checkpoint.json", Clock()
    job = runner(store, Dense(), clock, path)
    with pytest.raises(SourceError, match="nonempty"):
        job.run([], destination="test")
    with pytest.raises(SourceError, match="unique"):
        job.run(documents(1) * 2, destination="test")
    store.ingest(documents(1), Dense(), Sparse(), batch_size=1)
    with pytest.raises(SourceError, match="cannot adopt"):
        job.run(documents(2), destination="test")
    assert not path.exists()


def test_oversized_input_and_exhausted_retries_leave_checkpoint_retryable(
    store: HybridStore, tmp_path: Path
) -> None:
    path, clock = tmp_path / "checkpoint.json", Clock()
    job = runner(store, Dense(), clock, path)
    with pytest.raises(SourceError, match="2048-token"):
        job.run([IndexDocument(point_id=1, text="x" * 2049, payload={})], destination="test")
    saved = IngestionCheckpoint.model_validate_json(path.read_bytes())
    assert saved.quota is not None
    assert saved.next_offset == 0 and saved.quota.used_today == 0
    path.unlink()
    limited = ResumableIngestor(
        store,
        Dense(fail_at=1, error=EmbeddingRateLimitError()),
        Sparse(),
        Counter(),
        QuotaLimits(),
        path,
        max_retries=0,
        clock=clock.time,
        sleep=clock.sleep,
    )
    with pytest.raises(EmbeddingRateLimitError):
        limited.run(documents(1), destination="test")
    saved = IngestionCheckpoint.model_validate_json(path.read_bytes())
    assert saved.quota is not None and saved.quota.used_today == 1
    with pytest.raises(ValueError, match="Batch size"):
        ResumableIngestor(store, Dense(), Sparse(), Counter(), QuotaLimits(), path, batch_size=101)
