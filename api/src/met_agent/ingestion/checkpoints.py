"""Atomically persist acknowledged progress and reject concurrent or incompatible resumes."""

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from met_agent.ingestion.http import SourceError
from met_agent.ingestion.quota import QuotaState, quota_day
from met_agent.ingestion.storage import atomic_write
from met_agent.retrieval.schema import EmbeddingProvider


class RunIdentity(BaseModel):
    """Bind checkpoints to exact inputs, destination, model, and vector geometry."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    input_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    destination_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    collection: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding_provider: EmbeddingProvider = "gemini"
    dimensions: int = Field(gt=0)
    schema_version: Literal[3] = 3
    image_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    total: int = Field(gt=0)


class IngestionCheckpoint(BaseModel):
    """Reserve quota before requests and advance offsets only after acknowledged writes."""

    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    identity: RunIdentity
    next_offset: int = Field(default=0, ge=0)
    completed_ids: list[int | str] | None = None
    completed_tokens: int = Field(default=0, ge=0)
    quota: QuotaState | None = None
    elapsed_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    status: Literal["running", "waiting", "failed", "complete"] = "running"
    updated_at: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_progress(self) -> Self:
        if self.next_offset > self.identity.total or (
            self.status == "complete" and self.next_offset != self.identity.total
        ):
            raise ValueError("Checkpoint progress is inconsistent")
        if self.completed_ids is not None and (
            len(self.completed_ids) != self.next_offset
            or len(set(self.completed_ids)) != self.next_offset
        ):
            raise ValueError("Checkpoint completed IDs are inconsistent")
        return self


@contextmanager
def checkpoint_lock(path: Path) -> Iterator[None]:
    """An OS lock is released even when the process is interrupted or killed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SourceError("Another ingestion process owns this checkpoint") from None
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def load_checkpoint(
    path: Path, identity: RunIdentity, *, now: float, initial_daily_requests: int = 0
) -> IngestionCheckpoint:
    if path.exists():
        checkpoint = IngestionCheckpoint.model_validate_json(path.read_bytes())
        if checkpoint.identity != identity:
            raise SourceError("Checkpoint inputs, destination, model, or dimensions changed")
        return checkpoint
    return IngestionCheckpoint(
        identity=identity,
        updated_at=now,
        quota=QuotaState(day=quota_day(now), used_today=initial_daily_requests),
    )


def save_checkpoint(path: Path, checkpoint: IngestionCheckpoint, *, now: float) -> None:
    checkpoint.updated_at = now
    validated = IngestionCheckpoint.model_validate(checkpoint.model_dump())
    atomic_write(path, validated.model_dump_json(indent=2).encode())
