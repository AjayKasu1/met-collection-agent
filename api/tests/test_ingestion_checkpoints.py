"""Validate checkpoint identity, atomic writes, single-writer locking, and persisted quota state."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from met_agent.ingestion.checkpoints import (
    RunIdentity,
    checkpoint_lock,
    load_checkpoint,
    save_checkpoint,
)
from met_agent.ingestion.http import SourceError


def identity() -> RunIdentity:
    return RunIdentity(
        input_sha256="a" * 64,
        destination_sha256="b" * 64,
        collection="test",
        embedding_model="test-model",
        dimensions=768,
        total=200,
    )


def test_checkpoint_preserves_progress_and_rejects_changed_run_identity(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    original = identity()
    checkpoint = load_checkpoint(path, original, now=1000, initial_daily_requests=20)
    checkpoint.next_offset = 100
    checkpoint.completed_tokens = 1234
    checkpoint.quota.used_today = 120
    save_checkpoint(path, checkpoint, now=2000)
    assert load_checkpoint(path, original, now=3000) == checkpoint
    updates: list[dict[str, object]] = [
        {"dimensions": 3072},
        {"input_sha256": "c" * 64},
        {"destination_sha256": "c" * 64},
        {"collection": "another"},
        {"embedding_model": "another"},
        {"total": 300},
    ]
    for update in updates:
        with pytest.raises(SourceError, match="changed"):
            load_checkpoint(path, original.model_copy(update=update), now=3000)
    previous = path.read_bytes()
    checkpoint.next_offset = 201
    with pytest.raises(ValidationError):
        save_checkpoint(path, checkpoint, now=3000)
    assert path.read_bytes() == previous
    checkpoint.next_offset = 100
    checkpoint.status = "complete"
    with pytest.raises(ValidationError):
        save_checkpoint(path, checkpoint, now=3000)


def test_checkpoint_lock_rejects_a_second_writer_and_releases_after_failure(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    with pytest.raises(RuntimeError), checkpoint_lock(path):
        with pytest.raises(SourceError, match="Another ingestion"), checkpoint_lock(path):
            pytest.fail("Concurrent writer acquired the lock")
        raise RuntimeError("interrupted")
    with checkpoint_lock(path):
        save_checkpoint(path, load_checkpoint(path, identity(), now=1000), now=1000)
    assert path.exists()
