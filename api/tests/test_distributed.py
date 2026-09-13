"""Test distributed PostgreSQL session locks and cross-instance pacing budgets."""

import asyncio
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from met_agent.agent.distributed import (
    DistributedTokenPacer,
    LocalNullLock,
    PostgresSessionLock,
    get_session_lock,
)
from met_agent.config import Settings


def test_local_null_lock() -> None:
    async def run() -> None:
        async with LocalNullLock():
            pass

    asyncio.run(run())


def test_get_session_lock_fallback() -> None:
    session_id = uuid4()
    mock_store = MagicMock()
    mock_store.pool = None
    lock = get_session_lock(mock_store, session_id)
    assert isinstance(lock, LocalNullLock)


def test_postgres_session_lock_lifecycle() -> None:
    session_id = uuid4()
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_pool.getconn.return_value = mock_conn

    lock = PostgresSessionLock(mock_pool, session_id, timeout=10.0)

    async def run() -> None:
        async with lock:
            assert mock_pool.getconn.called
            # Verify lock timeout was set
            mock_conn.execute.assert_any_call("SET lock_timeout = %s", ("10000ms",))
            # Verify advisory lock was acquired with 63-bit integer hash
            expected_lock_id = session_id.int & 0x7FFFFFFFFFFFFFFF
            mock_conn.execute.assert_any_call("SELECT pg_advisory_lock(%s)", (expected_lock_id,))

    asyncio.run(run())

    # Verify advisory unlock and putconn were called upon exit
    expected_lock_id = session_id.int & 0x7FFFFFFFFFFFFFFF
    mock_conn.execute.assert_any_call("SELECT pg_advisory_unlock(%s)", (expected_lock_id,))
    mock_pool.putconn.assert_called_once_with(mock_conn)


def test_postgres_session_lock_failure_cleans_up() -> None:
    session_id = uuid4()
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = RuntimeError("db error")
    mock_pool.getconn.return_value = mock_conn

    lock = PostgresSessionLock(mock_pool, session_id)

    async def run() -> None:
        with pytest.raises(RuntimeError, match="db error"):
            async with lock:
                pass

    asyncio.run(run())
    mock_pool.putconn.assert_called_once_with(mock_conn)


def test_distributed_token_pacer_local_fallback(settings: Settings) -> None:
    pacer = DistributedTokenPacer(settings.model_copy(update={"llm_pacing_enabled": True}))

    async def run() -> None:
        res = await pacer.reserve("test/model", 100)
        assert res is not None
        assert res.tokens == 100
        await pacer.reconcile(res, 50)
        assert res.tokens == 50

    asyncio.run(run())


def test_distributed_token_pacer_postgres(settings: Settings) -> None:
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn
    mock_conn.transaction.return_value.__enter__.return_value = None

    # Simulate SELECT count, sum, min returning low usage
    mock_conn.execute.return_value.fetchone.side_effect = [
        (1, 200, None),  # req_count, current_tokens, earliest
        (42, 1726000000.0),  # RETURNING id, created_at
    ]

    pacer = DistributedTokenPacer(
        settings.model_copy(update={"llm_pacing_enabled": True}), pool=mock_pool
    )

    async def run() -> None:
        res = await pacer.reserve("test/model", 100)
        assert res is not None
        assert res.tokens == 100
        assert getattr(res, "db_id", None) == 42
        await pacer.reconcile(res, 40)
        mock_conn.execute.assert_any_call(
            "UPDATE met_agent_provider_pacing SET tokens = %s WHERE id = %s",
            (40, 42),
        )

    asyncio.run(run())
