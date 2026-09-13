"""Test distributed PostgreSQL session locks and cross-instance pacing budgets."""

import asyncio
import os
from unittest.mock import MagicMock
from uuid import uuid4

import psycopg
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


@pytest.mark.integration
def test_real_postgres_session_lock_concurrency_and_cancellation() -> None:
    database_url = os.environ.get("TEST_POSTGRES_URL")
    if not database_url:
        pytest.skip("TEST_POSTGRES_URL is not configured")

    from met_agent.agent.events import PostgresEventStore

    store = PostgresEventStore(database_url, pool_min_size=1, pool_max_size=4)
    session_id = uuid4()

    async def exercise() -> None:
        lock1 = PostgresSessionLock(store.pool, session_id, timeout=2.0)
        lock2 = PostgresSessionLock(store.pool, session_id, timeout=0.1)

        task1_acquired = asyncio.Event()
        task1_release = asyncio.Event()
        task2_failed = asyncio.Event()

        async def worker1() -> None:
            async with lock1:
                task1_acquired.set()
                await task1_release.wait()

        async def worker2() -> None:
            await task1_acquired.wait()
            try:
                async with lock2:
                    pass
            except (psycopg.errors.QueryCanceled, psycopg.errors.LockNotAvailable):
                task2_failed.set()

        t1 = asyncio.create_task(worker1())
        t2 = asyncio.create_task(worker2())

        await task2_failed.wait()
        task1_release.set()
        await t1
        await t2

        # Verify cancellation cleans up without pool exhaustion
        async def cancel_worker() -> None:
            lock3 = PostgresSessionLock(store.pool, session_id, timeout=5.0)
            async with lock3:
                pass

        task3 = asyncio.create_task(cancel_worker())
        await asyncio.sleep(0.01)
        task3.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task3

    try:
        asyncio.run(exercise())
    finally:
        store.close()


@pytest.mark.integration
def test_real_postgres_distributed_token_pacer_concurrency(settings: Settings) -> None:
    database_url = os.environ.get("TEST_POSTGRES_URL")
    if not database_url:
        pytest.skip("TEST_POSTGRES_URL is not configured")

    from met_agent.agent.events import PostgresEventStore
    from met_agent.config import LLMRateLimit

    store = PostgresEventStore(database_url, pool_min_size=1, pool_max_size=4)
    model = "test-concurrent-model"
    limit = LLMRateLimit(requests_per_minute=20, tokens_per_minute=500)
    config = settings.model_copy(
        update={
            "llm_pacing_enabled": True,
            "llm_rate_limits": {model: limit},
        }
    )
    pacer = DistributedTokenPacer(config, pool=store.pool)

    async def exercise() -> None:
        # Run 3 concurrent requests for 200 tokens each under 500 token limit
        # The first 2 succeed (400 <= 500), the 3rd must wait/timeout
        results: list[bool] = []

        async def attempt() -> None:
            try:
                res = await asyncio.wait_for(pacer.reserve(model, 200), timeout=0.3)
                results.append(res is not None)
            except TimeoutError:
                results.append(False)

        await asyncio.gather(attempt(), attempt(), attempt())
        # Exactly 2 should succeed (400 tokens) and 1 must have waited/timed out
        assert results.count(True) == 2
        assert results.count(False) == 1

    try:
        asyncio.run(exercise())
    finally:
        with store.pool.connection() as conn:
            conn.execute("DELETE FROM met_agent_provider_pacing WHERE model = %s", (model,))
        store.close()
