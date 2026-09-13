"""Test distributed PostgreSQL session locks and cross-instance pacing budgets."""

import asyncio
import os
import threading
from typing import Any
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
    mock_conn.execute.return_value.fetchone.return_value = (True,)
    mock_pool.getconn.return_value = mock_conn

    lock = PostgresSessionLock(mock_pool, session_id, timeout=10.0)

    async def run() -> None:
        async with lock:
            assert mock_pool.getconn.called
            expected_lock_id = session_id.int & 0x7FFFFFFFFFFFFFFF
            mock_conn.execute.assert_any_call(
                "SELECT pg_try_advisory_lock(%s)", (expected_lock_id,)
            )

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


def test_postgres_session_lock_cancellation_while_getconn_pending() -> None:
    session_id = uuid4()
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.return_value = (True,)

    getconn_started = threading.Event()
    release_getconn = threading.Event()

    def slow_getconn(*args: Any, **kwargs: Any) -> Any:
        getconn_started.set()
        release_getconn.wait(timeout=2.0)
        return mock_conn

    mock_pool.getconn.side_effect = slow_getconn

    lock = PostgresSessionLock(mock_pool, session_id, timeout=5.0)

    async def run() -> None:
        async def acquire_task() -> None:
            async with lock:
                pass

        t = asyncio.create_task(acquire_task())
        await asyncio.to_thread(getconn_started.wait, 1.0)
        t.cancel()
        release_getconn.set()
        with pytest.raises(asyncio.CancelledError):
            await t

    asyncio.run(run())
    assert mock_pool.putconn.call_count == 1
    mock_pool.putconn.assert_called_with(mock_conn)


def test_postgres_session_lock_cancellation_while_query_pending() -> None:
    session_id = uuid4()
    mock_pool = MagicMock()
    mock_conn = MagicMock()

    query_started = threading.Event()
    release_query = threading.Event()

    def slow_execute(query: str, *args: Any, **kwargs: Any) -> Any:
        if "pg_try_advisory_lock" in query:
            query_started.set()
            release_query.wait(timeout=2.0)
            mock_res = MagicMock()
            mock_res.fetchone.return_value = (True,)
            return mock_res
        return MagicMock()

    mock_conn.execute.side_effect = slow_execute
    mock_pool.getconn.return_value = mock_conn

    lock = PostgresSessionLock(mock_pool, session_id, timeout=5.0)

    async def run() -> None:
        async def acquire_task() -> None:
            async with lock:
                pass

        t = asyncio.create_task(acquire_task())
        await asyncio.to_thread(query_started.wait, 1.0)

        # Cancel while the query is actively in flight in the background thread
        t.cancel()
        # Verify connection has NOT been returned to the pool before query completes
        assert mock_pool.putconn.call_count == 0

        # Release background query to complete
        release_query.set()
        with pytest.raises(asyncio.CancelledError):
            await t

    asyncio.run(run())

    # Verify exactly one putconn call and that advisory unlock was executed
    assert mock_pool.putconn.call_count == 1
    mock_pool.putconn.assert_called_with(mock_conn)
    mock_conn.execute.assert_any_call("SELECT pg_advisory_unlock(%s)", (lock.lock_id,))


def test_postgres_session_lock_cancellation_while_query_pending_unacquired() -> None:
    session_id = uuid4()
    mock_pool = MagicMock()
    mock_conn = MagicMock()

    query_started = threading.Event()
    release_query = threading.Event()

    def slow_execute(query: str, *args: Any, **kwargs: Any) -> Any:
        if "pg_try_advisory_lock" in query:
            query_started.set()
            release_query.wait(timeout=2.0)
            mock_res = MagicMock()
            mock_res.fetchone.return_value = (False,)
            return mock_res
        return MagicMock()

    mock_conn.execute.side_effect = slow_execute
    mock_pool.getconn.return_value = mock_conn

    lock = PostgresSessionLock(mock_pool, session_id, timeout=5.0)

    async def run() -> None:
        async def acquire_task() -> None:
            async with lock:
                pass

        t = asyncio.create_task(acquire_task())
        await asyncio.to_thread(query_started.wait, 1.0)

        t.cancel()
        assert mock_pool.putconn.call_count == 0

        release_query.set()
        with pytest.raises(asyncio.CancelledError):
            await t

    asyncio.run(run())

    assert mock_pool.putconn.call_count == 1
    mock_pool.putconn.assert_called_with(mock_conn)
    assert not any("pg_advisory_unlock" in str(call) for call in mock_conn.execute.call_args_list)


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

        async def worker1() -> None:
            async with lock1:
                task1_acquired.set()
                await task1_release.wait()

        t1 = asyncio.create_task(worker1())
        await asyncio.wait_for(task1_acquired.wait(), timeout=5.0)

        # Concurrent attempt on held session lock must fail with TimeoutError
        with pytest.raises(TimeoutError):
            async with lock2:
                pass

        # Release first worker
        task1_release.set()
        await asyncio.wait_for(t1, timeout=5.0)

        # Verify lock can now be reacquired
        lock3 = PostgresSessionLock(store.pool, session_id, timeout=1.0)
        async with lock3:
            pass

        # Verify cancellation while waiting cleans up connection pool
        hold_release = asyncio.Event()
        held_event = asyncio.Event()
        lock_holder = PostgresSessionLock(store.pool, session_id, timeout=2.0)

        async def holder() -> None:
            async with lock_holder:
                held_event.set()
                await hold_release.wait()

        t_holder = asyncio.create_task(holder())
        await asyncio.wait_for(held_event.wait(), timeout=5.0)

        async def cancel_worker() -> None:
            lock4 = PostgresSessionLock(store.pool, session_id, timeout=5.0)
            async with lock4:
                pass

        t_cancel = asyncio.create_task(cancel_worker())
        await asyncio.sleep(0.05)
        t_cancel.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(t_cancel, timeout=5.0)

        hold_release.set()
        await asyncio.wait_for(t_holder, timeout=5.0)

        # Pool must still have all connections available (not exhausted/leaked)
        lock5 = PostgresSessionLock(store.pool, session_id, timeout=1.0)
        async with lock5:
            pass

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
