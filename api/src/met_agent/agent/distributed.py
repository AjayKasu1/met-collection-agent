"""Distributed PostgreSQL session locks and cross-instance provider pacing budgets."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any
from uuid import UUID

from psycopg_pool import ConnectionPool

from met_agent.agent.events import AuditStore
from met_agent.config import LLMRateLimit, Settings
from met_agent.llm.pacing import RequestBudgetError, Reservation, TokenPacer


class PostgresSessionLock:
    """Distributed session coordination using 64-bit PostgreSQL advisory locks."""

    def __init__(
        self,
        pool: ConnectionPool,
        session_id: UUID,
        timeout: float = 30.0,
    ) -> None:
        self.pool = pool
        self.session_id = session_id
        # 63-bit positive integer hash fits PostgreSQL 64-bit signed advisory lock
        self.lock_id = session_id.int & 0x7FFFFFFFFFFFFFFF
        self.timeout = timeout
        self.conn: Any = None
        self._acquired = False
        self._cancelled = False
        self._state_lock = threading.Lock()

    def _try_acquire(self) -> bool:
        with self._state_lock:
            if self._cancelled:
                return False
            existing_conn = self.conn

        if existing_conn is None:
            conn = self.pool.getconn(timeout=self.timeout)
            with self._state_lock:
                if self._cancelled:
                    self.pool.putconn(conn)
                    return False
                self.conn = conn
                existing_conn = conn

        row = existing_conn.execute("SELECT pg_try_advisory_lock(%s)", (self.lock_id,)).fetchone()
        acquired = bool(row and row[0])

        with self._state_lock:
            if self._cancelled:
                try:
                    if acquired:
                        with suppress(Exception):
                            existing_conn.execute("SELECT pg_advisory_unlock(%s)", (self.lock_id,))
                finally:
                    self.conn = None
                    self._acquired = False
                    self.pool.putconn(existing_conn)
                return False

            self._acquired = acquired
            return acquired

    def _cleanup(self) -> None:
        with self._state_lock:
            self._cancelled = True
            conn = self.conn
            acquired = self._acquired
            self.conn = None
            self._acquired = False

        if conn is not None:
            try:
                if acquired:
                    with suppress(Exception):
                        conn.execute("SELECT pg_advisory_unlock(%s)", (self.lock_id,))
            finally:
                self.pool.putconn(conn)

    async def __aenter__(self) -> PostgresSessionLock:
        deadline = time.monotonic() + self.timeout
        try:
            while True:
                self._acquired = await asyncio.to_thread(self._try_acquire)
                if self._acquired:
                    return self
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Session lock timeout ({self.timeout}s): {self.session_id}")
                await asyncio.sleep(min(0.05, max(0.01, deadline - time.monotonic())))
        except BaseException:
            await asyncio.to_thread(self._cleanup)
            raise

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await asyncio.to_thread(self._cleanup)


class LocalNullLock:
    """No-op distributed lock when running locally with SQLite."""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None


def get_session_lock(events: AuditStore, session_id: UUID) -> Any:
    """Select PostgreSQL advisory locking in production or process-local in development."""
    pool = getattr(events, "pool", None)
    if pool is not None and isinstance(pool, ConnectionPool):
        return PostgresSessionLock(pool, session_id)
    return LocalNullLock()


class DistributedTokenPacer(TokenPacer):
    """Coordinate token rate-limit budgets across multiple Cloud Run instances via PostgreSQL."""

    def __init__(
        self,
        settings: Settings,
        pool: ConnectionPool | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        super().__init__(settings, clock=clock, sleep=sleep)
        self.pool = pool

    def _reserve_postgres(
        self, model: str, tokens: int, limit: LLMRateLimit
    ) -> tuple[Reservation | None, float]:
        if self.pool is None:
            return None, 0.0
        with self.pool.connection() as conn, conn.transaction():
            # Serialize check-and-insert per model across instances using transaction advisory lock
            conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"pacing:{model}",))
            conn.execute(
                "DELETE FROM met_agent_provider_pacing "
                "WHERE created_at < now() - INTERVAL '60 seconds'"
            )
            row = conn.execute(
                """
                SELECT count(*), coalesce(sum(tokens), 0),
                       coalesce(extract(epoch from (now() - min(created_at))), 60)
                FROM met_agent_provider_pacing
                WHERE model = %s AND created_at >= now() - INTERVAL '60 seconds'
                """,
                (model,),
            ).fetchone()
            req_count = int(row[0]) if row and row[0] is not None else 0
            current_tokens = int(row[1]) if row and row[1] is not None else 0
            oldest_age = float(row[2]) if row and row[2] is not None else 60.0

            if (
                req_count < limit.requests_per_minute
                and (current_tokens + tokens) <= limit.tokens_per_minute
            ):
                res_row = conn.execute(
                    """
                    INSERT INTO met_agent_provider_pacing (model, tokens)
                    VALUES (%s, %s)
                    RETURNING id, extract(epoch from created_at)
                    """,
                    (model, tokens),
                ).fetchone()
                db_id = res_row[0] if res_row else 0
                ts = float(res_row[1]) if res_row and res_row[1] is not None else self.clock()
                reservation = Reservation(started=ts, tokens=tokens, db_id=db_id)
                return reservation, 0.0

            # Compute wait time until oldest entry rolls past 60s
            wait = max(0.1, min(1.0, 60.0 - oldest_age + 0.05))
            return None, wait

    def _reconcile_postgres(self, reservation: Reservation, actual_tokens: int) -> None:
        if self.pool is None:
            return
        if reservation.db_id is not None:
            with self.pool.connection() as conn:
                conn.execute(
                    "UPDATE met_agent_provider_pacing SET tokens = %s WHERE id = %s",
                    (max(1, actual_tokens), reservation.db_id),
                )

    async def reserve(self, model: str, tokens: int) -> Reservation | None:
        if not self.enabled:
            return None
        limit = self.limits.get(model, self.default)
        if not 0 < tokens <= limit.tokens_per_minute:
            raise RequestBudgetError(
                "Request exceeds the configured per-minute token budget; "
                "reduce context or set the model's verified account limit"
            )
        if self.pool is None:
            return await super().reserve(model, tokens)

        while True:
            reservation, wait = await asyncio.to_thread(
                self._reserve_postgres, model, tokens, limit
            )
            if reservation is not None:
                return reservation
            await self.sleep(max(0.05, wait))

    async def reconcile(self, reservation: Reservation | None, actual_tokens: int) -> None:
        if reservation is not None:
            if self.pool is not None:
                await asyncio.to_thread(self._reconcile_postgres, reservation, actual_tokens)
            else:
                await super().reconcile(reservation, actual_tokens)
