"""Distributed PostgreSQL session locks and cross-instance provider pacing budgets."""

from __future__ import annotations

import asyncio
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

    def _acquire(self) -> None:
        self.conn = self.pool.getconn(timeout=self.timeout)
        try:
            self.conn.execute("SET lock_timeout = %s", (f"{int(self.timeout * 1000)}ms",))
            self.conn.execute("SELECT pg_advisory_lock(%s)", (self.lock_id,))
        except Exception:
            if self.conn is not None:
                self.pool.putconn(self.conn)
                self.conn = None
            raise

    def _release(self) -> None:
        if self.conn is not None:
            try:
                with suppress(Exception):
                    self.conn.execute("SELECT pg_advisory_unlock(%s)", (self.lock_id,))
            finally:
                self.pool.putconn(self.conn)
                self.conn = None

    async def __aenter__(self) -> PostgresSessionLock:
        await asyncio.to_thread(self._acquire)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await asyncio.to_thread(self._release)


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
            conn.execute(
                "DELETE FROM met_agent_provider_pacing "
                "WHERE created_at < now() - INTERVAL '60 seconds'"
            )
            row = conn.execute(
                """
                SELECT count(*), coalesce(sum(tokens), 0), min(created_at)
                FROM met_agent_provider_pacing
                WHERE model = %s AND created_at >= now() - INTERVAL '60 seconds'
                """,
                (model,),
            ).fetchone()
            req_count = int(row[0]) if row and row[0] is not None else 0
            current_tokens = int(row[1]) if row and row[1] is not None else 0
            earliest = row[2] if row else None

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
            wait = 0.5
            if earliest is not None:
                wait = 0.5
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
