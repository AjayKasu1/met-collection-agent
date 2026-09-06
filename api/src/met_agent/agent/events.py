"""Persist immutable session events in SQLite, with redacted inputs and bounded audit reads."""

import json
import re
import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID

from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, JsonValue


class Event(BaseModel):
    sequence: int
    session_id: UUID
    turn_id: UUID
    timestamp: str
    kind: str
    data: JsonValue


class AuditStore(Protocol):
    """Storage contract shared by the local and managed audit backends."""

    def redact(self, value: JsonValue) -> JsonValue: ...

    def append(self, session: UUID, turn: UUID, kind: str, data: JsonValue) -> None: ...

    def append_many(
        self, session: UUID, turn: UUID, events: Sequence[tuple[str, JsonValue]]
    ) -> None: ...

    def read(self, session: UUID, *, after: int = 0, limit: int = 200) -> list[Event]: ...

    def history(self, session: UUID) -> list[dict[str, str]]: ...

    def ready(self) -> bool: ...

    def close(self) -> None: ...


class RedactingStore:
    """Apply one credential-redaction policy before any persistence boundary."""

    def __init__(self, *, secrets: Sequence[str] = ()) -> None:
        self.secrets = tuple(value for value in secrets if value)

    def redact(self, value: JsonValue) -> JsonValue:
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, "[REDACTED]")
            return re.sub(r"(?:AIza[\w-]{30,}|(?:ghp_|gsk_|hf_)[\w-]{20,})", "[REDACTED]", value)
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        if isinstance(value, dict):
            return {
                key: "[REDACTED]"
                if any(
                    word in key.lower() for word in ("api_key", "authorization", "secret", "token")
                )
                and not key.endswith("_tokens")
                else self.redact(item)
                for key, item in value.items()
            }
        return value


class EventStore(RedactingStore):
    """Single-process SQLite audit store used for development and offline evaluation."""

    def __init__(self, path: Path, *, secrets: Sequence[str] = ()) -> None:
        super().__init__(secrets=secrets)
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL, turn_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_session ON events(session_id, sequence);
                CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
                BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
                BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
            """)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def append(self, session: UUID, turn: UUID, kind: str, data: JsonValue) -> None:
        self.append_many(session, turn, [(kind, data)])

    def append_many(
        self, session: UUID, turn: UUID, events: Sequence[tuple[str, JsonValue]]
    ) -> None:
        if not events:
            return
        timestamp = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO events(session_id,turn_id,timestamp,kind,data) VALUES (?,?,?,?,?)",
                [
                    (
                        str(session),
                        str(turn),
                        timestamp,
                        kind,
                        json.dumps(self.redact(data), ensure_ascii=False),
                    )
                    for kind, data in events
                ],
            )

    def read(self, session: UUID, *, after: int = 0, limit: int = 200) -> list[Event]:
        if after < 0 or not 1 <= limit <= 500:
            raise ValueError("Invalid audit cursor or limit")
        with self._connect() as connection:
            rows = connection.execute(
                (
                    "SELECT sequence,session_id,turn_id,timestamp,kind,data FROM events "
                    "WHERE session_id=? AND sequence>? ORDER BY sequence LIMIT ?"
                ),
                (str(session), after, limit),
            ).fetchall()
        return [
            Event(
                sequence=row[0],
                session_id=UUID(row[1]),
                turn_id=UUID(row[2]),
                timestamp=row[3],
                kind=row[4],
                data=json.loads(row[5]),
            )
            for row in rows
        ]

    def history(self, session: UUID) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                (
                    "SELECT kind,data FROM events WHERE session_id=? AND kind IN "
                    "('user_message','final_answer') ORDER BY sequence DESC LIMIT 12"
                ),
                (str(session),),
            ).fetchall()
        return [
            {
                "role": "user" if kind == "user_message" else "assistant",
                "content": json.loads(data)["message" if kind == "user_message" else "text"],
            }
            for kind, data in reversed(rows)
        ]

    def ready(self) -> bool:
        with self._connect() as connection:
            row = connection.execute("SELECT 1").fetchone()
        return row is not None and int(row[0]) == 1

    def close(self) -> None:
        """SQLite connections are scoped to individual operations."""


class PostgresEventStore(RedactingStore):
    """Pooled, durable PostgreSQL audit store with append-only records and retention."""

    def __init__(
        self,
        database_url: str,
        *,
        secrets: Sequence[str] = (),
        retention_days: int = 30,
        pool_min_size: int = 1,
        pool_max_size: int = 4,
    ) -> None:
        super().__init__(secrets=secrets)
        self.retention_days = retention_days
        self._last_purge = 0.0
        self._purge_lock = threading.Lock()
        self.pool = ConnectionPool(
            database_url,
            min_size=pool_min_size,
            max_size=pool_max_size,
            open=True,
            check=ConnectionPool.check_connection,
            timeout=10,
            name="met-agent-audit",
        )
        self.pool.wait(timeout=15)
        self._initialize()

    def _initialize(self) -> None:
        with self.pool.connection() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (7_779_410_001,))
            connection.execute("""
                CREATE TABLE IF NOT EXISTS met_agent_schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS met_agent_events (
                    sequence BIGSERIAL PRIMARY KEY,
                    session_id UUID NOT NULL,
                    turn_id UUID NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    kind TEXT NOT NULL CHECK (length(kind) BETWEEN 1 AND 80),
                    data JSONB NOT NULL
                )
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS met_agent_events_session
                ON met_agent_events(session_id, sequence)
            """)
            connection.execute("""
                CREATE OR REPLACE FUNCTION met_agent_events_immutable()
                RETURNS trigger LANGUAGE plpgsql AS $$
                BEGIN
                    IF TG_OP = 'DELETE'
                       AND current_setting('met_agent.retention_delete', true) = 'on' THEN
                        RETURN OLD;
                    END IF;
                    RAISE EXCEPTION 'audit events are append-only';
                END;
                $$
            """)
            connection.execute("""
                DROP TRIGGER IF EXISTS met_agent_events_no_mutation ON met_agent_events
            """)
            connection.execute("""
                CREATE TRIGGER met_agent_events_no_mutation
                BEFORE UPDATE OR DELETE ON met_agent_events
                FOR EACH ROW EXECUTE FUNCTION met_agent_events_immutable()
            """)
            connection.execute("""
                INSERT INTO met_agent_schema_migrations(version) VALUES (1)
                ON CONFLICT (version) DO NOTHING
            """)

    def append(self, session: UUID, turn: UUID, kind: str, data: JsonValue) -> None:
        self.append_many(session, turn, [(kind, data)])

    def append_many(
        self, session: UUID, turn: UUID, events: Sequence[tuple[str, JsonValue]]
    ) -> None:
        if not events:
            return
        timestamp = datetime.now(UTC)
        with self.pool.connection() as connection, connection.cursor() as cursor:
            cursor.executemany(
                (
                    "INSERT INTO met_agent_events"
                    "(session_id,turn_id,timestamp,kind,data) VALUES (%s,%s,%s,%s,%s)"
                ),
                [
                    (session, turn, timestamp, kind, Jsonb(self.redact(data)))
                    for kind, data in events
                ],
            )
        self._maybe_purge()

    def _maybe_purge(self) -> None:
        now = time.monotonic()
        if now - self._last_purge < 86_400 or not self._purge_lock.acquire(blocking=False):
            return
        try:
            with self.pool.connection() as connection:
                connection.execute("SET LOCAL met_agent.retention_delete = 'on'")
                connection.execute(
                    (
                        "DELETE FROM met_agent_events "
                        "WHERE timestamp < now() - make_interval(days => %s)"
                    ),
                    (self.retention_days,),
                )
            self._last_purge = now
        finally:
            self._purge_lock.release()

    def read(self, session: UUID, *, after: int = 0, limit: int = 200) -> list[Event]:
        if after < 0 or not 1 <= limit <= 500:
            raise ValueError("Invalid audit cursor or limit")
        with self.pool.connection() as connection:
            rows = connection.execute(
                (
                    "SELECT sequence,session_id,turn_id,timestamp,kind,data "
                    "FROM met_agent_events WHERE session_id=%s AND sequence>%s "
                    "ORDER BY sequence LIMIT %s"
                ),
                (session, after, limit),
            ).fetchall()
        return [
            Event(
                sequence=row[0],
                session_id=row[1],
                turn_id=row[2],
                timestamp=row[3].isoformat(),
                kind=row[4],
                data=row[5],
            )
            for row in rows
        ]

    def history(self, session: UUID) -> list[dict[str, str]]:
        with self.pool.connection() as connection:
            rows = connection.execute(
                (
                    "SELECT kind,data FROM met_agent_events WHERE session_id=%s "
                    "AND kind IN ('user_message','final_answer') "
                    "ORDER BY sequence DESC LIMIT 12"
                ),
                (session,),
            ).fetchall()
        return [
            {
                "role": "user" if kind == "user_message" else "assistant",
                "content": data["message" if kind == "user_message" else "text"],
            }
            for kind, data in reversed(rows)
        ]

    def ready(self) -> bool:
        with self.pool.connection() as connection:
            row = connection.execute("SELECT 1").fetchone()
        return row is not None and int(row[0]) == 1

    def close(self) -> None:
        self.pool.close()
