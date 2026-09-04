"""Persist immutable session events in SQLite, with redacted inputs and bounded audit reads."""

import json
import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, JsonValue


class Event(BaseModel):
    sequence: int
    session_id: UUID
    turn_id: UUID
    timestamp: str
    kind: str
    data: JsonValue


class EventStore:
    def __init__(self, path: Path, *, secrets: Sequence[str] = ()) -> None:
        self.path, self.secrets = path, tuple(value for value in secrets if value)
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

    def append(self, session: UUID, turn: UUID, kind: str, data: JsonValue) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO events(session_id,turn_id,timestamp,kind,data) VALUES (?,?,?,?,?)",
                (
                    str(session),
                    str(turn),
                    datetime.now(UTC).isoformat(),
                    kind,
                    json.dumps(self.redact(data), ensure_ascii=False),
                ),
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
