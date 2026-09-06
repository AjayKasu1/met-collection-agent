"""Verify the managed audit backend against a real PostgreSQL process."""

import os
from uuid import uuid4

import psycopg
import pytest

from met_agent.agent.events import PostgresEventStore


@pytest.mark.integration
def test_postgres_audit_is_atomic_redacted_and_append_only() -> None:
    database_url = os.environ.get("TEST_POSTGRES_URL")
    if not database_url:
        pytest.skip("TEST_POSTGRES_URL is not configured")

    session, turn = uuid4(), uuid4()
    store = PostgresEventStore(
        database_url,
        secrets=["unit-test-private-value"],
        pool_min_size=1,
        pool_max_size=2,
    )
    try:
        store.append_many(
            session,
            turn,
            [
                ("user_message", {"message": "unit-test-private-value"}),
                ("final_answer", {"text": "Public answer"}),
            ],
        )
        events = store.read(session)
        assert [event.kind for event in events] == ["user_message", "final_answer"]
        assert events[0].data == {"message": "[REDACTED]"}
        assert store.history(session) == [
            {"role": "user", "content": "[REDACTED]"},
            {"role": "assistant", "content": "Public answer"},
        ]
        assert store.ready()

        with (
            pytest.raises(psycopg.errors.RaiseException, match="append-only"),
            psycopg.connect(database_url) as connection,
        ):
            connection.execute(
                "UPDATE met_agent_events SET kind='changed' WHERE session_id=%s",
                (session,),
            )

        failed_session = uuid4()
        with pytest.raises(psycopg.errors.CheckViolation):
            store.append_many(
                failed_session,
                turn,
                [("valid", {}), ("x" * 81, {})],
            )
        assert store.read(failed_session) == []
    finally:
        with psycopg.connect(database_url) as connection:
            connection.execute("SET LOCAL met_agent.retention_delete = 'on'")
            connection.execute(
                "DELETE FROM met_agent_events WHERE session_id IN (%s,%s)",
                (session, failed_session),
            )
        store.close()
