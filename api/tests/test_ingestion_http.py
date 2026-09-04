"""Verify bounded retries and safe failures for unreliable public-source responses."""

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest

from met_agent.ingestion.http import USER_AGENT, SourceClient, SourceError, download_csv


@pytest.mark.parametrize(
    "retry_after", ["2", "unparseable", format_datetime(datetime.now(UTC) + timedelta(seconds=30))]
)
def test_transient_errors_retry_with_identifying_agent(retry_after: str) -> None:
    calls = 0
    delays: list[float] = []

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["User-Agent"] == USER_AGENT
        return (
            httpx.Response(429, headers={"Retry-After": retry_after})
            if calls == 1
            else httpx.Response(200, json={"ok": True})
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        source = SourceClient(client, interval=0, sleep=delays.append)
        assert source.json("https://source.example") == {"ok": True}
    assert calls == 2 and len(delays) == 1 and 0 <= delays[0] <= 60


def test_connections_retry_and_authentication_does_not() -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("private upstream detail", request=request)

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        pytest.raises(SourceError, match="after retries"),
    ):
        SourceClient(client, interval=0, sleep=lambda _: None, retries=2).get(
            "https://source.example"
        )
    assert calls == 3
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(403, text="private"))
        ) as client,
        pytest.raises(SourceError, match="HTTP 403"),
    ):
        SourceClient(client).json("https://source.example")
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, text="not JSON"))
        ) as client,
        pytest.raises(SourceError, match="invalid JSON"),
    ):
        SourceClient(client).json("https://source.example")


def test_failed_download_retains_previous_cache(tmp_path: Path) -> None:
    cache = tmp_path / "MetObjects.csv"
    cache.write_text("existing artifact")
    with (
        httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503))) as client,
        pytest.raises(SourceError, match="HTTP 503"),
    ):
        download_csv(client, "https://source.example", cache, refresh=True)

    def interrupted(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("private details", request=request)

    with (
        httpx.Client(transport=httpx.MockTransport(interrupted)) as client,
        pytest.raises(SourceError, match="retained"),
    ):
        download_csv(client, "https://source.example", cache, refresh=True)
    assert cache.read_text() == "existing artifact"
    assert not list(tmp_path.glob(".download-*"))
