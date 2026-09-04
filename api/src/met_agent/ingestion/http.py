"""Fetch public sources with bounded retries, an identifying User-Agent, and safe failures."""

import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx

from met_agent.ingestion.storage import sha256_file, write_json

USER_AGENT = "met-collection-agent/0.1 (+https://github.com/AjayKasu1/met-collection-agent)"
RETRYABLE = frozenset({429, 500, 502, 503, 504})


class SourceError(RuntimeError):
    """A public-source failure that omits response bodies and credential-bearing URLs."""


class SourceClient:
    """A synchronous ingestion client with one explicit request budget."""

    def __init__(
        self,
        client: httpx.Client,
        *,
        interval: float = 0.2,
        retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.interval = interval
        self.retries = retries
        self.sleep = sleep
        self.clock = clock
        self._last_request: float | None = None

    def throttle(self) -> None:
        """Space request starts, including retries and redirects."""
        if self._last_request is not None:
            remaining = self.interval - (self.clock() - self._last_request)
            if remaining > 0:
                self.sleep(remaining)
        self._last_request = self.clock()

    def get(self, url: str, *, params: dict[str, str] | None = None) -> httpx.Response:
        """Retry transient failures only, honoring bounded Retry-After values."""
        for attempt in range(self.retries + 1):
            self.throttle()
            try:
                response = self.client.get(url, params=params, headers={"User-Agent": USER_AGENT})
            except httpx.TransportError:
                if attempt == self.retries:
                    raise SourceError("Source connection failed after retries") from None
                self.sleep(min(2**attempt, 30))
                continue
            if response.status_code not in RETRYABLE or attempt == self.retries:
                return response
            retry_after = response.headers.get("Retry-After", "")
            try:
                delay = float(retry_after)
            except ValueError:
                try:
                    delay = (parsedate_to_datetime(retry_after) - datetime.now(UTC)).total_seconds()
                except (TypeError, ValueError):
                    delay = float(2**attempt)
            self.sleep(max(0, min(delay, 60)))
        raise AssertionError("Retry loop must return or raise")

    def json(self, url: str, *, params: dict[str, str] | None = None) -> object:
        """Require a successful JSON response without including remote body text in errors."""
        response = self.get(url, params=params)
        if response.status_code != 200:
            raise SourceError(f"Source returned HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError:
            raise SourceError("Source returned invalid JSON") from None


def download_csv(
    client: httpx.Client, url: str, destination: Path, *, refresh: bool = False
) -> Path:
    """Stream the LFS media file to disk, rejecting pointers and incomplete downloads."""
    if destination.exists() and not refresh:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=".download-", delete=False
    ) as f:
        temporary = Path(f.name)
    try:
        with client.stream(
            "GET", url, headers={"User-Agent": USER_AGENT}, follow_redirects=True
        ) as r:
            if r.status_code != 200:
                raise SourceError(f"CSV download returned HTTP {r.status_code}")
            size = 0
            with temporary.open("wb") as handle:
                for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                    size += len(chunk)
                    if size > 1024 * 1024 * 1024:
                        raise SourceError("CSV download exceeded the 1 GiB safety limit")
                    handle.write(chunk)
        with temporary.open("r", encoding="utf-8-sig") as handle:
            header = handle.readline()
        if "Object ID" not in header or "Is Public Domain" not in header:
            raise SourceError("Downloaded file is not the Met CSV; use the Git LFS media URL")
        checksum = sha256_file(temporary)
        temporary.replace(destination)
        write_json(
            destination.with_suffix(".source.json"),
            {
                "source_url": url,
                "fetched_at": datetime.now(UTC).isoformat(),
                "sha256": checksum,
            },
        )
        return destination
    except httpx.TransportError:
        raise SourceError("CSV download connection failed; previous cache was retained") from None
    finally:
        temporary.unlink(missing_ok=True)
