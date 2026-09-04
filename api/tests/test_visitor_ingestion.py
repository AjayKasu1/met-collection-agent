"""Test robots enforcement, safe redirects, content extraction, and Unicode chunk boundaries."""

from datetime import UTC, datetime

import httpx
import pytest

from met_agent.ingestion.http import SourceClient, SourceError
from met_agent.ingestion.visitors import (
    VisitorCrawler,
    chunk_markdown,
    html_to_markdown,
    validate_visitor_url,
)


def test_html_extraction_preserves_headings_but_removes_navigation() -> None:
    title, text = html_to_markdown(
        "<html><title>Visiting</title><main><nav>Account</nav><h1>Hours</h1>"
        "<p>Open Monday.<br>Closed Wednesday.</p><h2>Access</h2><ul><li>Elevator</li></ul>"
        '<img alt="Map of floor one"><script>secret script</script></main>'
        "<footer>Footer</footer></html>"
    )
    assert title == "Visiting"
    assert "# Hours" in text and "## Access" in text and "- Elevator" in text
    assert "Map of floor one" in text
    assert all(word not in text for word in ("Account", "secret script", "Footer"))


def test_chunks_overlap_preserve_provenance_and_have_stable_ids() -> None:
    text = "# Hours\n" + "Monday Tuesday Wednesday Thursday Friday " * 25
    source_url = "https://www.metmuseum.org/visit"
    fetched_at = datetime.now(UTC)
    chunks = chunk_markdown(
        text,
        target_tokens=50,
        overlap_tokens=10,
        source_url=source_url,
        page_title="Visit",
        fetched_at=fetched_at,
    )
    assert len(chunks) > 2
    assert all(chunk.text in text for chunk in chunks)
    assert all(chunk.section_heading == "Hours" for chunk in chunks)
    assert chunks[0].text[-20:] in chunks[1].text
    repeated = chunk_markdown(
        text,
        target_tokens=50,
        overlap_tokens=10,
        source_url=source_url,
        page_title="Visit",
        fetched_at=fetched_at,
    )
    assert [c.point_id for c in repeated] == [c.point_id for c in chunks]
    assert chunks[0].document().payload["source_url"] == source_url


def test_unicode_windows_and_section_boundaries() -> None:
    chunks = chunk_markdown(
        "# 参观\n" + "开放时间欢迎参观" * 20 + "\n## Accès\nAscenseur accessible.",
        source_url="https://www.metmuseum.org/visit",
        page_title="Visit",
        fetched_at=datetime.now(UTC),
        target_tokens=20,
        overlap_tokens=5,
    )
    assert chunks[-1].section_heading == "Accès"
    assert all("\ufffd" not in chunk.text for chunk in chunks)
    with pytest.raises(ValueError):
        chunk_markdown(
            "text",
            source_url="x",
            page_title="x",
            fetched_at=datetime.now(UTC),
            target_tokens=10,
            overlap_tokens=10,
        )
    with pytest.raises(ValueError):
        chunk_markdown("text", source_url="x", page_title="x", fetched_at=datetime(2026, 1, 1))
    with pytest.raises(SourceError):
        html_to_markdown("<main><script>only script</script></main>")


@pytest.mark.parametrize(
    "url",
    [
        "http://www.metmuseum.org/visit",
        "https://evil.example/visit",
        "https://user@www.metmuseum.org/visit",
    ],
)
def test_external_or_credentialed_sources_are_rejected(url: str) -> None:
    with pytest.raises(SourceError):
        validate_visitor_url(url)


def test_disallowed_page_is_never_requested() -> None:
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, text="User-agent: *\nDisallow: /private")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        crawler = VisitorCrawler(SourceClient(client, sleep=lambda _: None))
        with pytest.raises(SourceError, match="disallowed"):
            crawler.fetch("https://www.metmuseum.org/private")
    assert calls == ["/robots.txt"]


def test_redirect_rechecks_robots_and_rate_limit() -> None:
    calls = []
    delays: list[float] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nCrawl-delay: 2\nDisallow: /blocked")
        return httpx.Response(302, headers={"location": "/blocked"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        source = SourceClient(client, sleep=delays.append, clock=lambda: 0.0)
        with pytest.raises(SourceError, match="disallowed"):
            VisitorCrawler(source).fetch("https://www.metmuseum.org/start")
    assert calls == ["/robots.txt", "/start"]
    assert delays == [2.0]


def test_successful_crawl_follows_permitted_redirect() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path == "/start":
            return httpx.Response(301, headers={"location": "/visit"})
        return httpx.Response(200, text="<main>Hours</main>", headers={"content-type": "text/html"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        crawler = VisitorCrawler(SourceClient(client, sleep=lambda _: None))
        url, html = crawler.fetch("https://www.metmuseum.org/start")
    assert url.endswith("/visit") and "Hours" in html


def test_links_and_table_cells_remain_separate() -> None:
    _, text = html_to_markdown(
        '<main><h1>Map</h1><p>Use <a href="https://maps.metmuseum.org/">the map</a>.</p>'
        "<table><tr><th>Type</th><th>Price</th></tr><tr><td>Adult</td><td>$30</td></tr></table>"
        '<a href="javascript:alert(1)">ignore script</a></main>'
    )
    assert "[the map](https://maps.metmuseum.org/)" in text
    assert "Adult | $30" in text
    assert "javascript:" not in text


@pytest.mark.parametrize(
    "scenario", ["robots_error", "non_html", "no_location", "loop", "missing_page"]
)
def test_crawler_fails_closed_on_unusable_responses(scenario: str) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return (
                httpx.Response(403)
                if scenario == "robots_error"
                else httpx.Response(200, text="User-agent: *\nRequest-rate: 1/2\nAllow: /")
            )
        if scenario == "non_html":
            return httpx.Response(200, headers={"content-type": "application/json"})
        if scenario == "no_location":
            return httpx.Response(302)
        if scenario == "missing_page":
            return httpx.Response(404)
        return httpx.Response(302, headers={"location": "/loop"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client, pytest.raises(SourceError):
        VisitorCrawler(SourceClient(client, sleep=lambda _: None)).fetch(
            "https://www.metmuseum.org/visit"
        )
