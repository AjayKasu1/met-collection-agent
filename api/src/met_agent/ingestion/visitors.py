"""Respect robots rules while converting curated visitor pages into attributed text windows."""

import re
from collections.abc import Callable
from datetime import datetime
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser
from uuid import NAMESPACE_URL, uuid5

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify

from met_agent.ingestion.http import USER_AGENT, SourceClient, SourceError
from met_agent.ingestion.models import VisitorChunk

_HOSTS = frozenset({"www.metmuseum.org", "metmuseum.org", "maps.metmuseum.org"})
_TOKENS = re.compile(r"[\u3400-\u9fff]|[^\W_]{1,4}|[^\w\s]", re.UNICODE)


def validate_visitor_url(url: str) -> str:
    """Restrict fetched content and redirect targets to the curated public website."""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in _HOSTS or parsed.username is not None:
        raise SourceError("Visitor sources must be HTTPS URLs on metmuseum.org")
    return url


class VisitorCrawler:
    """Check robots.txt for every origin and redirected URL before fetching page content."""

    def __init__(self, source: SourceClient) -> None:
        self.source = source
        self.source.interval = max(1.0, self.source.interval)
        self._robots: dict[str, RobotFileParser] = {}

    def _policy(self, url: str) -> RobotFileParser:
        parsed = urlsplit(validate_visitor_url(url))
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            response = self.source.get(origin + "/robots.txt")
            policy = RobotFileParser()
            if response.status_code == 404:
                policy.parse(["User-agent: *", "Allow: /"])
            elif response.status_code == 200:
                policy.parse(response.text.splitlines())
            else:
                raise SourceError(
                    f"Cannot establish robots permission: HTTP {response.status_code}"
                )
            delay = policy.crawl_delay(USER_AGENT) or policy.crawl_delay("*") or 0
            rate = policy.request_rate(USER_AGENT) or policy.request_rate("*")
            self.source.interval = max(self.source.interval, float(delay))
            if rate and rate.requests:
                self.source.interval = max(self.source.interval, rate.seconds / rate.requests)
            self._robots[origin] = policy
        return self._robots[origin]

    def fetch(self, url: str) -> tuple[str, str]:
        """Return final source URL and HTML; forbidden redirects are never requested."""
        for _ in range(6):
            validate_visitor_url(url)
            if not self._policy(url).can_fetch(USER_AGENT, url):
                raise SourceError("Visitor page is disallowed by robots.txt")
            response = self.source.get(url)
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise SourceError("Visitor redirect has no destination")
                url = validate_visitor_url(urljoin(url, location))
                continue
            if response.status_code != 200:
                raise SourceError(f"Visitor page returned HTTP {response.status_code}")
            if "html" not in response.headers.get("content-type", "").lower():
                raise SourceError("Visitor page is not HTML")
            return url, response.text
        raise SourceError("Visitor page exceeded the redirect limit")


def html_to_markdown(html: str) -> tuple[str, str]:
    """Extract main content, retain headings and list structure, and discard site navigation."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    candidate = soup.find("main") or soup.find("article") or soup.body
    content = candidate if isinstance(candidate, Tag) else soup
    for picture in content.find_all("picture"):
        picture.unwrap()
    for image in content.find_all("img"):
        image.replace_with(str(image.get("alt") or ""))
    # Complementary sections can contain factual hours/admission tables.
    for aside in content.find_all("aside"):
        if aside.find("table") is not None:
            aside.unwrap()
        else:
            aside.decompose()
    for element in content.find_all(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "noscript",
            "iframe",
            "svg",
        ]
    ):
        element.decompose()
    for anchor in content.find_all("a", href=True):
        href = anchor.get("href")
        if isinstance(href, str):
            absolute = urljoin("https://www.metmuseum.org", href)
            if urlsplit(absolute).scheme in {"https", "http", "mailto", "tel"}:
                anchor["href"] = absolute
            else:
                del anchor["href"]
    text = str(markdownify(str(content), heading_style="ATX", bullets="-")).strip()
    if not text:
        raise SourceError("Visitor page contains no usable main text")
    return title, text


def chunk_markdown(
    markdown: str,
    *,
    source_url: str,
    page_title: str,
    fetched_at: datetime,
    target_tokens: int = 500,
    overlap_tokens: int = 50,
    token_count: Callable[[str], int] | None = None,
    max_model_tokens: int = 512,
) -> list[VisitorChunk]:
    """Window sections using a deterministic, Unicode-safe approximate token counter.

    Latin words are split into up to four-character pieces; CJK characters count
    individually. This is a local estimate, not a claim about Gemini token billing.
    When supplied, the model counter includes its prefix and special tokens and
    bounds every emitted window. Windows never split a Unicode code point.
    """
    if target_tokens <= 0 or not 0 <= overlap_tokens < target_tokens or max_model_tokens <= 0:
        raise ValueError("Chunk size must exceed overlap and be positive")
    if fetched_at.tzinfo is None:
        raise ValueError("Fetch timestamp must have a timezone")
    sections: list[tuple[str, str]] = []
    heading = page_title
    start = 0
    for match in re.finditer(r"^#{1,6}\s+(.+)$", markdown, re.MULTILINE):
        if markdown[start : match.start()].strip():
            sections.append((heading, markdown[start : match.start()].strip()))
        heading = match.group(1).strip()
        start = match.start()
    if markdown[start:].strip():
        sections.append((heading, markdown[start:].strip()))
    result: list[VisitorChunk] = []
    for heading, text in sections:
        spans = list(_TOKENS.finditer(text))
        offset = 0
        while offset < len(spans):
            end = min(offset + target_tokens, len(spans))
            if token_count is not None:
                low, high, fitted = offset + 1, end, None
                while low <= high:
                    middle = (low + high) // 2
                    candidate = text[spans[offset].start() : spans[middle - 1].end()]
                    if token_count(candidate) <= max_model_tokens:
                        fitted, low = middle, middle + 1
                    else:
                        high = middle - 1
                if fitted is None:
                    raise SourceError("A visitor text span exceeds the model token limit")
                end = fitted
            index = len(result)
            result.append(
                VisitorChunk(
                    point_id=str(uuid5(NAMESPACE_URL, f"{source_url}|{index}")),
                    source_url=source_url,
                    page_title=page_title,
                    section_heading=heading,
                    fetched_at=fetched_at,
                    chunk_index=index,
                    text=text[spans[offset].start() : spans[end - 1].end()],
                )
            )
            if end == len(spans):
                break
            offset = max(offset + 1, end - overlap_tokens)
    return result
