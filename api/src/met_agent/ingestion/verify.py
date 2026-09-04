"""Compare golden object IDs with the live Met API and the actual prepared collection."""

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from met_agent.ingestion.http import SourceClient, SourceError


class GoldenRow(BaseModel):
    """Keep original manual-review notes while validating the verification inputs."""

    model_config = ConfigDict(extra="allow")
    id: str
    question: str
    expected_object_ids: list[int] = Field(default_factory=list)
    verify: bool = False


class Verification(BaseModel):
    """Distinguish absent records from upstream failures and sample membership."""

    row_id: str
    object_id: int
    exists: bool | None
    in_ingested_set: bool
    title: str
    error: str = ""


def read_golden(path: Path) -> list[GoldenRow]:
    """Validate JSONL rows without interpreting their question or notes as instructions."""
    return [
        GoldenRow.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def verify_golden(
    rows: Sequence[GoldenRow],
    ingested_ids: set[int],
    source: SourceClient,
    api_base: str,
) -> list[Verification]:
    """Fetch each distinct object once; never infer existence from a non-404 error."""
    fetched: dict[int, tuple[bool | None, str, str]] = {}
    result = []
    for row in rows:
        for object_id in row.expected_object_ids:
            if object_id not in fetched:
                try:
                    response = source.get(f"{api_base.rstrip('/')}/objects/{object_id}")
                    if response.status_code == 404:
                        fetched[object_id] = (False, "", "HTTP 404")
                    elif response.status_code != 200:
                        fetched[object_id] = (None, "", f"HTTP {response.status_code}")
                    else:
                        data = response.json()
                        if not isinstance(data, dict) or data.get("objectID") != object_id:
                            raise ValueError("Unexpected object record")
                        title = data.get("title", "")
                        fetched[object_id] = (True, title if isinstance(title, str) else "", "")
                except (SourceError, ValueError):
                    fetched[object_id] = (None, "", "Request or response validation failed")
            exists, title, error = fetched[object_id]
            result.append(
                Verification(
                    row_id=row.id,
                    object_id=object_id,
                    exists=exists,
                    in_ingested_set=object_id in ingested_ids,
                    title=title,
                    error=error,
                )
            )
    return result


def verification_markdown(rows: Sequence[GoldenRow], results: Sequence[Verification]) -> str:
    """Render factual status and preserve every requested manual-review row."""
    lines = [
        "# Golden object verification",
        "",
        "Membership describes the selected sample, not retrieval accuracy.",
        "",
        "| Row | Object ID | Exists | In sample | Live title | Error |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for item in results:
        exists = "yes" if item.exists else "no" if item.exists is False else "unknown"
        title = item.title.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {item.row_id} | {item.object_id} | {exists} | "
            f"{'yes' if item.in_ingested_set else 'no'} | {title} | {item.error} |"
        )
    lines.extend(["", "## Rows requiring manual confirmation", ""])
    for row in rows:
        if row.verify:
            lines.extend(["```json", row.model_dump_json(), "```", ""])
    return "\n".join(lines) + "\n"
