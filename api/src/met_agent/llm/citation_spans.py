"""Build verbatim citation choices exclusively from server-returned tool evidence."""

import json
from typing import Any

from met_agent.agent.models import Citation
from met_agent.llm.prompts import load_prompt
from met_agent.tools.models import Evidence


def excerpts(text: str, limit: int = 1200) -> list[str]:
    """Split at line boundaries when possible; every excerpt remains a literal substring."""
    if limit < 1:
        raise ValueError("Excerpt limit must be positive")
    parts = []
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            newline = text.rfind("\n", start, end)
            if newline > start:
                end = newline + 1
        part = text[start:end].strip()
        if part:
            parts.append(part)
        start = end
    return parts


def prepare_final(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Citation]]:
    catalog: dict[str, Citation] = {}
    context = []
    for message in messages:
        if message.get("role") == "user":
            context.append(message)
        elif message.get("role") == "tool":
            body = json.loads(message["content"])
            choices = []
            for raw in body.pop("evidence", []):
                evidence = Evidence.model_validate(raw)
                if evidence.object_id is None and evidence.source_url is None:
                    continue
                for quote in excerpts(evidence.text):
                    key = f"q{len(catalog)}"
                    citation = Citation(
                        object_id=evidence.object_id,
                        source_url=evidence.source_url if evidence.object_id is None else None,
                        quote=quote,
                    )
                    catalog[key] = citation
                    choices.append({"quote_key": key, **citation.model_dump(exclude_none=True)})
            body["citation_excerpts"] = choices
            context.append({"role": "tool", "content": body})
    return [
        {
            "role": "system",
            "content": load_prompt("system_v2") + "\n" + load_prompt("citations_v1"),
        },
        {"role": "user", "content": json.dumps({"context": context}, ensure_ascii=False)},
    ], catalog
