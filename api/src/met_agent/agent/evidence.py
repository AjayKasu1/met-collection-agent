"""Bound exact source excerpts across a whole turn, retaining the latest copy of each key."""

import json
from typing import Any

from met_agent.tools.models import Evidence

MAX_RECORDS = 8
MAX_TEXT_CHARACTERS = 6000
MAX_RECORD_CHARACTERS = 1400


def pack_evidence(messages: list[dict[str, Any]]) -> list[Evidence]:
    selected: dict[int, list[Evidence]] = {}
    seen: set[str] = set()
    remaining = MAX_TEXT_CHARACTERS
    records: list[Evidence] = []
    bodies: dict[int, dict[str, Any]] = {}
    for index in reversed(range(len(messages))):
        message = messages[index]
        if message.get("role") != "tool":
            continue
        body = json.loads(message["content"])
        bodies[index] = body
        for raw in body.get("evidence", []):
            item = Evidence.model_validate(raw)
            if item.key in seen:
                continue
            seen.add(item.key)
            if remaining <= 0 or len(records) >= MAX_RECORDS:
                continue
            limit = min(remaining, MAX_RECORD_CHARACTERS)
            text = item.text[:limit]
            if len(item.text) > limit:
                boundary = text.rfind("\n")
                if boundary > limit // 2:
                    text = text[:boundary]
            item = item.model_copy(update={"text": text})
            remaining -= len(text)
            records.append(item)
            selected.setdefault(index, []).append(item)
    for index, body in bodies.items():
        body["evidence"] = [
            item.model_dump(mode="json", exclude_none=True) for item in selected.get(index, [])
        ]
        messages[index] = {**messages[index], "content": json.dumps(body, ensure_ascii=False)}
    return records
