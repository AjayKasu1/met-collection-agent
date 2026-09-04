"""Adapt Pydantic schemas to Groq's strict constrained-decoding contract."""

import copy
from typing import Any

from pydantic import BaseModel


def strict_schema(schema: type[BaseModel]) -> dict[str, Any]:
    result = copy.deepcopy(schema.model_json_schema())

    def close(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("default", None)
            if value.get("type") == "object":
                value["additionalProperties"] = False
                value["required"] = list(value.get("properties", {}))
            for item in value.values():
                close(item)
        elif isinstance(value, list):
            for item in value:
                close(item)

    close(result)
    if schema.__name__ == "AgentDraft":
        citation = result["$defs"]["Citation"]
        alternatives = []
        for selected in ("object_id", "source_url"):
            branch = copy.deepcopy(citation)
            for name in ("object_id", "source_url"):
                if name != selected:
                    branch["properties"][name] = {"type": "null"}
                else:
                    branch["properties"][name] = next(
                        item
                        for item in branch["properties"][name]["anyOf"]
                        if item.get("type") != "null"
                    )
            alternatives.append(branch)
        result["$defs"]["Citation"] = {"anyOf": alternatives}
    return result


def response_format(schema: type[BaseModel], model: str) -> dict[str, Any]:
    native = model in {
        "groq/openai/gpt-oss-120b",
        "groq/openai/gpt-oss-20b",
        "groq/qwen/qwen3.8-27b",
    }
    if native:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": strict_schema(schema),
            },
        }
    return {"type": "json_object"}
