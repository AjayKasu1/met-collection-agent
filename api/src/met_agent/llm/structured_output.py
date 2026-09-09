"""Adapt Pydantic schemas to provider-native constrained decoding."""

import copy
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from met_agent.agent.models import AgentDraft, Citation, Language


def strict_schema(schema: type[BaseModel]) -> dict[str, Any]:
    wire_schema = FinalAnswerWire if schema is AgentDraft else schema
    result = copy.deepcopy(wire_schema.model_json_schema())

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
    return result


def response_format(
    schema: type[BaseModel], model: str, *, catalog: dict[str, Citation] | None = None
) -> dict[str, Any]:
    native = model in {
        "groq/openai/gpt-oss-120b",
        "groq/openai/gpt-oss-20b",
        "groq/qwen/qwen3.8-27b",
        "gemini/gemini-3.8-flash",
        "gemini/gemini-3.7-flash",
        "gemini/gemini-3.1-flash-lite",
    }
    if native:
        shape = strict_schema(FinalAnswerReferences if catalog is not None else schema)
        if catalog is not None:
            if catalog:
                shape["properties"]["citations"]["items"]["enum"] = list(catalog)
            else:
                shape["properties"]["citations"]["maxItems"] = 0
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": shape,
            },
        }
    return {"type": "json_object"}


class ObjectSource(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    object_id: int = Field(gt=0)


class PageSource(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_url: str


class CitationWire(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source: ObjectSource | PageSource
    quote: str = Field(min_length=1, max_length=2000)


class FinalAnswerWire(BaseModel):
    """Disjoint source keys keep cross-provider citation decoding unambiguous."""

    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=8000)
    citations: list[CitationWire] = Field(max_length=20)
    language: Language


class FinalAnswerReferences(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=8000)
    citations: list[str] = Field(max_length=20)
    language: Language


def decode_final(content: str, catalog: dict[str, Citation] | None = None) -> str:
    if catalog is not None:
        references = FinalAnswerReferences.model_validate_json(content)
        if any(key not in catalog for key in references.citations):
            raise ValueError("Unknown citation excerpt key")
        return AgentDraft(
            text=references.text,
            language=references.language,
            citations=[catalog[key] for key in references.citations],
        ).model_dump_json()
    wire = FinalAnswerWire.model_validate_json(content)
    answer = {
        "text": wire.text,
        "language": wire.language,
        "citations": [{**item.source.model_dump(), "quote": item.quote} for item in wire.citations],
    }
    return json.dumps(answer, ensure_ascii=False)
