"""Validate the same five tool contracts for chat and MCP before running sequential handlers."""

import asyncio
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Protocol, cast

from pydantic import BaseModel, ValidationError

from met_agent.config import Settings
from met_agent.retrieval.service import SearchService
from met_agent.tools.find_similar_objects import find_similar_objects
from met_agent.tools.get_object import LiveObjectClient, ObjectNotFound
from met_agent.tools.handoff import handoff
from met_agent.tools.models import (
    CollectionSearchResult,
    Evidence,
    GetObjectArguments,
    Handoff,
    HandoffArguments,
    LiveObject,
    SearchCollectionArguments,
    SearchVisitorArguments,
    SimilarObjectsResult,
    ToolError,
    ToolPayload,
    ToolResult,
    VisitorSearchResult,
)
from met_agent.tools.schemas import FindSimilarObjectsArguments
from met_agent.tools.search_collection import search_collection
from met_agent.tools.search_visitor_info import search_visitor_info


class ToolTelemetry(Protocol):
    def tool(self, name: str, arguments: str) -> AbstractContextManager[None]: ...


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    arguments: type[BaseModel]
    output: type[BaseModel]
    handler: Callable[[BaseModel], BaseModel]

    def schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.arguments.model_json_schema(),
            },
        }


class ToolRegistry:
    """Count errors at the agent boundary; this registry never retries or expands a tool call."""

    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}
        self.telemetry: ToolTelemetry | None = None

    def register[A: BaseModel, R: BaseModel](
        self,
        name: str,
        description: str,
        arguments: type[A],
        output: type[R],
        handler: Callable[[A], R],
    ) -> None:
        if name in self.tools:
            raise ValueError("Tool names must be unique")

        def invoke(value: BaseModel) -> BaseModel:
            return handler(cast(A, value))

        self.tools[name] = Tool(name, description, arguments, output, invoke)

    @property
    def schemas(self) -> list[dict[str, object]]:
        return [tool.schema() for tool in self.tools.values()]

    async def execute(self, name: str, arguments: str) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(
                name=name,
                error=ToolError(code="unknown_tool", message="Only registered tools may execute"),
            )
        try:
            if len(arguments) > 10000:
                raise ValueError("Tool arguments exceed the input limit")
            parsed = tool.arguments.model_validate_json(arguments)
        except (ValidationError, ValueError):
            return ToolResult(
                name=name,
                error=ToolError(
                    code="invalid_arguments", message="Arguments do not match the registered schema"
                ),
            )
        try:
            with self.telemetry.tool(name, arguments) if self.telemetry else nullcontext():
                raw = await asyncio.to_thread(tool.handler, parsed)
            validated = tool.output.model_validate(raw)
            output = cast(ToolPayload, validated)
            return ToolResult(name=name, output=output, evidence=evidence_from(output))
        except ObjectNotFound as error:
            return ToolResult(
                name=name,
                error=ToolError(
                    code="not_found", message="The Met API returned no object record for this ID"
                ),
                evidence=[error.evidence] if error.evidence else [],
            )
        except Exception as error:
            return ToolResult(
                name=name,
                error=ToolError(
                    code="upstream_unavailable",
                    message=f"Tool operation failed ({type(error).__name__})",
                ),
            )


def evidence_from(output: ToolPayload) -> list[Evidence]:
    if isinstance(output, CollectionSearchResult):
        return [
            Evidence(
                key=f"object:{obj.object_id}",
                object_id=obj.object_id,
                source_url=obj.source_url,
                text=obj.text,
                kind="collection",
            )
            for obj in output.objects
        ]
    if isinstance(output, LiveObject):
        return [
            Evidence(
                key=f"object:{output.object_id}",
                object_id=output.object_id,
                source_url=output.source_url,
                text=output.text,
                kind="live_object",
            )
        ]
    if isinstance(output, VisitorSearchResult):
        return [
            Evidence(
                key=f"page:{chunk.point_id}",
                source_url=chunk.source_url,
                text=chunk.text,
                kind="visitor_info",
            )
            for chunk in output.chunks
        ]
    if isinstance(output, SimilarObjectsResult):
        return [
            Evidence(
                key=f"object:{obj.object_id}",
                object_id=obj.object_id,
                source_url=obj.source_url,
                text=obj.text + f"\nImage cosine similarity: {obj.image_score}",
                kind="image_similarity",
            )
            for obj in output.objects
        ]
    return []


def create_registry(
    settings: Settings, search: SearchService, live: LiveObjectClient
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        "search_collection",
        (
            "Search factual collection records with hybrid retrieval. Use an "
            "English query rewrite for the English reranker. Gallery filters "
            "reflect ingestion time, not current display status."
        ),
        SearchCollectionArguments,
        CollectionSearchResult,
        partial(search_collection, search, settings.qdrant_collection),
    )
    registry.register(
        "get_object",
        (
            "Get authoritative Met object details, image links, current "
            "GalleryNumber and display status. Cached for at most five minutes."
        ),
        GetObjectArguments,
        LiveObject,
        live.get,
    )
    registry.register(
        "search_visitor_info",
        (
            "Search captured visitor information with URLs and capture timestamps. "
            "Use an English query rewrite. Do not infer floor-plan details absent "
            "from the returned passages."
        ),
        SearchVisitorArguments,
        VisitorSearchResult,
        partial(search_visitor_info, search, settings.qdrant_visitor_collection),
    )
    registry.register(
        "handoff",
        (
            "Terminal tool for out-of-scope, account, or purchase questions. "
            "Suggest a contact without sending anything."
        ),
        HandoffArguments,
        Handoff,
        handoff,
    )
    registry.register(
        "find_similar_objects",
        (
            "Find nearby objects in the named image vector using only published Met"
            " image embeddings. Exclude the source object. Missing images return "
            "image_unavailable without a text fallback."
        ),
        FindSimilarObjectsArguments,
        SimilarObjectsResult,
        partial(find_similar_objects, search, settings.qdrant_collection),
    )
    return registry
