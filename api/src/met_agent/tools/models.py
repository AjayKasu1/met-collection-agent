"""Define the bounded inputs, factual outputs, and citation evidence shared by chat and MCP."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from met_agent.retrieval.hybrid import RetrievedChunk, RetrievedObject, SearchFilters

ObjectId = Annotated[int, Field(gt=0)]
Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
Contact = Literal["info@metmuseum.org", "store.support@metmuseum.org"]
ToolName = Literal[
    "search_collection", "get_object", "search_visitor_info", "handoff", "find_similar_objects"
]


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SearchCollectionArguments(Arguments):
    query: Query
    filters: SearchFilters | None = None
    k: int = Field(default=8, ge=1, le=40)


class GetObjectArguments(Arguments):
    object_id: ObjectId


class SearchVisitorArguments(Arguments):
    query: Query
    k: int = Field(default=5, ge=1, le=20)


class HandoffArguments(Arguments):
    reason: str = Field(min_length=1, max_length=1000)
    suggested_contact: Contact = "info@metmuseum.org"


class Handoff(BaseModel):
    reason: str
    suggested_contact: Contact


class CollectionSearchResult(BaseModel):
    objects: list[RetrievedObject]
    freshness: str = (
        "Gallery values and on_view filters describe ingestion-time records. "
        "Use get_object for current display information."
    )


class VisitorSearchResult(BaseModel):
    chunks: list[RetrievedChunk]


class LiveObject(BaseModel):
    object_id: ObjectId
    title: str
    artist: str
    culture: str
    medium: str
    object_date: str
    department: str
    gallery_number: str
    is_on_view: bool
    is_public_domain: bool
    primary_image: str
    primary_image_small: str
    source_url: str
    fetched_at: str
    text: str


class SimilarObject(BaseModel):
    object_id: ObjectId
    title: str
    source_url: str
    image_score: float = Field(allow_inf_nan=False)
    text: str


class SimilarObjectsResult(BaseModel):
    source_object_id: ObjectId
    status: Literal["ok", "object_not_indexed", "image_unavailable"]
    objects: list[SimilarObject] = Field(default_factory=list)
    vector: Literal["image"] = "image"


class Evidence(BaseModel):
    """Only the server derives citation candidates from this turn's tool results."""

    key: str
    object_id: int | None = None
    source_url: str | None = None
    text: str
    kind: Literal["collection", "visitor_info", "live_object", "image_similarity"]


class ToolError(BaseModel):
    code: Literal["invalid_arguments", "unknown_tool", "upstream_unavailable", "not_found"]
    message: str


ToolPayload = (
    CollectionSearchResult | VisitorSearchResult | LiveObject | SimilarObjectsResult | Handoff
)


class ToolResult(BaseModel):
    name: str
    output: ToolPayload | None = None
    error: ToolError | None = None
    evidence: list[Evidence] = Field(default_factory=list)

    def model_evidence(self) -> list[Evidence]:
        """Bound model-visible evidence without changing full API/MCP results."""
        return [
            item.model_copy(
                update={
                    "text": "\n".join(
                        line
                        for line in item.text.splitlines()
                        if not line.startswith(("primary_image:", "primary_image_small:"))
                    )
                }
            )
            for item in self.evidence[:8]
        ]

    def model_context(self) -> str:
        """Send citation evidence once; retain full records only in tool APIs and audit logs."""
        import json

        if self.error or not self.evidence:
            return self.model_dump_json(exclude={"evidence"})
        payload = {
            "name": self.name,
            "evidence": [
                item.model_dump(mode="json", exclude_none=True) for item in self.model_evidence()
            ],
        }
        if isinstance(self.output, CollectionSearchResult):
            payload["freshness"] = self.output.freshness
        return json.dumps(payload, ensure_ascii=False)
