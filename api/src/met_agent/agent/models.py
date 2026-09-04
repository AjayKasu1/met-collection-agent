"""Validate public chat requests, grounded answers, and model-generated drafts."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from met_agent.config import ChatProvider
from met_agent.tools.models import Handoff

Language = Literal["en", "fr", "es", "zh"]
Route = Literal["lite", "main"]


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    object_id: int | None = Field(default=None, gt=0)
    source_url: str | None = None
    quote: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def one_source(self) -> "Citation":
        if (self.object_id is None) == (self.source_url is None):
            raise ValueError("Citations require exactly one object ID or source URL")
        if not self.quote.strip():
            raise ValueError("Citation quotes cannot be blank")
        return self


class AgentDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=8000)
    citations: list[Citation] = Field(default_factory=list, max_length=20)
    language: Language

    @field_validator("text")
    @classmethod
    def ordinary_display_spaces(cls, value: str) -> str:
        """Normalize presentation-only spaces in prose; never alter verbatim citation quotes."""
        return value.replace("\u00a0", " ").replace("\u202f", " ")


class ModelCall(BaseModel):
    model: str
    route: str
    provider: ChatProvider = "gemini"
    path: Literal["ai_gateway", "direct_google", "direct_groq", "direct_cerebras"]
    pacing_ms: float = Field(default=0, ge=0)
    provider_ms: float = Field(default=0, ge=0)
    retry_ms: float = Field(default=0, ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    latency_ms: float = Field(ge=0, allow_inf_nan=False)


class AgentAnswer(AgentDraft):
    session_id: UUID
    turn_id: UUID
    handoff: Handoff | None = None
    route: Route
    cost_usd: float | None = Field(ge=0, allow_inf_nan=False)
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    grounding_score: float = Field(ge=0, le=1)
    policy_refusal: bool = False
    model_calls: list[ModelCall] = Field(default_factory=list)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=4000)
    session_id: UUID | None = None
    language: Language | None = None

    @model_validator(mode="after")
    def nonempty_message(self) -> "ChatRequest":
        if not self.message.strip():
            raise ValueError("Message must not be blank")
        return self
