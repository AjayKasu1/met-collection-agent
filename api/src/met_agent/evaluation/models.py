"""Validate evaluation inputs, judgments and durable run results."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from met_agent.agent.models import AgentAnswer, ModelCall


class EvalRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    category: str
    question: str = Field(min_length=1)
    expect: Literal["contains", "retrieval", "refuse_interpretive", "handoff"]
    expected: list[str]
    expected_object_ids: list[int]
    verify: bool
    notes: str

    @model_validator(mode="after")
    def labels(self) -> "EvalRow":
        if self.expect == "contains" and not self.expected:
            raise ValueError("contains requires expected strings")
        if self.expect == "retrieval" and not self.expected_object_ids:
            raise ValueError("retrieval requires independently established object IDs")
        return self


class Judgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    faithfulness: float = Field(ge=0, le=1, allow_inf_nan=False)
    no_opinion: bool
    reason: str = Field(min_length=1, max_length=1500)


class RowResult(BaseModel):
    row: EvalRow
    session_id: str
    passed: bool = False
    reason: str = ""
    answer: AgentAnswer | None = None
    judgment: Judgment | None = None
    judge_calls: list[ModelCall] = Field(default_factory=list)
    judge_ms: float = Field(default=0, ge=0)
    judge_cost_usd: float | None = None
    latency_ms: float = Field(default=0, ge=0)
    retrieval_ids: list[int] = Field(default_factory=list)
    retrieval_ms: float = Field(default=0, ge=0)
    hits: dict[str, bool] = Field(default_factory=dict)
    error: str | None = None


class Report(BaseModel):
    schema_version: int = 1
    run_id: str
    git_sha: str
    started_at: str
    suite: Literal["quick", "full"]
    golden_sha256: str
    prompt_sha256: dict[str, str]
    models: dict[str, str]
    expected_ids: list[str]
    retrieval: dict[str, str] = Field(default_factory=dict)
    execution_notes: list[str] = Field(default_factory=list)
    results: list[RowResult] = Field(default_factory=list)
    complete: bool = False

    @model_validator(mode="after")
    def complete_membership(self) -> "Report":
        actual = [r.row.id for r in self.results]
        if len(set(self.expected_ids)) != len(self.expected_ids) or len(set(actual)) != len(actual):
            raise ValueError("Duplicate report IDs")
        if any(item not in self.expected_ids for item in actual):
            raise ValueError("Unexpected report row")
        if self.complete and (not self.expected_ids or actual != self.expected_ids):
            raise ValueError("Complete report requires every expected row in order")
        return self
