"""Represent lite-model routing and language decisions as a closed, validated taxonomy."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from met_agent.agent.models import Language, Route


class Intent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Literal["collection", "visitor_info", "multi_hop", "interpretive", "out_of_scope"]
    difficulty: Literal["simple", "complex"]
    language: Language
    search_query: str = Field(min_length=1, max_length=2000)

    @property
    def route(self) -> Route:
        return "lite" if self.category == "collection" and self.difficulty == "simple" else "main"
