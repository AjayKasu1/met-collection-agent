"""Represent and normalize routing decisions as a closed, validated taxonomy."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from met_agent.agent.models import Language, Route


class Intent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Literal["collection", "visitor_info", "multi_hop", "interpretive", "out_of_scope"]
    difficulty: Literal["simple", "complex"]
    language: Language
    search_query: str = Field(min_length=1, max_length=2000)

    constraints: list[str] = Field(default_factory=list, max_length=8)
    operation: Literal["general", "gallery_inventory", "gallery_wayfinding"] = "general"
    gallery_number: str | None = Field(default=None, pattern=r"^[0-9]{1,4}$")
    origin: Literal["unspecified", "fifth_avenue_entrance", "other"] = "unspecified"

    handoff_contact: Literal["info@metmuseum.org", "store.support@metmuseum.org"] = (
        "info@metmuseum.org"
    )

    @property
    def route(self) -> Route:
        return (
            "lite"
            if self.category in {"collection", "visitor_info"} and self.difficulty == "simple"
            else "main"
        )


def message_numbers(message: str) -> set[str]:
    """Extract numeric entities only; this does not classify what the user wants."""
    return set(re.findall(r"(?<!\w)[0-9]+(?:\.[0-9]+)?(?!\w)", message))


def normalize_intent(intent: Intent, message: str) -> tuple[Intent, str | None]:
    """Authorize bounded handlers from semantic intent and source-anchored entities."""
    if (
        intent.category in {"interpretive", "out_of_scope"}
        or intent.difficulty != "simple"
        or intent.language != "en"
        or intent.constraints
        or intent.gallery_number is None
        or message_numbers(message) != {intent.gallery_number}
        or not re.search(
            rf"\b(?:gallery|room)\s*#?\s*{re.escape(intent.gallery_number)}\b",
            message,
            re.IGNORECASE,
        )
    ):
        return intent, None
    if intent.operation == "gallery_inventory" and intent.category == "collection":
        return intent.model_copy(
            update={"search_query": f"Objects in Gallery {intent.gallery_number}"}
        ), "direct_gallery_question"
    if (
        intent.operation == "gallery_wayfinding"
        and intent.category == "visitor_info"
        and intent.origin == "fifth_avenue_entrance"
    ):
        return intent.model_copy(
            update={
                "search_query": (
                    f"Directions from the Fifth Avenue entrance to Gallery {intent.gallery_number}"
                )
            }
        ), "direct_gallery_wayfinding"
    return intent, None
