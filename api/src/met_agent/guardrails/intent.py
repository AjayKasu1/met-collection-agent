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

    handoff_contact: Literal["info@metmuseum.org", "store.support@metmuseum.org"] = (
        "info@metmuseum.org"
    )

    @property
    def route(self) -> Route:
        return "lite" if self.category == "collection" and self.difficulty == "simple" else "main"


_DIRECT_GALLERY_QUESTION = re.compile(
    r"^\s*(?:"
    r"what\s+can\s+i\s+see|"
    r"what(?:'s|\s+is)(?:\s+on\s+view)?|"
    r"show\s+me(?:\s+what(?:'s|\s+is))?|"
    r"which\s+(?:objects?|artworks?)\s+(?:are\s+)?(?:on\s+view)?"
    r")\s+(?:in\s+)?(?:the\s+)?gallery\s*#?\s*(\d{1,4})\s*[?!.]*\s*$",
    re.IGNORECASE,
)


def direct_gallery_number(message: str) -> str | None:
    """Return a bounded gallery number only for a complete direct-gallery question."""
    match = _DIRECT_GALLERY_QUESTION.fullmatch(message)
    return match.group(1) if match is not None else None


def normalize_intent(intent: Intent, message: str) -> tuple[Intent, str | None]:
    """Stabilize a narrow direct-gallery route while preserving semantic safety classes."""
    gallery_number = direct_gallery_number(message)
    if gallery_number is None or intent.category in {"interpretive", "out_of_scope"}:
        return intent, None
    normalized = intent.model_copy(
        update={
            "category": "collection",
            "difficulty": "simple",
            "language": "en",
            "search_query": f"Objects in Gallery {gallery_number}",
        }
    )
    if normalized == intent:
        return intent, None
    return normalized, "direct_gallery_question"
