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
    r"what(?:'s|\s+is)?\s+inside|"
    r"what\s+is\s+there|"
    r"what(?:'s|\s+is)\s+(?:displayed|on\s+display)|"
    r"what(?:'s|\s+is)(?:\s+on\s+view)?|"
    r"(?:list|show)\s+(?:me\s+)?(?:the\s+)?(?:objects?|artworks?)|"
    r"show\s+me(?:\s+what(?:'s|\s+is))?|"
    r"which\s+(?:objects?|artworks?)\s+(?:are\s+)?(?:on\s+view)?"
    r")\s+(?:in\s+)?(?:the\s+)?gallery\s*#?\s*(\d{1,4})\s*[?!.]*\s*$",
    re.IGNORECASE,
)

_DIRECT_GALLERY_WAYFINDING = re.compile(
    r"^\s*(?:"
    r"(?:please\s+)?(?:tell\s+me\s+)?how\s+(?:do|can|should)\s+i\s+"
    r"(?:get|go|walk)\s+to\s+(?:the\s+)?gallery\s*#?\s*(?P<to>\d{1,4})\s+"
    r"from\s+(?:the\s+)?(?:fifth\s+avenue\s+)?(?:main\s+)?entrance|"
    r"(?:please\s+)?(?:give\s+me\s+)?directions\s+to\s+(?:the\s+)?"
    r"gallery\s*#?\s*(?P<directions>\d{1,4})\s+from\s+(?:the\s+)?"
    r"(?:fifth\s+avenue\s+)?(?:main\s+)?entrance|"
    r"(?:please\s+)?(?:tell\s+me\s+)?how\s+(?:do|can|should)\s+i\s+"
    r"(?:get|go|walk)\s+from\s+(?:the\s+)?(?:fifth\s+avenue\s+)?"
    r"(?:main\s+)?entrance\s+to\s+(?:the\s+)?gallery\s*#?\s*(?P<from>\d{1,4})"
    r")(?:\s*,?\s*please)?\s*[?!.]*\s*$",
    re.IGNORECASE,
)


def direct_gallery_number(message: str) -> str | None:
    """Return a bounded gallery number only for a complete direct-gallery question."""
    match = _DIRECT_GALLERY_QUESTION.fullmatch(message)
    return match.group(1) if match is not None else None


def direct_gallery_wayfinding_number(message: str) -> str | None:
    """Return a gallery only for a complete Fifth Avenue entrance direction request."""
    match = _DIRECT_GALLERY_WAYFINDING.fullmatch(message)
    if match is None:
        return None
    return match.group("to") or match.group("directions") or match.group("from")


def normalize_intent(intent: Intent, message: str) -> tuple[Intent, str | None]:
    """Stabilize a narrow direct-gallery route while preserving semantic safety classes."""
    wayfinding_gallery = direct_gallery_wayfinding_number(message)
    if wayfinding_gallery is not None and intent.category not in {"interpretive", "out_of_scope"}:
        return (
            intent.model_copy(
                update={
                    "category": "visitor_info",
                    "difficulty": "simple",
                    "language": "en",
                    "search_query": (
                        f"Directions from the Fifth Avenue entrance to Gallery {wayfinding_gallery}"
                    ),
                }
            ),
            "direct_gallery_wayfinding",
        )
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
