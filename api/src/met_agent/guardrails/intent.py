"""Represent and normalize routing decisions as a closed, validated taxonomy."""

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from met_agent.agent.models import Language, Route

_GREETING_LANGUAGES: dict[str, Language] = {
    "hi": "en",
    "hi there": "en",
    "hello": "en",
    "hello there": "en",
    "hey": "en",
    "good morning": "en",
    "good afternoon": "en",
    "good evening": "en",
    "bonjour": "fr",
    "bonsoir": "fr",
    "salut": "fr",
    "hola": "es",
    "buenos días": "es",
    "buenas tardes": "es",
    "buenas noches": "es",
    "你好": "zh",
    "您好": "zh",
    "早上好": "zh",
    "晚上好": "zh",
}
_WELLBEING_LANGUAGES: dict[str, Language] = {
    "how are you": "en",
    "how are you doing": "en",
    "hows it going": "en",
    "comment allez vous": "fr",
    "ça va": "fr",
    "cómo estás": "es",
    "cómo está": "es",
    "你好吗": "zh",
}
_THANKS_LANGUAGES: dict[str, Language] = {
    "thanks": "en",
    "thank you": "en",
    "merci": "fr",
    "gracias": "es",
    "谢谢": "zh",
}
_FAREWELL_LANGUAGES: dict[str, Language] = {
    "bye": "en",
    "goodbye": "en",
    "see you": "en",
    "au revoir": "fr",
    "adiós": "es",
    "再见": "zh",
}
_LAUGHTER_LANGUAGES: dict[str, Language] = {
    "lol": "en",
    "haha": "en",
    "hahaha": "en",
    "lmao": "en",
    "rofl": "en",
    "😂": "en",
    "😆": "en",
    "😹": "en",
    "mdr": "fr",
    "ptdr": "fr",
    "jaja": "es",
    "jajaja": "es",
    "哈哈": "zh",
    "哈哈哈": "zh",
}
_ACKNOWLEDGEMENT_LANGUAGES: dict[str, Language] = {
    "cool": "en",
    "nice": "en",
    "great": "en",
    "awesome": "en",
    "sounds good": "en",
    "ok": "en",
    "okay": "en",
    "got it": "en",
    "perfect": "en",
    "super": "fr",
    "chouette": "fr",
    "daccord": "fr",
    "parfait": "fr",
    "genial": "es",
    "guay": "es",
    "vale": "es",
    "perfecto": "es",
    "真棒": "zh",
    "太棒了": "zh",
    "好的": "zh",
    "行": "zh",
    "明白": "zh",
}
_DISSATISFACTION_LANGUAGES: dict[str, Language] = {
    "that didnt help": "en",
    "that did not help": "en",
    "this didnt help": "en",
    "this did not help": "en",
    "that didnt answer my question": "en",
    "that did not answer my question": "en",
    "thanks for nothing": "en",
    "you didnt answer my question": "en",
    "that wasnt helpful": "en",
    "that was not helpful": "en",
    "great": "en",  # Ambiguous when following a failure/refusal
    "not helpful": "en",
    "didnt help": "en",
    "ca na pas aide": "fr",
    "cela na pas aide": "fr",
    "vous navez pas repondu": "fr",
    "pas utile": "fr",
    "eso no ayudo": "es",
    "no me sirvio": "es",
    "no respondiste a mi pregunta": "es",
    "no fue util": "es",
    "这没有帮助": "zh",
    "你没有回答我的问题": "zh",
    "没用": "zh",
    "没有帮助": "zh",
}
_FIRST_MESSAGE_RECALL = {
    "what did i ask first",
    "what did i ask you first",
    "what was my first question",
    "what was the first thing i asked",
    "what i asked first",
}
SocialIntent = Literal[
    "greeting",
    "wellbeing",
    "laughter",
    "acknowledgement",
    "thanks",
    "farewell",
    "dissatisfaction",
]


def _normalized_message(message: str) -> str:
    folded = unicodedata.normalize("NFKC", message).casefold()
    folded = re.sub(r"['\u2019]", "", folded)
    return " ".join(re.sub(r"[^\w\s]", " ", folded).split())


def greeting_language(message: str) -> Language | None:
    """Recognize a complete greeting without swallowing a museum question."""
    return _GREETING_LANGUAGES.get(_normalized_message(message))


def social_intent(
    message: str,
    *,
    last_answer_kind: str | None = None,
    last_failure_reason: str | None = None,
) -> tuple[SocialIntent, Language] | None:
    """Return a bounded social turn only when the whole message matches exact canonical phrases."""
    normalized = _normalized_message(message)
    raw_stripped = message.strip()

    # Exact match for laughter emojis if not folded away
    if raw_stripped in {"😂", "😆", "😹"}:
        return "laughter", "en"

    # Wellbeing allows question marks (e.g., "how are you?")
    if language := _WELLBEING_LANGUAGES.get(normalized):
        return "wellbeing", language

    # For other categories, question marks or conjunctions indicate substantive follow-ups
    if (
        "?" in message
        or " but " in message.lower()
        or " mais " in message.lower()
        or " pero " in message.lower()
    ):
        return None

    # Handle sarcastic/dissatisfied "great..." after failure
    is_failure = (
        last_answer_kind in {"unavailable", "policy_refusal"} or last_failure_reason is not None
    )
    if normalized == "great" and is_failure:
        return "dissatisfaction", "en"

    groups: tuple[tuple[SocialIntent, dict[str, Language]], ...] = (
        ("greeting", _GREETING_LANGUAGES),
        ("laughter", _LAUGHTER_LANGUAGES),
        ("acknowledgement", _ACKNOWLEDGEMENT_LANGUAGES),
        ("thanks", _THANKS_LANGUAGES),
        ("farewell", _FAREWELL_LANGUAGES),
        ("dissatisfaction", _DISSATISFACTION_LANGUAGES),
    )
    for kind, phrases in groups:
        if kind == "dissatisfaction" and normalized == "great":
            continue  # Handled above only conditionally
        if language := phrases.get(normalized):
            return kind, language
    return None


def asks_for_first_message(message: str) -> bool:
    """Recognize an exact request to recall the current session's first message."""
    return _normalized_message(message) in _FIRST_MESSAGE_RECALL


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
    factual_request: bool = True
    referenced_entity: str | None = None
    requires_clarification: bool = False

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
    if intent.operation == "gallery_wayfinding" and intent.category == "visitor_info":
        if intent.origin == "unspecified":
            return intent.model_copy(
                update={
                    "search_query": (
                        "Directions from the Fifth Avenue entrance to Gallery "
                        f"{intent.gallery_number}"
                    )
                }
            ), "direct_gallery_wayfinding_default_origin"
        if intent.origin != "fifth_avenue_entrance":
            return intent, None
        return intent.model_copy(
            update={
                "search_query": (
                    f"Directions from the Fifth Avenue entrance to Gallery {intent.gallery_number}"
                )
            }
        ), "direct_gallery_wayfinding"
    return intent, None
