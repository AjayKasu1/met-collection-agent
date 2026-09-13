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
_IDENTITY_LANGUAGES: dict[str, Language] = {
    "what model are you": "en",
    "which model are you": "en",
    "what model is this": "en",
    "what model do you use": "en",
    "what is your model": "en",
    "what llm are you": "en",
    "who are you": "en",
    "what are you": "en",
    "are you an ai": "en",
    "are you a bot": "en",
    "are you chatgpt": "en",
    "what is your name": "en",
    "who made you": "en",
    "who created you": "en",
    "quel modele etes vous": "fr",
    "qui etes vous": "fr",
    "etes vous une ia": "fr",
    "quel est votre nom": "fr",
    "qui vous a cree": "fr",
    "que modelo eres": "es",
    "quien eres": "es",
    "eres una ia": "es",
    "como te llamas": "es",
    "quien te creo": "es",
    "你是哪个模型": "zh",
    "你是什么模型": "zh",
    "你是谁": "zh",
    "你是ai吗": "zh",
    "你的名字是什么": "zh",
    "谁创造了你": "zh",
}
_RECALL_FIRST_PHRASES: dict[str, Language] = {
    "what did i ask first": "en",
    "what did i ask you first": "en",
    "what was my first question": "en",
    "what was the first thing i asked": "en",
    "what i asked first": "en",
    "first question": "en",
    "quai je demande en premier": "fr",
    "quelle etait ma premiere question": "fr",
    "premiere question": "fr",
    "que pregunte primero": "es",
    "cual fue mi primera pregunta": "es",
    "primera pregunta": "es",
    "我最先问了什么": "zh",
    "我的第一个问题是什么": "zh",
    "第一个问题": "zh",
}
_RECALL_PREVIOUS_PHRASES: dict[str, Language] = {
    "what question i asked recent": "en",
    "what question did i ask recent": "en",
    "what question did i ask recently": "en",
    "what question did i ask": "en",
    "what did i ask recently": "en",
    "what did i ask recent": "en",
    "what did i just ask": "en",
    "what was my last question": "en",
    "what was my previous question": "en",
    "what was the last thing i asked": "en",
    "what did i ask last": "en",
    "what did i ask before": "en",
    "what did i ask before this": "en",
    "what was my prior question": "en",
    "repeat my last question": "en",
    "repeat what i just asked": "en",
    "previous question": "en",
    "last question": "en",
    "what did i ask": "en",
    "quelle etait ma question precedente": "fr",
    "que viens je de demander": "fr",
    "quai je demande recemment": "fr",
    "ma derniere question": "fr",
    "question precedente": "fr",
    "cual fue mi pregunta anterior": "es",
    "que acabo de preguntar": "es",
    "que pregunte recientemente": "es",
    "mi ultima pregunta": "es",
    "pregunta anterior": "es",
    "我刚才问了什么": "zh",
    "我上一个问题是什么": "zh",
    "我最近问了什么": "zh",
    "上一个问题": "zh",
    "上个问题": "zh",
}
_FIRST_MESSAGE_RECALL = set(_RECALL_FIRST_PHRASES.keys())

RecallTarget = Literal["previous", "first", "topic"]
SocialIntent = Literal[
    "greeting",
    "wellbeing",
    "laughter",
    "acknowledgement",
    "thanks",
    "farewell",
    "dissatisfaction",
    "assistant_identity",
    "conversation_recall",
]


def _normalized_message(message: str) -> str:
    folded = unicodedata.normalize("NFKC", message).casefold()
    folded = re.sub(r"['\u2019]", "", folded)
    return " ".join(re.sub(r"[^\w\s]", " ", folded).split())


def greeting_language(message: str) -> Language | None:
    """Recognize a complete greeting without swallowing a museum question."""
    return _GREETING_LANGUAGES.get(_normalized_message(message))


def identity_intent(message: str) -> tuple[Literal["assistant_identity"], Language] | None:
    """Recognize questions about the assistant model or identity without swallowing art facts."""
    if re.search(
        (
            r"\b(?:gallery|room|painting|sculpture|armor|statue|artist|"
            r"exhibition|artifact|collection)\b"
        ),
        message,
        re.IGNORECASE,
    ):
        return None
    normalized = _normalized_message(message)
    if language := _IDENTITY_LANGUAGES.get(normalized):
        return "assistant_identity", language
    if re.search(
        r"\b(?:what|which)\s+(?:model|llm)\s+(?:are\s+you|is\s+this)\b",
        message,
        re.IGNORECASE,
    ):
        return "assistant_identity", "en"
    if re.search(r"\bwho\s+are\s+you\b", message, re.IGNORECASE):
        return "assistant_identity", "en"
    return None


def recall_intent(
    message: str,
) -> tuple[Literal["conversation_recall"], RecallTarget, Language, str | None] | None:
    """Recognize exact or pattern-based requests to recall earlier conversation questions."""
    normalized = _normalized_message(message)

    # 1. First message check
    if language := _RECALL_FIRST_PHRASES.get(normalized):
        return "conversation_recall", "first", language, None

    # 2. Topic-specific recall check (e.g. "What did I ask about the sphinx?")
    topic_match = re.search(
        r"\bwhat\s+(?:did\s+)?i\s+ask\s+(?:about|regarding)\s+(?:the\s+)?(?P<topic>[a-zA-Z0-9\s]+?)\??$",
        message.strip(),
        re.IGNORECASE,
    )
    if topic_match:
        topic = topic_match.group("topic").strip()
        if topic:
            return "conversation_recall", "topic", "en", topic

    fr_topic = re.search(
        r"\bqu['\s]*ai[\s-]*je\s+demande\s+(?:sur|a\s+propos\s+de)\s+(?:le\s+|la\s+|l['\s]*)?(?P<topic>[^?.,!]+)\??$",
        message.strip(),
        re.IGNORECASE,
    )
    if fr_topic:
        topic = fr_topic.group("topic").strip()
        if topic:
            return "conversation_recall", "topic", "fr", topic

    es_topic = re.search(
        r"\bque\s+pregunte\s+(?:sobre|acerca\s+de)\s+(?:el\s+|la\s+|los\s+|las\s+)?(?P<topic>[^?.,!]+)\??$",
        message.strip(),
        re.IGNORECASE,
    )
    if es_topic:
        topic = es_topic.group("topic").strip()
        if topic:
            return "conversation_recall", "topic", "es", topic

    zh_topic = re.search(
        r"\b我(?:刚才|之前)?问了关于(?P<topic>[^?.,!]+)的什么\b",
        message.strip(),
    )
    if zh_topic:
        topic = zh_topic.group("topic").strip()
        if topic:
            return "conversation_recall", "topic", "zh", topic

    # 3. Previous / last / recent message check
    if language := _RECALL_PREVIOUS_PHRASES.get(normalized):
        return "conversation_recall", "previous", language, None

    if re.search(
        (
            r"\bwhat\s+(?:question\s+)?(?:did\s+)?i\s+ask(?:ed)?\s+"
            r"(?:recent(?:ly)?|last|before|just\s+now)\b"
        ),
        message,
        re.IGNORECASE,
    ):
        return "conversation_recall", "previous", "en", None
    if re.search(
        r"\bwhat\s+(?:was|is)\s+my\s+(?:last|previous|prior|recent)\s+question\b",
        message,
        re.IGNORECASE,
    ):
        return "conversation_recall", "previous", "en", None
    if re.search(r"\bwhat\s+did\s+i\s+just\s+ask\b", message, re.IGNORECASE):
        return "conversation_recall", "previous", "en", None
    if re.search(
        r"\brepeat\s+(?:my\s+last\s+question|what\s+i\s+just\s+asked)\b",
        message,
        re.IGNORECASE,
    ):
        return "conversation_recall", "previous", "en", None

    return None


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


class ConversationalClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outcome: Literal[
        "assistant_identity",
        "conversation_recall",
        "capability_explanation",
        "museum_handoff",
        "collection_or_visitor",
    ]
    recall_target: Literal["previous", "first", "topic", "none"] = "none"
    recall_topic: str | None = None
    handoff_contact: Literal["info@metmuseum.org", "store.support@metmuseum.org"] = (
        "info@metmuseum.org"
    )
    handoff_reason: str | None = None


def classify_conversational_fallback(
    message: str,
    intent: "Intent",
) -> ConversationalClassification:
    """Route ambiguous or classifier-evaluated requests through explicit typed outcomes."""
    # 1. Check if user is asking to recall past conversation questions
    if recall := recall_intent(message):
        _, target, _, topic = recall
        return ConversationalClassification(
            outcome="conversation_recall",
            recall_target=target,
            recall_topic=topic,
        )

    # 2. Check if user is asking about assistant model or identity
    if identity_intent(message):
        return ConversationalClassification(outcome="assistant_identity")

    # 3. Handle out_of_scope classification from the model
    if intent.category == "out_of_scope":
        # General visitor queries (ticket price, hours, admission) belong to visitor_info, not staff
        is_visitor_info = bool(
            re.search(
                (
                    r"\b(?:ticket|tickets|admission|hours|opening|fee|cost|how\s+much|"
                    r"directions|parking|cafe|dining|bag\s+check|coat\s+check)\b"
                ),
                message,
                re.IGNORECASE,
            )
        )
        is_account_or_refund = bool(
            re.search(
                r"\b(?:refund|cancel|receipt|membership\s+account|login|password)\b",
                message,
                re.IGNORECASE,
            )
        )
        if is_visitor_info and not is_account_or_refund:
            return ConversationalClassification(outcome="collection_or_visitor")

        is_retail = intent.handoff_contact == "store.support@metmuseum.org" or bool(
            re.search(
                r"\b(?:order|shipping|delivery|merchandise|store|purchase)\b",
                message,
                re.IGNORECASE,
            )
        )
        is_museum_ops = bool(
            re.search(
                (
                    r"\b(?:refund|cancel\s+my|membership\s+account|login|password|"
                    r"donat|venue\s+rental|facility\s+rental|lost\s+and\s+found|appraisal)\b"
                ),
                message,
                re.IGNORECASE,
            )
        )
        if is_retail:
            return ConversationalClassification(
                outcome="museum_handoff",
                handoff_contact="store.support@metmuseum.org",
                handoff_reason="This request concerns museum store purchases, shipping, or orders.",
            )
        if is_museum_ops:
            return ConversationalClassification(
                outcome="museum_handoff",
                handoff_contact="info@metmuseum.org",
                handoff_reason="This request concerns account, ticketing refund, or operations.",
            )
        # General conversational or external query: explain capability without staff handoff
        return ConversationalClassification(outcome="capability_explanation")

    return ConversationalClassification(outcome="collection_or_visitor")


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
