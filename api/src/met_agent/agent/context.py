"""Compact, bounded session context and referent resolution for conversation turns."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from met_agent.agent.models import AnswerKind, Language

ClarificationType = Literal[
    "which_object",
    "missing_referent",
    "visit_details",
    "wayfinding_origin",
    "dissatisfaction",
]


class VerifiedObjectRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object_id: int = Field(gt=0)
    title: str = Field(min_length=1, max_length=200)


class PendingClarification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clarification_type: ClarificationType
    candidates: list[VerifiedObjectRef] = Field(default_factory=list, max_length=3)
    original_question: str = Field(max_length=500)
    failure_reason: str | None = Field(default=None, max_length=200)


class SessionContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    first_user_message: str | None = Field(default=None, max_length=500)
    current_topic: str | None = Field(default=None, max_length=200)
    verified_objects: list[VerifiedObjectRef] = Field(default_factory=list, max_length=3)
    relevant_galleries: list[str] = Field(default_factory=list, max_length=3)
    pending_clarification: PendingClarification | None = None
    last_answer_kind: AnswerKind | None = None
    last_turn_id: UUID | None = None
    language: Language = "en"
    last_social_response_id: str | None = Field(default=None, max_length=100)
    consecutive_social_turns: int = Field(default=0, ge=0)
    last_failure_reason: str | None = Field(default=None, max_length=200)

    @property
    def active_topic(self) -> str | None:
        return self.current_topic

    def record_turn(
        self,
        *,
        turn_id: UUID,
        user_message: str,
        answer_kind: AnswerKind,
        verified_object: VerifiedObjectRef | None = None,
        gallery_number: str | None = None,
        language: Language | None = None,
        social_response_id: str | None = None,
        failure_reason: str | None = None,
        pending_clarification: PendingClarification | None = None,
    ) -> "SessionContext":
        if not self.first_user_message:
            self.first_user_message = user_message[:500]
        if language:
            self.language = language
        self.last_turn_id = turn_id
        self.last_answer_kind = answer_kind

        if answer_kind == "social":
            self.consecutive_social_turns += 1
            if social_response_id:
                self.last_social_response_id = social_response_id
            return self

        # Non-social turns reset consecutive social counter
        self.consecutive_social_turns = 0
        self.last_social_response_id = None
        self.last_failure_reason = failure_reason

        if pending_clarification is not None:
            self.pending_clarification = pending_clarification
        elif answer_kind != "clarification":
            self.pending_clarification = None

        # Update verified objects only when verified from actual tools/records
        if verified_object is not None:
            filtered = [
                obj for obj in self.verified_objects if obj.object_id != verified_object.object_id
            ]
            self.verified_objects = [verified_object, *filtered][:3]
            self.current_topic = verified_object.title[:200]

        if gallery_number and gallery_number not in self.relevant_galleries:
            self.relevant_galleries = [gallery_number, *self.relevant_galleries][:3]

        return self

    def match_clarification_answer(self, message: str) -> VerifiedObjectRef | None:
        """Resolve ordinal, numeric, or title answers against candidate objects."""
        if (
            not self.pending_clarification
            or self.pending_clarification.clarification_type != "which_object"
        ):
            return None
        candidates = self.pending_clarification.candidates
        if not candidates:
            return None

        normalized = " ".join(message.strip().lower().split())
        first_indicators = {
            "1",
            "first",
            "the first",
            "the first one",
            "le premier",
            "premier",
            "el primero",
            "primero",
            "第一个",
            "第一件",
            "第一",
        }
        second_indicators = {
            "2",
            "second",
            "the second",
            "the second one",
            "le deuxieme",
            "deuxieme",
            "el segundo",
            "segundo",
            "第二个",
            "第二件",
            "第二",
        }

        if normalized in first_indicators and len(candidates) >= 1:
            return candidates[0]
        if normalized in second_indicators and len(candidates) >= 2:
            return candidates[1]

        # Check substring match against candidate titles
        for candidate in candidates:
            title_norm = candidate.title.lower()
            if title_norm in normalized or (len(normalized) >= 4 and normalized in title_norm):
                return candidate

        return None

    def render_clarification(self, language: Language | None = None) -> str:
        """Render approved multilingual templates for pending clarifications."""
        lang = language or self.language
        if not self.pending_clarification:
            return ""

        ctype = self.pending_clarification.clarification_type
        if ctype == "which_object":
            candidates = self.pending_clarification.candidates[:2]
            if lang == "fr":
                names = " ou ".join(f'"{c.title}"' for c in candidates)
                return f"De quelle oeuvre s'agit-il : {names} ?"
            if lang == "es":
                names = " o ".join(f'"{c.title}"' for c in candidates)
                return f"¿A que obra se refiere: {names}?"
            if lang == "zh":
                names = " or ".join(f'"{c.title}"' for c in candidates)
                return f"请问您指的是哪件作品: {names}?"
            names = " or ".join(f'"{c.title}"' for c in candidates)
            return f"Which work do you mean: {names}?"

        if ctype == "missing_referent":
            if lang == "fr":
                return "De quelle oeuvre ou de quelle salle parlez-vous ?"
            if lang == "es":
                return "¿Sobre que obra o sala esta preguntando?"
            if lang == "zh":
                return "请问您询问的是哪件艺术品或哪个展厅?"
            return "Which artwork or gallery are you asking about?"

        if ctype == "dissatisfaction":
            reason = self.pending_clarification.failure_reason
            if reason == "object_not_found":
                if lang == "fr":
                    return (
                        "Je n'ai trouve aucune notice pour cette oeuvre. Si vous avez "
                        "le nom de l'artiste ou un autre titre, je peux relancer la recherche."
                    )
                if lang == "es":
                    return (
                        "No pude encontrar un registro para esa obra. Si tiene "
                        "el nombre del artista u otro titulo, puedo buscar de nuevo."
                    )
                if lang == "zh":
                    return "未能找到该艺术品的记录. 如果您知道艺术家姓名或别名, 我可以再次为您检索."
                return (
                    "I couldn't find a record for that artwork. If you have "
                    "an artist's name or an alternate title, I can search again."
                )

            if reason == "wayfinding_missing_origin":
                if lang == "fr":
                    return (
                        "Je n'ai pas pu donner d'indications claires. De quelle entree "
                        "ou salle partez-vous ?"
                    )
                if lang == "es":
                    return (
                        "No pude darle indicaciones claras. ¿Desde que entrada "
                        "o sala esta comenzando?"
                    )
                if lang == "zh":
                    return "未能提供明确路线. 请问您从哪个入口或展厅出发?"
                return (
                    "I couldn't give clear directions. Which entrance or "
                    "gallery are you starting from?"
                )

            if lang == "fr":
                return (
                    "Desole pour cela. Souhaitez-vous que je recherche une oeuvre "
                    "precise ou des informations sur votre visite au Met ?"
                )
            if lang == "es":
                return (
                    "Disculpe las molestias. ¿Desea que busque una obra en particular "
                    "o informacion sobre la visita al Met?"
                )
            if lang == "zh":
                return (
                    "抱歉未能满足您的需求. 您希望我查询具体的艺术品, "
                    "还是参观大都会艺术博物馆的实用信息?"
                )
            return (
                "Sorry about that. Would you like me to look up a specific artwork, "
                "or details about visiting The Met?"
            )

        return "Could you clarify what you would like to know about The Met?"

    def resolve_referent(
        self, original_question: str = "referent_resolution"
    ) -> tuple[VerifiedObjectRef | None, PendingClarification | None]:
        """
        Resolution rules:
        - Exactly one clear recent object: return it.
        - Several plausible objects: return None with structured PendingClarification.
        - No plausible referent: return None with structured PendingClarification.
        """
        if len(self.verified_objects) == 1:
            return self.verified_objects[0], None
        if len(self.verified_objects) > 1:
            return None, PendingClarification(
                clarification_type="which_object",
                candidates=self.verified_objects[:2],
                original_question=original_question,
            )
        return None, PendingClarification(
            clarification_type="missing_referent",
            candidates=[],
            original_question=original_question,
        )
