"""Compact, bounded session context and referent resolution for conversation turns."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from met_agent.agent.models import AnswerKind, Language


class VerifiedObjectRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object_id: int = Field(gt=0)
    title: str = Field(min_length=1, max_length=200)


class SessionContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    first_user_message: str | None = Field(default=None, max_length=500)
    current_topic: str | None = Field(default=None, max_length=200)
    verified_objects: list[VerifiedObjectRef] = Field(default_factory=list, max_length=3)
    relevant_galleries: list[str] = Field(default_factory=list, max_length=3)
    pending_clarification: str | None = Field(default=None, max_length=500)
    last_answer_kind: AnswerKind | None = None
    last_turn_id: UUID | None = None
    language: Language = "en"

    def record_turn(
        self,
        *,
        turn_id: UUID,
        user_message: str,
        answer_kind: AnswerKind,
        verified_object: VerifiedObjectRef | None = None,
        gallery_number: str | None = None,
        language: Language | None = None,
    ) -> "SessionContext":
        if not self.first_user_message:
            self.first_user_message = user_message[:500]
        if language:
            self.language = language
        self.last_turn_id = turn_id
        self.last_answer_kind = answer_kind

        # Social reactions preserve existing context
        if answer_kind == "social":
            return self

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

    def resolve_referent(self) -> tuple[VerifiedObjectRef | None, str | None]:
        """
        Resolution rules:
        - Exactly one clear recent object: return it.
        - Several plausible objects: return None with clarification prompt.
        - No plausible referent: return None with missing referent prompt.
        """
        if len(self.verified_objects) == 1:
            return self.verified_objects[0], None
        if len(self.verified_objects) > 1:
            names = [f'"{obj.title}"' for obj in self.verified_objects[:2]]
            return None, f"Which work do you mean: {' or '.join(names)}?"
        return None, "Which artwork or gallery are you asking about?"
