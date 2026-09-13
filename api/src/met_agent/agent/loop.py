# ruff: noqa: RUF001
"""Run bounded sequential tools and release only drafts that pass citation and claim checks."""

import asyncio
import json
import re
import time
from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid4

import structlog
from pydantic import JsonValue

from met_agent.agent.context import SessionContext, VerifiedObjectRef
from met_agent.agent.distributed import get_session_lock
from met_agent.agent.events import AuditStore
from met_agent.agent.evidence import pack_evidence
from met_agent.agent.models import (
    AgentAnswer,
    AgentDraft,
    AnswerKind,
    ChatRequest,
    Citation,
    Language,
    Route,
    VerificationStatus,
)
from met_agent.agent.social_policy import (
    resolve_capability_explanation,
    resolve_identity_response,
    resolve_recall_response,
    resolve_social_response,
)
from met_agent.guardrails.grounding import (
    GroundingCheck,
    valid_citations,
    valid_wayfinding_evidence,
)
from met_agent.guardrails.intent import (
    Intent,
    RecallTarget,
    classify_conversational_fallback,
    identity_intent,
    message_numbers,
    normalize_intent,
    recall_intent,
    social_intent,
)
from met_agent.guardrails.interpretive import POLICY, UNAVAILABLE, UNVERIFIED
from met_agent.llm.chat import CURRENT_CALL, CallContext, ChatModel, ModelError, structured
from met_agent.llm.prompts import load_prompt, prompt_hash
from met_agent.tools.models import (
    CollectionSearchResult,
    Evidence,
    Handoff,
    LiveObject,
    SearchVisitorArguments,
    ToolResult,
    VisitorSearchResult,
    WayfindingResult,
)
from met_agent.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)

MAX_TOOL_CALLS = 6
MAX_TOOL_ROUNDS = 2
NOT_FOUND: dict[Language, str] = {
    "en": "The requested object was not found. The Met API returned no object record.",
    "fr": "L'objet demandé est introuvable. L'API du Met n'a renvoyé aucune notice d'objet.",
    "es": "No se encontró el objeto solicitado. La API del Met no devolvió ningún registro.",
    "zh": "未找到所请求的藏品。大都会艺术博物馆 API 未返回任何藏品记录。",
}
GREETING: dict[Language, str] = {
    "en": "Hello! What would you like to explore or know before your visit?",
    "fr": "Bonjour ! Que souhaitez-vous explorer ou savoir avant votre visite ?",
    "es": "¡Hola! ¿Qué le gustaría explorar o saber antes de su visita?",
    "zh": "您好！在参观前您想了解或探索哪些展品和信息？",
}
REACTION: dict[Language, str] = {
    "en": "😄",
    "fr": "😄",
    "es": "😄",
    "zh": "😄",
}
WELLBEING: dict[Language, str] = {
    "en": "I'm ready to help. What would you like to know about The Met?",
    "fr": "Je suis prêt à vous aider. Que souhaitez-vous savoir sur le Met ?",
    "es": "Estoy listo para ayudarle. ¿Qué desea saber sobre el Met?",
    "zh": "我已准备好为您提供帮助。您想了解大都会艺术博物馆的哪些信息？",
}
THANKS: dict[Language, str] = {
    "en": "You're welcome!",
    "fr": "Je vous en prie !",
    "es": "¡De nada!",
    "zh": "不客气！",
}
FAREWELL: dict[Language, str] = {
    "en": "Goodbye! I hope you enjoy your visit to The Met.",
    "fr": "Au revoir ! Je vous souhaite une excellente visite au Met.",
    "es": "¡Adiós! Espero que disfrute de su visita al Met.",
    "zh": "再见！祝您参观愉快。",
}
DISSATISFACTION: dict[Language, str] = {
    "en": "Sorry I missed what you needed. Which part should I clarify?",
    "fr": "Désolé d'avoir manqué ce dont vous aviez besoin. Quelle partie dois-je clarifier ?",
    "es": "Disculpe si no entendí lo que necesitaba. ¿Qué parte desea que aclare?",
    "zh": "抱歉未能解答您的疑问。请问您需要我澄清哪一部分？",
}
SOCIAL_RESPONSES = {
    "greeting": GREETING,
    "reaction": REACTION,
    "wellbeing": WELLBEING,
    "thanks": THANKS,
    "farewell": FAREWELL,
    "dissatisfaction": DISSATISFACTION,
}


class Agent:
    def __init__(self, model: ChatModel, tools: ToolRegistry, events: AuditStore) -> None:
        self.model, self.tools, self.events = model, tools, events
        self._sessions: dict[UUID, asyncio.Lock] = {}
        self._users: dict[UUID, int] = {}

    async def run(self, request: ChatRequest) -> AgentAnswer:
        session = request.session_id or uuid4()
        lock = self._sessions.setdefault(session, asyncio.Lock())
        self._users[session] = self._users.get(session, 0) + 1
        try:
            async with lock, get_session_lock(self.events, session):
                return await self._turn(request, session)
        finally:
            self._users[session] -= 1
            if self._users[session] == 0:
                del self._sessions[session]
                del self._users[session]

    async def _turn(self, request: ChatRequest, session: UUID) -> AgentAnswer:
        started, turn = time.monotonic(), uuid4()
        history = await asyncio.to_thread(self.events.history, session)
        buffered_events: list[tuple[str, JsonValue]] = []

        def audit(kind: str, data: JsonValue) -> None:
            buffered_events.append((kind, data))

        context = CallContext(audit=audit)
        token = CURRENT_CALL.set(context)
        audit("user_message", {"message": request.message, "language_hint": request.language})
        audit(
            "prompt_versions",
            {
                name: prompt_hash(name)
                for name in ("system_v2", "tools_v1", "intent_v4", "grounding_v1", "citations_v1")
            },
        )
        session_ctx = await asyncio.to_thread(
            self.events.get_session_context, session
        ) or SessionContext(language=request.language or "en")
        try:
            # 1. Handle clarification responses before social fast path
            if session_ctx.pending_clarification is not None and (
                matched_ref := session_ctx.match_clarification_answer(request.message)
            ):
                audit(
                    "route_policy",
                    {
                        "policy": "clarification_resolved",
                        "selected_object_id": matched_ref.object_id,
                        "selected_title": matched_ref.title,
                        "route": "lite",
                    },
                )
                obj_res = await self.tools.execute(
                    "get_object", json.dumps({"object_id": matched_ref.object_id})
                )
                audit(
                    "tool_call",
                    {"name": "get_object", "arguments": {"object_id": matched_ref.object_id}},
                )
                audit("tool_result", obj_res.model_dump(mode="json"))
                live_obj = obj_res.output if isinstance(obj_res.output, LiveObject) else None
                if live_obj:
                    session_ctx.record_turn(
                        turn_id=turn,
                        user_message=request.message,
                        answer_kind="factual",
                        verified_object=matched_ref,
                        gallery_number=live_obj.gallery_number,
                        language=request.language or "en",
                    )
                    await asyncio.to_thread(
                        self.events.save_session_context, session, turn, session_ctx
                    )
                    overview = (
                        f"{live_obj.title} by {live_obj.artist or 'Unknown artist'} "
                        f"({live_obj.object_date or 'undated'})."
                    )
                    if live_obj.is_on_view and live_obj.gallery_number:
                        overview += (
                            f" It is currently on view in Gallery {live_obj.gallery_number}."
                        )
                    citations = [
                        Citation(
                            object_id=matched_ref.object_id,
                            quote=live_obj.title,
                        )
                    ]
                    return self._answer(
                        session,
                        turn,
                        started,
                        request.language or "en",
                        "lite",
                        context,
                        overview,
                        citations,
                        1.0,
                        answer_kind="factual",
                        verification_status="verified",
                    )

            # 2. Assistant identity fast path
            if identity := identity_intent(request.message):
                _, language = identity
                reply_text, response_id = resolve_identity_response(language)
                audit(
                    "route_policy",
                    {
                        "policy": "assistant_identity",
                        "language": language,
                        "response_id": response_id,
                        "route": "lite",
                    },
                )
                session_ctx.record_turn(
                    turn_id=turn,
                    user_message=request.message,
                    answer_kind="social",
                    language=language,
                    social_response_id=response_id,
                )
                await asyncio.to_thread(
                    self.events.save_session_context, session, turn, session_ctx
                )
                return self._answer(
                    session,
                    turn,
                    started,
                    language,
                    "lite",
                    context,
                    reply_text,
                    [],
                    None,
                    answer_kind="social",
                    verification_status="not_applicable",
                )

            # 3. Conversation recall fast path
            if recall := recall_intent(request.message):
                _, target, language, topic = recall
                try:
                    history_messages = await asyncio.to_thread(
                        self.events.user_messages, session, exclude_turn=turn
                    )
                    db_failed = False
                except Exception:
                    logger.exception("Failed to query user messages for recall")
                    history_messages = []
                    db_failed = True

                is_expired = (
                    not db_failed
                    and len(history_messages) == 0
                    and (session_ctx.turn_count > 0 or session_ctx.first_user_message is not None)
                )
                reply_text, response_id = resolve_recall_response(
                    target,
                    history_messages,
                    language=language,
                    topic=topic,
                    is_expired=is_expired,
                    db_failed=db_failed,
                )
                audit(
                    "route_policy",
                    {
                        "policy": "deterministic_session_recall",
                        "target": target,
                        "topic": topic,
                        "language": language,
                        "response_id": response_id,
                        "route": "lite",
                    },
                )
                is_grounded = response_id in {"recall_first", "recall_previous", "recall_topic"}
                session_ctx.record_turn(
                    turn_id=turn,
                    user_message=request.message,
                    answer_kind="factual" if is_grounded else "social",
                    language=language,
                    social_response_id=response_id,
                )
                await asyncio.to_thread(
                    self.events.save_session_context, session, turn, session_ctx
                )
                return self._answer(
                    session,
                    turn,
                    started,
                    language,
                    "lite",
                    context,
                    reply_text,
                    [],
                    1.0 if is_grounded else None,
                    answer_kind="factual" if is_grounded else "social",
                    verification_status="verified" if is_grounded else "not_applicable",
                )

            # 4. Context-aware deterministic social fast path
            if social := social_intent(
                request.message,
                last_answer_kind=session_ctx.last_answer_kind,
                last_failure_reason=session_ctx.last_failure_reason,
            ):
                kind, language = social
                reply_text, response_id = resolve_social_response(kind, session_ctx, language)
                audit(
                    "route_policy",
                    {
                        "policy": "deterministic_social_turn",
                        "kind": kind,
                        "language": language,
                        "response_id": response_id,
                        "route": "lite",
                    },
                )
                session_ctx.record_turn(
                    turn_id=turn,
                    user_message=request.message,
                    answer_kind="social",
                    language=language,
                    social_response_id=response_id,
                )
                await asyncio.to_thread(
                    self.events.save_session_context, session, turn, session_ctx
                )
                return self._answer(
                    session,
                    turn,
                    started,
                    language,
                    "lite",
                    context,
                    reply_text,
                    [],
                    None,
                    answer_kind="social",
                    verification_status="not_applicable",
                )
            has_pronoun = bool(re.search(r"\b(?:it|that|this)\b", request.message, re.IGNORECASE))
            if has_pronoun:
                ref, pending_clarify = session_ctx.resolve_referent(
                    original_question=request.message
                )
                if pending_clarify is not None:
                    audit("route_policy", {"policy": "referent_clarification", "route": "lite"})
                    session_ctx.record_turn(
                        turn_id=turn,
                        user_message=request.message,
                        answer_kind="clarification",
                        language=request.language or "en",
                        pending_clarification=pending_clarify,
                    )
                    await asyncio.to_thread(
                        self.events.save_session_context, session, turn, session_ctx
                    )
                    clarify_text = session_ctx.render_clarification(request.language or "en")
                    return self._answer(
                        session,
                        turn,
                        started,
                        request.language or "en",
                        "lite",
                        context,
                        clarify_text,
                        [],
                        None,
                        answer_kind="clarification",
                        verification_status="not_applicable",
                    )
                if ref is not None:
                    msg_lower = request.message.lower()
                    if any(
                        word in msg_lower
                        for word in (
                            "upstairs",
                            "downstairs",
                            "gallery",
                            "where",
                            "floor",
                            "located",
                            "room",
                        )
                    ):
                        obj_res = await self.tools.execute(
                            "get_object", json.dumps({"object_id": ref.object_id})
                        )
                        audit(
                            "tool_call",
                            {"name": "get_object", "arguments": {"object_id": ref.object_id}},
                        )
                        audit("tool_result", obj_res.model_dump(mode="json"))
                        live_obj = (
                            obj_res.output if isinstance(obj_res.output, LiveObject) else None
                        )
                        if live_obj and live_obj.is_on_view and live_obj.gallery_number:
                            gallery = live_obj.gallery_number
                            gallery_num = int(gallery) if gallery.isdigit() else None
                            is_upstairs = gallery_num is not None and (300 <= gallery_num <= 999)
                            if "upstairs" in msg_lower:
                                ans_text = (
                                    f"Yes, {ref.title} is upstairs in Gallery {gallery} "
                                    "(second floor)."
                                    if is_upstairs
                                    else f"No, {ref.title} is on the first floor in Gallery "
                                    f"{gallery}, not upstairs."
                                )
                            elif "downstairs" in msg_lower:
                                ans_text = (
                                    f"No, {ref.title} is on the second floor in Gallery "
                                    f"{gallery}, not downstairs."
                                    if is_upstairs
                                    else f"No, {ref.title} is on the first floor in Gallery "
                                    f"{gallery}."
                                )
                            else:
                                ans_text = f"{ref.title} is currently on view in Gallery {gallery}."
                            citation = Citation(object_id=ref.object_id, quote=f"Gallery {gallery}")
                            session_ctx.record_turn(
                                turn_id=turn,
                                user_message=request.message,
                                answer_kind="factual",
                                verified_object=ref,
                                gallery_number=gallery,
                                language="en",
                            )
                            await asyncio.to_thread(
                                self.events.save_session_context, session, turn, session_ctx
                            )
                            return self._answer(
                                session,
                                turn,
                                started,
                                request.language or "en",
                                "lite",
                                context,
                                ans_text,
                                [citation],
                                1.0,
                                answer_kind="factual",
                                verification_status="verified",
                            )
            model_intent = await structured(
                self.model,
                "lite",
                load_prompt("intent_v4"),
                {
                    "message": request.message,
                    "history": json.loads(json.dumps(history)),
                    "language_hint": request.language,
                },
                Intent,
            )
            audit("intent", model_intent.model_dump(mode="json"))
            intent, routing_policy = normalize_intent(model_intent, request.message)
            if routing_policy is not None:
                selected_route: Route = (
                    "lite"
                    if routing_policy.startswith("direct_gallery_wayfinding")
                    else intent.route
                )
                audit(
                    "route_policy",
                    {
                        "policy": routing_policy,
                        "model_category": model_intent.category,
                        "model_difficulty": model_intent.difficulty,
                        "category": intent.category,
                        "difficulty": intent.difficulty,
                        "route": selected_route,
                        "search_query": intent.search_query,
                    },
                )
            if intent.category == "interpretive":
                audit("guardrail", {"policy": "non_interpretive", "decision": "refuse"})
                session_ctx.record_turn(
                    turn_id=turn,
                    user_message=request.message,
                    answer_kind="policy_refusal",
                    language=intent.language,
                )
                await asyncio.to_thread(
                    self.events.save_session_context, session, turn, session_ctx
                )
                return self._answer(
                    session,
                    turn,
                    started,
                    intent.language,
                    intent.route,
                    context,
                    POLICY[intent.language],
                    [],
                    None,
                    answer_kind="policy_refusal",
                    verification_status="not_applicable",
                    refusal=True,
                )
            conv_fallback = classify_conversational_fallback(request.message, intent)
            if conv_fallback.outcome == "assistant_identity":
                reply_text, response_id = resolve_identity_response(intent.language)
                session_ctx.record_turn(
                    turn_id=turn,
                    user_message=request.message,
                    answer_kind="social",
                    language=intent.language,
                    social_response_id=response_id,
                )
                await asyncio.to_thread(
                    self.events.save_session_context, session, turn, session_ctx
                )
                return self._answer(
                    session,
                    turn,
                    started,
                    intent.language,
                    intent.route,
                    context,
                    reply_text,
                    [],
                    None,
                    answer_kind="social",
                    verification_status="not_applicable",
                )
            if conv_fallback.outcome == "conversation_recall":
                try:
                    history_messages = await asyncio.to_thread(
                        self.events.user_messages, session, exclude_turn=turn
                    )
                    db_failed = False
                except Exception:
                    logger.exception("Failed to query user messages for recall")
                    history_messages = []
                    db_failed = True

                is_expired = (
                    not db_failed
                    and len(history_messages) == 0
                    and (session_ctx.turn_count > 0 or session_ctx.first_user_message is not None)
                )
                recall_target: RecallTarget = (
                    "previous"
                    if conv_fallback.recall_target == "none"
                    else conv_fallback.recall_target
                )
                reply_text, response_id = resolve_recall_response(
                    recall_target,
                    history_messages,
                    language=intent.language,
                    topic=conv_fallback.recall_topic,
                    is_expired=is_expired,
                    db_failed=db_failed,
                )
                is_grounded = response_id in {"recall_first", "recall_previous", "recall_topic"}
                session_ctx.record_turn(
                    turn_id=turn,
                    user_message=request.message,
                    answer_kind="factual" if is_grounded else "social",
                    language=intent.language,
                    social_response_id=response_id,
                )
                await asyncio.to_thread(
                    self.events.save_session_context, session, turn, session_ctx
                )
                return self._answer(
                    session,
                    turn,
                    started,
                    intent.language,
                    intent.route,
                    context,
                    reply_text,
                    [],
                    1.0 if is_grounded else None,
                    answer_kind="factual" if is_grounded else "social",
                    verification_status="verified" if is_grounded else "not_applicable",
                )
            if conv_fallback.outcome == "capability_explanation":
                reply_text, response_id = resolve_capability_explanation(intent.language)
                session_ctx.record_turn(
                    turn_id=turn,
                    user_message=request.message,
                    answer_kind="social",
                    language=intent.language,
                    social_response_id=response_id,
                )
                await asyncio.to_thread(
                    self.events.save_session_context, session, turn, session_ctx
                )
                return self._answer(
                    session,
                    turn,
                    started,
                    intent.language,
                    intent.route,
                    context,
                    reply_text,
                    [],
                    None,
                    answer_kind="social",
                    verification_status="not_applicable",
                )
            if (
                conv_fallback.outcome == "collection_or_visitor"
                and intent.category == "out_of_scope"
            ):
                intent = intent.model_copy(
                    update={"category": "visitor_info", "search_query": request.message}
                )
            if intent.category == "out_of_scope":
                contact = conv_fallback.handoff_contact or intent.handoff_contact
                reason = conv_fallback.handoff_reason or (
                    "This request concerns account, purchase, or information "
                    "outside the collection and visitor workspace."
                )
                result = await self.tools.execute(
                    "handoff",
                    json.dumps(
                        {
                            "reason": reason,
                            "suggested_contact": contact,
                        }
                    ),
                )
                audit("tool_call", {"name": "handoff", "arguments": {"reason": "out_of_scope"}})
                audit("tool_result", result.model_dump(mode="json"))
                handoff = result.output if isinstance(result.output, Handoff) else None
                session_ctx.record_turn(
                    turn_id=turn,
                    user_message=request.message,
                    answer_kind="handoff",
                    language=intent.language,
                )
                await asyncio.to_thread(
                    self.events.save_session_context, session, turn, session_ctx
                )
                return self._answer(
                    session,
                    turn,
                    started,
                    intent.language,
                    intent.route,
                    context,
                    UNVERIFIED[intent.language],
                    [],
                    None,
                    answer_kind="handoff",
                    verification_status="not_applicable",
                    handoff=handoff,
                )
            if routing_policy == "direct_gallery_question":
                gallery_number = intent.gallery_number
                if gallery_number is None:
                    raise RuntimeError("Direct gallery policy lost its validated gallery number")
                return await self._gallery_workspace(
                    request,
                    session,
                    turn,
                    started,
                    intent,
                    gallery_number,
                    context,
                    audit,
                )
            if routing_policy in {
                "direct_gallery_wayfinding",
                "direct_gallery_wayfinding_default_origin",
            }:
                gallery_number = intent.gallery_number
                if gallery_number is None:
                    raise RuntimeError("Wayfinding policy lost its validated gallery number")
                return await self._wayfinding_workspace(
                    session,
                    turn,
                    started,
                    gallery_number,
                    routing_policy == "direct_gallery_wayfinding_default_origin",
                    context,
                    audit,
                )
            if intent.category == "visitor_info" and intent.difficulty == "simple":
                audit(
                    "route_policy",
                    {
                        "policy": "direct_visitor_search",
                        "model_category": model_intent.category,
                        "model_difficulty": model_intent.difficulty,
                        "category": intent.category,
                        "difficulty": intent.difficulty,
                        "route": "lite",
                        "search_query": intent.search_query,
                    },
                )
                return await self._visitor_workspace(
                    request,
                    session,
                    turn,
                    started,
                    intent,
                    context,
                    audit,
                )
            answer = await self._workspace(
                request, session, turn, started, intent, context, history, audit
            )
            if (answer.grounding_score is None or answer.grounding_score < 1) and (
                intent.gallery_number is not None or message_numbers(request.message)
            ):
                deviation: dict[str, JsonValue] = {
                    "reason": "numeric_query_unverified_on_general_route",
                    "category": intent.category,
                    "operation": intent.operation,
                    "gallery_number": intent.gallery_number,
                    "session_id": str(session),
                    "turn_id": str(turn),
                }
                audit("routing_deviation", deviation)
                structlog.get_logger(__name__).warning("routing_deviation", **deviation)
            verified_obj = None
            if answer.citations:
                for cit in answer.citations:
                    if cit.object_id:
                        verified_obj = VerifiedObjectRef(
                            object_id=cit.object_id,
                            title=intent.search_query.strip() or f"Object {cit.object_id}",
                        )
                        break
            failure_reason = None
            if answer.answer_kind == "unavailable":
                failure_reason = "object_not_found"
            elif answer.answer_kind == "policy_refusal":
                failure_reason = "policy_refusal"
            session_ctx.record_turn(
                turn_id=turn,
                user_message=request.message,
                answer_kind=answer.answer_kind,
                verified_object=verified_obj,
                gallery_number=intent.gallery_number,
                language=answer.language,
                failure_reason=failure_reason,
            )
            await asyncio.to_thread(self.events.save_session_context, session, turn, session_ctx)
            return answer
        except ModelError as error:
            audit("turn_error", {"code": error.code, "message": str(error)})
            raise
        finally:
            CURRENT_CALL.reset(token)
            await asyncio.to_thread(self.events.append_many, session, turn, buffered_events)

    async def _visitor_workspace(
        self,
        request: ChatRequest,
        session: UUID,
        turn: UUID,
        started: float,
        intent: Intent,
        context: CallContext,
        audit: Callable[[str, JsonValue], None],
    ) -> AgentAnswer:
        """Search visitor evidence directly after a validated simple visitor intent."""

        arguments = SearchVisitorArguments(query=intent.search_query, k=5).model_dump_json()
        audit("tool_call", {"name": "search_visitor_info", "arguments": arguments, "number": 1})
        tool_started = time.monotonic()
        result = await self.tools.execute("search_visitor_info", arguments)
        audit(
            "tool_timing",
            {"name": "search_visitor_info", "latency_ms": (time.monotonic() - tool_started) * 1000},
        )
        audit(
            "validation_error" if result.error else "tool_result",
            result.model_dump(mode="json"),
        )
        evidence = result.model_evidence()
        audit(
            "evidence_context",
            [item.model_dump(mode="json", exclude_none=True) for item in evidence],
        )
        if not isinstance(result.output, VisitorSearchResult) or not evidence:
            audit("guardrail", {"policy": "direct_visitor_search", "decision": "fail_closed"})
            return self._answer(
                session,
                turn,
                started,
                intent.language,
                "lite",
                context,
                UNVERIFIED[intent.language],
                [],
                None,
                answer_kind="unavailable",
                verification_status="unverified",
            )

        reply = await self.model.complete(
            "lite",
            [
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": request.message,
                            "detected_language": intent.language,
                            "english_search_query": intent.search_query,
                        },
                        ensure_ascii=False,
                    ),
                },
                {
                    "role": "tool",
                    "tool_call_id": "direct-visitor-search",
                    "content": result.model_context(),
                },
            ],
            response_schema=AgentDraft,
        )
        try:
            draft = AgentDraft.model_validate_json(reply.text)
        except ValueError:
            draft = None
        citations_ok = (
            draft is not None
            and draft.language == intent.language
            and valid_citations(draft, evidence)
        )
        if draft is None or not citations_ok:
            audit(
                "guardrail",
                {
                    "policy": "direct_visitor_search",
                    "decision": "fail_closed",
                    "citations_valid": citations_ok,
                },
            )
            return self._answer(
                session,
                turn,
                started,
                intent.language,
                "lite",
                context,
                UNVERIFIED[intent.language],
                [],
                None,
                answer_kind="unavailable",
                verification_status="unverified",
            )

        check = await structured(
            self.model,
            "lite",
            load_prompt("grounding_v1"),
            {
                "question": request.message,
                "draft": draft.model_dump(mode="json"),
                "evidence": [item.model_dump(mode="json") for item in evidence],
            },
            GroundingCheck,
        )
        score = check.score(evidence, draft_text=draft.text)
        audit(
            "guardrail",
            {
                "policy": "direct_visitor_search",
                "score": score,
                "result": check.model_dump(mode="json"),
            },
        )
        if not check.fully_supported or score != 1 or not draft.citations or not check.claims:
            audit("guardrail", {"policy": "citation_or_grounding", "decision": "fail_closed"})
            return self._answer(
                session,
                turn,
                started,
                intent.language,
                "lite",
                context,
                UNAVAILABLE[intent.language],
                [],
                None,
                answer_kind="unavailable",
                verification_status="unverified",
            )
        return self._answer(
            session,
            turn,
            started,
            intent.language,
            "lite",
            context,
            draft.text,
            draft.citations,
            score,
            answer_kind="factual",
            verification_status="verified",
        )

    async def _wayfinding_workspace(
        self,
        session: UUID,
        turn: UUID,
        started: float,
        gallery_number: str,
        assumed_origin: bool,
        context: CallContext,
        audit: Callable[[str, JsonValue], None],
    ) -> AgentAnswer:
        """Resolve one entrance-to-gallery route without loading retrieval models."""
        arguments = json.dumps(
            {
                "origin": "fifth_avenue_entrance",
                "destination_gallery": gallery_number,
            }
        )
        audit("tool_call", {"name": "get_directions", "arguments": arguments, "number": 1})
        tool_started = time.monotonic()
        result = await self.tools.execute("get_directions", arguments)
        audit(
            "tool_timing",
            {"name": "get_directions", "latency_ms": (time.monotonic() - tool_started) * 1000},
        )
        audit(
            "validation_error" if result.error else "tool_result",
            result.model_dump(mode="json"),
        )
        if not isinstance(result.output, WayfindingResult) or len(result.evidence) != 1:
            audit("guardrail", {"policy": "official_map_wayfinding", "decision": "fail_closed"})
            return self._answer(
                session,
                turn,
                started,
                "en",
                "lite",
                context,
                UNVERIFIED["en"],
                [],
                None,
                answer_kind="unavailable",
                verification_status="unverified",
            )

        route = result.output
        evidence = result.model_evidence()
        audit(
            "evidence_context",
            [item.model_dump(mode="json", exclude_none=True) for item in evidence],
        )
        assumption = (
            "I will use the Fifth Avenue entrance as your starting assumption. "
            if assumed_origin
            else ""
        )
        text = (
            f"{assumption}The Met's official map route starts at {route.origin} and "
            f"continues on {route.floor} to {route.destination}, about "
            f"{route.distance_feet} feet ({route.duration_minutes} minutes). Open the cited "
            "live route before you start."
        )
        draft = AgentDraft(
            text=text,
            language="en",
            citations=[Citation(source_url=route.source_url, quote=evidence[0].text)],
        )
        if not valid_citations(draft, evidence) or not valid_wayfinding_evidence(route, evidence):
            audit(
                "guardrail",
                {
                    "policy": "official_map_wayfinding",
                    "decision": "fail_closed",
                    "reason": "evidence_mismatch",
                },
            )
            return self._answer(
                session,
                turn,
                started,
                "en",
                "lite",
                context,
                UNVERIFIED["en"],
                [],
                None,
                answer_kind="unavailable",
                verification_status="unverified",
            )
        audit(
            "guardrail",
            {
                "policy": "official_map_wayfinding",
                "method": "deterministic_template",
                "score": 1.0,
                "evidence_key": evidence[0].key,
            },
        )
        return self._answer(
            session,
            turn,
            started,
            "en",
            "lite",
            context,
            draft.text,
            draft.citations,
            1.0,
            answer_kind="factual",
            verification_status="verified",
        )

    async def _gallery_workspace(
        self,
        request: ChatRequest,
        session: UUID,
        turn: UUID,
        started: float,
        intent: Intent,
        gallery_number: str,
        context: CallContext,
        audit: Callable[[str, JsonValue], None],
    ) -> AgentAnswer:
        """Resolve a direct gallery inventory with exact filtering and live records."""

        async def execute(number: int, name: str, arguments: str) -> ToolResult:
            audit("tool_call", {"name": name, "arguments": arguments, "number": number})
            tool_started = time.monotonic()
            result = await self.tools.execute(name, arguments)
            audit(
                "tool_timing",
                {"name": name, "latency_ms": (time.monotonic() - tool_started) * 1000},
            )
            audit(
                "validation_error" if result.error else "tool_result",
                result.model_dump(mode="json"),
            )
            return result

        query = f"Objects in Gallery {gallery_number}"
        search = await execute(
            1,
            "search_collection",
            json.dumps(
                {
                    "query": query,
                    "filters": {"gallery_number": gallery_number},
                    "k": 5,
                }
            ),
        )
        if not isinstance(search.output, CollectionSearchResult):
            audit(
                "guardrail",
                {"policy": "deterministic_gallery_inventory", "decision": "fail_closed"},
            )
            return self._answer(
                session,
                turn,
                started,
                "en",
                "lite",
                context,
                UNVERIFIED["en"],
                [],
                None,
                answer_kind="unavailable",
                verification_status="unverified",
            )

        confirmed: list[tuple[LiveObject, Evidence]] = []
        for number, candidate in enumerate(search.output.objects[:5], 2):
            result = await execute(
                number, "get_object", json.dumps({"object_id": candidate.object_id})
            )
            if not isinstance(result.output, LiveObject):
                continue
            if not result.output.is_on_view or result.output.gallery_number != gallery_number:
                continue
            live_evidence = next(
                (item for item in result.model_evidence() if item.kind == "live_object"), None
            )
            if live_evidence is not None:
                confirmed.append((result.output, live_evidence))

        evidence = [item for _, item in confirmed]
        audit(
            "evidence_context",
            [item.model_dump(mode="json", exclude_none=True) for item in evidence],
        )
        if not confirmed:
            audit(
                "guardrail",
                {"policy": "deterministic_gallery_inventory", "decision": "fail_closed"},
            )
            return self._answer(
                session,
                turn,
                started,
                "en",
                "lite",
                context,
                UNVERIFIED["en"],
                [],
                None,
                answer_kind="unavailable",
                verification_status="unverified",
            )

        labels = [obj.title.strip() or f"Object {obj.object_id}" for obj, _ in confirmed]
        if len(labels) == 1:
            text = (
                f"The live Met record currently lists {labels[0]} as on view "
                f"in Gallery {gallery_number}."
            )
        else:
            text = (
                f"The live Met records currently list these objects as on view in Gallery "
                f"{gallery_number}: " + "; ".join(labels) + "."
            )
        draft = AgentDraft(
            text=text,
            language="en",
            citations=[
                Citation(object_id=obj.object_id, quote=item.text[:2000]) for obj, item in confirmed
            ],
        )
        if not valid_citations(draft, evidence):
            raise RuntimeError("Server-produced gallery citations are invalid")
        check = await structured(
            self.model,
            "lite",
            load_prompt("grounding_v1"),
            {
                "question": request.message,
                "draft": draft.model_dump(mode="json"),
                "evidence": [item.model_dump(mode="json") for item in evidence],
            },
            GroundingCheck,
        )
        score = check.score(evidence, draft_text=draft.text)
        audit(
            "guardrail",
            {
                "policy": "deterministic_gallery_inventory",
                "score": score,
                "result": check.model_dump(mode="json"),
            },
        )
        if not check.fully_supported or score != 1 or not check.claims:
            audit(
                "guardrail",
                {"policy": "citation_or_grounding", "decision": "fail_closed"},
            )
            return self._answer(
                session,
                turn,
                started,
                "en",
                "lite",
                context,
                UNAVAILABLE["en"],
                [],
                None,
                answer_kind="unavailable",
                verification_status="unverified",
            )
        return self._answer(
            session,
            turn,
            started,
            "en",
            "lite",
            context,
            draft.text,
            draft.citations,
            score,
            answer_kind="factual",
            verification_status="verified",
        )

    async def _workspace(
        self,
        request: ChatRequest,
        session: UUID,
        turn: UUID,
        started: float,
        intent: Intent,
        context: CallContext,
        history: list[dict[str, str]],
        audit: Callable[[str, JsonValue], None],
    ) -> AgentAnswer:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": load_prompt("tools_v1")},
            *history,
            {"role": "user", "content": request.message},
            {
                "role": "system",
                "content": json.dumps(
                    {
                        "detected_language": intent.language,
                        "english_search_query": intent.search_query,
                    }
                ),
            },
        ]
        evidence: list[Evidence] = []
        tool_count = 0
        tool_rounds = 0
        repairs = 0
        for _ in range(MAX_TOOL_CALLS + 3):
            reply = await self.model.complete(
                intent.route,
                messages,
                tools=(
                    self.tools.schemas
                    if tool_count < MAX_TOOL_CALLS
                    and tool_rounds < MAX_TOOL_ROUNDS
                    and repairs == 0
                    else None
                ),
                response_schema=AgentDraft,
            )
            calls = reply.message.get("tool_calls") or []
            if not isinstance(calls, list) or any(
                not isinstance(call, dict) or not isinstance(call.get("function"), dict)
                for call in calls
            ):
                raise ModelError("invalid_response", "Invalid model tool-call envelope")
            messages.append(reply.message)
            if calls:
                tool_rounds += 1
                for call in calls:
                    if tool_count >= MAX_TOOL_CALLS:
                        audit("tool_limit", {"limit": MAX_TOOL_CALLS})
                        return self._answer(
                            session,
                            turn,
                            started,
                            intent.language,
                            intent.route,
                            context,
                            UNVERIFIED[intent.language],
                            [],
                            None,
                            answer_kind="unavailable",
                            verification_status="unverified",
                        )
                    tool_count += 1
                    function = call.get("function") or {}
                    name, arguments = str(function.get("name", "")), function.get("arguments", "")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments)
                    audit("tool_call", {"name": name, "arguments": arguments, "number": tool_count})
                    tool_started = time.monotonic()
                    result = await self.tools.execute(name, arguments)
                    audit(
                        "tool_timing",
                        {"name": name, "latency_ms": (time.monotonic() - tool_started) * 1000},
                    )
                    audit(
                        "validation_error" if result.error else "tool_result",
                        result.model_dump(mode="json"),
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", ""),
                            "content": result.model_context(),
                        }
                    )
                    evidence = pack_evidence(messages)
                    audit(
                        "evidence_context",
                        [item.model_dump(mode="json", exclude_none=True) for item in evidence],
                    )
                    lookup = next(
                        (
                            item
                            for item in evidence
                            if result.error
                            and result.error.code == "not_found"
                            and item.kind == "lookup_status"
                            and item.source_url is not None
                        ),
                        None,
                    )
                    if lookup is not None:
                        citation = Citation(source_url=lookup.source_url, quote=lookup.text)
                        lookup_draft = AgentDraft(
                            text=NOT_FOUND[intent.language],
                            language=intent.language,
                            citations=[citation],
                        )
                        if not valid_citations(lookup_draft, evidence):
                            raise RuntimeError("Server-produced lookup citation is invalid")
                        audit(
                            "guardrail",
                            {
                                "policy": "deterministic_lookup_status",
                                "score": 1.0,
                                "evidence_key": lookup.key,
                            },
                        )
                        return self._answer(
                            session,
                            turn,
                            started,
                            intent.language,
                            intent.route,
                            context,
                            lookup_draft.text,
                            lookup_draft.citations,
                            1.0,
                            answer_kind="factual",
                            verification_status="verified",
                        )
                    if isinstance(result.output, Handoff):
                        audit("terminal_handoff", result.output.model_dump(mode="json"))
                        return self._answer(
                            session,
                            turn,
                            started,
                            intent.language,
                            intent.route,
                            context,
                            UNVERIFIED[intent.language],
                            [],
                            None,
                            answer_kind="handoff",
                            verification_status="not_applicable",
                            handoff=result.output,
                        )
                if tool_rounds == MAX_TOOL_ROUNDS:
                    audit(
                        "tool_round_limit",
                        {"limit": MAX_TOOL_ROUNDS, "decision": "finalize_with_current_evidence"},
                    )
                continue
            try:
                draft = AgentDraft.model_validate_json(reply.text)
            except ValueError:
                draft = None
            citations_ok = (
                draft is not None
                and draft.language == intent.language
                and valid_citations(draft, evidence)
            )
            if draft is not None and citations_ok:
                check = await structured(
                    self.model,
                    "lite",
                    load_prompt("grounding_v1"),
                    {
                        "question": request.message,
                        "draft": draft.model_dump(mode="json"),
                        "evidence": [item.model_dump(mode="json") for item in evidence],
                    },
                    GroundingCheck,
                )
                score = check.score(evidence, draft_text=draft.text)
                audit(
                    "guardrail",
                    {
                        "policy": "grounding",
                        "score": score,
                        "result": check.model_dump(mode="json"),
                    },
                )
                if check.fully_supported and score == 1 and draft.citations and check.claims:
                    return self._answer(
                        session,
                        turn,
                        started,
                        intent.language,
                        intent.route,
                        context,
                        draft.text,
                        draft.citations,
                        score,
                        answer_kind="factual",
                        verification_status="verified",
                    )
            audit(
                "guardrail",
                {
                    "policy": "citation_or_grounding",
                    "decision": "regenerate" if repairs == 0 else "fail_closed",
                    "citations_valid": citations_ok,
                },
            )
            if repairs:
                break
            repairs += 1
            # Remove the rejected draft from the regeneration context, including unseen citations.
            messages.pop()
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The draft failed server verification. Regenerate once using only "
                        "evidence already returned in this turn. Remove unsupported claims and "
                        "unseen citations. If sources are insufficient, say that honestly. "
                        "Return the final JSON schema."
                    ),
                }
            )
        return self._answer(
            session,
            turn,
            started,
            intent.language,
            intent.route,
            context,
            UNAVAILABLE[intent.language],
            [],
            None,
            answer_kind="unavailable",
            verification_status="unverified",
        )

    def _answer(
        self,
        session: UUID,
        turn: UUID,
        started: float,
        language: Language,
        route: Route,
        context: CallContext,
        text: str,
        citations: list[Citation],
        score: float | None = None,
        *,
        answer_kind: AnswerKind = "factual",
        verification_status: VerificationStatus = "verified",
        refusal: bool = False,
        handoff: Handoff | None = None,
    ) -> AgentAnswer:
        answer = AgentAnswer(
            session_id=session,
            turn_id=turn,
            text=text,
            citations=citations,
            language=language,
            answer_kind=answer_kind,
            verification_status=verification_status,
            handoff=handoff,
            route=route,
            cost_usd=context.ledger.cost,
            latency_ms=(time.monotonic() - started) * 1000,
            grounding_score=score,
            policy_refusal=refusal,
            model_calls=context.ledger.calls,
        )
        context.audit("final_answer", answer.model_dump(mode="json"))
        return answer
