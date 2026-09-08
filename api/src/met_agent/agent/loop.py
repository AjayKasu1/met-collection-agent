"""Run bounded sequential tools and release only drafts that pass citation and claim checks."""

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid4

from pydantic import JsonValue

from met_agent.agent.events import AuditStore
from met_agent.agent.evidence import pack_evidence
from met_agent.agent.models import AgentAnswer, AgentDraft, ChatRequest, Citation, Language, Route
from met_agent.guardrails.grounding import GroundingCheck, valid_citations
from met_agent.guardrails.intent import Intent, normalize_intent
from met_agent.guardrails.interpretive import POLICY, UNVERIFIED
from met_agent.llm.chat import CURRENT_CALL, CallContext, ChatModel, ModelError, structured
from met_agent.llm.prompts import load_prompt, prompt_hash
from met_agent.tools.models import Evidence, Handoff
from met_agent.tools.registry import ToolRegistry

MAX_TOOL_CALLS = 6
MAX_TOOL_ROUNDS = 2
NOT_FOUND: dict[Language, str] = {
    "en": "The requested object was not found. The Met API returned no object record.",
    "fr": "L'objet demandé est introuvable. L'API du Met n'a renvoyé aucune notice d'objet.",
    "es": "No se encontró el objeto solicitado. La API del Met no devolvió ningún registro.",
    "zh": "未找到所请求的藏品。大都会艺术博物馆 API 未返回藏品记录。",
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
            async with lock:
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
                for name in ("system_v2", "tools_v1", "intent_v2", "grounding_v1", "citations_v1")
            },
        )
        try:
            model_intent = await structured(
                self.model,
                "lite",
                load_prompt("intent_v2"),
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
                audit(
                    "route_policy",
                    {
                        "policy": routing_policy,
                        "model_category": model_intent.category,
                        "model_difficulty": model_intent.difficulty,
                        "category": intent.category,
                        "difficulty": intent.difficulty,
                        "route": intent.route,
                        "search_query": intent.search_query,
                    },
                )
            if intent.category == "interpretive":
                audit("guardrail", {"policy": "non_interpretive", "decision": "refuse"})
                return self._answer(
                    session,
                    turn,
                    started,
                    intent.language,
                    intent.route,
                    context,
                    POLICY[intent.language],
                    [],
                    1.0,
                    refusal=True,
                )
            if intent.category == "out_of_scope":
                result = await self.tools.execute(
                    "handoff",
                    json.dumps(
                        {
                            "reason": (
                                "This request concerns account, purchase, or information "
                                "outside the "
                                "collection and visitor workspace."
                            ),
                            "suggested_contact": intent.handoff_contact,
                        }
                    ),
                )
                audit("tool_call", {"name": "handoff", "arguments": {"reason": "out_of_scope"}})
                audit("tool_result", result.model_dump(mode="json"))
                handoff = result.output if isinstance(result.output, Handoff) else None
                return self._answer(
                    session,
                    turn,
                    started,
                    intent.language,
                    intent.route,
                    context,
                    UNVERIFIED[intent.language],
                    [],
                    1.0,
                    handoff=handoff,
                )
            return await self._workspace(
                request, session, turn, started, intent, context, history, audit
            )
        except ModelError as error:
            audit("turn_error", {"code": error.code, "message": str(error)})
            raise
        finally:
            CURRENT_CALL.reset(token)
            await asyncio.to_thread(self.events.append_many, session, turn, buffered_events)

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
                            0.0,
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
                            1.0,
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
                score = check.score(evidence)
                audit(
                    "guardrail",
                    {
                        "policy": "grounding",
                        "score": score,
                        "result": check.model_dump(mode="json"),
                    },
                )
                if check.fully_supported and score == 1 and (draft.citations or not check.claims):
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
            UNVERIFIED[intent.language],
            [],
            0.0,
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
        score: float,
        *,
        refusal: bool = False,
        handoff: Handoff | None = None,
    ) -> AgentAnswer:
        answer = AgentAnswer(
            session_id=session,
            turn_id=turn,
            text=text,
            citations=citations,
            language=language,
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
