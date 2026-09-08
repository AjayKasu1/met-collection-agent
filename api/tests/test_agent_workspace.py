"""Verify bounded tools, multilingual policy, citation isolation, and fail-closed answers."""

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel, ValidationError

from met_agent.agent.events import EventStore
from met_agent.agent.loop import Agent
from met_agent.agent.models import AgentDraft, ChatRequest, Citation, Language, Route
from met_agent.guardrails.grounding import GroundingCheck, valid_citations
from met_agent.guardrails.interpretive import POLICY
from met_agent.ingestion.verify import read_golden
from met_agent.llm.chat import ModelError, Reply
from met_agent.llm.prompts import load_prompt, prompt_hash
from met_agent.tools.get_object import ObjectNotFound
from met_agent.tools.handoff import handoff
from met_agent.tools.models import (
    Evidence,
    GetObjectArguments,
    Handoff,
    HandoffArguments,
    LiveObject,
)
from met_agent.tools.registry import ToolRegistry


class ScriptedModel:
    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[Route, list[dict[str, Any]], object]] = []

    async def complete(
        self,
        route: Route,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, object]] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> Reply:
        self.calls.append((route, json.loads(json.dumps(messages)), tools))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Reply(
            reply
            if isinstance(reply, dict) and "role" in reply
            else {"role": "assistant", "content": json.dumps(reply)}
        )


def intent(language: Language = "en", category: str = "collection") -> dict[str, str]:
    return {
        "category": category,
        "difficulty": "simple",
        "language": language,
        "search_query": "Temple",
    }


def calls(*names: str, arguments: str | dict[str, int] = '{"object_id":1}') -> dict[str, Any]:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": f"call-{i}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
            for i, name in enumerate(names)
        ],
    }


def draft(language: Language = "en", object_id: int = 1) -> dict[str, Any]:
    return {
        "text": "Temple",
        "language": language,
        "citations": [{"object_id": object_id, "quote": "Temple"}],
    }


def judgment(supported: bool = True) -> dict[str, Any]:
    return {
        "fully_supported": supported,
        "claims": [{"text": "Temple", "supported": True, "evidence_keys": ["object:1"]}]
        if supported
        else [
            {"text": "Temple", "supported": True, "evidence_keys": ["object:1"]},
            {"text": "Unsupported date", "supported": False, "evidence_keys": []},
        ],
    }


def live_object(object_id: int = 1) -> LiveObject:
    return LiveObject(
        object_id=object_id,
        title="Temple",
        artist="",
        culture="",
        medium="Stone",
        object_date="",
        department="",
        gallery_number="131",
        is_on_view=True,
        is_public_domain=True,
        primary_image="",
        primary_image_small="",
        source_url=f"https://www.metmuseum.org/art/collection/search/{object_id}",
        fetched_at="2026-09-04T00:00:00Z",
        text="Temple\nStone\ngallery_number: 131",
    )


def registry_with_calls(executed: list[int]) -> ToolRegistry:
    registry = ToolRegistry()

    def get(value: GetObjectArguments) -> LiveObject:
        executed.append(value.object_id)
        return live_object(value.object_id)

    registry.register("get_object", "Live facts", GetObjectArguments, LiveObject, get)
    registry.register("handoff", "Terminal contact", HandoffArguments, Handoff, handoff)
    return registry


@pytest.mark.parametrize("language", ["en", "fr", "es", "zh"])
def test_verified_answer_and_current_turn_citations(tmp_path: Path, language: Language) -> None:
    executed: list[int] = []
    model = ScriptedModel([intent(language), calls("get_object"), draft(language), judgment()])
    store = EventStore(tmp_path / "events.sqlite3")
    agent = Agent(model, registry_with_calls(executed), store)
    answer = asyncio.run(agent.run(ChatRequest(message="Temple", language=language)))
    assert answer.language == language and answer.route == "lite" and answer.grounding_score == 1
    assert answer.citations[0].object_id == 1 and executed == [1]
    assert [call[0] for call in model.calls] == ["lite"] * 4
    assert not agent._sessions
    events = store.read(answer.session_id)
    assert events[0].kind == "user_message" and events[-1].kind == "final_answer"
    assert len(store.history(answer.session_id)) == 2
    # An earlier turn's valid object is not citation authority for a new turn.
    model.replies.extend([intent(language), draft(language), draft(language)])
    failed = asyncio.run(agent.run(ChatRequest(message="Repeat", session_id=answer.session_id)))
    assert failed.grounding_score == 0 and not failed.citations
    assert executed == [1]


@pytest.mark.parametrize(
    "bad",
    [
        draft(object_id=999),
        {
            "text": "Temple",
            "language": "en",
            "citations": [{"source_url": "https://evil.test", "quote": "Temple"}],
        },
        {"role": "assistant", "content": "not JSON"},
        draft("fr"),
    ],
)
def test_invalid_draft_regenerates_once_without_unseen_citations(
    tmp_path: Path, bad: dict[str, Any]
) -> None:
    model = ScriptedModel([intent(), calls("get_object"), bad, draft(), judgment()])
    agent = Agent(model, registry_with_calls([]), EventStore(tmp_path / "audit.sqlite3"))
    answer = asyncio.run(agent.run(ChatRequest(message="Temple")))
    assert answer.grounding_score == 1 and answer.citations[0].object_id == 1
    assert "evil.test" not in json.dumps(model.calls[3][1])
    assert "999" not in json.dumps(model.calls[3][1])


def test_unsupported_claim_rewritten_and_scored(tmp_path: Path) -> None:
    model = ScriptedModel(
        [intent(), calls("get_object"), draft(), judgment(False), draft(), judgment()]
    )
    store = EventStore(tmp_path / "audit.sqlite3")
    result = asyncio.run(
        Agent(model, registry_with_calls([]), store).run(ChatRequest(message="Facts"))
    )
    scores = [
        event.data["score"]
        for event in store.read(result.session_id)
        if isinstance(event.data, dict) and "score" in event.data
    ]
    assert scores == [0.5, 1.0]
    evidence = [Evidence(key="object:1", object_id=1, text="Temple", kind="collection")]
    assert GroundingCheck.model_validate(judgment(False)).score(evidence) == 0.5
    assert GroundingCheck.model_validate(judgment()).score([]) == 0
    assert GroundingCheck(fully_supported=False, claims=[]).score([]) == 0


@pytest.mark.parametrize("invalid", [True, False])
def test_six_attempt_budget_and_sequential_execution(tmp_path: Path, invalid: bool) -> None:
    executed: list[int] = []
    model = ScriptedModel(
        [
            intent(),
            calls(
                *(["get_object"] * 7), arguments='{"object_id":0}' if invalid else '{"object_id":1}'
            ),
        ]
    )
    store = EventStore(tmp_path / "audit.sqlite3")
    result = asyncio.run(
        Agent(model, registry_with_calls(executed), store).run(ChatRequest(message="many"))
    )
    assert len(executed) == (0 if invalid else 6)
    assert result.grounding_score == 0
    events = store.read(result.session_id)
    assert len([e for e in events if e.kind == "tool_call"]) == 6
    assert any(e.kind == "tool_limit" for e in events)


def test_invalid_argument_feedback_and_terminal_handoff(tmp_path: Path) -> None:
    executed: list[int] = []
    model = ScriptedModel(
        [
            intent(),
            calls("get_object", arguments="{}"),
            calls("get_object", arguments={"object_id": 1}),
            draft(),
            judgment(),
        ]
    )
    agent = Agent(model, registry_with_calls(executed), EventStore(tmp_path / "audit.sqlite3"))
    answer = asyncio.run(agent.run(ChatRequest(message="Temple")))
    assert answer.grounding_score == 1
    assert "invalid_arguments" in json.dumps(model.calls[2][1])
    assert model.calls[1][2] is not None and model.calls[2][2] is not None
    assert model.calls[3][2] is None
    assert any(
        event.kind == "tool_round_limit"
        and isinstance(event.data, dict)
        and event.data.get("decision") == "finalize_with_current_evidence"
        for event in agent.events.read(answer.session_id)
    )
    model.replies.extend(
        [
            intent(),
            calls(
                "handoff",
                "get_object",
                arguments='{"reason":"account","suggested_contact":"info@metmuseum.org"}',
            ),
        ]
    )
    answer = asyncio.run(agent.run(ChatRequest(message="account")))
    assert answer.handoff and executed == [1]
    model.replies.append(intent(category="out_of_scope"))
    answer = asyncio.run(agent.run(ChatRequest(message="purchase")))
    assert answer.handoff and answer.route == "main"


REF_ROWS = [
    row
    for row in read_golden(Path(__file__).parents[2] / "evals/golden.jsonl")
    if row.id.startswith("ref-")
]


@pytest.mark.parametrize("row", REF_ROWS, ids=lambda row: row.id)
def test_golden_interpretive_contract(tmp_path: Path, row: Any) -> None:
    # Deterministic responses verify policy wiring, not live classifier accuracy.
    model = ScriptedModel([intent(category="interpretive")])
    executed: list[int] = []
    answer = asyncio.run(
        Agent(model, registry_with_calls(executed), EventStore(tmp_path / "audit.sqlite3")).run(
            ChatRequest(message=row.question)
        )
    )
    assert answer.policy_refusal and answer.text == POLICY["en"] and not executed
    assert row.question in str(model.calls[0][1])
    assert "interpretive" in load_prompt("intent_v1")


def test_model_error_malformed_envelope_and_honest_no_claims(tmp_path: Path) -> None:
    for reply in (
        ModelError("access_denied", "Provider denied"),
        {"role": "assistant", "tool_calls": ["invalid"]},
    ):
        store = EventStore(tmp_path / (str(uuid4()) + ".sqlite3"))
        agent = Agent(ScriptedModel([intent(), reply]), registry_with_calls([]), store)
        session = uuid4()
        with pytest.raises(ModelError):
            asyncio.run(agent.run(ChatRequest(message="Temple", session_id=session)))
        assert store.read(session)[-1].kind == "turn_error"
    model = ScriptedModel(
        [
            intent(category="visitor_info"),
            {"text": "I have no matching information.", "language": "en", "citations": []},
            {"fully_supported": True, "claims": []},
        ]
    )
    answer = asyncio.run(
        Agent(model, registry_with_calls([]), EventStore(tmp_path / "empty.sqlite3")).run(
            ChatRequest(message="hours")
        )
    )
    assert answer.route == "main" and answer.grounding_score == 1


def test_confirmed_missing_object_returns_verified_answer_without_third_model_call(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()

    def missing(_: GetObjectArguments) -> LiveObject:
        raise ObjectNotFound(
            "missing",
            evidence=Evidence(
                key="lookup:999",
                source_url="https://collectionapi.metmuseum.org/public/collection/v1/objects/999",
                text=(
                    "Met API lookup for Object ID 999: not found (HTTP 404).\n"
                    "No object record was returned for this requested ID."
                ),
                kind="lookup_status",
            ),
        )

    registry.register("get_object", "Lookup", GetObjectArguments, LiveObject, missing)
    model = ScriptedModel([intent(), calls("get_object", arguments='{"object_id":999}')])
    store = EventStore(tmp_path / "missing.sqlite3")
    answer = asyncio.run(
        Agent(model, registry, store).run(ChatRequest(message="What gallery is object 999 in?"))
    )
    assert (
        answer.text == "The requested object was not found. The Met API returned no object record."
    )
    assert answer.grounding_score == 1 and len(answer.citations) == 1
    assert len(model.calls) == 2
    assert any(
        event.kind == "guardrail" and event.data.get("policy") == "deterministic_lookup_status"
        for event in store.read(answer.session_id)
        if isinstance(event.data, dict)
    )


def test_quote_identity_and_append_only_redacted_audit(tmp_path: Path) -> None:
    for value in (
        {"quote": "Temple"},
        {"object_id": 1, "source_url": "https://met.test", "quote": "Temple"},
        {"object_id": 1, "quote": " "},
    ):
        with pytest.raises(ValidationError):
            Citation.model_validate(value)
    with pytest.raises(ValidationError):
        ChatRequest(message="  ")
    evidence = [
        Evidence(
            key="page:1",
            source_url="https://www.metmuseum.org/plan-your-visit",
            text="Open\n daily",
            kind="visitor_info",
        )
    ]
    assert valid_citations(
        AgentDraft(
            text="Open",
            language="en",
            citations=[Citation(source_url=evidence[0].source_url, quote="Open daily")],
        ),
        evidence,
    )
    assert not valid_citations(
        AgentDraft(text="Temple", language="en", citations=[Citation(object_id=1, quote="wrong")]),
        evidence,
    )
    store = EventStore(tmp_path / "audit.sqlite3", secrets=["private-value"])
    session, turn = uuid4(), uuid4()
    store.append(
        session,
        turn,
        "test",
        {
            "message": "private-value",
            "authorization": "hidden",
            "nested": ["private-value", True],
            "input_tokens": 10,
        },
    )
    events = store.read(session)
    assert "private-value" not in events[0].model_dump_json()
    assert events[0].data == {
        "message": "[REDACTED]",
        "authorization": "[REDACTED]",
        "nested": ["[REDACTED]", True],
        "input_tokens": 10,
    }
    assert store.read(uuid4()) == [] and store.read(session, after=events[0].sequence) == []
    for kwargs in ({"after": -1}, {"limit": 0}, {"limit": 501}):
        with pytest.raises(ValueError):
            store.read(session, **kwargs)
    with sqlite3.connect(store.path) as connection:
        for command in ("UPDATE events SET kind='changed'", "DELETE FROM events"):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(command)
    assert len(prompt_hash("system_v1")) == 64
    with pytest.raises(ValueError):
        load_prompt("../../private")  # type: ignore[arg-type]


def test_out_of_scope_uses_bounded_retail_contact(tmp_path: Path) -> None:
    decision = {**intent(category="out_of_scope"), "handoff_contact": "store.support@metmuseum.org"}
    model = ScriptedModel([decision])
    store = EventStore(tmp_path / "audit.sqlite3")
    answer = asyncio.run(
        Agent(model, registry_with_calls([]), store).run(
            ChatRequest(message="An order delivery question")
        )
    )
    assert answer.handoff and answer.handoff.suggested_contact == "store.support@metmuseum.org"
    assert any(e.kind == "tool_call" for e in store.read(answer.session_id))
    assert len(model.calls) == 1
