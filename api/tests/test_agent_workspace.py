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
from met_agent.guardrails.intent import (
    Intent,
    asks_for_first_message,
    greeting_language,
    normalize_intent,
    social_intent,
)
from met_agent.guardrails.interpretive import POLICY
from met_agent.ingestion.verify import read_golden
from met_agent.llm.chat import ModelError, Reply
from met_agent.llm.prompts import load_prompt, prompt_hash
from met_agent.retrieval.hybrid import (
    RetrievedChunk,
    RetrievedObject,
    ScoreBreakdown,
    SearchFilters,
)
from met_agent.tools.get_object import ObjectNotFound
from met_agent.tools.handoff import handoff
from met_agent.tools.models import (
    CollectionSearchResult,
    Evidence,
    GetDirectionsArguments,
    GetObjectArguments,
    Handoff,
    HandoffArguments,
    LiveObject,
    SearchCollectionArguments,
    SearchVisitorArguments,
    VisitorSearchResult,
    WayfindingResult,
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


@pytest.mark.parametrize(
    "message",
    [
        "What can I see in Gallery 131?",
        "What inside Gallery 131?",
        "What's inside Gallery 131?",
        "What is inside the Gallery 131?",
        "What is there in Gallery 131?",
        "What is displayed in Gallery 131?",
        "Show me the objects in Gallery 131.",
        "What is in gallery #131?",
        "Which artworks are on view in the Gallery 131?",
    ],
)
def test_direct_gallery_questions_have_a_stable_lite_route(message: str) -> None:
    model_intent = Intent(
        category="collection",
        difficulty="simple",
        operation="gallery_inventory",
        gallery_number="131",
        language="en",
        search_query="Objects in Gallery 131",
    )
    normalized, policy = normalize_intent(model_intent, message)
    assert policy == "direct_gallery_question"
    assert normalized.category == "collection"
    assert normalized.difficulty == "simple"
    assert normalized.search_query == "Objects in Gallery 131"
    assert normalized.route == "lite"


def test_simple_visitor_facts_use_lite_while_complex_requests_use_main() -> None:
    simple = Intent.model_validate(
        {
            **intent(category="visitor_info"),
            "search_query": "large luggage and wine coat-check policy",
        }
    )
    complex_request = simple.model_copy(update={"difficulty": "complex"})

    assert simple.route == "lite"
    assert complex_request.route == "main"


@pytest.mark.parametrize(
    ("message", "language"),
    [
        ("hi", "en"),
        ("Hello!", "en"),
        ("Bonjour", "fr"),
        ("¡Hola!", "es"),
        ("你好!", "zh"),
    ],
)
def test_complete_greetings_are_detected(message: str, language: Language) -> None:
    assert greeting_language(message) == language


@pytest.mark.parametrize(
    "message",
    [
        "Hi, is the museum open Wednesday?",
        "Hello, where is Gallery 131?",
        "Bonjour, quels sont les horaires ?",
    ],
)
def test_greetings_do_not_swallow_museum_questions(message: str) -> None:
    assert greeting_language(message) is None
    assert social_intent(message) is None


@pytest.mark.parametrize(
    ("message", "kind", "language"),
    [
        ("How are you?", "wellbeing", "en"),
        ("How's it going?", "wellbeing", "en"),
        ("Merci!", "thanks", "fr"),
        ("Adiós", "farewell", "es"),
    ],
)
def test_bounded_social_turns_are_detected(message: str, kind: str, language: Language) -> None:
    assert social_intent(message) == (kind, language)


def test_greeting_short_circuits_models_and_tools(tmp_path: Path) -> None:
    model = ScriptedModel([])
    store = EventStore(tmp_path / "greeting.sqlite3")
    answer = asyncio.run(Agent(model, ToolRegistry(), store).run(ChatRequest(message="Hi!")))

    assert answer.text.startswith("Hello!")
    assert answer.route == "lite"
    assert answer.language == "en"
    assert answer.grounding_score == 1
    assert answer.citations == []
    assert answer.model_calls == []
    assert answer.cost_usd == 0
    assert model.calls == []
    assert any(
        event.kind == "route_policy"
        and isinstance(event.data, dict)
        and event.data.get("policy") == "deterministic_social_turn"
        for event in store.read(answer.session_id)
    )


def test_wellbeing_short_circuits_models_and_tools(tmp_path: Path) -> None:
    model = ScriptedModel([])
    store = EventStore(tmp_path / "wellbeing.sqlite3")
    answer = asyncio.run(
        Agent(model, ToolRegistry(), store).run(ChatRequest(message="How are you?"))
    )

    assert answer.text == "I'm ready to help. What would you like to know about The Met?"
    assert answer.route == "lite"
    assert answer.grounding_score == 1
    assert answer.citations == []
    assert answer.model_calls == []
    assert answer.cost_usd == 0
    assert model.calls == []


@pytest.mark.parametrize(
    "message",
    [
        "What did I ask first?",
        "what i asked first",
        "What was my first question?",
    ],
)
def test_first_message_recall_is_bounded(message: str) -> None:
    assert asks_for_first_message(message)


def test_first_message_recall_does_not_swallow_compound_questions() -> None:
    assert not asks_for_first_message("What did I ask first, and when does the museum close?")


def test_first_message_recall_uses_only_the_current_session(tmp_path: Path) -> None:
    model = ScriptedModel([])
    store = EventStore(tmp_path / "recall.sqlite3")
    agent = Agent(model, ToolRegistry(), store)
    first = asyncio.run(agent.run(ChatRequest(message="Hi!")))
    other = asyncio.run(agent.run(ChatRequest(message="Bonjour!")))
    recalled = asyncio.run(
        agent.run(ChatRequest(message="What did I ask first?", session_id=first.session_id))
    )

    assert recalled.text == 'You first asked: "Hi!"'
    assert recalled.model_calls == []
    assert recalled.citations == []
    assert recalled.grounding_score == 1
    assert store.first_user_message(other.session_id) == "Bonjour!"


def test_simple_visitor_facts_bypass_model_tool_selection(tmp_path: Path) -> None:
    source_url = "https://www.metmuseum.org/policies/visitor-guidelines"
    policy = (
        "The following items are not allowed in the building or at coat check:\n"
        "- Large bags, luggage\n"
        "- Glass containers and liquids other than water"
    )
    searches: list[SearchVisitorArguments] = []

    def search(arguments: SearchVisitorArguments) -> VisitorSearchResult:
        searches.append(arguments)
        return VisitorSearchResult(
            chunks=[
                RetrievedChunk(
                    point_id="policy",
                    source_url=source_url,
                    page_title="Visitor Guidelines",
                    section_heading="Policy",
                    fetched_at="2026-09-04T10:11:54Z",
                    text=policy,
                    scores=ScoreBreakdown(dense=0.9, sparse=0.8, rrf=0.7, rerank=0.95),
                )
            ]
        )

    registry = ToolRegistry()
    registry.register(
        "search_visitor_info",
        "Visitor search",
        SearchVisitorArguments,
        VisitorSearchResult,
        search,
    )
    decision = {
        **intent(category="visitor_info"),
        "search_query": "large luggage and wine coat-check policy",
    }
    answer_text = "No. Large luggage and wine are not allowed in the building or at coat check."
    model = ScriptedModel(
        [
            decision,
            {
                "text": answer_text,
                "language": "en",
                "citations": [{"source_url": source_url, "quote": policy}],
            },
            {
                "fully_supported": True,
                "claims": [
                    {
                        "text": answer_text,
                        "supported": True,
                        "evidence_keys": ["page:policy"],
                    }
                ],
            },
        ]
    )
    store = EventStore(tmp_path / "visitor-fast-path.sqlite3")

    answer = asyncio.run(
        Agent(model, registry, store).run(
            ChatRequest(
                message=(
                    "Can I bring a large suitcase and wine inside if I leave them at coat check?"
                )
            )
        )
    )

    assert answer.text == answer_text
    assert answer.route == "lite" and answer.grounding_score == 1
    assert [call[0] for call in model.calls] == ["lite", "lite", "lite"]
    assert all(call[2] is None for call in model.calls)
    assert searches == [
        SearchVisitorArguments(query="large luggage and wine coat-check policy", k=5)
    ]
    events = store.read(answer.session_id)
    assert len([event for event in events if event.kind == "tool_call"]) == 1
    assert any(
        event.kind == "route_policy"
        and isinstance(event.data, dict)
        and event.data.get("policy") == "direct_visitor_search"
        for event in events
    )


def test_gallery_normalization_does_not_collapse_complex_or_policy_requests() -> None:
    collection = Intent.model_validate(intent())
    normalized, policy = normalize_intent(
        collection, "Compare Gallery 131 and Gallery 132, then plan my route"
    )
    assert normalized == collection and policy is None
    interpretive = collection.model_copy(
        update={"category": "interpretive", "difficulty": "complex"}
    )
    normalized, policy = normalize_intent(interpretive, "What is in Gallery 131?")
    assert normalized == interpretive and policy is None


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"category": "interpretive"}, "What inside gallery 131?"),
        ({"category": "out_of_scope"}, "What inside gallery 131?"),
        ({"difficulty": "complex"}, "Compare the sculptures inside 131"),
        ({"operation": "general"}, "Which bronze objects are in room 131?"),
        ({"constraints": ["bronze", "sculptures"]}, "Which bronze sculptures are in room 131?"),
        ({"language": "fr"}, "Que voir dans la salle 131 ?"),
        ({"gallery_number": "132"}, "what inside gallery 131"),
        ({}, "Compare gallery 131 with gallery 132"),
        ({}, "what is there"),
        ({}, "object 547802"),
        ({}, "gimme info on 131"),
        ({}, "gallery 131.5"),
    ],
)
def test_semantic_routing_rejects_unsupported_or_unanchored_decisions(
    updates: dict[str, Any], message: str
) -> None:
    decision = Intent.model_validate(
        {**intent(), "operation": "gallery_inventory", "gallery_number": "131", **updates}
    )
    normalized, policy = normalize_intent(decision, message)
    assert normalized == decision and policy is None


def test_unverified_numeric_general_route_emits_review_event(tmp_path: Path) -> None:
    model = ScriptedModel([intent(), draft(object_id=999), draft(object_id=999)])
    store = EventStore(tmp_path / "deviation.sqlite3")
    answer = asyncio.run(
        Agent(model, registry_with_calls([]), store).run(
            ChatRequest(message="gimme info on room 131")
        )
    )
    assert answer.grounding_score == 0
    events = [e for e in store.read(answer.session_id) if e.kind == "routing_deviation"]
    assert len(events) == 1
    assert isinstance(events[0].data, dict)
    assert events[0].data["session_id"] == str(answer.session_id)


@pytest.mark.parametrize(
    "message",
    [
        "How do I get to Gallery 131 from the Fifth Avenue entrance?",
        "How can I walk from the main entrance to gallery #131?",
        "Directions to Gallery 131 from the main entrance, please.",
    ],
)
def test_direct_gallery_wayfinding_has_a_stable_lite_route(message: str) -> None:
    model_intent = Intent(
        category="visitor_info",
        difficulty="simple",
        operation="gallery_wayfinding",
        gallery_number="131",
        origin="fifth_avenue_entrance",
        language="en",
        search_query="route to room",
    )
    normalized, policy = normalize_intent(model_intent, message)
    assert policy == "direct_gallery_wayfinding"
    assert normalized.category == "visitor_info"
    assert normalized.difficulty == "simple"
    assert normalized.search_query == "Directions from the Fifth Avenue entrance to Gallery 131"


@pytest.mark.parametrize(
    "message",
    [
        "How do I get to Gallery 131?",
        "how to go gallery 131",
        "directions for gallery 131",
    ],
)
def test_gallery_wayfinding_discloses_default_when_origin_is_omitted(message: str) -> None:
    model_intent = Intent(
        category="visitor_info",
        difficulty="simple",
        operation="gallery_wayfinding",
        gallery_number="131",
        origin="unspecified",
        language="en",
        search_query="route to room",
    )
    normalized, policy = normalize_intent(model_intent, message)
    assert policy == "direct_gallery_wayfinding_default_origin"
    assert normalized.search_query == "Directions from the Fifth Avenue entrance to Gallery 131"


def test_direct_gallery_wayfinding_uses_one_live_map_call(tmp_path: Path) -> None:
    registry = ToolRegistry()
    executed: list[GetDirectionsArguments] = []
    source_url = "https://maps.metmuseum.org/navigate/" + "a" * 32 + "/" + "b" * 32

    def directions(arguments: GetDirectionsArguments) -> WayfindingResult:
        executed.append(arguments)
        return WayfindingResult(
            requested_origin="Fifth Avenue entrance",
            origin="The Great Hall",
            destination="Gallery 131",
            floor="Floor 1",
            distance_metres=178,
            distance_feet=584,
            duration_minutes=2,
            source_url=source_url,
            fetched_at="2026-09-08T00:00:00Z",
            text=(
                "The Met Interactive Map route\nRequested origin: Fifth Avenue entrance\n"
                "Start: The Great Hall\n"
                "Destination: Gallery 131\nFloor: Floor 1\n"
                "Estimated walking time: 2 minutes\nDistance: 178 metres (584 feet)"
            ),
        )

    registry.register(
        "get_directions",
        "Official route",
        GetDirectionsArguments,
        WayfindingResult,
        directions,
    )
    model_decision = {
        **intent(category="visitor_info"),
        "operation": "gallery_wayfinding",
        "gallery_number": "131",
        "origin": "fifth_avenue_entrance",
    }
    store = EventStore(tmp_path / "wayfinding.sqlite3")
    model = ScriptedModel([model_decision])
    answer = asyncio.run(
        Agent(model, registry, store).run(
            ChatRequest(message="How do I get to Gallery 131 from the Fifth Avenue entrance?")
        )
    )
    assert answer.route == "lite" and answer.grounding_score == 1
    assert answer.text == (
        "The Met's official map route starts at The Great Hall and continues on Floor 1 to "
        "Gallery 131, about 584 feet (2 minutes). Open the cited live route before you start."
    )
    assert answer.citations[0].source_url == source_url
    assert executed == [
        GetDirectionsArguments(origin="fifth_avenue_entrance", destination_gallery="131")
    ]
    assert [call[0] for call in model.calls] == ["lite"]
    events = store.read(answer.session_id)
    assert len([event for event in events if event.kind == "tool_call"]) == 1
    assert any(
        event.kind == "route_policy"
        and isinstance(event.data, dict)
        and event.data.get("policy") == "direct_gallery_wayfinding"
        for event in events
    )


def test_gallery_wayfinding_default_is_disclosed_and_grounded(tmp_path: Path) -> None:
    registry = ToolRegistry()
    source_url = "https://maps.metmuseum.org/navigate/" + "a" * 32 + "/" + "b" * 32

    def directions(_: GetDirectionsArguments) -> WayfindingResult:
        return WayfindingResult(
            requested_origin="Fifth Avenue entrance",
            origin="The Great Hall",
            destination="Gallery 131",
            floor="Floor 1",
            distance_metres=178,
            distance_feet=584,
            duration_minutes=2,
            source_url=source_url,
            fetched_at="2026-09-08T00:00:00Z",
            text=(
                "The Met Interactive Map route\nRequested origin: Fifth Avenue entrance\n"
                "Start: The Great Hall\n"
                "Destination: Gallery 131\nFloor: Floor 1\n"
                "Estimated walking time: 2 minutes\nDistance: 178 metres (584 feet)"
            ),
        )

    registry.register(
        "get_directions",
        "Official route",
        GetDirectionsArguments,
        WayfindingResult,
        directions,
    )
    model = ScriptedModel(
        [
            {
                **intent(category="visitor_info"),
                "operation": "gallery_wayfinding",
                "gallery_number": "131",
                "origin": "unspecified",
            }
        ]
    )
    answer = asyncio.run(
        Agent(model, registry, EventStore(tmp_path / "wayfinding-default.sqlite3")).run(
            ChatRequest(message="directions for gallery 131")
        )
    )
    assert answer.route == "lite" and answer.grounding_score == 1
    assert answer.text.startswith(
        "I will use the Fifth Avenue entrance as your starting assumption. "
        "The Met's official map route starts at The Great Hall"
    )
    assert answer.citations[0].source_url == source_url
    assert [call[0] for call in model.calls] == ["lite"]


def test_gallery_wayfinding_fails_closed_when_evidence_disagrees(tmp_path: Path) -> None:
    registry = ToolRegistry()
    source_url = "https://maps.metmuseum.org/navigate/" + "a" * 32 + "/" + "b" * 32

    def directions(_: GetDirectionsArguments) -> WayfindingResult:
        return WayfindingResult(
            requested_origin="Fifth Avenue entrance",
            origin="The Great Hall",
            destination="Gallery 131",
            floor="Floor 1",
            distance_metres=178,
            distance_feet=584,
            duration_minutes=2,
            source_url=source_url,
            fetched_at="2026-09-08T00:00:00Z",
            text=(
                "The Met Interactive Map route\nRequested origin: Fifth Avenue entrance\n"
                "Start: The Great Hall\nDestination: Gallery 131\nFloor: Floor 1\n"
                "Estimated walking time: 2 minutes\nDistance: 178 metres (999 feet)"
            ),
        )

    registry.register(
        "get_directions",
        "Official route",
        GetDirectionsArguments,
        WayfindingResult,
        directions,
    )
    model = ScriptedModel(
        [
            {
                **intent(category="visitor_info"),
                "operation": "gallery_wayfinding",
                "gallery_number": "131",
                "origin": "unspecified",
            }
        ]
    )
    answer = asyncio.run(
        Agent(model, registry, EventStore(tmp_path / "wayfinding-mismatch.sqlite3")).run(
            ChatRequest(message="directions for gallery 131")
        )
    )
    assert answer.text == (
        "I couldn't verify that from the available museum sources. Please try a more "
        "specific question or contact info@metmuseum.org."
    )
    assert answer.citations == [] and answer.grounding_score == 0
    assert [call[0] for call in model.calls] == ["lite"]


def test_direct_gallery_wayfinding_fails_closed_without_map_evidence(tmp_path: Path) -> None:
    registry = ToolRegistry()

    def unavailable(_: GetDirectionsArguments) -> WayfindingResult:
        raise RuntimeError("private upstream detail")

    registry.register(
        "get_directions",
        "Official route",
        GetDirectionsArguments,
        WayfindingResult,
        unavailable,
    )
    model = ScriptedModel(
        [
            {
                **intent(category="visitor_info"),
                "operation": "gallery_wayfinding",
                "gallery_number": "131",
                "origin": "fifth_avenue_entrance",
            }
        ]
    )
    answer = asyncio.run(
        Agent(model, registry, EventStore(tmp_path / "wayfinding-failure.sqlite3")).run(
            ChatRequest(message="Directions to Gallery 131 from the Fifth Avenue entrance")
        )
    )
    assert answer.grounding_score == 0 and not answer.citations
    assert (
        answer.text
        == "I couldn't verify that from the available museum sources. Please try a more "
        "specific question or contact info@metmuseum.org."
    )
    assert "private upstream detail" not in answer.model_dump_json()


def test_direct_gallery_policy_is_audited_and_used(tmp_path: Path) -> None:
    model_decision = {
        **intent(),
        "operation": "gallery_inventory",
        "gallery_number": "131",
        "search_query": "Objects in Gallery 131",
    }
    executed: list[int] = []
    registry = registry_with_calls(executed)

    def gallery_search(arguments: SearchCollectionArguments) -> CollectionSearchResult:
        assert arguments.query == "Objects in Gallery 131"
        assert arguments.filters == SearchFilters(gallery_number="131")
        return CollectionSearchResult(
            objects=[
                RetrievedObject(
                    object_id=1,
                    title="Temple",
                    source_url="https://www.metmuseum.org/art/collection/search/1",
                    text="Temple\ngallery_number: 131",
                    record={"gallery_number": "131"},
                    scores=ScoreBreakdown(rrf=1 / 61, rerank=1.0),
                )
            ]
        )

    registry.register(
        "search_collection",
        "Collection search",
        SearchCollectionArguments,
        CollectionSearchResult,
        gallery_search,
    )
    model = ScriptedModel([model_decision, judgment()])
    store = EventStore(tmp_path / "gallery-route.sqlite3")
    answer = asyncio.run(
        Agent(model, registry, store).run(ChatRequest(message="What can I see in Gallery 131?"))
    )
    assert answer.route == "lite" and answer.grounding_score == 1
    assert answer.text == "The live Met record currently lists Temple as on view in Gallery 131."
    assert answer.citations[0].object_id == 1 and executed == [1]
    assert [call[0] for call in model.calls] == ["lite"] * 2
    policy_events = [
        event for event in store.read(answer.session_id) if event.kind == "route_policy"
    ]
    assert len(policy_events) == 1
    assert policy_events[0].data == {
        "policy": "direct_gallery_question",
        "model_category": "collection",
        "model_difficulty": "simple",
        "category": "collection",
        "difficulty": "simple",
        "route": "lite",
        "search_query": "Objects in Gallery 131",
    }
    assert any(
        event.kind == "guardrail"
        and isinstance(event.data, dict)
        and event.data.get("policy") == "deterministic_gallery_inventory"
        for event in store.read(answer.session_id)
    )


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
            intent(category="multi_hop"),
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
