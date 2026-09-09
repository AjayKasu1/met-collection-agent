"""Exercise HTTP SSE, audit access, official MCP tool dispatch, and resource ownership."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from mcp.types import CallToolResult
from qdrant_client import QdrantClient
from test_agent_workspace import (
    ScriptedModel,
    calls,
    draft,
    intent,
    judgment,
    live_object,
    registry_with_calls,
)

from met_agent.agent.events import EventStore
from met_agent.agent.loop import Agent
from met_agent.agent.models import AgentAnswer, ChatRequest
from met_agent.config import Settings
from met_agent.llm.chat import ModelError
from met_agent.main import create_app
from met_agent.mcp.server import create_server
from met_agent.retrieval.hybrid import RetrievedChunk, ScoreBreakdown
from met_agent.retrieval.service import SearchService
from met_agent.tools.get_directions import WayfindingClient
from met_agent.tools.get_object import LiveObjectClient, ObjectNotFound
from met_agent.tools.models import (
    GetObjectArguments,
    SearchVisitorArguments,
    VisitorSearchResult,
)
from met_agent.tools.registry import create_registry
from scripts.demo_check import parse_answer
from scripts.demo_check import run as demo_run


class Service:
    def __init__(self, tmp_path: Path, replies: list[Any]) -> None:
        self.events = EventStore(tmp_path / "events.sqlite3")
        self.model = ScriptedModel(replies)
        registry = registry_with_calls([])
        registry.register(
            "search_visitor_info",
            "Visitor search",
            SearchVisitorArguments,
            VisitorSearchResult,
            self.search_visitor,
        )
        self.agent = Agent(self.model, registry, self.events)
        self.live = SimpleNamespace(get=self.get)
        self.closed = False
        self.failure: Exception | None = None

    def get(self, args: GetObjectArguments) -> Any:
        if args.object_id == 404:
            raise ObjectNotFound("Absent")
        if args.object_id == 500:
            raise RuntimeError("private-token")
        return live_object(args.object_id)

    def search_visitor(self, _: SearchVisitorArguments) -> VisitorSearchResult:
        return VisitorSearchResult(
            chunks=[
                RetrievedChunk(
                    point_id="fixture",
                    source_url="https://www.metmuseum.org/plan-your-visit",
                    page_title="Plan Your Visit",
                    section_heading="Hours",
                    fetched_at="2026-09-04T00:00:00Z",
                    text="Temple",
                    scores=ScoreBreakdown(dense=1, sparse=1, rrf=1, rerank=1),
                )
            ]
        )

    async def chat(self, request: ChatRequest) -> AgentAnswer:
        if self.failure:
            raise self.failure
        return await self.agent.run(request)

    def close(self) -> None:
        self.closed = True


def test_verified_sse_audit_and_live_proxy(settings: Settings, tmp_path: Path) -> None:
    service = Service(
        tmp_path, [intent(), calls("get_object"), draft(object_id=999), draft(), judgment()]
    )
    app = create_app(settings)
    app.state.runtime = service
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "Temple"})
        assert response.headers["content-type"].startswith("text/event-stream")
        answer = parse_answer(response.text)
        assert all(citation.object_id != 999 for citation in answer.citations)
        pieces = [
            json.loads(block.split("data: ", 1)[1])["text"]
            for block in response.text.split("\n\n")
            if block.startswith("event: token")
        ]
        assert "999" not in "".join(pieces)
        assert "".join(pieces) == answer.text
        events = client.get(f"/sessions/{answer.session_id}/events").json()
        assert events[-1]["kind"] == "final_answer"
        assert (
            client.get(
                f"/sessions/{answer.session_id}/events", params={"after": events[-1]["sequence"]}
            ).json()
            == []
        )
        assert client.get(f"/sessions/{uuid4()}/events").status_code == 404
        assert client.get(f"/sessions/{answer.session_id}/events?limit=501").status_code == 422
        assert client.get("/sessions/not-an-id/events").status_code == 422
        assert client.post("/chat", json={"message": " "}).status_code == 422
        assert client.get("/objects/1").json()["gallery_number"] == "131"
        assert client.get("/objects/0").status_code == 422
        assert client.get("/objects/404").status_code == 404
        assert client.get("/objects/500").status_code == 503
        for error, expected in (
            (ModelError("access_denied", "Provider denied"), "access_denied"),
            (RuntimeError("private-token"), "service_unavailable"),
        ):
            service.failure = error
            text = client.post("/chat", json={"message": "Temple"}).text
            assert expected in text and "private-token" not in text and "event: token" not in text
            with pytest.raises(RuntimeError):
                parse_answer(text)
    assert service.closed
    with pytest.raises(RuntimeError, match="without a final"):
        parse_answer("event: session\ndata: {}\n\n")


def test_runtime_is_lazy_and_reuses_agent(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from met_agent import runtime as module

    created: list[str] = []

    def qdrant(_: Settings) -> QdrantClient:
        created.append("qdrant")
        return QdrantClient(location=":memory:")

    model = ScriptedModel([intent(category="interpretive"), intent(category="interpretive")])
    monkeypatch.setattr(module, "qdrant_client", qdrant)
    monkeypatch.setattr(module, "LiteLLMChat", lambda *args, **kwargs: model)
    app = create_app(settings.model_copy(update={"data_dir": tmp_path}))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200 and not created
        for _ in range(2):
            assert parse_answer(
                client.post("/chat", json={"message": "Meaning?"}).text
            ).policy_refusal
        assert created == ["qdrant"]
    assert app.state.runtime.http.is_closed


def test_mcp_six_tools_use_same_validation_and_results(settings: Settings) -> None:
    async def exercise() -> None:
        with httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json={
                        "objectID": 1,
                        "objectURL": "https://www.metmuseum.org/art/collection/search/1",
                    },
                )
            )
        ) as http:
            client = QdrantClient(location=":memory:")
            try:
                registry = create_registry(
                    settings,
                    SearchService(client, settings),
                    LiveObjectClient(http, "https://met.test"),
                    WayfindingClient(http, "https://map.test", "https://maps.test"),
                )
                server = create_server(registry)
                listed = await server.list_tools()
                assert {tool.name for tool in listed} == set(registry.tools)
                for name, args in (
                    ("get_object", {"object_id": 1}),
                    ("handoff", {"reason": "account"}),
                    ("search_collection", {"query": "vase", "k": 0}),
                    ("search_visitor_info", {"query": "hours", "k": 0}),
                    ("find_similar_objects", {"object_id": 0}),
                    ("get_directions", {"destination_gallery": 131}),
                ):
                    result = await server.call_tool(name, args)
                    assert isinstance(result, CallToolResult)
                    data = result.structured_content
                    assert data is not None
                    if name == "get_object":
                        assert data["output"]["object_id"] == 1
                    elif name == "handoff":
                        assert data["output"]["suggested_contact"] == "info@metmuseum.org"
                    else:
                        assert data["error"]["code"] == "invalid_arguments"
                assert server.sse_app() is not None
            finally:
                client.close()

    asyncio.run(exercise())


def test_demo_prints_required_fields_and_returns_failure(
    settings: Settings,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(Path(__file__).parents[2])
    app = create_app(settings)
    app.state.runtime = Service(
        tmp_path,
        [
            intent(),
            calls("get_object"),
            draft(),
            judgment(),
            intent(category="visitor_info"),
            {
                "text": "Temple",
                "language": "en",
                "citations": [
                    {
                        "source_url": "https://www.metmuseum.org/plan-your-visit",
                        "quote": "Temple",
                    }
                ],
            },
            {
                "fully_supported": True,
                "claims": [
                    {
                        "text": "Temple",
                        "supported": True,
                        "evidence_keys": ["page:fixture"],
                    }
                ],
            },
            intent(category="interpretive"),
        ],
    )
    assert asyncio.run(demo_run(settings.model_copy(update={"data_dir": tmp_path}), app=app)) == 0
    printed = capsys.readouterr().out
    assert all(
        value in printed
        for value in (
            "col-003",
            "vis-001",
            "ref-001",
            "Answer:",
            "Citations:",
            "Route:",
            "latency:",
            "estimated cost USD:",
        )
    )
    records = json.loads((tmp_path / "demo-check.json").read_text())
    assert len(records) == 3
    assert all(record["events"][-1]["kind"] == "final_answer" for record in records)
    app.state.runtime.failure = ModelError("access_denied", "Provider denied")
    assert asyncio.run(demo_run(settings.model_copy(update={"data_dir": tmp_path}), app=app)) == 1
    printed = capsys.readouterr().out
    assert "FAILED: access_denied" in printed
    assert "Session events (" in printed
    failed_records = json.loads((tmp_path / "demo-check.json").read_text())
    assert all(record["session_id"] and record["latency_ms"] >= 0 for record in failed_records)
