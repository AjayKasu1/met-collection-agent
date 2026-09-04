"""Check opt-in trace nesting, credential masking, usage callbacks, and shutdown offline."""

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from pydantic import SecretStr
from test_chat_routing import response

from met_agent.agent.events import EventStore
from met_agent.agent.models import AgentAnswer, ChatRequest
from met_agent.config import Settings
from met_agent.observability.tracing import Telemetry


class Span:
    def __init__(self, fields: dict[str, Any]) -> None:
        self.fields = fields
        self.ended = False

    def update(self, **kwargs: Any) -> None:
        self.fields.update(kwargs)

    def end(self) -> None:
        self.ended = True


class Client:
    def __init__(self) -> None:
        self.spans: list[Span] = []
        self.updates: list[dict[str, Any]] = []
        self.flushed = self.closed = False

    def start_observation(self, **kwargs: Any) -> Span:
        span = Span(kwargs)
        self.spans.append(span)
        return span

    def get_current_trace_id(self) -> str:
        return "1" * 32

    def get_current_observation_id(self) -> str:
        return "2" * 16

    def update_current_span(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def score_current_trace(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def flush(self) -> None:
        self.flushed = True

    def shutdown(self) -> None:
        self.closed = True


def test_trace_callback_links_usage_masks_and_ends_spans(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import langfuse

    from met_agent.llm import cost

    store = EventStore(tmp_path / "audit.sqlite3", secrets=["private-test"])
    with pytest.raises(ValueError):
        Telemetry(settings, store)
    config = settings.model_copy(
        update={
            "langfuse_public_key": SecretStr("pk-test"),
            "langfuse_secret_key": SecretStr("sk-test"),
            "langfuse_base_url": "https://telemetry.test",
        }
    )
    client = Client()
    constructed: dict[str, Any] = {}

    def factory(**kwargs: Any) -> Client:
        constructed.update(kwargs)
        return client

    monkeypatch.setattr(langfuse, "Langfuse", factory)
    telemetry = Telemetry(config, store)
    assert constructed["mask"]({"secret": "private-test"}) == {"secret": "[REDACTED]"}
    callback = telemetry.callback
    fields = {
        "litellm_call_id": "1",
        "litellm_params": {"metadata": {"trace_context": callback.metadata()}},
    }
    callback.log_pre_api_call("gemini-test", [{"content": "private-test"}], fields)
    callback.log_pre_api_call("gemini-test", [], fields)
    assert len(client.spans) == 1
    monkeypatch.setattr(cost, "estimate_cost", lambda _: 0.001)
    now = datetime.now(UTC)
    asyncio.run(
        callback.async_log_success_event(fields, response(), now, now + timedelta(seconds=2))
    )
    assert client.spans[0].fields["usage_details"] == {"input": 30, "output": 20}
    assert client.spans[0].fields["cost_details"] == {"total": 0.001}
    assert client.spans[0].fields["metadata"]["latency_ms"] == 2000
    assert client.spans[0].fields["trace_context"]["trace_id"] == "1" * 32
    assert "private-test" not in json.dumps(client.spans[0].fields)
    assert client.spans[0].ended
    asyncio.run(callback.async_log_success_event(fields, response(), now, now))
    callback.log_pre_api_call("gemini-test", [], fields)
    asyncio.run(callback.async_log_failure_event(fields, None, now, now))
    asyncio.run(callback.async_log_failure_event(fields, None, now, now))
    assert client.spans[1].fields["status_message"] == "Provider call failed"
    with telemetry.tool("get_object", '{"key":"private-test"}'):
        pass
    with pytest.raises(RuntimeError), telemetry.tool("get_object", "{}"):
        raise RuntimeError("private-test")
    assert client.spans[-1].fields["status_message"] == "Tool operation failed"
    callback.log_pre_api_call("gemini-test", [], fields)
    telemetry.close()
    assert client.flushed and client.closed and all(span.ended for span in client.spans)


def test_observe_root_uses_explicit_key_and_request_session(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import langfuse

    config = settings.model_copy(
        update={
            "langfuse_public_key": SecretStr("pk-test"),
            "langfuse_secret_key": SecretStr("sk-test"),
            "langfuse_base_url": "https://telemetry.test",
        }
    )
    store = EventStore(tmp_path / "audit.sqlite3")
    client = Client()
    telemetry = Telemetry(config, store, client=client)
    observed: list[dict[str, Any]] = []

    def decorator(**options: Any) -> Any:
        observed.append(options)

        def decorate(fn: Any) -> Any:
            async def wrapped(**kwargs: Any) -> Any:
                assert kwargs == {"langfuse_public_key": "pk-test"}
                return await fn()

            return wrapped

        return decorate

    @contextmanager
    def attributes(**kwargs: Any) -> Iterator[None]:
        observed.append(kwargs)
        yield

    monkeypatch.setattr(langfuse, "observe", decorator)
    monkeypatch.setattr(langfuse, "propagate_attributes", attributes)
    request = ChatRequest(message="Temple", session_id=uuid4())

    async def answer(req: ChatRequest) -> AgentAnswer:
        return AgentAnswer(
            session_id=req.session_id or uuid4(),
            turn_id=uuid4(),
            text="Facts",
            language="en",
            route="lite",
            cost_usd=0.001,
            latency_ms=100,
            grounding_score=1,
        )

    result = asyncio.run(telemetry.run(answer, request))
    assert result.session_id == request.session_id
    assert observed[0] == {
        "name": "chat",
        "as_type": "agent",
        "capture_input": False,
        "capture_output": False,
    }
    assert observed[1]["session_id"] == str(request.session_id)
    assert client.updates[-1] == {"name": "grounding_score", "value": 1.0}
