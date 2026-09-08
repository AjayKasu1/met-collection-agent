"""Exercise provider isolation, safe diagnostics, evidence bounds and deadline cancellation."""

import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest
from litellm.exceptions import BadRequestError, RateLimitError
from pydantic import SecretStr
from test_chat_routing import Router

from met_agent.agent.events import EventStore
from met_agent.agent.evidence import MAX_TEXT_CHARACTERS, pack_evidence
from met_agent.agent.models import ChatRequest
from met_agent.config import Settings
from met_agent.llm.chat import CURRENT_CALL, CallContext, LiteLLMChat, ModelError
from met_agent.llm.providers import configured_models
from met_agent.llm.recovery import failure_details, retry_delay
from met_agent.runtime import Runtime


def test_stored_fallback_is_inactive_without_opt_in(settings: Settings) -> None:
    config = settings.model_copy(
        update={"llm_model_fallback": "groq/test", "groq_api_key": SecretStr("private")}
    )
    assert list(configured_models(config)) == ["lite", "main"]
    assert not config.optional_services.fallback_llm


def test_rate_limit_audit_precedes_cooldown_and_retains_no_body(settings: Settings) -> None:
    failure = RateLimitError(
        "private-body",
        llm_provider="groq",
        model="test",
        response=httpx.Response(
            429,
            headers={
                "retry-after": "120",
                "x-ratelimit-remaining-tokens": "0",
                "authorization": "private-header",
            },
            request=httpx.Request("POST", "https://test.local"),
        ),
    )
    router = Router([failure])
    model = LiteLLMChat(settings, router=router)
    events: list[tuple[str, Any]] = []
    token = CURRENT_CALL.set(CallContext(audit=lambda k, d: events.append((k, d))))
    try:
        for _ in range(2):
            with pytest.raises(ModelError):
                asyncio.run(model.complete("lite", []))
    finally:
        CURRENT_CALL.reset(token)
    assert len(router.calls) == 1
    attempts = [d for k, d in events if k == "provider_attempt_failed"]
    assert attempts[0]["status"] == 429
    assert attempts[0]["remaining_tokens"] == "0"
    assert attempts[0]["retry_after"] == "120"
    assert "private" not in json.dumps(events)


def test_permission_failure_quarantines_model_until_reconfigured(settings: Settings) -> None:
    failure = BadRequestError(
        "private",
        llm_provider="groq",
        model="test",
        response=httpx.Response(403, request=httpx.Request("POST", "https://test.local")),
    )
    router = Router([failure])
    model = LiteLLMChat(settings, router=router)
    for _ in range(2):
        with pytest.raises(ModelError) as caught:
            asyncio.run(model.complete("main", []))
        assert caught.value.code == "access_denied"
    assert len(router.calls) == 1


def test_untrusted_quota_headers_and_nonfinite_retry_are_rejected() -> None:
    failure = BadRequestError(
        "private",
        llm_provider="groq",
        model="test",
        response=httpx.Response(
            400,
            headers={"retry-after": "private-value"},
            request=httpx.Request("POST", "https://test.local"),
        ),
    )
    assert "retry_after" not in failure_details(failure, model="test", route="main")
    for value in ("nan", "inf", "private", "-10"):
        assert retry_delay({"retry_after": value}) == 0


@pytest.mark.parametrize(
    "value,expected",
    [("7.66s", 7.66), ("2m59.56s", 179.56), ("250ms", 0.25), ("1h2m3s", 3723)],
)
def test_provider_reset_durations_are_parsed(value: str, expected: float) -> None:
    assert retry_delay({"token_reset_after": value}) == pytest.approx(expected)


def test_turn_evidence_is_deduplicated_bounded_and_verbatim() -> None:
    def message(records: list[dict[str, Any]]) -> dict[str, Any]:
        return {"role": "tool", "content": json.dumps({"name": "search", "evidence": records})}

    def record(key: str, text: str) -> dict[str, Any]:
        return {"key": key, "text": text, "kind": "collection", "object_id": 1}

    latest = "Current source\n" + "Exact source text. " * 300
    messages = [
        message([record("same", "outdated")]),
        message([record("same", latest)] + [record(str(i), latest) for i in range(12)]),
    ]
    packed = pack_evidence(messages)
    assert len(packed) <= 8
    assert sum(len(item.text) for item in packed) <= MAX_TEXT_CHARACTERS
    assert len({item.key for item in packed}) == len(packed)
    assert all(item.text in latest for item in packed)
    assert json.loads(messages[0]["content"])["evidence"] == []
    assert pack_evidence(messages) == packed


def test_interactive_deadline_cancels_unverified_answer(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = Runtime.__new__(Runtime)
    service.settings = settings.model_copy(update={"chat_deadline_seconds": 0.01})
    service.events = EventStore(tmp_path / "events.sqlite3")
    service.telemetry = None
    cancelled: list[bool] = []

    class SlowAgent:
        async def run(self, request: ChatRequest) -> Any:
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.append(True)

    monkeypatch.setattr(service, "agent", SlowAgent(), raising=False)
    session = uuid4()
    with pytest.raises(ModelError) as error:
        asyncio.run(service.chat(ChatRequest(message="Facts", session_id=session)))
    assert error.value.code == "verification_unavailable"
    assert cancelled == [True]
    events = service.events.read(session)
    assert [e.kind for e in events] == ["turn_error"]
