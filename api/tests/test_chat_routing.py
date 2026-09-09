"""Verify explicit gateway routing, narrow fallback, structured replies, and cost accounting."""

import asyncio
import json
from typing import Any

import httpx
import pytest
from litellm.exceptions import AuthenticationError, BadRequestError, RateLimitError, Timeout
from litellm.types.utils import ModelResponse
from pydantic import SecretStr

from met_agent.agent.models import ModelCall
from met_agent.config import Settings
from met_agent.guardrails.intent import Intent
from met_agent.llm import chat, cost
from met_agent.llm.chat import (
    CURRENT_CALL,
    CallContext,
    LiteLLMChat,
    ModelError,
    chat_routes,
    google_model,
    structured,
)


class Router:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def acompletion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def response() -> ModelResponse:
    return ModelResponse(
        model="gemini-test",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": (
                        '{"category":"collection","difficulty":"simple",'
                        '"language":"en","search_query":"Temple"}'
                    ),
                    "reasoning_content": "private reasoning",
                },
            }
        ],
        usage={"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},
    )


def test_explicit_routes_and_gateway(settings: Settings) -> None:
    for name in (
        "google-ai-studio/gemini-test",
        "gemini/gemini-test",
        "models/gemini-test",
        "gemini-test",
    ):
        assert google_model(name) == "gemini/gemini-test"
    assert "api_base" not in chat_routes(settings)[0]["litellm_params"]
    config = settings.model_copy(
        update={
            "use_ai_gateway": True,
            "cf_ai_gateway_url": "https://gateway.ai.cloudflare.com/v1/account/gateway",
            "cf_ai_gateway_token": SecretStr("gateway-private"),
            "llm_fallback_enabled": True,
            "llm_model_fallback": "groq/fallback",
            "groq_api_key": SecretStr("groq-private"),
        }
    )
    routes = chat_routes(config)
    assert len(routes) == 3
    assert routes[0]["litellm_params"]["api_base"].endswith("/google-ai-studio/v1beta")
    assert (
        routes[0]["litellm_params"]["extra_headers"]["cf-aig-authorization"]
        == "Bearer gateway-private"
    )
    assert routes[2]["litellm_params"]["api_base"].endswith("/groq")
    with pytest.raises(ValueError):
        chat_routes(config.model_copy(update={"cf_ai_gateway_token": None}))


def test_router_retry_policy_and_json_tools_exclusivity(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import litellm.router

    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> Router:
        captured.update(kwargs)
        return Router([response(), response()])

    monkeypatch.setattr(litellm.router, "Router", factory)
    model = LiteLLMChat(settings)
    assert captured["fallbacks"] == [] and captured["max_fallbacks"] == 0
    assert captured["retry_policy"].AuthenticationErrorRetries == 0
    asyncio.run(
        model.complete("main", [{"role": "user", "content": "facts"}], tools=[{"type": "function"}])
    )
    asyncio.run(model.complete("lite", []))
    assert "response_format" not in model.router.calls[0]
    assert model.router.calls[1]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    "failure",
    [
        RateLimitError("private", llm_provider="gemini", model="test"),
        Timeout("private", model="test", llm_provider="gemini"),
    ],
)
def test_fallback_is_only_for_quota_and_timeout(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    config = settings.model_copy(
        update={
            "llm_max_retries": 0,
            "llm_fallback_enabled": True,
            "llm_model_fallback": "groq/fallback",
            "groq_api_key": SecretStr("synthetic"),
        }
    )
    router = Router([failure, response()])
    monkeypatch.setattr(chat, "estimate_cost", lambda _: 0.002)
    context = CallContext()
    token = CURRENT_CALL.set(context)
    try:
        result = asyncio.run(LiteLLMChat(config, router=router).complete("main", []))
    finally:
        CURRENT_CALL.reset(token)
    assert [call["model"] for call in router.calls] == ["main", "fallback"]
    assert context.ledger.calls[0].path == "direct_groq" and context.ledger.cost == 0.002
    assert "reasoning_content" not in result.message
    with pytest.raises(ModelError):
        asyncio.run(LiteLLMChat(settings, router=Router([failure])).complete("lite", []))


def test_groq_tool_protocol_failure_moves_immediately_to_distinct_fallback(
    settings: Settings,
) -> None:
    failure = BadRequestError(
        "private failed generation",
        model="openai/gpt-oss-20b",
        llm_provider="groq",
        response=httpx.Response(
            400,
            request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        ),
        body={
            "error": {
                "code": "tool_use_failed",
                "failed_generation": "private model output",
            }
        },
    )
    config = settings.model_copy(
        update={
            "llm_model": "groq/openai/gpt-oss-120b",
            "llm_model_lite": "groq/openai/gpt-oss-20b",
            "groq_api_key": SecretStr("synthetic-groq"),
            "llm_fallback_enabled": True,
            "llm_model_fallback": "gemini/gemini-3.7-flash",
            "gemini_api_key": SecretStr("synthetic-gemini"),
        }
    )
    router = Router([failure, response()])
    events: list[tuple[str, Any]] = []
    token = CURRENT_CALL.set(CallContext(audit=lambda kind, data: events.append((kind, data))))
    try:
        asyncio.run(
            LiteLLMChat(config, router=router).complete(
                "lite",
                [{"role": "user", "content": "visitor policy"}],
                tools=[{"type": "function"}],
            )
        )
    finally:
        CURRENT_CALL.reset(token)

    assert [call["model"] for call in router.calls] == ["lite", "fallback"]
    assert (
        "model_fallback",
        {"from": "lite", "to": "fallback", "reason": "tool_protocol_failure"},
    ) in events
    attempt = next(data for kind, data in events if kind == "provider_attempt_failed")
    assert attempt["provider_code"] == "tool_use_failed"
    assert "private" not in json.dumps(events)


def test_groq_tool_protocol_failure_never_hides_non_tool_request_errors(
    settings: Settings,
) -> None:
    failure = BadRequestError(
        "private failed generation",
        model="openai/gpt-oss-20b",
        llm_provider="groq",
        response=httpx.Response(
            400,
            request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        ),
        body={"error": {"code": "tool_use_failed"}},
    )
    config = settings.model_copy(
        update={
            "llm_model_lite": "groq/openai/gpt-oss-20b",
            "groq_api_key": SecretStr("synthetic-groq"),
            "llm_fallback_enabled": True,
            "llm_model_fallback": "gemini/gemini-3.7-flash",
            "gemini_api_key": SecretStr("synthetic-gemini"),
        }
    )
    router = Router([failure])

    with pytest.raises(ModelError):
        asyncio.run(LiteLLMChat(config, router=router).complete("lite", []))

    assert [call["model"] for call in router.calls] == ["lite"]


@pytest.mark.parametrize(
    "failure,code",
    [
        (
            AuthenticationError("private credential", llm_provider="gemini", model="test"),
            "access_denied",
        ),
        (
            BadRequestError(
                "private",
                llm_provider="gemini",
                model="test",
                response=httpx.Response(403, request=httpx.Request("POST", "https://test.local")),
            ),
            "access_denied",
        ),
        (
            BadRequestError(
                "private",
                llm_provider="gemini",
                model="test",
                response=httpx.Response(404, request=httpx.Request("POST", "https://test.local")),
            ),
            "model_not_found",
        ),
        (RuntimeError("private"), "provider_unavailable"),
    ],
)
def test_provider_failure_is_sanitized_without_fallback(
    settings: Settings, failure: Exception, code: str
) -> None:
    router = Router([failure])
    context = CallContext()
    token = CURRENT_CALL.set(context)
    try:
        with pytest.raises(ModelError, match="Model request failed") as caught:
            asyncio.run(
                LiteLLMChat(
                    settings.model_copy(
                        update={
                            "llm_fallback_enabled": True,
                            "llm_model_fallback": "groq/fallback",
                            "groq_api_key": SecretStr("synthetic"),
                        }
                    ),
                    router=router,
                ).complete("lite", [])
            )
    finally:
        CURRENT_CALL.reset(token)
    assert caught.value.code == code and "private" not in str(caught.value)
    assert len(router.calls) == 1


def test_structured_validation_usage_and_callback(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Callback:
        def metadata(self) -> dict[str, str]:
            return {"trace_id": "trace"}

    router = Router(
        [
            response(),
            ModelResponse(
                model="test",
                choices=[{"message": {"role": "assistant", "content": "invalid"}}],
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            ),
        ]
    )
    model = LiteLLMChat(
        settings.model_copy(update={"use_ai_gateway": True}), router=router, callback=Callback()
    )
    audit: list[tuple[str, object]] = []
    context = CallContext(audit=lambda name, data: audit.append((name, data)))
    token = CURRENT_CALL.set(context)
    monkeypatch.setattr(chat, "estimate_cost", lambda _: None)
    try:
        answer = asyncio.run(structured(model, "lite", "Classify", "Temple", Intent))
        assert answer.route == "lite"
        with pytest.raises(ModelError, match="structured validation"):
            asyncio.run(structured(model, "lite", "Classify", "Temple", Intent))
    finally:
        CURRENT_CALL.reset(token)
    assert context.ledger.calls[0].input_tokens == 30
    assert context.ledger.calls[0].path == "ai_gateway" and context.ledger.cost is None
    assert "private reasoning" not in json.dumps(audit)
    assert all(
        key in router.calls[0]
        for key in ("input_callback", "success_callback", "failure_callback", "metadata")
    )


def test_unknown_pricing_is_not_reported_as_free(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    for value, expected in ((0.01, 0.01), (-1, None), (float("inf"), None), ("invalid", None)):
        monkeypatch.setattr(litellm, "completion_cost", lambda value=value, **kwargs: value)
        assert cost.estimate_cost(response()) == expected

    def missing(**kwargs: Any) -> float:
        raise KeyError("not priced")

    monkeypatch.setattr(litellm, "completion_cost", missing)
    assert cost.estimate_cost(response()) is None
    calls = [
        ModelCall(
            model="test",
            route="main",
            path="direct_google",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            cost_usd=value,
        )
        for value in (0.01, 0.02)
    ]
    assert cost.CallLedger(calls).cost == 0.03


def test_missing_first_key_uses_second_and_403_stops_chain(settings: Settings) -> None:
    config = settings.model_copy(
        update={
            "llm_max_retries": 0,
            "llm_fallback_enabled": True,
            "llm_model_fallback": "cerebras/gpt-oss-120b",
            "llm_model_fallback_2": "groq/llama-3.3-70b-versatile",
            "groq_api_key": SecretStr("synthetic"),
        }
    )
    quota = RateLimitError("quota", llm_provider="gemini", model="test")
    router = Router([quota, response()])
    asyncio.run(LiteLLMChat(config, router=router).complete("main", []))
    assert [call["model"] for call in router.calls] == ["main", "fallback_2"]
    denied = BadRequestError(
        "private",
        llm_provider="cerebras",
        model="test",
        response=httpx.Response(403, request=httpx.Request("POST", "https://test.local")),
    )
    router = Router([quota, denied])
    config = config.model_copy(update={"cerebras_api_key": SecretStr("synthetic")})
    with pytest.raises(ModelError) as error:
        asyncio.run(LiteLLMChat(config, router=router).complete("main", []))
    assert error.value.code == "access_denied"
    assert [call["model"] for call in router.calls] == ["main", "fallback"]


def test_rate_limit_spills_to_lite_without_retrying_exhausted_model(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []
    reservations: list[str] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    config = settings.model_copy(update={"llm_max_retries": 1})
    quota = RateLimitError(
        "quota",
        llm_provider="gemini",
        model="test",
        response=httpx.Response(
            429, headers={"Retry-After": "3"}, request=httpx.Request("POST", "https://test.local")
        ),
    )
    model = LiteLLMChat(config, router=Router([quota, response()]))
    events: list[tuple[str, Any]] = []
    token = CURRENT_CALL.set(CallContext(audit=lambda kind, data: events.append((kind, data))))

    async def reserve(name: str, tokens: int) -> None:
        reservations.append(name)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    monkeypatch.setattr(model.pacer, "reserve", reserve)
    try:
        asyncio.run(model.complete("main", []))
    finally:
        CURRENT_CALL.reset(token)
    assert sleeps == []
    assert reservations == [model.models["main"], model.models["lite"]]
    assert [call["model"] for call in model.router.calls] == ["main", "lite"]
    assert ("model_fallback", {"from": "main", "to": "lite", "reason": "rate_limit"}) in events


def test_gateway_500_is_retried_with_safe_telemetry(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    class GatewayInternalError(RuntimeError):
        def __init__(self) -> None:
            super().__init__("private provider body")
            self.response = httpx.Response(
                500,
                headers={"retry-after": "2", "authorization": "private-header"},
                request=httpx.Request("POST", "https://test.local"),
            )

    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    events: list[tuple[str, Any]] = []
    model = LiteLLMChat(
        settings.model_copy(update={"llm_max_retries": 1}),
        router=Router([GatewayInternalError(), response()]),
    )
    token = CURRENT_CALL.set(CallContext(audit=lambda k, d: events.append((k, d))))
    monkeypatch.setattr(asyncio, "sleep", sleep)
    try:
        asyncio.run(model.complete("lite", []))
    finally:
        CURRENT_CALL.reset(token)
    assert sleeps == [2]
    assert len(model.router.calls) == 2
    attempts = [data for kind, data in events if kind == "provider_attempt_failed"]
    assert attempts == [
        {
            "model": model.models["lite"],
            "route": "lite",
            "error_type": "GatewayInternalError",
            "status": 500,
            "retry_after": "2",
            "attempt": 1,
        }
    ]
    assert "private" not in json.dumps(events)


def test_native_final_schema_is_separate_from_tool_selection(settings: Settings) -> None:
    from met_agent.agent.models import AgentDraft

    ready = response()
    ready.choices[0].message.content = "READY"
    final_reply = response()
    final_reply.choices[0].message.content = json.dumps(
        {
            "text": "Gallery 131",
            "language": "en",
            "citations": ["q0"],
        }
    )
    router = Router([ready, final_reply])
    config = settings.model_copy(
        update={"llm_model": "groq/openai/gpt-oss-120b", "groq_api_key": SecretStr("synthetic")}
    )
    asyncio.run(
        LiteLLMChat(config, router=router).complete(
            "main",
            [
                {"role": "user", "content": "Where?"},
                {
                    "role": "tool",
                    "tool_call_id": "call",
                    "content": json.dumps(
                        {
                            "name": "get_object",
                            "evidence": [
                                {
                                    "key": "object:547802",
                                    "object_id": 547802,
                                    "kind": "live_object",
                                    "text": "Gallery 131",
                                }
                            ],
                        }
                    ),
                },
            ],
            tools=[{"type": "function"}],
            response_schema=AgentDraft,
        )
    )
    first, final = router.calls
    assert "tools" in first and "response_format" not in first
    assert "tools" not in final
    fmt = final["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["citations"]["items"]["enum"] == ["q0"]
    assert "Gallery 131" in final["messages"][1]["content"]
    assert all(message["role"] != "tool" for message in final["messages"])


@pytest.mark.parametrize(
    "model,input_price,output_price",
    [
        ("groq/openai/gpt-oss-120b", 0.15, 0.60),
        ("groq/openai/gpt-oss-20b", 0.075, 0.30),
    ],
)
def test_groq_prices_use_reported_tokens(
    model: str, input_price: float, output_price: float
) -> None:
    result = response().model_copy(update={"model": model})
    assert cost.estimate_cost(result) == pytest.approx((30 * input_price + 20 * output_price) / 1e6)


def test_wire_citations_preserve_existing_source_and_quote_rules() -> None:
    from met_agent.agent.models import AgentDraft
    from met_agent.llm.structured_output import decode_final

    for source in ({"object_id": 1}, {"source_url": "https://www.metmuseum.org/plan-your-visit"}):
        wire = {
            "text": "Fact",
            "language": "en",
            "citations": [{"source": source, "quote": "Fact"}],
        }
        decoded = AgentDraft.model_validate_json(decode_final(json.dumps(wire)))
        assert decoded.citations[0].quote == "Fact"
    for invalid_source in (
        {},
        {"object_id": 1, "source_url": "https://example.org"},
        {"object_id": 0},
    ):
        with pytest.raises(ValueError):
            decode_final(
                json.dumps(
                    {
                        "text": "Fact",
                        "language": "en",
                        "citations": [{"source": invalid_source, "quote": "Fact"}],
                    }
                )
            )
