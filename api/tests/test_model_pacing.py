"""Verify rolling token budgets, credential-based routes, and compact citation evidence."""

import asyncio
import json
import logging
from typing import Any

import pytest
from pydantic import SecretStr

from met_agent.config import LLMRateLimit, Settings, load_settings
from met_agent.llm.chat import LiteLLMChat
from met_agent.llm.pacing import RequestBudgetError, TokenPacer, prompt_tokens
from met_agent.llm.providers import chat_routes, configured_models
from met_agent.tools.models import Evidence, ToolResult


def test_missing_fallback_keys_and_provider_keys(settings: Settings, caplog: Any) -> None:
    config = settings.model_copy(
        update={
            "llm_model": "groq/openai/gpt-oss-120b",
            "llm_model_lite": "groq/openai/gpt-oss-20b",
            "groq_api_key": SecretStr("synthetic-groq"),
            "llm_model_fallback": "cerebras/gpt-oss-120b",
            "llm_model_fallback_2": "groq/llama-3.3-70b-versatile",
            "cerebras_api_key": None,
        }
    )
    assert "fallback" not in configured_models(config)
    assert configured_models(config)["fallback_2"].startswith("groq/")
    assert all(r["litellm_params"]["api_key"] == "synthetic-groq" for r in chat_routes(config))
    with caplog.at_level(logging.INFO):
        LiteLLMChat(config, router=object())
    assert "llama-3.3-70b-versatile" in caplog.text
    assert "synthetic-groq" not in caplog.text
    config = config.model_copy(update={"cerebras_api_key": SecretStr("synthetic-cerebras")})
    assert chat_routes(config)[2]["litellm_params"]["api_key"] == "synthetic-cerebras"


def test_groq_local_does_not_require_google_key(
    valid_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("LLM_MODEL", "groq/main")
    monkeypatch.setenv("LLM_MODEL_LITE", "groq/lite")
    monkeypatch.setenv("GROQ_API_KEY", "synthetic")
    monkeypatch.delenv("GEMINI_API_KEY")
    assert load_settings(env_file=None).gemini_api_key is None


def test_token_and_request_limits_reconcile_and_model_isolation(settings: Settings) -> None:
    now = [0.0]
    waits: list[float] = []

    async def sleep(delay: float) -> None:
        waits.append(delay)
        now[0] += delay

    config = settings.model_copy(
        update={
            "llm_pacing_enabled": True,
            "llm_rate_limits": {
                "model": LLMRateLimit(tokens_per_minute=100, requests_per_minute=2)
            },
        }
    )
    pacer = TokenPacer(config, clock=lambda: now[0], sleep=sleep)

    async def scenario() -> None:
        first = await pacer.reserve("model", 80)
        await pacer.reconcile(first, 20)
        await pacer.reserve("model", 80)
        assert not waits
        await pacer.reserve("other", 90)
        assert not waits
        await pacer.reserve("model", 1)
        assert waits == [60]
        await pacer.reserve("model", 100)
        assert waits == [60, 60]
        with pytest.raises(RequestBudgetError):
            await pacer.reserve("model", 101)
        await pacer.reconcile(None, 0)

    asyncio.run(scenario())
    disabled = TokenPacer(settings)
    assert asyncio.run(disabled.reserve("model", 100000)) is None


def test_prompt_token_count_and_conservative_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    monkeypatch.setattr(litellm, "token_counter", lambda **kw: 100)
    assert prompt_tokens("model", [], None) >= 174

    def fail(**kw: Any) -> int:
        raise ValueError("unknown tokenizer")

    monkeypatch.setattr(litellm, "token_counter", fail)
    assert prompt_tokens("model", [{"role": "user", "content": "中文"}], None) > 64


def test_compact_context_keeps_exact_citation_evidence() -> None:
    evidence = Evidence(
        key="object:1",
        object_id=1,
        source_url="https://example.org/1",
        text="The exact evidence.",
        kind="collection",
    )
    result = ToolResult(name="get_object", evidence=[evidence])
    payload = json.loads(result.model_context())
    assert payload["evidence"] == [evidence.model_dump(mode="json")]
    assert "output" not in payload
    assert json.loads(ToolResult(name="handoff").model_context())["name"] == "handoff"
