"""Route chat through LiteLLM with explicit credentials, bounded retries, and narrow fallback."""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, JsonValue

from met_agent.agent.models import ModelCall, Route
from met_agent.config import Settings, model_provider
from met_agent.llm.cost import CallLedger, estimate_cost
from met_agent.llm.pacing import RequestBudgetError, TokenPacer, prompt_tokens
from met_agent.llm.prompts import load_prompt
from met_agent.llm.providers import chat_routes as chat_routes
from met_agent.llm.providers import configured_models
from met_agent.llm.providers import google_model as google_model
from met_agent.llm.structured_output import response_format


class ModelError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class CallContext:
    ledger: CallLedger = field(default_factory=CallLedger)
    audit: Callable[[str, JsonValue], None] = lambda kind, data: None


CURRENT_CALL: ContextVar[CallContext | None] = ContextVar("model_call_context", default=None)


@dataclass
class Reply:
    message: dict[str, Any]

    @property
    def text(self) -> str:
        return str(self.message.get("content") or "")


class ChatModel(Protocol):
    async def complete(
        self,
        route: Route,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, object]] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> Reply: ...


async def structured[T: BaseModel](
    model: ChatModel, route: Route, prompt: str, content: JsonValue, schema: type[T]
) -> T:
    reply = await model.complete(
        route,
        [
            {
                "role": "system",
                "content": prompt + "\nOutput schema:\n" + json.dumps(schema.model_json_schema()),
            },
            {"role": "user", "content": json.dumps(content, ensure_ascii=False)},
        ],
        response_schema=schema,
    )
    try:
        return schema.model_validate_json(reply.text)
    except ValueError:
        raise ModelError(
            "invalid_response", "Model response failed structured validation"
        ) from None


class LiteLLMChat:
    """Fallback only after rate-limit or timeout exhaustion, never after authentication failure."""

    def __init__(self, settings: Settings, *, router: Any = None, callback: Any = None) -> None:
        self.settings, self.callback = settings, callback
        self.models = configured_models(settings)
        self.pacer = TokenPacer(settings)
        logging.getLogger(__name__).info(
            "Active model fallbacks: %s",
            {k: v for k, v in self.models.items() if k.startswith("fallback")},
        )
        if router is None:
            from litellm.router import Router
            from litellm.types.router import RetryPolicy

            for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
                log = logging.getLogger(name)
                log.handlers = [logging.NullHandler()]
                log.propagate = False
            router = Router(
                model_list=chat_routes(settings),
                timeout=settings.llm_timeout_seconds,
                num_retries=0,
                retry_policy=RetryPolicy(
                    BadRequestErrorRetries=0,
                    AuthenticationErrorRetries=0,
                    TimeoutErrorRetries=0,
                    RateLimitErrorRetries=0,
                    InternalServerErrorRetries=0,
                ),
                fallbacks=[],
                max_fallbacks=0,
                disable_cooldowns=True,
            )
        self.router = router

    async def complete(
        self,
        route: Route,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, object]] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> Reply:
        from litellm.exceptions import RateLimitError, Timeout

        if (
            response_schema is not None
            and response_schema.__name__ == "AgentDraft"
            and not tools
            and any(item.get("role") == "tool" for item in messages)
        ):
            messages = [
                {"role": "system", "content": load_prompt("system_v1")},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "context": [
                                item for item in messages if item.get("role") in {"user", "tool"}
                            ]
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        context = CURRENT_CALL.get()
        if context:
            context.audit(
                "model_request", {"route": route, "messages": json.loads(json.dumps(messages))}
            )
        output_limit = min(
            self.settings.llm_max_output_tokens,
            512
            if tools
            else 384
            if response_schema and response_schema.__name__ == "Intent"
            else 1024,
        )
        options: dict[str, Any] = {
            "messages": messages,
            "temperature": 0,
            "max_tokens": output_limit,
        }
        if tools:
            options["tools"] = tools
        elif response_schema is None:
            options["response_format"] = {"type": "json_object"}
        if self.callback is not None:
            options["input_callback"] = [self.callback]
            options["success_callback"] = [self.callback]
            options["failure_callback"] = [self.callback]
            options["metadata"] = {"trace_context": self.callback.metadata()}
        started = time.monotonic()
        actual_route = str(route)
        pacing_ms = 0.0
        provider_ms = 0.0
        retry_ms = 0.0
        response = None
        try:
            candidates = [str(route)] + [k for k in self.models if k.startswith("fallback")]
            for candidate in candidates:
                actual_route = candidate
                model_name = self.models[candidate]
                if response_schema is not None and not tools:
                    options["response_format"] = response_format(response_schema, model_name)
                if "gpt-oss" in model_name:
                    options["reasoning_effort"] = "low"
                else:
                    options.pop("reasoning_effort", None)
                for attempt in range(self.settings.llm_max_retries + 1):
                    pacing_started = time.monotonic()
                    reservation = await self.pacer.reserve(
                        model_name,
                        prompt_tokens(model_name, messages, tools)
                        + output_limit
                        + (
                            len(json.dumps(options.get("response_format", {})).encode("utf-8")) // 2
                        ),
                    )
                    pacing_ms += (time.monotonic() - pacing_started) * 1000
                    provider_started = time.monotonic()
                    try:
                        response = await self.router.acompletion(model=candidate, **options)
                        provider_ms += (time.monotonic() - provider_started) * 1000
                        await self.pacer.reconcile(reservation, int(response.usage.total_tokens))
                        break
                    except (RateLimitError, Timeout) as error:
                        provider_ms += (time.monotonic() - provider_started) * 1000
                        headers = getattr(getattr(error, "response", None), "headers", {})
                        try:
                            retry_after = max(0.0, float(headers.get("retry-after", 0)))
                        except (TypeError, ValueError):
                            retry_after = 0.0
                        if attempt < self.settings.llm_max_retries and retry_after <= 60:
                            retry_started = time.monotonic()
                            await asyncio.sleep(max(2**attempt, retry_after))
                            retry_ms += (time.monotonic() - retry_started) * 1000
                            continue
                        if candidate == candidates[-1]:
                            raise
                        if context:
                            context.audit(
                                "model_fallback",
                                {
                                    "from": candidate,
                                    "to": candidates[candidates.index(candidate) + 1],
                                },
                            )
                        break
                if response is not None:
                    break
        except Exception as error:
            status = getattr(getattr(error, "response", None), "status_code", None) or getattr(
                error, "status_code", None
            )
            code = (
                "context_budget_exceeded"
                if isinstance(error, RequestBudgetError)
                else "access_denied"
                if status in (401, 403)
                else "model_not_found"
                if status == 404
                else "provider_unavailable"
            )
            if context:
                context.audit("model_error", {"code": code, "error_type": type(error).__name__})
            raise ModelError(
                code,
                f"Model request failed ({type(error).__name__}, HTTP {status}); "
                "check provider access and configured model identifiers",
            ) from None
        if response is None:
            raise ModelError("provider_unavailable", "No provider response")
        usage = response.usage
        call = ModelCall(
            model=str(response.model),
            route=actual_route,
            provider=model_provider(self.models[actual_route]),
            path="ai_gateway"
            if self.settings.use_ai_gateway
            else "direct_groq"
            if model_provider(self.models[actual_route]) == "groq"
            else "direct_cerebras"
            if model_provider(self.models[actual_route]) == "cerebras"
            else "direct_google",
            input_tokens=int(usage.prompt_tokens),
            output_tokens=int(usage.completion_tokens),
            cost_usd=estimate_cost(
                response.model_copy(update={"model": self.models[actual_route]})
            ),
            pacing_ms=pacing_ms,
            provider_ms=provider_ms,
            retry_ms=retry_ms,
            latency_ms=(time.monotonic() - started) * 1000,
        )
        message = response.choices[0].message.model_dump(exclude_none=True)
        # Preserve provider tool signatures for subsequent calls, but do not expose reasoning text.
        message.pop("reasoning_content", None)
        message.pop("reasoning", None)
        if context:
            context.ledger.calls.append(call)
            context.audit(
                "model_response",
                {
                    "message": {
                        k: v for k, v in message.items() if k in {"role", "content", "tool_calls"}
                    },
                    "usage": call.model_dump(mode="json"),
                },
            )
        if tools and response_schema is not None and not message.get("tool_calls"):
            # Groq forbids tools with structured output. Start a separate final-answer call.
            final_messages = [
                {"role": "system", "content": load_prompt("system_v1")},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "context": [
                                item for item in messages if item.get("role") in {"user", "tool"}
                            ]
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
            return await self.complete(route, final_messages, response_schema=response_schema)
        return Reply(message)
