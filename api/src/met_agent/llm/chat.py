"""Route chat through LiteLLM with explicit credentials, bounded retries, and narrow fallback."""

import json
import logging
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, JsonValue

from met_agent.agent.models import ModelCall, Route
from met_agent.config import Settings
from met_agent.llm.cost import CallLedger, estimate_cost
from met_agent.llm.router import _gateway_base


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
    )
    try:
        return schema.model_validate_json(reply.text)
    except ValueError:
        raise ModelError(
            "invalid_response", "Model response failed structured validation"
        ) from None


def google_model(name: str) -> str:
    return "gemini/" + name.removeprefix("google-ai-studio/").removeprefix("gemini/").removeprefix(
        "models/"
    )


def chat_routes(settings: Settings) -> list[dict[str, Any]]:
    routes = []
    for alias, name in (("lite", settings.llm_model_lite), ("main", settings.llm_model)):
        params: dict[str, Any] = {
            "model": google_model(name),
            "api_key": settings.gemini_api_key.get_secret_value(),
        }
        if settings.use_ai_gateway:
            if settings.cf_ai_gateway_url is None or settings.cf_ai_gateway_token is None:
                raise ValueError("AI Gateway configuration is incomplete")
            params.update(
                api_base=_gateway_base(str(settings.cf_ai_gateway_url)),
                extra_headers={
                    "cf-aig-authorization": "Bearer "
                    + settings.cf_ai_gateway_token.get_secret_value()
                },
            )
        routes.append({"model_name": alias, "litellm_params": params})
    if settings.llm_model_fallback and settings.groq_api_key:
        routes.append(
            {
                "model_name": "fallback",
                "litellm_params": {
                    "model": settings.llm_model_fallback,
                    "api_key": settings.groq_api_key.get_secret_value(),
                },
            }
        )
    return routes


class LiteLLMChat:
    """Fallback only after rate-limit or timeout exhaustion, never after authentication failure."""

    def __init__(self, settings: Settings, *, router: Any = None, callback: Any = None) -> None:
        self.settings, self.callback = settings, callback
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
                num_retries=settings.llm_max_retries,
                retry_policy=RetryPolicy(
                    BadRequestErrorRetries=0,
                    AuthenticationErrorRetries=0,
                    TimeoutErrorRetries=settings.llm_max_retries,
                    RateLimitErrorRetries=settings.llm_max_retries,
                    InternalServerErrorRetries=settings.llm_max_retries,
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
    ) -> Reply:
        from litellm.exceptions import RateLimitError, Timeout

        context = CURRENT_CALL.get()
        if context:
            context.audit(
                "model_request", {"route": route, "messages": json.loads(json.dumps(messages))}
            )
        options: dict[str, Any] = {"messages": messages, "temperature": 0, "max_tokens": 2000}
        if tools:
            options["tools"] = tools
        else:
            options["response_format"] = {"type": "json_object"}
        if self.callback is not None:
            options["input_callback"] = [self.callback]
            options["success_callback"] = [self.callback]
            options["failure_callback"] = [self.callback]
            options["metadata"] = {"trace_context": self.callback.metadata()}
        started = time.monotonic()
        actual_route = str(route)
        try:
            try:
                response = await self.router.acompletion(model=route, **options)
            except (RateLimitError, Timeout):
                if not (self.settings.llm_model_fallback and self.settings.groq_api_key):
                    raise
                actual_route = "fallback"
                if context:
                    context.audit("model_fallback", {"from": route, "to": "fallback"})
                response = await self.router.acompletion(model="fallback", **options)
        except Exception as error:
            status = getattr(getattr(error, "response", None), "status_code", None) or getattr(
                error, "status_code", None
            )
            code = (
                "access_denied"
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
        usage = response.usage
        call = ModelCall(
            model=str(response.model),
            route=actual_route,
            path="direct_groq"
            if actual_route == "fallback"
            else "ai_gateway"
            if self.settings.use_ai_gateway
            else "direct_google",
            input_tokens=int(usage.prompt_tokens),
            output_tokens=int(usage.completion_tokens),
            cost_usd=estimate_cost(response),
            latency_ms=(time.monotonic() - started) * 1000,
        )
        message = response.choices[0].message.model_dump(exclude_none=True)
        # Preserve provider tool signatures for subsequent calls, but do not expose reasoning text.
        message.pop("reasoning_content", None)
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
        return Reply(message)
