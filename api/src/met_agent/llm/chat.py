"""Route chat through LiteLLM with explicit credentials and bounded transient recovery."""

import asyncio
import json
import logging
import random
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, JsonValue

from met_agent.agent.models import Citation, ModelCall, Route
from met_agent.config import Settings, model_provider
from met_agent.llm.citation_spans import prepare_final
from met_agent.llm.cost import CallLedger, estimate_cost
from met_agent.llm.pacing import RequestBudgetError, TokenPacer, prompt_tokens
from met_agent.llm.providers import chat_routes as chat_routes
from met_agent.llm.providers import configured_models
from met_agent.llm.providers import google_model as google_model
from met_agent.llm.recovery import failure_details, is_groq_tool_protocol_failure, retry_delay
from met_agent.llm.structured_output import decode_final, response_format


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
    """Recover from bounded transient failures without retrying client or access errors."""

    def __init__(self, settings: Settings, *, router: Any = None, callback: Any = None) -> None:
        self.settings, self.callback = settings, callback
        self.models = configured_models(settings)
        self.pacer = TokenPacer(settings)
        self.quarantined: set[str] = set()
        self.cooldown_until: dict[str, float] = {}
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

    def _candidate_routes(self, route: Route) -> list[str]:
        """Prefer configured fallbacks, then use the lite model as a capacity reserve."""
        aliases = [str(route), *[key for key in self.models if key.startswith("fallback")]]
        if route == "main":
            aliases.append("lite")
        candidates: list[str] = []
        seen_models: set[str] = set()
        for alias in aliases:
            model = self.models[alias]
            if model not in seen_models:
                candidates.append(alias)
                seen_models.add(model)
        return candidates

    async def complete(
        self,
        route: Route,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, object]] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> Reply:
        from litellm.exceptions import RateLimitError, Timeout

        citation_catalog: dict[str, Citation] | None = None
        if (
            response_schema is not None
            and response_schema.__name__ == "AgentDraft"
            and not tools
            and any(item.get("role") == "tool" for item in messages)
        ):
            messages, citation_catalog = prepare_final(messages)
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
            candidates = self._candidate_routes(route)
            for candidate in candidates:
                actual_route = candidate
                model_name = self.models[candidate]
                if model_name in self.quarantined:
                    raise ModelError("access_denied", "Provider access needs operator review")
                if self.cooldown_until.get(model_name, 0) > time.monotonic():
                    if candidate != candidates[-1]:
                        if context:
                            context.audit(
                                "model_fallback",
                                {
                                    "from": candidate,
                                    "to": candidates[candidates.index(candidate) + 1],
                                    "reason": "active_cooldown",
                                },
                            )
                        continue
                    raise ModelError("provider_unavailable", "Provider is temporarily unavailable")
                if response_schema is not None and not tools:
                    options["response_format"] = response_format(
                        response_schema, model_name, catalog=citation_catalog
                    )
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
                    except Exception as error:
                        provider_ms += (time.monotonic() - provider_started) * 1000
                        details = failure_details(error, model=model_name, route=candidate)
                        status = details.get("status")
                        tool_protocol_failure = tools is not None and is_groq_tool_protocol_failure(
                            error, model=model_name
                        )
                        if tool_protocol_failure:
                            details["provider_code"] = "tool_use_failed"
                            details["attempt"] = attempt + 1
                            if context:
                                context.audit("provider_attempt_failed", details)
                            if candidate == candidates[-1]:
                                raise
                            if context:
                                context.audit(
                                    "model_fallback",
                                    {
                                        "from": candidate,
                                        "to": candidates[candidates.index(candidate) + 1],
                                        "reason": "tool_protocol_failure",
                                    },
                                )
                            break
                        rate_limited = isinstance(error, RateLimitError) or status == 429
                        retryable = (
                            isinstance(error, (RateLimitError, Timeout))
                            or status in (408, 429)
                            or (isinstance(status, int) and 500 <= status <= 599)
                        )
                        if not retryable:
                            raise
                        details["attempt"] = attempt + 1
                        if context:
                            context.audit("provider_attempt_failed", details)
                        retry_after = retry_delay(details)
                        if (
                            not rate_limited
                            and attempt < self.settings.llm_max_retries
                            and retry_after <= 60
                        ):
                            retry_started = time.monotonic()
                            await asyncio.sleep(
                                max(random.SystemRandom().uniform(0, 2**attempt), retry_after)
                            )
                            retry_ms += (time.monotonic() - retry_started) * 1000
                            continue
                        self.cooldown_until[model_name] = time.monotonic() + max(30, retry_after)
                        if candidate == candidates[-1]:
                            raise
                        if context:
                            context.audit(
                                "model_fallback",
                                {
                                    "from": candidate,
                                    "to": candidates[candidates.index(candidate) + 1],
                                    "reason": "rate_limit" if rate_limited else "transient_failure",
                                },
                            )
                        break
                if response is not None:
                    break
        except ModelError:
            raise
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
            if status in (401, 403):
                self.quarantined.add(self.models[actual_route])
            if context:
                context.audit(
                    "provider_failure",
                    failure_details(error, model=self.models[actual_route], route=actual_route),
                )
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
        if (
            response_schema is not None
            and response_schema.__name__ == "AgentDraft"
            and not tools
            and (
                citation_catalog is not None
                or options.get("response_format", {}).get("type") == "json_schema"
            )
        ):
            try:
                message["content"] = decode_final(
                    str(message.get("content") or ""), citation_catalog
                )
            except ValueError:
                raise ModelError(
                    "invalid_response", "Final answer failed structured validation"
                ) from None
        if tools and response_schema is not None and not message.get("tool_calls"):
            # Groq forbids tools with structured output. Start a separate final-answer call.
            final_messages = [item for item in messages if item.get("role") in {"user", "tool"}]
            return await self.complete(route, final_messages, response_schema=response_schema)
        return Reply(message)
