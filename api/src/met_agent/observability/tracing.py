"""Opt-in Langfuse traces with explicit configuration and a per-call LiteLLM callback."""

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any, cast

from met_agent.agent.events import EventStore
from met_agent.agent.models import AgentAnswer, ChatRequest
from met_agent.config import Settings


class Telemetry:
    def __init__(self, settings: Settings, events: EventStore, *, client: Any = None) -> None:
        from langfuse import Langfuse

        if not settings.optional_services.langfuse:
            raise ValueError("Langfuse configuration is incomplete")
        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            raise ValueError("Langfuse keys are required")
        self.public_key = settings.langfuse_public_key.get_secret_value()
        self.events = events
        self.client = (
            client
            if client is not None
            else Langfuse(
                public_key=self.public_key,
                secret_key=settings.langfuse_secret_key.get_secret_value(),
                base_url=str(settings.langfuse_base_url),
                environment=settings.app_env,
                release=settings.git_sha,
                mask=lambda data, **kwargs: events.redact(data),
            )
        )
        self.callback = make_callback(self.client, events)

    async def run(
        self, action: Callable[[ChatRequest], Awaitable[AgentAnswer]], request: ChatRequest
    ) -> AgentAnswer:
        from langfuse import observe, propagate_attributes

        @observe(name="chat", as_type="agent", capture_input=False, capture_output=False)
        async def observed() -> AgentAnswer:
            with propagate_attributes(session_id=str(request.session_id), trace_name="chat"):
                self.client.update_current_span(
                    input=self.events.redact(request.model_dump(mode="json"))
                )
                answer = await action(request)
                self.client.update_current_span(
                    output=self.events.redact(answer.model_dump(mode="json"))
                )
                self.client.score_current_trace(
                    name="grounding_score", value=answer.grounding_score
                )
                return answer

        return await cast(Callable[..., Awaitable[AgentAnswer]], observed)(
            langfuse_public_key=self.public_key
        )

    @contextmanager
    def tool(self, name: str, arguments: str) -> Iterator[None]:
        span = self.client.start_observation(
            name=name, as_type="tool", input=self.events.redact(arguments)
        )
        try:
            yield
        except Exception:
            span.update(level="ERROR", status_message="Tool operation failed")
            raise
        finally:
            span.end()

    def close(self) -> None:
        self.callback.close()
        self.client.flush()
        self.client.shutdown()


def make_callback(client: Any, events: EventStore) -> Any:
    """Use the supported CustomLogger hooks, without SDK-global credentials or callbacks."""
    from litellm.integrations.custom_logger import CustomLogger

    from met_agent.llm import cost

    class Callback(CustomLogger):
        def __init__(self) -> None:
            super().__init__()
            self.spans: dict[str, Any] = {}

        def metadata(self) -> dict[str, str]:
            return {
                "trace_id": client.get_current_trace_id() or "",
                "parent_span_id": client.get_current_observation_id() or "",
            }

        def log_pre_api_call(self, model: Any, messages: Any, kwargs: Any) -> None:
            call_id = str(kwargs.get("litellm_call_id", ""))
            context = kwargs.get("litellm_params", {}).get("metadata", {}).get("trace_context")
            if call_id not in self.spans:
                self.spans[call_id] = client.start_observation(
                    name="model",
                    as_type="generation",
                    model=str(model),
                    input=events.redact(messages),
                    trace_context=context,
                )

        async def async_log_success_event(
            self, kwargs: Any, response_obj: Any, start_time: datetime, end_time: datetime
        ) -> None:
            span = self.spans.pop(str(kwargs.get("litellm_call_id", "")), None)
            if span is None:
                return
            estimated_cost = cost.estimate_cost(response_obj)
            span.update(
                output=events.redact(response_obj.choices[0].message.content or ""),
                usage_details={
                    "input": response_obj.usage.prompt_tokens,
                    "output": response_obj.usage.completion_tokens,
                },
                cost_details={"total": estimated_cost} if estimated_cost is not None else None,
                metadata={"latency_ms": (end_time - start_time).total_seconds() * 1000},
            )
            span.end()

        async def async_log_failure_event(
            self, kwargs: Any, response_obj: Any, start_time: datetime, end_time: datetime
        ) -> None:
            span = self.spans.pop(str(kwargs.get("litellm_call_id", "")), None)
            if span is not None:
                span.update(
                    level="ERROR",
                    status_message="Provider call failed",
                    metadata={"latency_ms": (end_time - start_time).total_seconds() * 1000},
                )
                span.end()

        def close(self) -> None:
            for span in self.spans.values():
                span.update(level="WARNING", status_message="Model call interrupted")
                span.end()
            self.spans.clear()

    return Callback()
