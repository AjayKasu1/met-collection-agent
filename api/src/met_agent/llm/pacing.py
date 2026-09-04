"""Reserve prompt and completion tokens in shared per-model rolling minute budgets."""

import asyncio
import json
import math
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from met_agent.config import LLMRateLimit, Settings


class RequestBudgetError(ValueError):
    """A request must fit one model minute; waiting cannot make an oversized request fit."""


@dataclass
class Reservation:
    started: float
    tokens: int


def prompt_tokens(
    model: str, messages: list[dict[str, Any]], tools: list[dict[str, object]] | None
) -> int:
    """Count the full chat envelope and tools, with a margin for provider tokenization drift."""
    from litellm import token_counter

    try:
        options: dict[str, Any] = {"model": model, "messages": messages, "tools": tools}
        count = int(token_counter(**options))
        if count < 0:
            raise ValueError("Negative token estimate")
        return math.ceil(count * 1.1) + 64
    except Exception:
        # A UTF-8 byte bound is conservative when the tokenizer is unavailable.
        payload = json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False)
        return len(payload.encode("utf-8")) + 64


class TokenPacer:
    """Share limits across aliases in one process; external traffic still needs provider 429s."""

    def __init__(
        self,
        settings: Settings,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.enabled = settings.llm_pacing_enabled
        self.limits = settings.llm_rate_limits
        self.default = LLMRateLimit(
            tokens_per_minute=settings.llm_default_tokens_per_minute,
            requests_per_minute=settings.llm_default_requests_per_minute,
        )
        self.clock, self.sleep = clock, sleep
        self.events: dict[str, deque[Reservation]] = defaultdict(deque)
        self.lock = asyncio.Lock()

    async def reserve(self, model: str, tokens: int) -> Reservation | None:
        if not self.enabled:
            return None
        limit = self.limits.get(model, self.default)
        if not 0 < tokens <= limit.tokens_per_minute:
            raise RequestBudgetError(
                "Request exceeds the configured per-minute token budget; "
                "reduce context or set the model's verified account limit"
            )
        while True:
            async with self.lock:
                now = self.clock()
                events = self.events[model]
                while events and events[0].started <= now - 60:
                    events.popleft()
                if (
                    len(events) < limit.requests_per_minute
                    and sum(item.tokens for item in events) + tokens <= limit.tokens_per_minute
                ):
                    reservation = Reservation(now, tokens)
                    events.append(reservation)
                    return reservation
                wait = max(0.01, events[0].started + 60 - now)
            await self.sleep(wait)

    async def reconcile(self, reservation: Reservation | None, actual_tokens: int) -> None:
        """Return unused completion allowance, retaining conservative reservations on failures."""
        if reservation is not None:
            async with self.lock:
                reservation.tokens = max(1, actual_tokens)
