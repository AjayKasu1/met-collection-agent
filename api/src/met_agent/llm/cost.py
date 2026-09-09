"""Estimate provider charges from reported usage; unknown pricing stays unknown."""

import math
from dataclasses import dataclass, field
from typing import Any

from met_agent.agent.models import ModelCall

# USD per million tokens. Groq prices verified 2026-09-04; Gemini prices
# verified 2026-09-09. Sources: https://console.groq.com/docs/models and
# https://ai.google.dev/gemini-api/docs/pricing
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "groq/openai/gpt-oss-120b": (0.15, 0.60),
    "groq/openai/gpt-oss-20b": (0.075, 0.30),
    "gemini/gemini-3.7-flash": (0.75, 3.75),
    "gemini/gemini-3.1-flash-lite": (0.25, 1.50),
}


@dataclass
class CallLedger:
    calls: list[ModelCall] = field(default_factory=list)

    @property
    def cost(self) -> float | None:
        if any(call.cost_usd is None for call in self.calls):
            return None
        return sum(call.cost_usd or 0 for call in self.calls)


def estimate_cost(response: Any) -> float | None:
    from litellm import completion_cost

    if response.model in MODEL_PRICES:
        input_price, output_price = MODEL_PRICES[response.model]
        return (
            int(response.usage.prompt_tokens) * input_price
            + int(response.usage.completion_tokens) * output_price
        ) / 1_000_000
    try:
        result = float(completion_cost(completion_response=response))
        return result if math.isfinite(result) and result >= 0 else None
    except Exception:
        # Pricing discovery is optional and must not discard a valid provider response.
        return None
