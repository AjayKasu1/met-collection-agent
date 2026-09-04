"""Estimate provider charges from reported usage; unknown pricing stays unknown."""

import math
from dataclasses import dataclass, field
from typing import Any

from met_agent.agent.models import ModelCall


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

    try:
        result = float(completion_cost(completion_response=response))
        return result if math.isfinite(result) and result >= 0 else None
    except Exception:
        # Pricing discovery is optional and must not discard a valid provider response.
        return None
