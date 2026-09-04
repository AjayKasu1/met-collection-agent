"""Account for input requests and tokens across rolling minutes and Pacific quota days."""

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

PACIFIC = ZoneInfo("America/Los_Angeles")


class QuotaLimits(BaseModel):
    """Explicit project limits, verified in AI Studio before a long ingestion run."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    requests_per_minute: int = Field(default=100, gt=0)
    tokens_per_minute: int = Field(default=30_000, gt=0)
    requests_per_day: int = Field(default=1000, gt=0)


class Reservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    at: float = Field(ge=0, allow_inf_nan=False)
    requests: int = Field(gt=0)
    tokens: int = Field(gt=0)


class QuotaState(BaseModel):
    """Serialize reservations before a request so interruptions cannot reset the budget."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    day: str
    used_today: int = Field(default=0, ge=0)
    minute: list[Reservation] = Field(default_factory=list)
    not_before: float = Field(default=0, ge=0, allow_inf_nan=False)


def quota_day(now: float) -> str:
    return datetime.fromtimestamp(now, PACIFIC).date().isoformat()


def next_reset(now: float) -> float:
    """Use the provider's midnight-Pacific boundary, including daylight-saving transitions."""
    tomorrow = datetime.fromtimestamp(now, PACIFIC).date() + timedelta(days=1)
    return datetime.combine(tomorrow, time(), PACIFIC).timestamp()


class QuotaBudget:
    """Each batched input consumes one RPM/RPD request, regardless of HTTP batching."""

    def __init__(self, limits: QuotaLimits, state: QuotaState) -> None:
        self.limits = limits
        self.state = state

    def refresh(self, now: float) -> None:
        day = quota_day(now)
        if day != self.state.day:
            self.state.day = day
            self.state.used_today = 0
        self.state.minute = [item for item in self.state.minute if item.at + 60.1 > now]

    def remaining_today(self, now: float) -> int:
        self.refresh(now)
        return max(0, self.limits.requests_per_day - self.state.used_today)

    def delay(self, now: float, *, requests: int, tokens: int) -> float:
        """Return the earliest safe admission time for a whole batch without reserving it."""
        if not 0 < requests <= min(self.limits.requests_per_minute, self.limits.requests_per_day):
            raise ValueError("Batch exceeds a request quota or has no inputs")
        if not 0 < tokens <= self.limits.tokens_per_minute:
            raise ValueError("Batch exceeds the token quota or has no tokens")
        self.refresh(now)
        until = max(now, self.state.not_before)
        if self.state.used_today + requests > self.limits.requests_per_day:
            until = max(until, next_reset(now) + 1)
        minute_requests = sum(item.requests for item in self.state.minute)
        minute_tokens = sum(item.tokens for item in self.state.minute)
        for item in self.state.minute:
            if (
                minute_requests + requests <= self.limits.requests_per_minute
                and minute_tokens + tokens <= self.limits.tokens_per_minute
            ):
                break
            until = max(until, item.at + 60.1)
            minute_requests -= item.requests
            minute_tokens -= item.tokens
        return max(0.0, until - now)

    def reserve(self, now: float, *, requests: int, tokens: int) -> None:
        if self.delay(now, requests=requests, tokens=tokens) > 0:
            raise ValueError("Quota reservation requires waiting first")
        self.state.used_today += requests
        self.state.minute.append(Reservation(at=now, requests=requests, tokens=tokens))

    def defer(self, now: float, *, seconds: float, daily: bool = False) -> None:
        """Honor a provider rejection, including usage from other clients in the same project."""
        self.refresh(now)
        if daily:
            self.state.used_today = self.limits.requests_per_day
        self.state.not_before = max(self.state.not_before, now + seconds)

    def estimate_finish(self, now: float, *, remaining: int, tokens_per_object: float) -> float:
        """Estimate quota-constrained completion, excluding downtime and upstream failures."""
        if remaining <= 0:
            return now
        available = self.remaining_today(now)
        per_minute = min(
            self.limits.requests_per_minute,
            self.limits.tokens_per_minute / max(1.0, tokens_per_object),
        )
        if remaining <= available:
            return max(now, self.state.not_before) + remaining / per_minute * 60
        later = remaining - available
        extra_days = (later - 1) // self.limits.requests_per_day
        last_count = (later - 1) % self.limits.requests_per_day + 1
        final_day = datetime.fromtimestamp(next_reset(now), PACIFIC).date() + timedelta(
            days=extra_days
        )
        return (
            datetime.combine(final_day, time(), PACIFIC).timestamp() + last_count / per_minute * 60
        )
