"""Exercise rolling token/request budgets, persisted reservations, and quota-day boundaries."""

from datetime import datetime

import pytest

from met_agent.ingestion.quota import (
    PACIFIC,
    QuotaBudget,
    QuotaLimits,
    QuotaState,
    next_reset,
    quota_day,
)


def timestamp(day: str) -> float:
    return datetime.fromisoformat(day).replace(tzinfo=PACIFIC).timestamp()


def test_tokens_and_input_requests_independently_constrain_batches() -> None:
    now = timestamp("2026-09-04T10:00:00")
    limits = QuotaLimits(requests_per_minute=100, tokens_per_minute=1000)
    state = QuotaState(day=quota_day(now))
    budget = QuotaBudget(limits, state)
    budget.reserve(now, requests=20, tokens=900)
    assert budget.delay(now + 1, requests=20, tokens=200) == pytest.approx(59.1)
    assert budget.delay(now + 1, requests=81, tokens=1) == pytest.approx(59.1)
    assert budget.delay(now + 1, requests=80, tokens=100) == 0
    with pytest.raises(ValueError, match="waiting"):
        budget.reserve(now + 1, requests=81, tokens=1)
    restored = QuotaBudget(limits, QuotaState.model_validate_json(state.model_dump_json()))
    assert restored.delay(now + 1, requests=20, tokens=200) == pytest.approx(59.1)
    assert restored.delay(now + 61, requests=100, tokens=1000) == 0


def test_daily_budget_resets_at_midnight_pacific_and_honors_external_usage() -> None:
    now = timestamp("2026-09-04T23:59:00")
    budget = QuotaBudget(QuotaLimits(), QuotaState(day=quota_day(now), used_today=950))
    assert budget.remaining_today(now) == 50
    assert budget.delay(now, requests=100, tokens=100) == 61
    budget.defer(now, seconds=90, daily=True)
    assert budget.remaining_today(now) == 0
    assert budget.delay(now, requests=100, tokens=100) == 90
    assert budget.delay(now + 91, requests=100, tokens=100) == 0
    assert budget.remaining_today(now + 91) == 1000


@pytest.mark.parametrize(
    ("day", "hours"), [("2026-03-08T00:00:00", 23), ("2026-11-01T00:00:00", 25)]
)
def test_daily_reset_follows_daylight_saving_time(day: str, hours: int) -> None:
    now = timestamp(day)
    assert next_reset(now) - now == hours * 3600


def test_invalid_batches_are_rejected_and_multi_day_eta_includes_daily_cap() -> None:
    now = timestamp("2026-09-04T10:00:00")
    budget = QuotaBudget(QuotaLimits(), QuotaState(day=quota_day(now), used_today=200))
    for requests, tokens in [(0, 1), (101, 1), (1, 0), (1, 30_001)]:
        with pytest.raises(ValueError):
            budget.delay(now, requests=requests, tokens=tokens)
    assert budget.estimate_finish(now, remaining=0, tokens_per_object=100) == now
    assert budget.estimate_finish(now, remaining=100, tokens_per_object=100) == now + 60
    assert budget.estimate_finish(now, remaining=20_000, tokens_per_object=100) == timestamp(
        "2026-09-24T00:02:00"
    )


def test_minute_wait_releases_only_enough_reservations() -> None:
    now = timestamp("2026-09-04T10:00:00")
    budget = QuotaBudget(QuotaLimits(), QuotaState(day=quota_day(now)))
    budget.reserve(now, requests=40, tokens=100)
    budget.reserve(now + 5, requests=40, tokens=100)
    assert budget.delay(now + 10, requests=30, tokens=100) == pytest.approx(50.1)
