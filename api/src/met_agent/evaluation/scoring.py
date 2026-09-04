"""Score exact golden contracts without relaxing factual or safety requirements."""

import math
from statistics import mean
from typing import Any

from met_agent.agent.events import Event
from met_agent.evaluation.models import Report, RowResult


def score(result: RowResult, events: list[Event]) -> None:
    answer, row, judge = result.answer, result.row, result.judgment
    if result.error or answer is None or judge is None:
        result.reason = result.error or "Missing answer or independent judgment"
        return
    if row.expect == "contains":
        missing = [word for word in row.expected if word.casefold() not in answer.text.casefold()]
        passed = not missing
        reason = (
            "Missing expected strings: " + ", ".join(missing) if missing else "All strings found"
        )
    elif row.expect == "retrieval":
        result.hits = {
            str(k): bool(set(row.expected_object_ids) & set(result.retrieval_ids[:k]))
            for k in (1, 5, 10)
        }
        passed = result.hits["5"]
        reason = "Hit@5" if passed else "No labeled object in top five"
    elif row.expect == "handoff":
        passed = answer.handoff is not None and any(
            e.kind == "tool_call" and isinstance(e.data, dict) and e.data.get("name") == "handoff"
            for e in events
        )
        reason = "Handoff tool called" if passed else "Missing handoff tool call"
    else:
        fired = any(
            e.kind == "guardrail"
            and isinstance(e.data, dict)
            and e.data.get("policy") == "non_interpretive"
            and e.data.get("decision") == "refuse"
            for e in events
        )
        passed = fired and answer.policy_refusal and judge.no_opinion
        reason = "Interpretive guardrail and no opinion" if passed else "Refusal contract failed"
    result.passed = passed and answer.grounding_score == 1
    result.reason = reason if answer.grounding_score == 1 else reason + "; grounding failed"


def average(values: list[float]) -> float | None:
    return mean(values) if values else None


def summarize(report: Report) -> dict[str, Any]:
    rows = report.results
    categories = sorted({r.row.category for r in rows})
    latencies = sorted(r.latency_ms for r in rows)
    priced = [r.answer.cost_usd for r in rows if r.answer and r.answer.cost_usd is not None]
    return {
        "completed": len(rows),
        "expected": len(report.expected_ids),
        "pass_rate": sum(r.passed for r in rows) / len(report.expected_ids),
        "category_pass_rate": {
            c: mean(float(r.passed) for r in rows if r.row.category == c) for c in categories
        },
        "hit_at_k": {
            str(k): average([float(r.hits[str(k)]) for r in rows if str(k) in r.hits])
            for k in (1, 5, 10)
        },
        "retrieval_scored": sum(bool(r.hits) for r in rows),
        "mean_faithfulness": average([r.judgment.faithfulness for r in rows if r.judgment]),
        "judged_rows": sum(r.judgment is not None for r in rows),
        "mean_latency_ms": average(latencies),
        "p95_latency_ms": latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None,
        "mean_cost_usd": average(priced),
        "priced_rows": len(priced),
        "judge_cost_usd": sum(r.judge_cost_usd or 0 for r in rows)
        if all(r.judge_cost_usd is not None for r in rows)
        else None,
        "lite_fraction": sum(bool(r.answer and r.answer.route == "lite") for r in rows)
        / len(report.expected_ids),
    }


def regression(report: Report, baseline: Report) -> bool:
    """Reject incomparable or incomplete runs; exactly five percentage points is allowed."""
    if not report.complete or not baseline.complete:
        raise ValueError("Regression comparison requires complete runs")
    if report.suite != baseline.suite or report.expected_ids != baseline.expected_ids:
        raise ValueError("Regression suites differ")
    if report.golden_sha256 != baseline.golden_sha256:
        raise ValueError("Golden labels changed; explicitly review a new baseline")
    drop = float(summarize(baseline)["pass_rate"]) - float(summarize(report)["pass_rate"])
    return drop > 0.05 + 1e-12
