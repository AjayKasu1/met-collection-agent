"""Run independent sessions sequentially through the production agent and its shared pacer."""

import json
import time
from pathlib import Path
from uuid import UUID, uuid4

import httpx

from met_agent.agent.events import Event
from met_agent.agent.models import ChatRequest
from met_agent.evaluation.models import EvalRow, Judgment, Report, RowResult
from met_agent.evaluation.reporting import save
from met_agent.evaluation.scoring import score
from met_agent.llm.chat import CURRENT_CALL, CallContext, ModelError, structured
from met_agent.llm.prompts import load_prompt
from met_agent.runtime import Runtime
from met_agent.tools.models import CollectionSearchResult, ToolResult


def evidence(events: list[Event]) -> list[dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for event in events:
        if event.kind in {"tool_result", "validation_error"}:
            result = ToolResult.model_validate(event.data)
            for item in result.model_evidence():
                records[item.key] = item.model_dump(mode="json", exclude_none=True)
    return list(records.values())


async def evaluate(runtime: Runtime, result: RowResult) -> None:
    started = time.monotonic()
    session = UUID(result.session_id)
    try:
        result.answer = await runtime.chat(
            ChatRequest(message=result.row.question, session_id=session)
        )
        result.latency_ms = result.answer.latency_ms
        events = runtime.events.read(session, limit=500)
        # The same instance is essential: judging must share the agent's per-model quota ledger.
        if runtime.agent is None:
            raise RuntimeError("Runtime did not initialize its agent")
        context = CallContext(
            audit=lambda kind, data: (
                runtime.events.append(session, result.answer.turn_id, "eval_" + kind, data)
                if result.answer
                else None
            )
        )
        token = CURRENT_CALL.set(context)
        judged = time.monotonic()
        try:
            result.judgment = await structured(
                runtime.agent.model,
                "lite",
                load_prompt("evaluation_v1"),
                {
                    "question": result.row.question,
                    "answer": result.answer.text,
                    "evidence": json.loads(json.dumps(evidence(events))),
                },
                Judgment,
            )
        finally:
            result.judge_ms = (time.monotonic() - judged) * 1000
            result.judge_calls = context.ledger.calls
            result.judge_cost_usd = context.ledger.cost if context.ledger.calls else None
            CURRENT_CALL.reset(token)
        if result.row.expect == "retrieval":
            # A separate measured probe supplies a full top 10, without expanding model context.
            probe = time.monotonic()
            found = await runtime.tools.execute(
                "search_collection", json.dumps({"query": result.row.question, "k": 10})
            )
            result.retrieval_ms = (time.monotonic() - probe) * 1000
            if found.error or not isinstance(found.output, CollectionSearchResult):
                raise RuntimeError("Retrieval evaluation probe failed")
            result.retrieval_ids = [item.object_id for item in found.output.objects]
        score(result, events)
    except (ModelError, httpx.HTTPError, ValueError, RuntimeError) as error:
        # Provider exceptions may include credentials or request bodies. Retain safe codes only.
        result.error = error.code if isinstance(error, ModelError) else type(error).__name__
        result.reason = "Evaluation failed: " + result.error
        if result.answer is None:
            result.latency_ms = (time.monotonic() - started) * 1000


async def run(runtime: Runtime, report: Report, rows: list[EvalRow], path: Path) -> None:
    completed = {r.row.id for r in report.results}
    for row in rows:
        if row.id in completed:
            continue
        result = RowResult(row=row, session_id=str(uuid4()))
        await evaluate(runtime, result)
        report.results.append(result)
        save(report, path)
        print(
            f"{row.id}: {'PASS' if result.passed else 'FAIL'}; {result.reason}; "
            f"{result.latency_ms:.0f} ms; session={result.session_id}",
            flush=True,
        )
    report.complete = True
    save(report, path)
