"""Protect scoring, independent judgments, checkpoints and publication semantics."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from pydantic import ValidationError

from met_agent.agent.events import Event, EventStore
from met_agent.agent.models import AgentAnswer
from met_agent.evaluation.cli import read_rows
from met_agent.evaluation.models import EvalRow, Judgment, Report, RowResult
from met_agent.evaluation.publishing import publish
from met_agent.evaluation.reporting import save
from met_agent.evaluation.runner import evaluate, run
from met_agent.evaluation.scoring import regression, score, summarize
from met_agent.llm.chat import ModelError, Reply
from met_agent.runtime import Runtime


def row(**values: Any) -> EvalRow:
    return EvalRow.model_validate(
        {
            "id": "a",
            "category": "collection",
            "question": "Which gallery?",
            "expect": "contains",
            "expected": ["Gallery", "131"],
            "expected_object_ids": [1],
            "verify": False,
            "notes": "fixture",
            **values,
        }
    )


def result(**values: Any) -> RowResult:
    session, turn = uuid4(), uuid4()
    return RowResult(
        row=row(**values),
        session_id=str(session),
        answer=AgentAnswer(
            text="GALLERY 131",
            language="en",
            session_id=session,
            turn_id=turn,
            route="lite",
            cost_usd=0.001,
            latency_ms=10,
            grounding_score=1,
        ),
        judgment=Judgment(faithfulness=1, no_opinion=True, reason="Supported"),
        latency_ms=10,
    )


def report(results: list[RowResult], *, complete: bool = True) -> Report:
    return Report(
        run_id="fixture",
        git_sha="abc",
        started_at="2026-09-04",
        suite="quick",
        golden_sha256="abc",
        prompt_sha256={},
        models={},
        expected_ids=[r.row.id for r in results],
        results=results,
        complete=complete,
    )


def event(r: RowResult, kind: str, data: Any) -> Event:
    assert r.answer
    return Event(
        sequence=1,
        session_id=r.answer.session_id,
        turn_id=r.answer.turn_id,
        timestamp="2026-09-04",
        kind=kind,
        data=data,
    )


def test_all_strings_and_unchanged_grounding() -> None:
    r = result()
    score(r, [])
    assert r.passed
    r.row.expected.append("second")
    score(r, [])
    assert not r.passed and "second" in r.reason
    assert r.answer
    r.answer.grounding_score = 0
    score(r, [])
    assert "grounding failed" in r.reason


def test_deadline_validation_and_baseline_comparability() -> None:
    import argparse

    from met_agent.evaluation.cli import deadline

    assert deadline("300") == 300
    for value in ("0", "-1", "601", "nan", "inf"):
        with pytest.raises(argparse.ArgumentTypeError):
            deadline(value)
    previous = report([result()])
    current = previous.model_copy(deep=True)
    current.execution["chat_deadline_seconds"] = "300"
    with pytest.raises(ValueError, match="deadlines differ"):
        regression(current, previous)


def test_refusal_requires_event_and_independent_no_opinion() -> None:
    r = result(expect="refuse_interpretive")
    assert r.answer and r.judgment
    r.answer.policy_refusal = True
    score(r, [])
    assert not r.passed
    e = event(r, "guardrail", {"policy": "non_interpretive", "decision": "refuse"})
    score(r, [e])
    assert r.passed
    r.judgment.no_opinion = False
    score(r, [e])
    assert not r.passed


def test_handoff_requires_actual_tool_call() -> None:
    from met_agent.tools.models import Handoff

    r = result(expect="handoff")
    assert r.answer
    r.answer.handoff = Handoff(reason="Outside scope", suggested_contact="info@metmuseum.org")
    score(r, [])
    assert not r.passed
    score(r, [event(r, "tool_call", {"name": "handoff"})])
    assert r.passed


def test_hits_and_invalid_labels() -> None:
    with pytest.raises(ValidationError):
        row(expect="retrieval", expected_object_ids=[])
    with pytest.raises(ValidationError):
        row(expected=[])
    r = result(expect="retrieval")
    r.retrieval_ids = [2, 3, 4, 5, 1, 6, 7, 8, 9, 10]
    score(r, [])
    assert r.hits == {"1": False, "5": True, "10": True} and r.passed
    r.retrieval_ids = []
    score(r, [])
    assert not r.passed


def test_summary_unknowns_failures_and_p95() -> None:
    rows = [result(id=str(i)) for i in range(20)]
    for i, r in enumerate(rows):
        r.latency_ms = i + 1
    rows[0].answer = None
    rows[0].judgment = None
    score(rows[0], [])
    summary = summarize(report(rows))
    assert summary["p95_latency_ms"] == 19
    assert summary["mean_latency_ms"] == 10.5
    assert summary["judged_rows"] == summary["priced_rows"] == 19
    assert summary["hit_at_k"]["10"] is None
    assert summary["judge_cost_usd"] is None


def test_regression_exact_five_points_and_incomparable_runs() -> None:
    rows = [result(id=str(i)) for i in range(20)]
    for r in rows:
        r.passed = True
    baseline = report(rows)
    current = baseline.model_copy(deep=True)
    current.results[0].passed = False
    assert not regression(current, baseline)
    current.results[1].passed = False
    assert regression(current, baseline)
    current.golden_sha256 = "changed"
    with pytest.raises(ValueError, match="labels changed"):
        regression(current, baseline)
    current.golden_sha256 = baseline.golden_sha256
    current.expected_ids = ["different"]
    with pytest.raises(ValueError, match="suites differ"):
        regression(current, baseline)
    current.complete = False
    with pytest.raises(ValueError, match="complete"):
        regression(current, baseline)


def test_read_and_write_reports(tmp_path: Path) -> None:
    golden = tmp_path / "golden.jsonl"
    golden.write_text("\n".join(row(id=str(i)).model_dump_json() for i in range(10)))
    assert len(read_rows(golden, quick=True)) == 10
    r = report([result()])
    path = tmp_path / "run.json"
    save(r, path)
    assert Report.model_validate_json(path.read_text()) == r
    assert "**FAIL**" in path.with_suffix(".md").read_text()
    golden.write_text(row().model_dump_json())
    with pytest.raises(ValueError, match="ten"):
        read_rows(golden, quick=True)
    golden.write_text(row().model_dump_json() + "\n" + row().model_dump_json())
    with pytest.raises(ValueError, match="Duplicate"):
        read_rows(golden, quick=False)


class FakeModel:
    async def complete(self, *args: Any, **kwargs: Any) -> Reply:
        assert "expected" not in args[1][1]["content"]
        return Reply(
            {
                "content": Judgment(
                    faithfulness=1, no_opinion=True, reason="Supported"
                ).model_dump_json()
            }
        )


class FakeRuntime:
    def __init__(self, tmp_path: Path, *, fail: bool = False) -> None:
        self.events = EventStore(tmp_path / "events.sqlite")
        self.agent = SimpleNamespace(model=FakeModel())
        self.fail = fail

    async def chat(self, request: Any) -> AgentAnswer:
        if self.fail:
            raise ModelError("authentication_error", "sensitive provider body")
        answer = result().answer
        assert answer
        return answer.model_copy(update={"session_id": request.session_id})


def test_runner_judge_checkpoint_resume_and_error_redaction(tmp_path: Path) -> None:
    runtime = cast(Runtime, FakeRuntime(tmp_path))
    r = report([], complete=False)
    r.expected_ids = ["a"]
    path = tmp_path / "run.json"
    asyncio.run(run(runtime, r, [row()], path))
    assert r.complete and r.results[0].passed
    assert r.results[0].judgment
    asyncio.run(run(runtime, r, [row()], path))
    assert len(r.results) == 1
    failure = result()
    failure.answer = None
    asyncio.run(evaluate(cast(Runtime, FakeRuntime(tmp_path, fail=True)), failure))
    assert failure.error == "authentication_error"
    assert "sensitive" not in failure.model_dump_json()


def test_publication_records_results_without_inference() -> None:
    class Client:
        def create_dataset(self, **kwargs: Any) -> None:
            assert kwargs["name"].startswith("met-golden-")

        def create_dataset_item(self, **kwargs: Any) -> Any:
            return SimpleNamespace(input=kwargs["input"])

        def run_experiment(self, **kwargs: Any) -> None:
            output = kwargs["task"](item=kwargs["data"][0])
            assert len(kwargs["evaluators"][0](output=output)) == 2
            assert kwargs["metadata"]["execution"] == "recorded-results-import"
            assert kwargs["max_concurrency"] == 1

        def flush(self) -> None:
            pass

    publish(report([result()]), Client())


def test_cli_checkpoint_and_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: Any
) -> None:
    from met_agent.evaluation import cli

    class RuntimeStub(FakeRuntime):
        telemetry = None

        def close(self) -> None:
            pass

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "Runtime", lambda settings: RuntimeStub(tmp_path))
    golden = tmp_path / "golden.jsonl"
    golden.write_text("\n".join(row(id=str(i)).model_dump_json() for i in range(10)))
    args = ["--quick", "--golden", str(golden), "--output-dir", str(tmp_path / "reports")]
    assert cli.main(args) == 0
    path = next((tmp_path / "reports").glob("*.json"))
    assert cli.main([*args, "--resume", str(path), "--baseline", str(path)]) == 0
    data = json.loads(path.read_text())
    data["git_sha"] = "changed"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Resume identity"):
        cli.main([*args, "--resume", str(path)])


def test_incomplete_baseline_cannot_claim_complete() -> None:
    r = report([result()])
    data = r.model_dump()
    data["results"] = []
    with pytest.raises(ValidationError, match="every expected row"):
        Report.model_validate(data)


def test_retrieval_probe_and_evidence_are_independent(tmp_path: Path) -> None:
    from met_agent.retrieval.hybrid import RetrievedObject, ScoreBreakdown
    from met_agent.tools.models import CollectionSearchResult, Evidence, ToolResult

    class Tools:
        async def execute(self, name: str, arguments: str) -> ToolResult:
            assert name == "search_collection" and json.loads(arguments)["k"] == 10
            return ToolResult(
                name="search_collection",
                output=CollectionSearchResult(
                    objects=[
                        RetrievedObject(
                            object_id=1,
                            title="Test",
                            source_url="https://example.org/1",
                            text="Gallery 131",
                            record={},
                            scores=ScoreBreakdown(rrf=0.1, rerank=1),
                        )
                    ]
                ),
            )

    class SearchRuntime(FakeRuntime):
        tools = Tools()

        async def chat(self, request: Any) -> AgentAnswer:
            answer = await super().chat(request)
            self.events.append(
                answer.session_id,
                answer.turn_id,
                "tool_result",
                ToolResult(
                    name="get_object",
                    evidence=[
                        Evidence(key="1", object_id=1, text="Gallery 131", kind="live_object")
                    ],
                ).model_dump(mode="json"),
            )
            return answer

    r = result(expect="retrieval")
    asyncio.run(evaluate(cast(Runtime, SearchRuntime(tmp_path)), r))
    assert r.passed and r.retrieval_ids == [1] and r.retrieval_ms > 0
