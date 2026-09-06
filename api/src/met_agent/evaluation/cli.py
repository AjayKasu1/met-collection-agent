"""Command-line entry point for reproducible full and quick evaluations."""

import argparse
import asyncio
import hashlib
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from met_agent.config import load_settings
from met_agent.evaluation.models import EvalRow, Report
from met_agent.evaluation.publishing import publish
from met_agent.evaluation.reporting import save
from met_agent.evaluation.runner import run
from met_agent.evaluation.scoring import regression, summarize
from met_agent.llm.prompts import prompt_hash
from met_agent.llm.providers import configured_models
from met_agent.retrieval.rerank import MODEL as RERANKER_MODEL
from met_agent.retrieval.rerank import REVISION as RERANKER_REVISION
from met_agent.runtime import Runtime


def read_rows(path: Path, *, quick: bool) -> list[EvalRow]:
    rows = [
        EvalRow.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if len({r.id for r in rows}) != len(rows):
        raise ValueError("Duplicate golden IDs")
    selected = [r for r in rows if not r.verify] if quick else rows
    if not selected or (quick and len(selected) != 10):
        raise ValueError("Quick suite requires exactly ten verify:false rows; full cannot be empty")
    return selected


def deadline(value: str) -> float:
    seconds = float(value)
    if not 0 < seconds <= 600:
        raise argparse.ArgumentTypeError("Deadline must be between 0 and 600 seconds")
    return seconds


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--golden", type=Path, default=Path("evals/golden.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("evals/reports"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--chat-deadline-seconds",
        type=deadline,
        help="Explicit batch deadline override; recorded in report, never written to .env",
    )
    args = parser.parse_args(argv)
    rows = read_rows(args.golden, quick=args.quick)
    settings = load_settings()
    configured_deadline = settings.chat_deadline_seconds
    interactive_pacing = settings.llm_pacing_enabled
    settings = settings.model_copy(update={"llm_pacing_enabled": settings.eval_pacing_enabled})
    if args.chat_deadline_seconds is not None:
        settings = settings.model_copy(update={"chat_deadline_seconds": args.chat_deadline_seconds})
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # noqa: S607
    now = datetime.now(UTC)
    report = Report(
        run_id=f"{now:%Y-%m-%dT%H%M%S}-{sha[:8]}-{'quick' if args.quick else 'full'}",
        git_sha=sha,
        started_at=now.isoformat(),
        suite="quick" if args.quick else "full",
        golden_sha256=hashlib.sha256(args.golden.read_bytes()).hexdigest(),
        prompt_sha256={
            name: prompt_hash(name)
            for name in (
                "system_v2",
                "tools_v1",
                "intent_v2",
                "grounding_v1",
                "evaluation_v1",
                "citations_v1",
            )
        },
        retrieval={
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "dimensions": str(settings.embedding_dimensions),
            "reranker_model": RERANKER_MODEL,
            "reranker_revision": RERANKER_REVISION,
            "collection": settings.qdrant_collection,
            "visitor_collection": settings.qdrant_visitor_collection,
        },
        models=configured_models(settings),
        execution={
            "chat_deadline_seconds": str(settings.chat_deadline_seconds),
            "configured_chat_deadline_seconds": str(configured_deadline),
            "pacing_enabled": str(settings.llm_pacing_enabled),
            "interactive_pacing_enabled": str(interactive_pacing),
            "rate_limits": json.dumps(
                {k: v.model_dump() for k, v in settings.llm_rate_limits.items()}, sort_keys=True
            ),
            "fallback_enabled": str(settings.llm_fallback_enabled),
            "gateway_enabled": str(settings.use_ai_gateway),
        },
        execution_notes=[
            f"Chat deadline: {settings.chat_deadline_seconds:g} seconds. "
            f"Saved interactive deadline: {configured_deadline:g} seconds. "
            f"Evaluation pacing: {settings.llm_pacing_enabled}. "
            "Batch pacing and deadline settings do not establish interactive latency acceptance."
        ],
        expected_ids=[r.id for r in rows],
    )
    path = args.output_dir / (report.run_id + ".json")
    if args.resume:
        previous = Report.model_validate_json(args.resume.read_text())
        for field in (
            "git_sha",
            "suite",
            "golden_sha256",
            "prompt_sha256",
            "models",
            "expected_ids",
            "retrieval",
            "execution",
        ):
            if getattr(previous, field) != getattr(report, field):
                raise ValueError("Resume identity mismatch: " + field)
        report, path = previous, args.resume
    save(report, path)
    runtime = Runtime(settings)
    try:
        asyncio.run(run(runtime, report, rows, path))
        if runtime.telemetry:
            publish(report, runtime.telemetry.client)
    finally:
        runtime.close()
    print(json.dumps(summarize(report), indent=2))
    print(f"Report: {path}")
    if args.baseline:
        return int(regression(report, Report.model_validate_json(args.baseline.read_text())))
    return int(any(r.error for r in report.results))
