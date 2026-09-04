"""Persist complete and interrupted evaluation reports with explicit provenance."""

import json
from pathlib import Path

from met_agent.evaluation.models import Report
from met_agent.evaluation.scoring import summarize


def cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def save(report: Report, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = report.model_dump(mode="json")
    document["summary"] = summarize(report)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
    lines = [
        f"# Evaluation {report.run_id}",
        "",
        f"Suite: {report.suite}. Commit: `{report.git_sha}`. Complete: {report.complete}.",
        "",
        "Agent latency excludes the independent judge and the retrieval-only top-10 probe. "
        "Provider pacing is included. Costs are standard-rate token estimates, not invoices.",
        "",
        *report.execution_notes,
        "",
        "| Metric | Value |",
        "| --- | --- |",
        *[f"| {key} | {cell(value)} |" for key, value in summarize(report).items()],
        "",
        "| Row | Result | Route | Latency ms | Cost USD | Faithfulness | Reason |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in report.results:
        lines.append(
            f"| {r.row.id} | {'PASS' if r.passed else '**FAIL**'} | "
            f"{r.answer.route if r.answer else 'N/A'} | {r.latency_ms:.1f} | "
            f"{r.answer.cost_usd if r.answer else 'N/A'} | "
            f"{r.judgment.faithfulness if r.judgment else 'N/A'} | {cell(r.reason)} |"
        )
    for r in report.results:
        lines += [
            "",
            f"## {r.row.id}",
            "",
            r.row.question,
            "",
            r.answer.text if r.answer else r.reason,
            "",
            f"Session: `{r.session_id}`.",
        ]
        if r.answer:
            for citation in r.answer.citations:
                lines.append(
                    f"- Source {citation.object_id or citation.source_url}: {cell(citation.quote)}"
                )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
