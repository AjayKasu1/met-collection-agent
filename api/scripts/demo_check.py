"""Exercise three golden questions through the actual HTTP chat interface and print evidence."""

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from met_agent.agent.models import AgentAnswer
from met_agent.config import Settings, load_settings
from met_agent.ingestion.verify import read_golden
from met_agent.main import create_app

DEMO_IDS = ("col-003", "vis-001", "ref-001")


def parse_answer(text: str) -> AgentAnswer:
    for block in text.split("\n\n"):
        lines = block.splitlines()
        name = next((line[7:] for line in lines if line.startswith("event: ")), "")
        data = "\n".join(line[6:] for line in lines if line.startswith("data: "))
        if name == "error":
            error = json.loads(data)
            raise RuntimeError(f"{error['code']}: {error['message']}")
        if name == "answer":
            return AgentAnswer.model_validate_json(data)
    raise RuntimeError("Chat stream ended without a final answer")


async def run(settings: Settings, *, app: Any = None) -> int:
    questions = {row.id: row for row in read_golden(Path("evals/golden.jsonl"))}
    app = app if app is not None else create_app(settings)
    records: list[dict[str, Any]] = []
    failed = False
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://demo.local", timeout=180
        ) as client,
    ):
        for question_id in DEMO_IDS:
            row = questions[question_id]
            print(f"\n{question_id}: {row.question}", flush=True)
            try:
                response = await client.post("/chat", json={"message": row.question})
                response.raise_for_status()
                answer = parse_answer(response.text)
                print(f"Answer: {answer.text}")
                print(
                    "Citations: "
                    + json.dumps(
                        [item.model_dump(mode="json") for item in answer.citations],
                        ensure_ascii=False,
                    )
                )
                print(
                    f"Route: {answer.route}; latency: {answer.latency_ms:.0f} ms; "
                    "estimated cost USD: "
                    f"{answer.cost_usd if answer.cost_usd is not None else 'unavailable'}"
                )
                print("Model paths: " + ", ".join(call.path for call in answer.model_calls))
                records.append({"id": question_id, "answer": answer.model_dump(mode="json")})
                if (
                    answer.grounding_score < 1
                    or (question_id.startswith("ref-") and not answer.policy_refusal)
                    or (not question_id.startswith("ref-") and not answer.citations)
                ):
                    failed = True
            except (RuntimeError, ValueError, ValidationError, httpx.HTTPError) as error:
                print(f"FAILED: {error}", flush=True)
                records.append({"id": question_id, "error": str(error)})
                failed = True
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "demo-check.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(run(load_settings()))


if __name__ == "__main__":
    raise SystemExit(main())
