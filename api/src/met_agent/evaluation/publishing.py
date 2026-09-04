"""Publish recorded evaluations as dataset experiments without repeating paid inference."""

from typing import Any
from uuid import NAMESPACE_URL, uuid5

from met_agent.evaluation.models import Report


def publish(report: Report, client: Any) -> None:
    from langfuse import Evaluation

    name = "met-golden-" + report.golden_sha256[:12] + "-" + report.suite
    client.create_dataset(name=name, description="Versioned Met agent golden evaluation")
    items = []
    results = {r.row.id: r for r in report.results}
    for r in report.results:
        items.append(
            client.create_dataset_item(
                dataset_name=name,
                id=str(uuid5(NAMESPACE_URL, name + "/" + r.row.id)),
                input={"id": r.row.id, "question": r.row.question},
                expected_output=r.row.model_dump(mode="json"),
            )
        )

    def task(*, item: Any, **kwargs: Any) -> dict[str, Any]:
        return results[item.input["id"]].model_dump(mode="json")

    def evaluator(*, output: Any, **kwargs: Any) -> list[Evaluation]:
        values = [Evaluation(name="pass", value=float(output["passed"]))]
        if output["judgment"]:
            values.append(Evaluation(name="faithfulness", value=output["judgment"]["faithfulness"]))
        return values

    client.run_experiment(
        name=name,
        run_name=report.run_id,
        data=items,
        task=task,
        evaluators=[evaluator],
        max_concurrency=1,
        metadata={
            "git_sha": report.git_sha,
            "execution": "recorded-results-import",
            "timing": "Use output latency_ms; import span duration is not inference time",
        },
    )
    client.flush()
