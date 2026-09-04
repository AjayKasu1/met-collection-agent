"""Keep golden API failures distinct from absent objects and sample membership."""

from pathlib import Path

import httpx

from met_agent.ingestion.http import SourceClient
from met_agent.ingestion.verify import read_golden, verification_markdown, verify_golden


def test_verification_deduplicates_requests_and_preserves_manual_rows(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"id":"first","question":"Lookup","expected_object_ids":[1,2,3,4,5],'
        '"verify":true,"notes":"Confirm title"}\n\n'
        '{"id":"second","question":"Repeat","expected_object_ids":[1],"verify":false}\n'
    )
    calls: list[int] = []

    def respond(request: httpx.Request) -> httpx.Response:
        object_id = int(request.url.path.split("/")[-1])
        calls.append(object_id)
        if object_id == 1:
            return httpx.Response(200, json={"objectID": 1, "title": "Title | line\nnext"})
        if object_id == 2:
            return httpx.Response(404)
        if object_id == 3:
            return httpx.Response(403)
        if object_id == 4:
            return httpx.Response(200, json={"objectID": 999})
        return httpx.Response(200, text="invalid JSON")

    rows = read_golden(path)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        results = verify_golden(
            rows, {1, 5}, SourceClient(client, interval=0), "https://api.example"
        )
    assert calls == [1, 2, 3, 4, 5]
    assert [result.exists for result in results] == [True, False, None, None, None, True]
    report = verification_markdown(rows, results)
    assert "Title \\| line next" in report and "Confirm title" in report
    assert '"question":"Repeat"' not in report
    assert "| first | 5 | unknown | yes |" in report
