"""Test factual field preservation, deterministic selection, source freshness, and Parquet reuse."""

import csv
from pathlib import Path

import httpx
import pytest

from met_agent.ingestion.collection import (
    enrich_object,
    image_inventory,
    object_from_row,
    reuse_selection,
    select_objects,
)
from met_agent.ingestion.http import SourceClient, SourceError, download_csv
from met_agent.ingestion.storage import load_objects, save_objects, sha256_file, write_json


@pytest.fixture
def raw_record() -> dict[str, str]:
    """Provide a synthetic public-domain record with BCE dates and preserved source fields."""
    return {
        "Object ID": "101",
        "Is Public Domain": "True",
        "Is Highlight": "True",
        "Title": "Bronze vessel",
        "Artist Display Name": "",
        "Object Number": "12.34",
        "Culture": "Roman",
        "Period": "Republican",
        "Object Date": "100 BCE",
        "Object Begin Date": "-100",
        "Object End Date": "-90",
        "Medium": "Bronze",
        "Department": "Greek and Roman Art",
        "Gallery Number": "100",
        "Classification": "Metalwork",
        "Credit Line": "Gift, 1912",
        "Tags": "Vessels",
        "Link Resource": "https://www.metmuseum.org/art/collection/search/101",
        "Curatorial Description": "The record describes a bronze vessel.",
        "Extra Field": "retained",
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_document_contains_only_record_facts(raw_record: dict[str, str]) -> None:
    obj = object_from_row(raw_record, {101})
    document = obj.document()
    assert obj.object_begin_date == -100
    assert obj.raw_fields == raw_record
    assert obj.has_image
    for text in (
        "Bronze vessel",
        "Roman",
        "Republican",
        "100 BCE",
        "Bronze",
        "Metalwork",
        "Gift, 1912",
        "Vessels",
    ):
        assert text in document.text
    assert "The record describes a bronze vessel." in document.text
    assert "Artist:" not in document.text
    assert document.payload["raw_fields"] == raw_record
    assert document.payload["source_kind"] == "collection"


def test_selection_prefers_highlights_then_images_and_excludes_ineligible(
    tmp_path: Path, raw_record: dict[str, str]
) -> None:
    rows = []
    for object_id, highlight, public, link in [
        (101, "False", "True", True),
        (102, "True", "True", True),
        (103, "False", "True", True),
        (104, "True", "False", True),
        (105, "True", "True", False),
    ]:
        rows.append(
            {
                **raw_record,
                "Object ID": str(object_id),
                "Is Highlight": highlight,
                "Is Public Domain": public,
                "Link Resource": raw_record["Link Resource"] if link else "",
            }
        )
    path = tmp_path / "objects.csv"
    write_csv(path, rows)
    assert [o.object_id for o in select_objects(path, limit=3, image_ids={103}).objects] == [
        102,
        103,
        101,
    ]
    with pytest.raises(ValueError):
        select_objects(path, limit=0, image_ids=set())


def test_invalid_source_rows_fail_closed(tmp_path: Path, raw_record: dict[str, str]) -> None:
    with pytest.raises(SourceError, match="public-domain"):
        object_from_row({**raw_record, "Is Public Domain": "False"}, set())
    with pytest.raises(SourceError, match="year"):
        object_from_row({**raw_record, "Object Begin Date": "not a year"}, set())
    path = tmp_path / "bad.csv"
    path.write_text("unrelated,columns\n1,2\n")
    with pytest.raises(SourceError, match="required"):
        select_objects(path, limit=1, image_ids=set())
    write_csv(path, [{**raw_record, "Is Public Domain": "False"}])
    with pytest.raises(SourceError, match="No eligible"):
        select_objects(path, limit=1, image_ids=set())
    write_csv(path, [raw_record, raw_record])
    with pytest.raises(SourceError, match="duplicate"):
        select_objects(path, limit=2, image_ids=set())


def test_golden_reservations_precede_ranking_without_exceeding_limit(
    tmp_path: Path, raw_record: dict[str, str]
) -> None:
    path = tmp_path / "objects.csv"
    rows = [
        {**raw_record, "Object ID": str(i), "Is Highlight": "False" if i == 105 else "True"}
        for i in range(101, 106)
    ]
    rows.extend(
        [
            {**raw_record, "Object ID": "106", "Is Public Domain": "False"},
            {**raw_record, "Object ID": "107", "Link Resource": ""},
        ]
    )
    write_csv(path, rows)
    selected = select_objects(
        path, limit=2, image_ids={101}, must_include_ids=frozenset({105, 106, 107})
    )
    assert [obj.object_id for obj in selected.objects] == [105, 101]
    assert selected.report.included_ids == [105]
    assert selected.report.excluded_ids == {106: "not_public_domain", 107: "missing_source_url"}
    assert all(obj.is_public_domain for obj in selected.objects)
    with pytest.raises(SourceError, match="accommodate 3"):
        select_objects(path, limit=2, image_ids=set(), must_include_ids=frozenset({101, 102, 103}))
    with pytest.raises(SourceError, match="absent from the CSV"):
        select_objects(path, limit=2, image_ids=set(), must_include_ids=frozenset({999}))
    with pytest.raises(ValueError, match="positive"):
        select_objects(path, limit=2, image_ids=set(), must_include_ids=frozenset({0}))
    write_csv(path, [raw_record, raw_record])
    with pytest.raises(SourceError, match="duplicate required"):
        select_objects(path, limit=1, image_ids=set(), must_include_ids=frozenset({101}))


def test_reuse_rejects_changed_golden_ids_artifacts_and_omitted_reservations(
    tmp_path: Path, raw_record: dict[str, str]
) -> None:
    path = tmp_path / "objects.csv"
    required = frozenset({101, 102})
    with pytest.raises(SourceError, match="report is missing"):
        reuse_selection(tmp_path, limit=2, must_include_ids=required)
    write_csv(path, [raw_record, {**raw_record, "Object ID": "102"}])
    selected = select_objects(path, limit=2, image_ids=set(), must_include_ids=required)
    artifact = tmp_path / "objects.parquet"
    save_objects(artifact, selected.objects)
    selected.report.artifact_sha256 = sha256_file(artifact)
    report = tmp_path / "selection.json"
    write_json(report, selected.report.model_dump(mode="json"))
    assert reuse_selection(tmp_path, limit=2, must_include_ids=required) == selected
    with pytest.raises(SourceError, match="Golden IDs changed"):
        reuse_selection(tmp_path, limit=2, must_include_ids=frozenset({101}))
    with pytest.raises(SourceError, match="omits required"):
        reuse_selection(tmp_path, limit=1, must_include_ids=required)
    inconsistent = selected.report.model_copy(update={"included_ids": [101]})
    write_json(report, inconsistent.model_dump(mode="json"))
    with pytest.raises(SourceError, match="inconsistent"):
        reuse_selection(tmp_path, limit=2, must_include_ids=required)
    write_json(report, selected.report.model_dump(mode="json"))
    save_objects(artifact, selected.objects[:1])
    with pytest.raises(SourceError, match="does not match"):
        reuse_selection(tmp_path, limit=2, must_include_ids=required)


def test_parquet_round_trip_retains_raw_and_live_fields(
    tmp_path: Path, raw_record: dict[str, str]
) -> None:
    record = object_from_row(raw_record, {101})
    path = tmp_path / "nested" / "objects.parquet"
    save_objects(path, [record])
    assert load_objects(path) == [record]
    assert len(sha256_file(path)) == 64
    with pytest.raises(ValueError, match="empty"):
        save_objects(path, [])
    assert load_objects(path) == [record]


def test_enrichment_preserves_csv_and_uses_live_gallery(raw_record: dict[str, str]) -> None:
    payload = {
        "objectID": 101,
        "isPublicDomain": True,
        "GalleryNumber": "202",
        "primaryImage": "https://images.metmuseum.org/item.jpg",
    }
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    ) as client:
        result = enrich_object(
            object_from_row(raw_record, set()), SourceClient(client), "https://api.example"
        )
    assert result.gallery_number == "202"
    assert result.raw_fields["Gallery Number"] == "100"
    assert result.live_fields == payload
    assert result.has_image and result.fetched_at is not None


@pytest.mark.parametrize("payload", [{"objectID": 999}, {"objectID": 101, "isPublicDomain": False}])
def test_live_identity_and_rights_must_match(
    raw_record: dict[str, str], payload: dict[str, object]
) -> None:
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
        ) as client,
        pytest.raises(SourceError),
    ):
        enrich_object(
            object_from_row(raw_record, set()), SourceClient(client), "https://api.example"
        )


def test_image_inventory_caches_and_validates_count(tmp_path: Path) -> None:
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"total": 2, "objectIDs": [1, 2]})
        )
    ) as client:
        source = SourceClient(client, interval=0)
        cache = tmp_path / "images.json"
        assert image_inventory(source, "https://api.example", cache) == {1, 2}
        assert image_inventory(source, "https://api.example", cache) == {1, 2}
        cache.write_text('{"total":3,"objectIDs":[1,2]}')
        with pytest.raises(SourceError, match="inventory"):
            image_inventory(source, "https://api.example", cache)


def test_csv_download_is_atomic_and_rejects_lfs_pointers(tmp_path: Path) -> None:
    path = tmp_path / "MetObjects.csv"
    content = b"Object ID,Is Public Domain\n101,True\n"
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=content))
    ) as client:
        download_csv(client, "https://media.example/MetObjects.csv", path)
    assert path.read_bytes() == content
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, text="version https://git-lfs.github.com/spec/v1")
        )
    ) as client:
        assert download_csv(client, "https://media.example/MetObjects.csv", path) == path
        with pytest.raises(SourceError, match="LFS"):
            download_csv(client, "https://media.example/MetObjects.csv", path, refresh=True)
    assert path.read_bytes() == content
