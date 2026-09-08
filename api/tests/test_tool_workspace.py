"""Exercise the shared tool boundary, live cache, and published image-only neighbors."""

import asyncio
import json
from contextlib import closing
from pathlib import Path
from typing import Any

import httpx
import pytest
from qdrant_client import QdrantClient, models

from met_agent.config import Settings
from met_agent.ingestion.http import SourceError
from met_agent.retrieval.hybrid import ScoreBreakdown
from met_agent.retrieval.qdrant_store import HybridStore, IndexCompatibilityError
from met_agent.retrieval.schema import ImageVectorSpec
from met_agent.retrieval.service import SearchService
from met_agent.tools.find_similar_objects import find_similar_objects
from met_agent.tools.get_directions import WayfindingClient
from met_agent.tools.get_object import LiveObjectClient, ObjectNotFound
from met_agent.tools.models import (
    GetDirectionsArguments,
    GetObjectArguments,
    Handoff,
    HandoffArguments,
    LiveObject,
)
from met_agent.tools.registry import ToolRegistry, create_registry
from met_agent.tools.schemas import FindSimilarObjectsArguments

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")


def map_feature(feature_id: str, label: str, *, closed: bool = False) -> dict[str, Any]:
    return {
        "id": feature_id,
        "is_temporarily_closed": closed,
        "label": {"name": [{"lang": "en-GB", "text": label}]},
        "location": {
            "center": {"latitude": 40.779448, "longitude": -73.963517},
            "floor": {
                "id": 2,
                "floor": "1",
                "name": [{"lang": "en-GB", "text": "Floor 1"}],
                "short_name": "1",
            },
        },
    }


def test_wayfinding_uses_exact_official_features_route_and_cache() -> None:
    calls: list[httpx.Request] = []
    clock = [0.0]
    hall_id = "a" * 32
    gallery_id = "b" * 32

    def transport(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["user-agent"]
        if request.method == "POST":
            assert json.loads(request.content) == {
                "from": {"lmId": hall_id},
                "to": {"lmId": gallery_id},
                "options": {},
                "project": "the_met",
            }
            return httpx.Response(
                200,
                json={"routeMetadata": [{"totalTime": 2, "totalLength": 0.178}]},
            )
        query = request.url.params["query"]
        data = (
            [map_feature(hall_id, "The Great Hall")]
            if query == "The Great Hall"
            else [
                map_feature("c" * 32, "Gallery 131 South Walkway"),
                map_feature(gallery_id, "131"),
            ]
        )
        return httpx.Response(200, json={"data": data})

    with httpx.Client(transport=httpx.MockTransport(transport)) as http:
        client = WayfindingClient(
            http,
            "https://map-api.prod.livingmap.com",
            "https://maps.metmuseum.org",
            clock=lambda: clock[0],
        )
        result = client.get(GetDirectionsArguments(destination_gallery="131"))
        assert result.origin == "The Great Hall" and result.destination == "Gallery 131"
        assert result.floor == "Floor 1"
        assert result.distance_metres == 178 and result.distance_feet == 584
        assert result.duration_minutes == 2
        assert result.source_url == (
            f"https://maps.metmuseum.org/navigate/{hall_id}/{gallery_id}?floor=1&lang=en-GB"
        )
        assert "Fetched at:" in result.text
        result.text = "locally changed"
        assert (
            "The Met Interactive Map route"
            in client.get(GetDirectionsArguments(destination_gallery="131")).text
        )
        assert len(calls) == 3
        clock[0] = 300
        client.get(GetDirectionsArguments(destination_gallery="131"))
        assert len(calls) == 6


def test_wayfinding_fails_closed_for_closed_or_ambiguous_destination() -> None:
    hall_id = "a" * 32
    gallery_id = "b" * 32
    destinations = [[map_feature(gallery_id, "131", closed=True)], []]
    for gallery_results in destinations:

        def transport(
            request: httpx.Request, gallery_results: list[dict[str, Any]] = gallery_results
        ) -> httpx.Response:
            query = request.url.params["query"]
            data = (
                [map_feature(hall_id, "The Great Hall")]
                if query == "The Great Hall"
                else gallery_results
            )
            return httpx.Response(200, json={"data": data})

        with (
            httpx.Client(transport=httpx.MockTransport(transport)) as http,
            pytest.raises(SourceError),
        ):
            WayfindingClient(
                http,
                "https://map-api.prod.livingmap.com",
                "https://maps.metmuseum.org",
            ).get(GetDirectionsArguments(destination_gallery="131"))


def test_live_cache_expiry_identity_and_safe_errors() -> None:
    calls: list[int] = []
    clock = [0.0]
    mode: list[Any] = [200]

    def transport(request: httpx.Request) -> httpx.Response:
        object_id = int(request.url.path.rsplit("/", 1)[-1])
        calls.append(object_id)
        return httpx.Response(
            mode[0],
            json={
                "objectID": object_id if mode[0] != 201 else -1,
                "title": "Temple",
                "GalleryNumber": "131" if object_id == 1 else "",
                "objectURL": f"https://www.metmuseum.org/art/collection/search/{object_id}",
                "isPublicDomain": True,
            },
        )

    with httpx.Client(transport=httpx.MockTransport(transport)) as http:
        live = LiveObjectClient(http, "https://met.test", clock=lambda: clock[0])
        one = live.get(GetObjectArguments(object_id=1))
        assert one.is_on_view and one.gallery_number == "131"
        one.title = "changed locally"
        assert live.get(GetObjectArguments(object_id=1)).title == "Temple"
        assert calls == [1]
        clock[0] = 300
        live.get(GetObjectArguments(object_id=1))
        assert calls == [1, 1]
        assert not live.get(GetObjectArguments(object_id=2)).is_on_view
        for object_id in range(3, 515):
            live.get(GetObjectArguments(object_id=object_id))
        assert len(live._cache) == 512
        assert 1 not in live._cache
        mode[0] = 404
        with pytest.raises(ObjectNotFound):
            live.get(GetObjectArguments(object_id=800))
        mode[0] = 500
        with pytest.raises(SourceError, match="HTTP 500"):
            live.get(GetObjectArguments(object_id=800))
    for data in ([1], {"objectID": 2}, {"objectID": 1, "objectURL": "https://evil.test"}):
        with (
            httpx.Client(
                transport=httpx.MockTransport(lambda _, data=data: httpx.Response(200, json=data))
            ) as http,
            pytest.raises(SourceError),
        ):
            LiveObjectClient(http, "https://met.test").get(GetObjectArguments(object_id=1))


def test_registry_never_executes_invalid_input_or_leaks_exception() -> None:
    registry = ToolRegistry()
    called: list[int] = []

    def handler(value: GetObjectArguments) -> Handoff:
        called.append(value.object_id)
        if value.object_id == 404:
            raise ObjectNotFound("private upstream text")
        raise RuntimeError("unit-test-secret-do-not-use")

    registry.register("get_object", "Object", GetObjectArguments, Handoff, handler)
    with pytest.raises(ValueError):
        registry.register("get_object", "Object", GetObjectArguments, Handoff, handler)
    for arguments in (
        '{"object_id": "1"}',
        '{"object_id":0}',
        '{"object_id":true}',
        '{"object_id":1,"extra":true}',
        "[]",
        "{",
        " " * 10001,
    ):
        result = asyncio.run(registry.execute("get_object", arguments))
        assert result.error and result.error.code == "invalid_arguments"
    assert called == []
    assert asyncio.run(registry.execute("unknown", "{}")).error is not None
    result = asyncio.run(registry.execute("get_object", '{"object_id":1}'))
    assert result.error and result.error.code == "upstream_unavailable"
    assert "secret" not in result.model_dump_json()
    result = asyncio.run(registry.execute("get_object", '{"object_id":404}'))
    assert result.error and result.error.code == "not_found"
    assert len(registry.schemas) == 1


def test_all_six_tools_and_image_space(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    config = settings.model_copy(
        update={
            "embedding_provider": "local",
            "embedding_model": "fixture",
            "embedding_dimensions": 2,
            "qdrant_collection": "objects",
            "qdrant_visitor_collection": "visitors",
        }
    )
    image = ImageVectorSpec(
        dataset="metmuseum/fixture", revision="a" * 40, model="published", dimensions=3
    )
    with (
        closing(QdrantClient(location=":memory:")) as client,
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json={
                        "objectID": 1,
                        "objectURL": "https://www.metmuseum.org/art/collection/search/1",
                    },
                )
            )
        ) as http,
    ):
        store = HybridStore(
            client, "objects", "fixture", 2, embedding_provider="local", image=image
        )
        store.ensure_collection(2)
        vectors: dict[int, dict[str, list[float]]] = {
            1: {"dense": [1.0, 0.0], "image": [1.0, 0.0, 0.0]},
            2: {"dense": [0.0, 1.0], "image": [0.9, 0.1, 0.0]},
            3: {"dense": [1.0, 0.0]},
        }
        client.upsert(
            "objects",
            [
                models.PointStruct(
                    id=i,
                    vector={key: value for key, value in v.items()},
                    payload={
                        "title": f"Object {i}",
                        "text": f"Object {i}",
                        "source_url": f"https://www.metmuseum.org/art/collection/search/{i}",
                    },
                )
                for i, v in vectors.items()
            ],
        )
        service = SearchService(client, config)
        registry = create_registry(
            config,
            service,
            LiveObjectClient(http, "https://met.test"),
            WayfindingClient(http, "https://map.test", "https://maps.test"),
        )
        assert set(registry.tools) == {
            "search_collection",
            "search_visitor_info",
            "get_object",
            "handoff",
            "find_similar_objects",
            "get_directions",
        }
        result = asyncio.run(registry.execute("find_similar_objects", '{"object_id":1,"k":5}'))
        assert result.error is None and result.output is not None
        assert result.evidence[0].object_id == 2
        assert "cosine similarity" in result.evidence[0].text
        for object_id, expected in ((3, "image_unavailable"), (900, "object_not_indexed")):
            value = find_similar_objects(
                service, "objects", FindSimilarObjectsArguments(object_id=object_id)
            )
            assert value.status == expected and value.objects == []
        result = asyncio.run(registry.execute("get_object", '{"object_id":1}'))
        assert isinstance(result.output, LiveObject) and result.evidence[0].kind == "live_object"
        result = asyncio.run(
            registry.execute(
                "handoff",
                HandoffArguments(
                    reason="Purchase", suggested_contact="store.support@metmuseum.org"
                ).model_dump_json(),
            )
        )
        assert isinstance(result.output, Handoff) and not result.evidence
        monkeypatch.setattr(
            service,
            "search",
            lambda *args, **kwargs: [
                (
                    models.ScoredPoint(
                        id=1,
                        version=0,
                        score=1.0,
                        payload={
                            "text": "Facts",
                            "source_url": "https://www.metmuseum.org/plan-your-visit",
                            "fetched_at": "2026-09-04T00:00:00Z",
                        },
                    ),
                    ScoreBreakdown(rrf=0.02, rerank=0.5),
                )
            ],
        )
        for name, kind in (
            ("search_collection", "collection"),
            ("search_visitor_info", "visitor_info"),
        ):
            result = asyncio.run(registry.execute(name, '{"query":"Facts"}'))
            assert result.error is None and result.evidence[0].kind == kind
        visitor = HybridStore(client, "visitors", "fixture", 2, embedding_provider="local")
        visitor.ensure_collection(2)
        client.upsert("visitors", [models.PointStruct(id=1, vector={"dense": [1.0, 0.0]})])
        assert (
            find_similar_objects(
                service, "visitors", FindSimilarObjectsArguments(object_id=1)
            ).status
            == "image_unavailable"
        )
        bad = SearchService(client, config.model_copy(update={"embedding_model": "mismatch"}))
        with pytest.raises(IndexCompatibilityError):
            bad.store("objects")


def test_search_service_loads_models_once(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from met_agent.retrieval import service as module

    created: list[str] = []

    def created_model(name: str) -> object:
        created.append(name)
        return object()

    monkeypatch.setattr(module, "create_embedder", lambda _: created_model("dense"))
    monkeypatch.setattr(module, "BM25Embedder", lambda _: created_model("sparse"))
    monkeypatch.setattr(module, "LocalReranker", lambda *args, **kwargs: created_model("rerank"))

    class Retriever:
        def __init__(self, *args: object) -> None:
            pass

        def search(self, query: str, **kwargs: object) -> list[object]:
            return []

    monkeypatch.setattr(module, "HybridRetriever", Retriever)
    with closing(QdrantClient(location=":memory:")) as client:
        service = SearchService(client, settings)
        monkeypatch.setattr(service, "store", lambda _: object())
        assert not created
        service.search("objects", "vase")
        service.search("objects", "vase")
        assert created == ["dense", "sparse", "rerank"]


def test_only_confirmed_404_produces_citable_lookup_status() -> None:
    from met_agent.agent.models import AgentDraft, Citation
    from met_agent.guardrails.grounding import valid_citations
    from met_agent.tools.models import LiveObject

    mode = [404]
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(mode[0]))) as http:
        live = LiveObjectClient(http, "https://collectionapi.metmuseum.org/public/collection/v1")
        registry = ToolRegistry()
        registry.register("get_object", "Lookup", GetObjectArguments, LiveObject, live.get)
        result = asyncio.run(registry.execute("get_object", '{"object_id":800}'))
        assert result.error and result.error.code == "not_found"
        assert len(result.evidence) == 1
        evidence = result.evidence[0]
        assert evidence.object_id is None and evidence.source_url
        assert "HTTP 404" in evidence.text and "Fetched at:" in evidence.text
        assert evidence.text in json.loads(result.model_context())["evidence"][0]["text"]
        draft = AgentDraft(
            text="No object record was returned.",
            language="en",
            citations=[Citation(source_url=evidence.source_url, quote=evidence.text)],
        )
        assert valid_citations(draft, result.evidence)
        draft.citations = [Citation(object_id=800, quote=evidence.text)]
        assert not valid_citations(draft, result.evidence)
        mode[0] = 403
        unavailable = asyncio.run(registry.execute("get_object", '{"object_id":800}'))
        assert unavailable.error and unavailable.error.code == "upstream_unavailable"
        assert not unavailable.evidence
