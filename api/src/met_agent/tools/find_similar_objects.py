"""Search only the Met's published image space, excluding the source object and text fallback."""

from qdrant_client import models

from met_agent.retrieval.service import SearchService
from met_agent.tools.models import SimilarObject, SimilarObjectsResult
from met_agent.tools.schemas import FindSimilarObjectsArguments


def find_similar_objects(
    service: SearchService, collection: str, arguments: FindSimilarObjectsArguments
) -> SimilarObjectsResult:
    store = service.store(collection)
    points = store.client.retrieve(
        collection,
        ids=[arguments.object_id],
        with_payload=False,
        with_vectors=["image"] if store.image else False,
    )
    if not points:
        return SimilarObjectsResult(
            source_object_id=arguments.object_id, status="object_not_indexed"
        )
    vectors = points[0].vector
    image = vectors.get("image") if isinstance(vectors, dict) else None
    if store.image is None or not isinstance(image, list):
        return SimilarObjectsResult(
            source_object_id=arguments.object_id, status="image_unavailable"
        )
    matches = store.client.query_points(
        collection,
        query=image,
        using="image",
        query_filter=models.Filter(must_not=[models.HasIdCondition(has_id=[arguments.object_id])]),
        limit=arguments.k,
        with_payload=True,
    ).points
    return SimilarObjectsResult(
        source_object_id=arguments.object_id,
        status="ok",
        objects=[
            SimilarObject(
                object_id=int(point.id),
                title=str((point.payload or {}).get("title", "")),
                source_url=str((point.payload or {}).get("source_url", "")),
                text=str((point.payload or {}).get("text", "")),
                image_score=point.score,
            )
            for point in matches
        ],
    )
