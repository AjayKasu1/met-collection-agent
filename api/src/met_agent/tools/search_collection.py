"""Expose bounded hybrid collection retrieval and its score breakdown to both tool transports."""

from met_agent.retrieval.hybrid import RetrievedObject
from met_agent.retrieval.service import SearchService
from met_agent.tools.models import CollectionSearchResult, SearchCollectionArguments


def search_collection(
    service: SearchService, collection: str, arguments: SearchCollectionArguments
) -> CollectionSearchResult:
    matches = service.search(collection, arguments.query, filters=arguments.filters, k=arguments.k)
    return CollectionSearchResult(
        objects=[
            RetrievedObject(
                object_id=int(point.id),
                title=str((point.payload or {}).get("title", "")),
                source_url=str((point.payload or {}).get("source_url", "")),
                text=str((point.payload or {}).get("text", "")),
                record=point.payload or {},
                scores=scores,
            )
            for point, scores in matches
        ]
    )
