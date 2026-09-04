"""Return visitor passages with their original URL, capture time, heading, and retrieval scores."""

from met_agent.retrieval.hybrid import RetrievedChunk
from met_agent.retrieval.service import SearchService
from met_agent.tools.models import SearchVisitorArguments, VisitorSearchResult


def search_visitor_info(
    service: SearchService, collection: str, arguments: SearchVisitorArguments
) -> VisitorSearchResult:
    chunks = []
    for point, scores in service.search(collection, arguments.query, k=arguments.k):
        payload = point.payload or {}
        chunks.append(
            RetrievedChunk(
                point_id=str(point.id),
                source_url=str(payload.get("source_url", "")),
                page_title=str(payload.get("page_title", "")),
                section_heading=str(payload.get("section_heading", "")),
                fetched_at=str(payload.get("fetched_at", "")),
                text=str(payload.get("text", "")),
                scores=scores,
            )
        )
    return VisitorSearchResult(chunks=chunks)
