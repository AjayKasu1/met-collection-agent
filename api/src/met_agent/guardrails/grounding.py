"""Validate citation identity and verbatim evidence before accepting an atomic-claim judgment."""

from pydantic import BaseModel, ConfigDict, Field

from met_agent.agent.models import AgentDraft
from met_agent.tools.models import Evidence, WayfindingResult


class ClaimCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    supported: bool
    evidence_keys: list[str]


class GroundingCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fully_supported: bool
    claims: list[ClaimCheck]

    def score(self, evidence: list[Evidence]) -> float:
        keys = {item.key for item in evidence}
        if not self.claims:
            return 1.0 if self.fully_supported else 0.0
        return sum(
            claim.supported and bool(claim.evidence_keys) and set(claim.evidence_keys) <= keys
            for claim in self.claims
        ) / len(self.claims)


def valid_citations(draft: AgentDraft, evidence: list[Evidence]) -> bool:
    for citation in draft.citations:
        matches = [
            item
            for item in evidence
            if (citation.object_id is not None and item.object_id == citation.object_id)
            or (citation.source_url is not None and item.source_url == citation.source_url)
        ]
        if not any(
            " ".join(citation.quote.split()) in " ".join(item.text.split()) for item in matches
        ):
            return False
    return True


def valid_wayfinding_evidence(route: WayfindingResult, evidence: list[Evidence]) -> bool:
    """Verify every templated route field against one canonical map evidence record."""
    if len(evidence) != 1:
        return False
    item = evidence[0]
    expected_key = f"route:{route.origin}:{route.destination}"
    expected_lines = {
        "The Met Interactive Map route",
        f"Requested origin: {route.requested_origin}",
        f"Start: {route.origin}",
        f"Destination: {route.destination}",
        f"Floor: {route.floor}",
        f"Estimated walking time: {route.duration_minutes} minutes",
        f"Distance: {route.distance_metres} metres ({route.distance_feet} feet)",
    }
    return (
        item.kind == "wayfinding"
        and item.key == expected_key
        and item.source_url == route.source_url
        and route.source_url.startswith("https://maps.metmuseum.org/navigate/")
        and expected_lines <= set(item.text.splitlines())
    )
