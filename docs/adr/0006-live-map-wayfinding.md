# ADR 0006: Bound wayfinding to the live museum map

- Status: Accepted
- Date: 2026-09-08

## Context

Collection records can identify an object's gallery, but they do not describe corridors, entrances, or connections between rooms. The saved museum-map page is a rendered application shell and its indexed text contains controls and labels rather than a route graph. Generic retrieval therefore cannot support directions from the Fifth Avenue entrance to Gallery 131. Repeated searches add latency without creating evidence.

## Decision

Add a typed `get_directions` tool for one narrow route class: a Fifth Avenue entrance request to an exact numbered Floor 1 gallery. Resolve The Great Hall and the gallery by exact English labels through the backend used by The Met's live interactive map, then request its route. The Great Hall is the explicit indoor starting point. Validate feature identifiers, coordinates, floor identity, closure status, route distance, and route duration before producing evidence.

Return the floor, distance, estimated time, capture time, and a `maps.metmuseum.org` navigation URL. Cache a validated route for five minutes. Use a five-second timeout for each of the three upstream requests. Keep the existing citation identity and atomic-claim grounding check. If the map response is unavailable, ambiguous, malformed, on another floor, or marks the destination closed, return the existing verification-unavailable response.

Do not infer turn-by-turn steps, accessibility routes, or an entrance-to-Great-Hall path that the validated response does not supply. Do not query the collection or visitor vector indexes after this route is selected.

## Consequences

The first uncached request makes two exact feature searches and one route request. Warm requests avoid those calls for five minutes. The route remains dependent on a live external service and users should open the cited map before walking. Other origins, other floors, named destinations, and step-free routing remain outside this contract until their source behavior is verified and covered by tests.
