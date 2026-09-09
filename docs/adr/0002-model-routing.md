# ADR 0002: Use explicit model routes with bounded fallback

- Status: Accepted
- Date: 2026-09-05, amended 2026-09-09

## Context

Intent classification and grounding checks need less capacity than tool planning and factual synthesis. Free-tier provider limits make routing and retry behavior visible parts of correctness. Automatic provider failover can hide access errors, multiply latency, and send data to an unintended service.

## Decision

Configure exact `main` and `lite` model identifiers and credentials through the typed settings boundary. Use the lite route for structured classification and verification; select the answer route from classified intent. Pace by model token and request limits. Disable fallbacks by default. When explicitly enabled, use separate main and lite fallback settings before legacy shared fallbacks. Skip routes with missing keys, retry only timeouts, rate limits, and 5xx responses, and quarantine authentication or permission failures for operator review. Deduplicate the chain by model identifier and never call a fallback after a successful response.

## Consequences

Startup logs show active fallback routes without printing keys. Access failures stay visible and cannot silently cross providers. Process-local pacing is sufficient for one API process but requires distributed coordination for horizontal scaling. A paid provider still needs measured limits configured explicitly.
