# Production operations

This runbook covers the public Cloudflare Worker and Google Cloud Run deployment. A release is ready for public traffic only when the activation checklist and smoke test both pass.

## Launch objectives

The initial public-demo objectives are 99.5 percent monthly availability, less than 1 percent server errors over five minutes, and p95 Cloud Run latency below 10 seconds over five minutes. These are operational targets, not guarantees from the current free model tier. The Terraform root in `infra/gcp/monitoring` alerts at 5 percent server errors, sustained readiness failure, and 10-second p95 latency. Tighten the server-error alert after normal traffic produces a stable baseline.

## Secret and configuration boundaries

Use Google Secret Manager for backend credentials and Cloudflare Worker secrets for edge credentials. Never put these values in GitHub variables, Wrangler configuration, build logs, Docker arguments, or the browser bundle.

| Location | Secret values |
| --- | --- |
| Google Secret Manager | `GROQ_API_KEY`, `QDRANT_API_KEY`, `AUDIT_DATABASE_URL`, `EDGE_ORIGIN_TOKEN` |
| Cloudflare Worker secrets | `TURNSTILE_SECRET`, `ORIGIN_AUTH_TOKEN` |
| Cloudflare build variables | `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_TURNSTILE_SITE_KEY` |
| Checked Worker configuration | `TURNSTILE_HOSTNAMES` and `CHAT_RATE_LIMITER` |

`EDGE_ORIGIN_TOKEN` and `ORIGIN_AUTH_TOKEN` must contain the same randomly generated value. Rotate them by adding the new backend secret version, deploying Cloud Run, updating the Worker secret immediately, and completing the smoke test. This brief order can reject requests but never opens the origin. For zero interruption, introduce a two-key overlap in a reviewed change before rotating.

## Activation checklist

1. Create a managed PostgreSQL database with encrypted connections, automated backups, point-in-time recovery, a private application role, and a connection limit larger than `AUDIT_POOL_MAX_SIZE` multiplied by Cloud Run's maximum instances. Store its TLS connection string as `AUDIT_DATABASE_URL` in Secret Manager.
2. Store a random edge credential in Secret Manager as `EDGE_ORIGIN_TOKEN` and the same value as the Worker secret `ORIGIN_AUTH_TOKEN`.
3. Create the Turnstile widget for `met-collection-agent.ajaykasu7.workers.dev`. Store its secret as the Worker secret `TURNSTILE_SECRET` and its site key as the build variable `NEXT_PUBLIC_TURNSTILE_SITE_KEY`.
4. Set Cloud Run to `APP_ENV=production`, `EDGE_AUTH_REQUIRED=true`, `AUDIT_STORE_REQUIRED=true`, `QDRANT_VISITOR_COLLECTION=met_visitor_info_live`, and the exact Worker URL in `CORS_ORIGINS`. Bind all backend secrets from Secret Manager.
5. Set the Worker build variable `NEXT_PUBLIC_API_URL` to the HTTPS Cloud Run origin. Confirm `TURNSTILE_HOSTNAMES` contains only the production Worker hostname.
6. Configure the GitHub `data-refresh` environment with write-scoped `QDRANT_URL` and `QDRANT_API_KEY` secrets. Set repository variable `ENABLE_VISITOR_REFRESH=true`, then run `Refresh visitor information` manually once. The run creates the initial `met_visitor_info_live` alias.
7. Apply `infra/gcp/monitoring` with at least one verified notification channel.
8. Run the smoke test below. Record the Git revision, Cloud Run revision, Worker deployment ID, Qdrant visitor release, and timestamp in the release record.

## Release smoke test

Check these in order after each deployment:

1. `GET /health` returns HTTP 200 and the expected Git SHA.
2. `GET /ready` returns HTTP 200 with both `audit_store` and `qdrant` true.
3. A direct request to a protected Cloud Run route without `X-Origin-Auth` returns HTTP 401.
4. The public Worker loads with the security headers in `web/next.config.ts`.
5. An incomplete Turnstile challenge cannot submit chat.
6. A completed challenge answers one object lookup with a working Met citation.
7. A visitor-hours question cites a visitor page and displays its capture time.
8. Six immediate chat attempts from one source receive at least one HTTP 429 without reaching Cloud Run.
9. The corresponding PostgreSQL session contains an ordered, complete event sequence and no provider credential.

Turnstile tokens are single-use and expire after five minutes. Every chat attempt must obtain a new token. Server-side validation checks success, action `chat`, hostname, and source IP.

## Visitor data refresh

The Monday workflow uses the project User-Agent, honors robots policy, and waits at least three seconds between every source request, redirect, and retry. It prepares all configured pages before writing vectors. The candidate uses a timestamped physical collection name. Promotion requires exact point count and exact source-URL coverage, then uses one atomic Qdrant alias operation. The prior physical collection remains available for rollback.

If the workflow fails, inspect the job summary without rerunning repeatedly against the museum site. A robots denial, changed dynamic map page, or missing main content requires human review. Update the curated source list or save an approved page capture, test locally, and start one manual run. Do not weaken source-host or robots checks.

To roll back, use the Qdrant dashboard's collection-alias control to atomically move `met_visitor_info_live` to the `previous_collection` recorded in the successful workflow artifact. Then verify `/ready` and repeat the visitor-hours smoke test. Keep at least the active and previous physical visitor collections. Delete older versions only after a successful release review.

## Incident response

| Signal | First check | Safe response |
| --- | --- | --- |
| Worker 429 | Cloudflare rate-limit events | Confirm attack or shared-network traffic, then tune only with measured evidence |
| Worker verification failures | Turnstile analytics and hostname/action | Check widget domains and secret version; never bypass verification |
| Cloud Run 401 from Worker | Edge credential versions | Restore the matching Worker and backend secret pair |
| `/ready` audit false | PostgreSQL availability and pool exhaustion | Stop scaling up, restore database connectivity, and preserve audit durability |
| `/ready` Qdrant false | Qdrant health and collection metadata | Restore connectivity or roll back the last visitor alias change |
| Provider 429 | Groq account limits and request tokens | Let bounded retries finish, reduce concurrency, or move to paid capacity |
| Provider 400, 401, or 403 | Model ID, key scope, provider incident | Quarantine that route and escalate to an operator; do not trigger fallback |
| Citation or grounding rejection | Session event sequence | Preserve the fail-closed answer and review evidence; do not relax guardrails |

Correlate Worker and Cloud Run events with `CF-Ray` and `X-Request-ID` where available. Application logs omit request bodies, raw paths, headers, and exception messages. Audit rows can contain user questions and retrieved text, so limit database access and retain them for the configured 30 days.

## Rollback

- Cloudflare: roll back to the prior Worker deployment, then verify the public security headers and one Turnstile-protected chat.
- Cloud Run: direct traffic to the prior healthy revision. Do not disable edge authentication to recover availability.
- Visitor index: atomically move `met_visitor_info_live` to the recorded previous collection.
- Model routing: restore the last evaluated model IDs and rate-limit map. Authentication failures and grounding failures remain ineligible for fallback.

After recovery, preserve sanitized logs and release identifiers, write a short timeline, identify the failed control, and add a regression test or monitor before closing the incident.
