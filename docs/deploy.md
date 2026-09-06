# Deployment runbook

The web client runs on Cloudflare Workers. The API runs as a Python container and requires a reachable Qdrant server that already contains the two verified collections. Deploy the API first because the Worker embeds its public API origin at build time.

## Local container demo

Install Docker Desktop, copy `.env.example` to `.env`, and supply the same chat settings used for local development. Set `HF_DATASET_REPO=AJAYKASU/met-collection-index` so the seed command can fetch the published bundle.

```sh
make demo
```

The target starts Qdrant, restores missing collections from the verified bundle, then builds and starts the API and web services. Open `http://localhost:3000`. API, web, and Qdrant ports bind to loopback only. Qdrant state, session audits, and the Hugging Face model cache use named Docker volumes.

Run the structural and image build checks independently with:

```sh
make compose-check
make build-containers
```

The API image uses Python 3.12, a locked production dependency set, a non-root user, a read-only root filesystem in Compose, and an HTTP health check. The web image is for local development only. Production web delivery uses the OpenNext Worker bundle.

For a one-click Codespace, first add Codespaces secrets for `GROQ_API_KEY`, `LLM_MODEL`, `LLM_MODEL_LITE`, and `HF_DATASET_REPO`. Use `AJAYKASU/met-collection-index` for the dataset value. The post-create command installs both locked dependency sets, starts Qdrant, waits for readiness, and restores the published indexes. Secret values remain in the Codespace environment and are not written to the repository.

## API on Google Cloud Run

Cloud Run remains publicly reachable so the Cloudflare Worker can call it without Google-specific identity exchange. Production data routes require a shared edge credential, while `/health` and `/ready` remain public for probes. Turnstile and the distributed chat rate limit run at Cloudflare before an expensive model request reaches Cloud Run.

The following commands use the current `gcloud` project and deploy to `us-east1`. Qdrant must be a managed HTTPS endpoint with `met_objects` and `met_visitor_info` restored from the pinned bundle first.

```sh
export GCP_REGION=us-east1
printf 'Qdrant HTTPS URL: '
read -r QDRANT_URL
export QDRANT_URL

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com

read -rsp 'Groq API key: ' GROQ_API_KEY_INPUT && printf '\n'
printf '%s' "$GROQ_API_KEY_INPUT" | gcloud secrets create met-agent-groq-api-key --data-file=- 2>/dev/null \
  || printf '%s' "$GROQ_API_KEY_INPUT" | gcloud secrets versions add met-agent-groq-api-key --data-file=-
unset GROQ_API_KEY_INPUT

read -rsp 'Qdrant API key: ' QDRANT_API_KEY_INPUT && printf '\n'
printf '%s' "$QDRANT_API_KEY_INPUT" | gcloud secrets create met-agent-qdrant-api-key --data-file=- 2>/dev/null \
  || printf '%s' "$QDRANT_API_KEY_INPUT" | gcloud secrets versions add met-agent-qdrant-api-key --data-file=-
unset QDRANT_API_KEY_INPUT

read -rsp 'Managed PostgreSQL TLS connection URL: ' AUDIT_DATABASE_URL_INPUT && printf '\n'
printf '%s' "$AUDIT_DATABASE_URL_INPUT" | gcloud secrets create met-agent-audit-database-url --data-file=- 2>/dev/null \
  || printf '%s' "$AUDIT_DATABASE_URL_INPUT" | gcloud secrets versions add met-agent-audit-database-url --data-file=-
unset AUDIT_DATABASE_URL_INPUT

read -rsp 'Shared edge-to-origin credential: ' EDGE_ORIGIN_TOKEN_INPUT && printf '\n'
printf '%s' "$EDGE_ORIGIN_TOKEN_INPUT" | gcloud secrets create met-agent-edge-origin-token --data-file=- 2>/dev/null \
  || printf '%s' "$EDGE_ORIGIN_TOKEN_INPUT" | gcloud secrets versions add met-agent-edge-origin-token --data-file=-
unset EDGE_ORIGIN_TOKEN_INPUT

gcloud run deploy met-collection-agent-api \
  --source api \
  --region "$GCP_REGION" \
  --allow-unauthenticated \
  --port 8000 \
  --cpu 4 \
  --memory 8Gi \
  --concurrency 4 \
  --min 1 \
  --max 3 \
  --set-secrets GROQ_API_KEY=met-agent-groq-api-key:latest,QDRANT_API_KEY=met-agent-qdrant-api-key:latest,AUDIT_DATABASE_URL=met-agent-audit-database-url:latest,EDGE_ORIGIN_TOKEN=met-agent-edge-origin-token:latest \
  --set-env-vars "^@^APP_ENV=production@LOG_LEVEL=INFO@LLM_MODEL=groq/openai/gpt-oss-120b@LLM_MODEL_LITE=groq/openai/gpt-oss-20b@LLM_FALLBACK_ENABLED=false@LLM_PACING_ENABLED=false@CHAT_DEADLINE_SECONDS=120@USE_AI_GATEWAY=false@EMBEDDING_PROVIDER=local@EMBEDDING_MODEL=intfloat/multilingual-e5-large@EMBEDDING_DIMENSIONS=1024@EMBEDDING_THREADS=4@QDRANT_URL=${QDRANT_URL}@QDRANT_COLLECTION=met_objects@QDRANT_VISITOR_COLLECTION=met_visitor_info_live@AUDIT_STORE_REQUIRED=true@EDGE_AUTH_REQUIRED=true@CORS_ORIGINS=[\"https://met-collection-agent.ajaykasu7.workers.dev\"]@DATA_DIR=/tmp/met-agent@GIT_SHA=$(git rev-parse HEAD)"
```

Cloud Run injects `PORT`; the image honors it and exposes liveness at `/health` and dependency readiness at `/ready`. The first semantic query downloads the pinned 2.24 GB local embedding model. Four CPUs, 8 GiB memory, minimum instance count one, and concurrency four reduce cold-start and memory pressure. Adjust them only after measuring production traffic and memory. PostgreSQL retains audits across revisions and instances. With the default pool maximum of four and three Cloud Run instances, reserve at least twelve application connections plus database administration headroom.

Capture the deployed origin and verify it before building the Worker:

```sh
export API_URL="$(gcloud run services describe met-collection-agent-api --region "$GCP_REGION" --format='value(status.url)')"
curl --fail --silent --show-error "$API_URL/health"
curl --fail --silent --show-error "$API_URL/ready"
```

## Web on Cloudflare Workers

The Worker name is `met-collection-agent`, so the expected account URL is `https://met-collection-agent.ajaykasu7.workers.dev`.

### Cloudflare Git integration

Use this option when Cloudflare should own deployments from GitHub:

1. In **Workers & Pages**, select **Create application**, then **Import a repository**.
2. Authorize only `AjayKasu1/met-collection-agent` in the Cloudflare Workers and Pages GitHub App.
3. Select branch `main` and use these build settings:

| Setting | Value |
| --- | --- |
| Worker name | `met-collection-agent` |
| Root directory | `web` |
| Build command | `pnpm build:worker` |
| Deploy command | `pnpm exec wrangler deploy` |
| Non-production deploy command | `pnpm exec wrangler versions upload` |

4. Add build variables `NEXT_PUBLIC_API_URL` with the verified Cloud Run `API_URL` and `NEXT_PUBLIC_TURNSTILE_SITE_KEY` with the production widget's site key. The API URL must use HTTPS and must not end with `/`.
5. Add Worker secrets `TURNSTILE_SECRET` and `ORIGIN_AUTH_TOKEN`. The latter must match Cloud Run's `EDGE_ORIGIN_TOKEN`.
6. Keep the GitHub repository variable `ENABLE_WEB_DEPLOY` unset or `false`. This prevents the separate GitHub Actions deploy workflow from competing with Cloudflare Builds.
7. Deploy and complete the release smoke test in [production operations](operations.md).

Cloudflare Builds installs the package manager declared in `web/package.json` and uses the locked Wrangler dependency. The checked-in `wrangler.jsonc` enables Workers logs and uses a current compatibility date with `nodejs_compat`.

### GitHub Actions deployment

Use this option when GitHub should own deployments. Do not also connect automatic Cloudflare Builds.

Create a GitHub `production` environment with these settings:

| Kind | Name | Value |
| --- | --- | --- |
| Environment secret | `CLOUDFLARE_API_TOKEN` | Scoped token with Workers Scripts Edit |
| Environment secret | `CF_ACCOUNT_ID` | Cloudflare account ID |
| Repository variable | `NEXT_PUBLIC_API_URL` | Verified HTTPS Cloud Run origin |
| Repository variable | `NEXT_PUBLIC_TURNSTILE_SITE_KEY` | Production Turnstile site key |
| Repository variable | `ENABLE_WEB_DEPLOY` | `true` |

Before the first workflow deployment, add `TURNSTILE_SECRET` and `ORIGIN_AUTH_TOKEN` directly to the existing Worker with the Cloudflare dashboard. The `Deploy web Worker` workflow then runs for web changes on `main` and can also be started manually. It fails before building when required public configuration is absent. Keep `ENABLE_WEB_DEPLOY=false` to disable it without modifying the workflow.

## API image publication

Pushing a release tag such as `v0.1.0` publishes these registry references:

```text
ghcr.io/ajaykasu1/met-collection-agent-api:v0.1.0
ghcr.io/ajaykasu1/met-collection-agent-api:latest
```

The tag workflow authenticates with GitHub's short-lived workflow token, builds from `api/Dockerfile`, and attaches build provenance and an SBOM. Protect release tags in repository rules so only reviewed commits can publish `latest`.

## Render fallback

Create a Render Web Service from `AjayKasu1/met-collection-agent` with the Docker runtime, root directory `api`, health check path `/health`, and region nearest the Qdrant cluster. Set the same non-secret values used for Cloud Run, add `GROQ_API_KEY` and `QDRANT_API_KEY` as secrets, set `DATA_DIR=/tmp/met-agent`, and allocate at least 8 GiB memory for the local embedding model. Point the Cloudflare build variable at the resulting HTTPS service URL and add the Worker origin to `CORS_ORIGINS`.

Render's local filesystem is also unsuitable for durable multi-instance session audits. The same authentication, rate-limit, model-cache, and durable-audit requirements apply.
