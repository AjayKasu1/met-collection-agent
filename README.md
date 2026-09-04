# met-collection-agent

An independent, grounded collection assistant being built over The Metropolitan Museum of Art's Open Access records.

[![CI](https://github.com/AjayKasu1/met-collection-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/AjayKasu1/met-collection-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

The API now provides hybrid retrieval, five typed tools, a constrained multilingual chat loop, verified-answer SSE, session audits, and MCP over stdio and SSE. The local index contains 20,000 objects, published image vectors, and captured visitor information. Live retrieval and all three golden demo questions pass through Groq and AI Gateway, with native structured final answers and unchanged citation/grounding checks. See [provider routing](docs/provider-routing.md). Evaluation scoring, the web client, and deployment remain later phases. No retrieval-quality or faithfulness scores are claimed yet.

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git. Run commands from this repository's root. uv installs Python 3.12 if needed and uses the committed dependency lockfile.

```sh
make setup
cp .env.example .env
```

Edit `.env` locally and supply `LLM_MODEL`, `LLM_MODEL_LITE`, and the corresponding provider key (`GROQ_API_KEY`, `GEMINI_API_KEY`, or optional `CEREBRAS_API_KEY`) for chat configuration. Text embeddings default to local FastEmbed with `EMBEDDING_MODEL=intfloat/multilingual-e5-large` and `EMBEDDING_DIMENSIONS=1024`. The optional Gemini embedding path requires an explicit provider, model, and dimension change. Keep `USE_AI_GATEWAY=false` for direct provider access, or configure the gateway as described in the configuration guide.

```sh
make dev
```

Open [API documentation](http://localhost:8000/docs), or inspect liveness:

```sh
curl -i -H 'X-Request-ID: local-health-check' http://localhost:8000/health
```

`/health` returns package version, `GIT_SHA` (or `unknown` for an unlabelled local build), and boolean optional-service configuration flags. It never calls those services. A `true` flag means configuration is available, not that a service is reachable or its integration has been implemented. Qdrant's local URL is configured by default. No API key, model identifier, endpoint URL, or request content appears in the health response.

Missing or invalid required settings fail before the application accepts requests. Error messages name the settings without printing their values. See [configuration details](docs/configuration.md).

## Development checks

```sh
make lint
make typecheck
make test
make check
```

Tests supply synthetic settings and never load the developer's `.env` or call an LLM. Start `make qdrant` to include the real index and snapshot integration checks; CI supplies that service automatically. Coverage includes configuration validation, concurrent request isolation, CORS preflights, error redaction, and incremental ASGI streaming. The coverage gate is 90% with branch coverage enabled.

`make setup` installs pre-commit hooks. Hooks and CI use the same pinned lint, type, and test tools. The repository check inspects the Git index, rejects private file paths before reading their contents, and detects recognizable provider tokens. It is an additional safeguard, not proof that arbitrary secrets cannot be committed. Stage files explicitly and review `git diff --cached` before committing.

## Ingest a pilot

```sh
make qdrant
make ingest ARGS="--limit 200 --data-dir data/pilot --collection met_objects_pilot_local --prepare-only"
make verify-golden ARGS="--data-dir data/pilot"
make ingest ARGS="--limit 200 --data-dir data/pilot --collection met_objects_pilot_local --reuse-prepared --batch-size 32"
```

Preparation makes no model calls. The final command downloads pinned ONNX weights on first use, runs text inference locally, and writes to the configured Qdrant server. Review the golden titles and membership before a larger run. See [ingestion behavior and source limitations](docs/ingestion.md) and [snapshot publication and seeding](docs/index-artifacts.md).

The bounded sample reserves eligible golden IDs before filling its remaining slots. `selection.json` records exclusions such as non-public-domain objects. To prepare browser-saved visitor HTML without crawling, use `make ingest-visitors ARGS="--html-dir data/visitor_pages --prepare-only"` to derive source URLs from canonical links or Chrome original-URL comments and capture times from file mtime. The generated local manifest preserves that provenance.

The [full ingestion guide](docs/full-ingestion.md) covers local inference, published SigLIP 2 image vectors, durable checkpoints, and detached execution with `nohup`. Local inference has no API quotas. Gemini remains available with token-aware pacing and persisted daily quota accounting.

See the [completed Phase 1 build report](docs/phase-1-results.md) for measured duration, image coverage, golden verification, and the pinned published snapshot.

## Chat and MCP

```sh
make dev
# In another terminal, from the repository root:
make demo-check
```

The demo exercises three golden questions through the actual HTTP chat route and prints answer, citations, route, latency, and estimated cost. It exits nonzero if the provider is unavailable or answer verification fails. See [Phase 2 behavior and verification status](docs/phase-2.md) for the API contract, retrieval scores, guardrails, and current limitations.

```sh
uv run --project api --locked met-agent-mcp
uv run --project api --locked met-agent-mcp --transport sse --port 8001
```

See [MCP client setup](docs/mcp.md). Both transports use the same validation and five tool schemas as chat. Local embeddings remain the default; image similarity uses only the Met's published vectors.

## Project layout

```text
api/
  src/met_agent/
    config.py                 Environment loading and validation
    main.py, routes.py        Application factory, chat SSE, objects, audit
    middleware.py             Request context and safe HTTP errors
    observability/            Safe logs and optional Langfuse traces
    ingestion/                Public sources, documents, verification, snapshots
    retrieval/                Embeddings, dense/BM25 fusion, local reranking
    llm/                      Explicit routing, cost estimates, prompt loading
    tools/                    Five shared typed tool implementations
    agent/, guardrails/       Bounded loop, session events, evidence checks
    mcp/                      Official SDK stdio and SSE transports
  prompts/                    Versioned runtime instructions and changelog
  scripts/                    Ingestion, verification, publication, seed, privacy check
  data_sources/               Curated public visitor pages
  tests/                      Offline tests and local Qdrant integration tests
  pyproject.toml              Exact direct dependency pins and tool settings
  uv.lock                     Reproducible transitive dependency resolution
evals/golden.jsonl             Original evaluation questions, not yet scored
docs/configuration.md         Settings and operational behavior
```

## Boundaries

The current service has no authentication or rate limiting. The development server binds to loopback. CORS restricts browser origins and is not an authorization mechanism. Request logs omit bodies, raw paths, query strings, and headers; application code must continue to avoid interpolating sensitive values into free-form log messages. Session audits under `DATA_DIR` and optional Langfuse traces contain conversation content. They require access and retention controls before a public deployment. Phase 2 uses one API process with local session locks.

## Data and attribution

Collection metadata is provided by The Metropolitan Museum of Art through [metmuseum/openaccess](https://github.com/metmuseum/openaccess), under CC0. Preparation uses its official CSV. The [Met's Hugging Face dataset card](https://huggingface.co/datasets/metmuseum/openaccess) supplies the accompanying attribution and usage guidance.

Image embeddings are provided by The Metropolitan Museum of Art through [metmuseum/openaccess-embeddings-siglip2](https://huggingface.co/datasets/metmuseum/openaccess-embeddings-siglip2), also CC0. The pinned source revision is `45fe67456ef1cffbe5ba801d5cd341fcf31cd806`, using `google/siglip2-so400m-patch14-384` with 1,152 dimensions. Its published vectors match 19,964 of the selected 20,000 Object IDs (99.82%); the remaining 36 objects have no image vector. No image embeddings are computed by this project.

This is a modified derivative: 20,000 public-domain records are selected, metadata is normalized, retrieval text is assembled, local text embeddings are generated, and published image vectors are joined by Object ID into a Qdrant index. Raw metadata fields and source URLs are retained. Text uses `intfloat/multilingual-e5-large` with 1,024 dimensions; model weights have their own MIT license. Snapshot manifests record the provider, model, dimensions, image source revision, coverage, and artifact hashes.

This project is not affiliated with or endorsed by The Metropolitan Museum of Art. Museum logos and trademarks are not used as project branding. The source code is MIT licensed. Visitor website extracts retain their original ownership and are not covered by the collection dataset's CC0 license.

## Phase 2 demo measurement

Measured September 4, 2026 with GPT-OSS 120B main, GPT-OSS 20B lite, local E5 retrieval, and free-tier pacing:

| Question | Verified result | Route | End-to-end latency | Standard-price estimate |
| --- | --- | --- | --- | --- |
| Temple of Dendur gallery | Gallery 131, Object 547802 | lite | 76.156 s | $0.000763275 |
| Fifth Avenue on Wednesdays | Closed, captured Plan Your Visit hours table | main | 7.168 s | $0.001022625 |
| Meaning of Wheat Field with Cypresses | Non-interpretive policy refusal | main policy, lite classification | 0.147 s | $0.000061200 |

Dendur spent 58.209 s waiting for the 8,000-TPM model budget, 2.602 s in provider calls, and 15.342 s in tools/service overhead including cold model startup. No retries or fallback occurred. These results do not imply sub-second end-to-end retrieval or unlimited free throughput. The checked-in [price table](api/src/met_agent/llm/cost.py) uses Groq's published standard input/output rates and reported tokens for every model call, including guardrails; it estimates equivalent inference cost, not a charge on the free tier. Local CPU and infrastructure costs are excluded.

See [measured latency and operating limits](docs/limitations.md) for the timing breakdown and paid-tier pacing configuration.
