# met-collection-agent

An independent, grounded collection assistant being built over The Metropolitan Museum of Art's Open Access records.

[![CI](https://github.com/AjayKasu1/met-collection-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/AjayKasu1/met-collection-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Phase 0 implements the API foundation: typed configuration, health reporting, request IDs, JSON logs, strict static checks, and offline tests. Collection ingestion, retrieval, chat, evaluations, MCP, and the web client are subsequent phases. No model calls or external-service connections are made by this scaffold, and no evaluation results are claimed yet.

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git. Run commands from this repository's root. uv installs Python 3.12 if needed and uses the committed dependency lockfile.

```sh
make setup
cp .env.example .env
```

Edit `.env` locally and supply `GEMINI_API_KEY`, `LLM_MODEL`, `LLM_MODEL_LITE`, and `EMBEDDING_MODEL`. Use exact model identifiers accepted by your provider through LiteLLM. Model names are intentionally configuration, with no guessed defaults. Keep `USE_AI_GATEWAY=false` for direct provider access.

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

Tests supply synthetic settings and never load the developer's `.env`. They require neither an LLM key nor a running Qdrant instance. Coverage includes configuration validation, concurrent request isolation, CORS preflights, error redaction, and incremental ASGI streaming. The coverage gate is 90% with branch coverage enabled.

`make setup` installs pre-commit hooks. Hooks and CI use the same pinned lint, type, and test tools. The repository check inspects the Git index, rejects private file paths before reading their contents, and detects recognizable provider tokens. It is an additional safeguard, not proof that arbitrary secrets cannot be committed. Stage files explicitly and review `git diff --cached` before committing.

## Project layout

```text
api/
  src/met_agent/
    config.py                 Environment loading and validation
    main.py                   Application factory and health endpoint
    middleware.py             Request context and safe HTTP errors
    observability/logging.py  JSON logging and structured redaction
  scripts/check_repository.py Git index privacy check
  tests/                      Offline unit and HTTP tests
  pyproject.toml              Exact direct dependency pins and tool settings
  uv.lock                     Reproducible transitive dependency resolution
evals/golden.jsonl             Original evaluation questions, not yet scored
docs/configuration.md         Settings and operational behavior
```

## Boundaries

The current service has no authentication or rate limiting. The development server binds to loopback. CORS restricts browser origins and is not an authorization mechanism. Request logs omit bodies, raw paths, query strings, and headers; application code must continue to avoid interpolating sensitive values into free-form log messages. Future tracing and audit storage require their own content-access and retention controls.

The source code is MIT licensed. Collection data comes from [The Met Open Access initiative](https://www.metmuseum.org/about-the-met/policies-and-documents/open-access) under its applicable CC0 terms. This project is not affiliated with or endorsed by The Metropolitan Museum of Art.
