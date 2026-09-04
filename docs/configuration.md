# Configuration and API foundation

`met_agent.config` is the only application module that reads environment variables or dotenv files. `create_app()` loads configuration before creating the FastAPI application. Importing modules does not load configuration or connect to providers.

Run development commands from the repository root. The loader reads `.env` in the current working directory, with process variables taking precedence, and never searches parent directories. Deployments can supply process environment variables instead. Tests call `load_settings(env_file=None)` or use explicitly created synthetic files.

## Required settings

| Key | Meaning |
| --- | --- |
| `GEMINI_API_KEY` | Provider key, represented internally as `SecretStr` |
| `LLM_MODEL` | Exact LiteLLM identifier for the main model |
| `LLM_MODEL_LITE` | Exact LiteLLM identifier for lightweight tasks |
| `EMBEDDING_MODEL` | Exact embedding model identifier |

Required strings cannot be empty or whitespace. No model name is inferred. The scaffold validates these names as configuration but does not call the provider to check availability. A later model-not-found response must be resolved by checking the configured provider model.

`.env.example` lists every supported setting and documents defaults. A test checks the set of keys against the settings schema. Blank optional environment values are ignored. `LOG_LEVEL` accepts upper or lower case and is normalized before logging setup. Settings are immutable after loading; restart the process after changing configuration.

## Optional integration activation

| Integration | Configuration needed for its health flag |
| --- | --- |
| Qdrant | `QDRANT_URL`, default `http://localhost:6333` |
| AI Gateway | `USE_AI_GATEWAY=true`, `CF_AI_GATEWAY_URL`, `CF_AI_GATEWAY_TOKEN` |
| Fallback model | `GROQ_API_KEY`, `LLM_MODEL_FALLBACK` |
| Langfuse | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` |
| Hugging Face dataset | `HF_DATASET_REPO`; a public download needs no token |
| R2 | `R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET` |

Incomplete optional integrations remain disabled. An explicitly enabled AI Gateway with missing URL or token fails startup, avoiding an accidental switch to direct provider requests. The API health endpoint does not contact optional services. Ingestion uses the configured embedding route and Qdrant server. Dataset publication and seeding use `HF_DATASET_REPO` and optional `HF_TOKEN` explicitly. `CLOUDFLARE_API_TOKEN`, `CF_ACCOUNT_ID`, and `GCP_REGION` are retained for later deployment tooling.

Service URLs must use HTTP or HTTPS and cannot carry user credentials, query strings, or fragments. Keys belong in dedicated secret fields. Set `GIT_SHA` to a 7 to 40 character lowercase hexadecimal revision for a labelled build; otherwise `/health` reports `unknown`.

## HTTP behavior

- `GET /health` is a typed liveness response. It does not prove retrieval or model readiness.
- `GET /docs` and `GET /openapi.json` expose the implemented API contract.
- `CORS_ORIGINS` accepts a JSON array or comma-separated list of explicit HTTP origins. Wildcards, credentials, paths, query strings, and fragments are rejected. An empty array disables cross-origin access. Cookies are not allowed by CORS.
- Each HTTP response, including errors and CORS preflights, carries `X-Request-ID`. A supplied ID is accepted only if it is a single header, 1 to 128 characters, starts with an ASCII letter or digit, and contains only ASCII letters, digits, `.`, `_`, `:`, or `-`. Other values are replaced by a generated ID.
- Requests log the route template, method, status, elapsed milliseconds, and request ID. Raw unmatched paths, query strings, headers, and bodies are omitted.
- Unhandled exceptions return a generic JSON error with the request ID. Logs contain the exception class, not its potentially sensitive message. If a response stream already started, the connection fails with a sanitized exception instead of sending a second response.

Middleware is ordered as request context, CORS, then error boundary. It is implemented directly in ASGI so response chunks pass through immediately and context remains attached during streaming.

## Verification and release discipline

`make check` runs Ruff lint and formatting checks, `mypy --strict`, tests with a 90% branch-inclusive coverage gate and local Qdrant integration coverage, and the Git index privacy check. The GitHub workflow runs the same checks with Python 3.12 and an immutable revision of each action. It needs no repository secrets.

Application versions come from package metadata. Direct dependencies and the build backend are pinned in `api/pyproject.toml`; transitive versions and hashes live in `api/uv.lock`. Update dependencies deliberately, regenerate the lockfile, and run all checks before committing.

The original build instructions, root-level duplicate golden file, credentials, raw data, virtual environments, and generated output are ignored. Versioned runtime prompts will live in `api/prompts/` and are intentionally not covered by the build-prompt exclusion.
