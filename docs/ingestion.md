# Collection and visitor ingestion

Run commands from the repository root after `make setup`. Configuration loads only through `met_agent.config`; no script searches a parent directory for credentials. Set the required provider fields in local `.env`. Start a local index with `make qdrant`, or configure an existing Qdrant server. Docker Compose also packages the API and local web client; `make demo` restores the verified snapshots before starting the full stack.

## A bounded pilot

Use a separate directory and collection so a small sample cannot be confused with a complete index:

```sh
make ingest ARGS="--limit 200 --data-dir data/pilot --collection met_objects_pilot_local --prepare-only"
make verify-golden ARGS="--data-dir data/pilot"
make ingest ARGS="--limit 200 --data-dir data/pilot --collection met_objects_pilot_local --reuse-prepared --batch-size 32"
```

`--prepare-only` downloads and validates public sources without calling an embedding model or Qdrant. `--reuse-prepared` indexes the existing Parquet artifact and makes no Met requests. The default limit comes from `INGEST_MAX_OBJECTS`; a positive `--limit` overrides it. Use a distinct collection for each differently sized sample. An upsert does not delete objects outside the current sample.

Preparation streams the official [Met CSV](https://github.com/metmuseum/openaccess) from its Git LFS media endpoint. A checksum and fetch timestamp accompany the cached CSV. The complete `hasImages=true` inventory from the [Met Collection API](https://metmuseum.github.io/) supplies image priority because the CSV has no image URLs. Missing or inconsistent inventory responses stop preparation.

Records must explicitly be public domain and have a nonempty Met resource URL. Selection reserves every eligible `expected_object_ids` value from `evals/golden.jsonl` first, deduplicating repeated references. `--golden PATH` selects another evaluation file. Remaining slots prioritize highlights, images, then object ID for deterministic ties. Reserved objects count toward the limit. An unknown required ID or a limit too small for the eligible golden set stops preparation.

`selection.json` records the required and included IDs, explicit eligibility exclusions, and the Parquet SHA-256. Required records lacking public-domain status or a resource URL are excluded and logged with a reason. Pollock's `488978` currently falls outside the public-domain subset. `--reuse-prepared` checks the artifact hash, current golden IDs, and coverage after applying the limit; stale preparation must be rebuilt. This selection deliberately favors the golden set and highlights, so it is not a representative or held-out evaluation sample. `--refresh` replaces the cached CSV and image inventory only after valid downloads.

Every record keeps its raw CSV fields. Retrieval text uses supplied title, artist, culture, period, date, medium, classification, department, credit line, gallery, tags, and description. Missing artist names and curatorial descriptions stay absent. Gallery values are observations from the source, not a claim that an object is currently on view. `--enrich-live` explicitly refreshes images, galleries, and available descriptions from the API, and refuses records whose live public-domain flag changed. A failed live request stops that optional enrichment run and retains the previous completed artifact.

`objects.parquet` exposes scalar columns, retrieval text, and a complete validated `payload_json` for lossless reuse. Data and model caches remain under ignored `data/`.

## Hybrid index

Text embeddings default to local FastEmbed `intfloat/multilingual-e5-large`, a multilingual 1,024-dimensional model with explicit E5 query and passage prefixes. The model is verified against the installed registry and its ONNX weights are pinned to an immutable revision. Invalid model identifiers, oversized inputs, malformed vectors, and incompatible collections stop ingestion. There is no model fallback. See [configuration](configuration.md) for the local model's cache, CPU threads, and token limit.

`EMBEDDING_PROVIDER=gemini` retains the optional Google path. Its adapter accepts the bare model name, Google's `models/` form, and LiteLLM's `gemini/` prefix. With `EMBEDDING_DIMENSIONS=768`, it requests reduced output using Gemini's native `outputDimensionality` through LiteLLM's `dimensions` parameter. Responses must match the requested size and are normalized to unit length. Ordinary Gemini ingestion uses bounded Router retries; checkpoint mode owns quota reservations and rate-limit retries.

Collection schema version 3 records text provider, model, dimension, sparse model, and optional image provenance. Query embedding and snapshot restore reject incompatible model identities before inference or upload. Earlier Gemini schema-version-2 checkpoints and the schema-version-1 3,072-dimensional pilot remain separate; they are not silently relabelled as local vectors.

For direct Gemini, set `USE_AI_GATEWAY=false`. When enabled, AI Gateway uses its Google AI Studio endpoint, the gateway token in `cf-aig-authorization`, and the Google key separately. Authentication errors do not silently switch routes. See [Cloudflare's provider documentation](https://developers.cloudflare.com/ai-gateway/usage/providers/google-ai-studio/).

Set `CF_AI_GATEWAY_URL` to `https://gateway.ai.cloudflare.com/v1/ACCOUNT/GATEWAY`. The native `/google-ai-studio`, `/google-ai-studio/v1`, and `/google-ai-studio/v1beta` suffixes are also accepted. Workers AI URLs on `api.cloudflare.com`, OpenAI-compatible `/compat` routes, and complete inference URLs are rejected before credentials are sent. The gateway token needs AI Gateway Run permission. Workers AI Edit does not change this application's Google provider route.

Qdrant stores named `dense` cosine vectors and `sparse` FastEmbed `Qdrant/bm25` vectors. With `--with-images`, it also stores the named cosine vector `image`. Dense and image vectors use disk storage and INT8 scalar quantization; sparse vectors use disk storage and Qdrant's IDF modifier. Payload indexes cover department, object ID, highlight status, gallery, begin/end dates, and source URL.

To join published image vectors to a prepared collection:

```sh
make prepare-images
make ingest ARGS="--limit 20000 --reuse-prepared --with-images --batch-size 32"
```

`prepare-images` downloads only Parquet files from the pinned Met SigLIP 2 dataset, checks its immutable revision, and joins on `objectID`. It rejects duplicate selected IDs, model or dimension drift, nonfinite values, and non-normalized vectors. It writes `image_vectors.parquet` and an integrity manifest with the source dataset, revision, model, dimensions, source file hashes, and included/missing IDs. `--with-images` requires this exact artifact and selection; it never computes an image vector or substitutes a text vector. The 20,000-record selection has 19,964 published image vectors, so the older fallback dataset is unnecessary.

Batch writes pair text, sparse, and available published image vectors with the same document and wait for acknowledgment. Local collection ingestion is always checkpointed, resumes by completed Object ID, and groups remaining records by text length to reduce padding work. It never schedules quota waits. Completion verifies every expected ID and payload and compares each image vector with the published source. See [detached execution and recovery](full-ingestion.md).

The Phase 2 contract `find_similar_objects(object_id, k)` selects the `image` vector, excludes the source object, and bounds `k` to 1 through 50. An object without a published image vector produces `image_unavailable`; no text fallback is allowed. Only the typed tool schema is present in this phase.

## Visitor pages

```sh
make ingest-visitors ARGS="--data-dir data/pilot --collection met_visitor_info_pilot --prepare-only"
make ingest-visitors ARGS="--data-dir data/pilot --collection met_visitor_info_pilot"
```

The curated YAML covers hours, admission, directions, accessibility, families, group visits, The Met Cloisters, visitor policies, current exhibitions, the interactive map, gallery closures, and contact information. Requests identify this project, wait three seconds by default, honor stricter robots crawl rules, and recheck permission after redirects. `--request-interval-seconds` can increase that delay but cannot reduce it below one second. A failed robots request stops the crawl. HTTP 404 for `robots.txt` permits crawling; access failures and exhausted rate limits do not.

To ingest Chrome "Webpage, Complete" saves, put top-level UTF-8 `.html` or `.htm` files under `data/visitor_pages/`. When no `sources.yaml` exists, the loader derives each source URL from its canonical link, falling back to Chrome's explicit saved-from comment, and uses file modification time in UTC as `fetched_at`. It writes a reproducible manifest and records how the provenance was obtained. Companion asset directories are ignored. File mtime is an approximation of capture time; copying or editing a page can change it. Supply an explicit manifest when the URL is missing or ambiguous, or when you have a more accurate capture time. Each manual entry requires a label, original Met HTTPS URL, relative HTML filename, and a timestamp with a timezone. This example illustrates the format; use the time you captured your page:

```yaml
pages:
  - label: Hours, admission, directions, and planning
    url: https://www.metmuseum.org/plan-your-visit
    html_file: plan-your-visit.html
    fetched_at: "2026-09-03T18:00:00-04:00"
```

```sh
make ingest-visitors ARGS="--html-dir data/visitor_pages --data-dir data/pilot --prepare-only"
make ingest-visitors ARGS="--html-dir data/visitor_pages --data-dir data/pilot --collection met_visitor_info_pilot"
```

`--html-dir` without a value defaults to `data/visitor_pages`. `--sources PATH` overrides the local manifest location. Saved mode makes no visitor HTTP or robots requests and never falls back to crawling. An explicitly selected missing manifest is an error. Indexing uses the configured text embedder and contacts Qdrant. Local inference does not contact Google. Source URLs, capture timestamps, headings, and stable chunk IDs flow through the same pipeline as live pages. The loader rejects missing timezones, duplicate URLs, paths or symlinks outside the HTML directory, non-HTML filenames, non-UTF-8 text, and files larger than 20 MiB. A saved manifest takes precedence on subsequent runs; editing a file does not silently revise its recorded timestamp. Text keeps meaningful image alt descriptions while omitting local asset paths and tracking frames. Files, manifests, and generated chunks remain under ignored `data/`.

Main content becomes Markdown with headings, tables, lists, and links retained. Each section is divided into approximately 500-token windows with 50-token overlap. The deterministic Unicode-safe counter estimates tokens locally and is not a Gemini billing measure. Each chunk records its source URL, fetch timestamp, heading, and stable ID. Stale chunks for a page are removed only after all replacement chunks have been acknowledged.

Production refreshes use a stable alias and an immutable timestamped collection:

```sh
make ingest-visitors ARGS="--promote-alias met_visitor_info_live --request-interval-seconds 3"
```

The command crawls and prepares every page before vector writes, builds a separate physical collection, validates embedding metadata, exact point count, and exact source-URL coverage, then changes the alias in one atomic Qdrant operation. A validation or ingestion failure never moves the alias. `data/visitor_refresh.json` records the new and previous physical collections, capture range, embedding identity, counts, and source URLs. The prior collection is retained for rollback. `--prepare-only` and `--promote-alias` are mutually exclusive, and a promotion alias cannot collide with a physical collection.

The weekly `Refresh visitor information` workflow uses this release path. It remains disabled until `ENABLE_VISITOR_REFRESH=true` and the `data-refresh` environment has `QDRANT_URL` and a write-scoped `QDRANT_API_KEY`. Its report is retained as a non-secret workflow artifact for 30 days.

The current [interactive museum map](https://maps.metmuseum.org/) requires JavaScript and supplies no usable static floor-plan text. The accessibility page contributes the museum's published map guidance and map link. The index does not infer restroom proximity, gallery adjacency, or an elevator's location relative to a particular artwork. Those questions require a verified map source or a handoff in the later agent phase.

## Golden verification

`make verify-golden ARGS="--collection met_objects"` first verifies the actual Qdrant IDs and payloads against all prepared records. Omitting `--collection` checks only prepared membership. It then writes and prints `data/verify_golden.md`, including live titles, existence, sample membership, and original rows explicitly marked `verify: true`. HTTP 404 means absent; authentication, rate limits, connection failures, and malformed responses mean unknown. Unknown or absent objects produce exit status 1. An object outside the sample is reported but does not cause that exit status.

This checks identities and membership, not retrieval accuracy. A valid ID may still identify the wrong artwork. Review the title column before changing golden expectations. The original evaluation file is not changed by verification.

The two corrected references were resolved through the Met search API and confirmed against object records: [Gubbio search](https://collectionapi.metmuseum.org/public/collection/v1/search?q=Gubbio) identifies [Studiolo, 198556](https://collectionapi.metmuseum.org/public/collection/v1/objects/198556) for `col-009`; [Ugolino search](https://collectionapi.metmuseum.org/public/collection/v1/search?title=true&q=Ugolino%20and%20His%20Sons) identifies [Ugolino and His Sons, 204812](https://collectionapi.metmuseum.org/public/collection/v1/objects/204812) for `col-010`. The previous IDs identified an unrelated patio and cream pitcher. Artist attribution and cultural expectations still require the manual review recorded in the golden rows; the Studiolo's live artist field and CSV attribution list differ, and Ugolino's live culture field is blank.

## Verification

`make check` runs strict types, lint, unit tests, a 90% branch-inclusive coverage gate, and the Git index privacy check. Tests never load local credentials or call Gemini. Integration tests use only loopback Qdrant, generate unique collection names, and remove only their own collections. They test a 20-object ingestion, repeated upserts, vector and payload configuration, and visitor chunk replacement. CI starts Qdrant 1.19.0 for these tests.
