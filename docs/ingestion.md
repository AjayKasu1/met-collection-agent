# Collection and visitor ingestion

Run commands from the repository root after `make setup`. Configuration loads only through `met_agent.config`; no script searches a parent directory for credentials. Set the required provider fields in local `.env`. Start a local index with `make qdrant`, or configure an existing Qdrant server. Docker Compose currently supplies the Phase 1 Qdrant service; API and web containers belong to later phases.

## A bounded pilot

Use a separate directory and collection so a small sample cannot be confused with a complete index:

```sh
make ingest ARGS="--limit 200 --data-dir data/pilot --collection met_objects_pilot_200 --prepare-only"
make verify-golden ARGS="--data-dir data/pilot"
make ingest ARGS="--limit 200 --data-dir data/pilot --collection met_objects_pilot_200 --reuse-prepared"
```

`--prepare-only` downloads and validates public sources without calling an embedding model or Qdrant. `--reuse-prepared` indexes the existing Parquet artifact and makes no Met requests. The default limit comes from `INGEST_MAX_OBJECTS`; a positive `--limit` overrides it. Use a distinct collection for each differently sized sample. An upsert does not delete objects outside the current sample.

Preparation streams the official [Met CSV](https://github.com/metmuseum/openaccess) from its Git LFS media endpoint. A checksum and fetch timestamp accompany the cached CSV. The complete `hasImages=true` inventory from the [Met Collection API](https://metmuseum.github.io/) supplies image priority because the CSV has no image URLs. Missing or inconsistent inventory responses stop preparation.

Records must explicitly be public domain and have a nonempty Met resource URL. Selection orders highlights first, image-bearing records second, then object ID for deterministic ties. This can bias a small pilot toward departments with lower IDs. It is not a representative evaluation sample. `--refresh` replaces the cached CSV and image inventory only after valid downloads.

Every record keeps its raw CSV fields. Retrieval text uses supplied title, artist, culture, period, date, medium, classification, department, credit line, gallery, tags, and description. Missing artist names and curatorial descriptions stay absent. Gallery values are observations from the source, not a claim that an object is currently on view. `--enrich-live` explicitly refreshes images, galleries, and available descriptions from the API, and refuses records whose live public-domain flag changed. A failed live request stops that optional enrichment run and retains the previous completed artifact.

`objects.parquet` exposes scalar columns, retrieval text, and a complete validated `payload_json` for lossless reuse. Data and model caches remain under ignored `data/`.

## Hybrid index

The dense provider uses the exact `EMBEDDING_MODEL` identifier. The adapter accepts Google's bare or `models/` form and LiteLLM's `gemini/` prefix. Authentication, invalid model identifiers, malformed vectors, and incompatible existing collections stop ingestion. Rate limits, timeouts, and transient failures use bounded Router retries. Embeddings never fall back to another model because that would mix vector spaces.

For direct Gemini, set `USE_AI_GATEWAY=false`. When enabled, AI Gateway uses its Google AI Studio endpoint, the gateway token in `cf-aig-authorization`, and the Google key separately. Authentication errors do not silently switch routes. See [Cloudflare's provider documentation](https://developers.cloudflare.com/ai-gateway/usage/providers/google-ai-studio/).

Set `CF_AI_GATEWAY_URL` to `https://gateway.ai.cloudflare.com/v1/ACCOUNT/GATEWAY`. The native `/google-ai-studio`, `/google-ai-studio/v1`, and `/google-ai-studio/v1beta` suffixes are also accepted. Workers AI URLs on `api.cloudflare.com`, OpenAI-compatible `/compat` routes, and complete inference URLs are rejected before credentials are sent. The gateway token needs AI Gateway Run permission. Workers AI Edit does not change this application's Google provider route.

Qdrant stores named `dense` cosine vectors and `sparse` FastEmbed `Qdrant/bm25` vectors. Both use disk storage; dense vectors also use INT8 scalar quantization. The sparse index applies Qdrant's IDF modifier. Payload indexes cover department, object ID, highlight status, gallery, begin/end dates, and source URL. Collection metadata records the embedding model, sparse model, and schema version.

Batch responses must contain one finite, nonzero dense vector per input with a consistent dimension. Writes pair both vectors with the same document and wait for acknowledgment. Repeating a completed ingestion upserts stable IDs without duplicates. A failed run can leave acknowledged earlier batches; rerun against the same prepared artifact to finish. It recomputes embeddings and may incur provider usage again.

## Visitor pages

```sh
make ingest-visitors ARGS="--data-dir data/pilot --collection met_visitor_info_pilot --prepare-only"
make ingest-visitors ARGS="--data-dir data/pilot --collection met_visitor_info_pilot"
```

The curated YAML covers hours, admission, directions, accessibility, bag/stroller policies, map guidance, gallery closures, and contact information. Requests identify this project, run at most once per second, honor stricter robots crawl rules, and recheck permission after redirects. A failed robots request stops the crawl. HTTP 404 for `robots.txt` permits crawling; access failures and exhausted rate limits do not.

Main content becomes Markdown with headings, tables, lists, and links retained. Each section is divided into approximately 500-token windows with 50-token overlap. The deterministic Unicode-safe counter estimates tokens locally and is not a Gemini billing measure. Each chunk records its source URL, fetch timestamp, heading, and stable ID. Stale chunks for a page are removed only after all replacement chunks have been acknowledged.

The current [interactive museum map](https://maps.metmuseum.org/) requires JavaScript and supplies no usable static floor-plan text. The accessibility page contributes the museum's published map guidance and map link. The index does not infer restroom proximity, gallery adjacency, or an elevator's location relative to a particular artwork. Those questions require a verified map source or a handoff in the later agent phase.

## Golden verification

`make verify-golden` writes and prints `data/verify_golden.md`, including live titles, existence, sample membership, and original rows explicitly marked `verify: true`. HTTP 404 means absent; authentication, rate limits, connection failures, and malformed responses mean unknown. Unknown or absent objects produce exit status 1. An object outside the sample is reported but does not cause that exit status.

This checks identities and membership, not retrieval accuracy. A valid ID may still identify the wrong artwork. Review the title column before changing golden expectations. The original evaluation file is not changed by verification.

## Verification

`make check` runs strict types, lint, unit tests, a 90% branch-inclusive coverage gate, and the Git index privacy check. Tests never load local credentials or call Gemini. Integration tests use only loopback Qdrant, generate unique collection names, and remove only their own collections. They test a 20-object ingestion, repeated upserts, vector and payload configuration, and visitor chunk replacement. CI starts Qdrant 1.19.0 for these tests.
