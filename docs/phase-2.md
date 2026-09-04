# Phase 2: retrieval and the constrained agent

The HTTP service and MCP server share five validated tools. Chat adds a bounded model loop, multilingual routing, citation checks, and claim verification. Evaluation scoring, the web client, and deployment remain later phases.

## Retrieval

`search_collection` and `search_visitor_info` query 40 dense and 40 sparse candidates with the same payload filters. Reciprocal rank fusion uses `1 / (60 + rank)` per list. A local cross-encoder reranks the union, with deterministic score and ID tie-breaking. Collection search returns eight results by default; visitor search returns five. Each result includes dense, sparse, fusion, and reranker scores. These scores are ranking signals, not probabilities of correctness.

Text embeddings default to `intfloat/multilingual-e5-large` through FastEmbed. Gemini remains optional. Provider, model name, and dimensions must match collection metadata before any query embedding is computed. The reranker is FastEmbed's Apache-2.0 `Xenova/ms-marco-MiniLM-L-6-v2`, pinned to revision `a09144355adeed5f58c8ed011d209bf8ee5a1fec`. The first search downloads its weights. Model instances are shared and CPU inference is serialized within each process.

The reranker and collection metadata are English-focused. Chat's lite-model classifier produces an English search rewrite while preserving the user's answer language. MCP clients should supply English search queries for best reranking. Multilingual dense embeddings alone do not establish end-to-end multilingual retrieval quality; that needs evaluation.

Department filters use exact values. Date filters select overlapping object date intervals. `on_view` filters use the indexed gallery value, which describes ingestion time. Current display questions require `get_object`; the service never presents that filter as a live availability check.

`find_similar_objects(object_id, k)` retrieves the source's published `image` vector and performs a cosine search in that named vector only. It excludes the source object and returns at most 50 neighbors. An absent source returns `object_not_indexed`; an object without a published vector returns `image_unavailable`. There is no text or generated-image fallback. Similarity is a model distance, not an artistic judgment.

## Visitor capture

The supplied Chrome save was ingested into `met_visitor_info` as 28 chunks. The source is `https://www.metmuseum.org/plan-your-visit`, captured at `2026-09-04T08:49:28.645305+00:00` from file mtime. The document has no canonical link, so discovery used Chrome's original-URL comment and persisted that provenance in the local manifest. No manual manifest was required. Browser asset folders were excluded; meaningful image alt text was retained, without inferring floor-plan details.

Visitor answers cite the source URL and can use the returned capture timestamp. File mtime is a capture-time proxy chosen for these saved files, not a claim about when the museum last updated the page. Copying or editing a file may change it. Reuse the generated manifest to preserve the accepted capture metadata.

Hours and admission are sections of the saved Plan Your Visit page. Accessibility, visitor guidelines, families, groups, the Cloisters, contact, the museum map, and the exhibitions listing still need separate captures. The browser interface available during this build exposed an accessibility tree but no full HTML export or native save dialog, so no additional captures are claimed. Saved pages from the official `maps.metmuseum.org` host are accepted alongside `www.metmuseum.org` and `metmuseum.org`. A map save provides only its captured text, not inferred room connections or live navigation.

## HTTP contract

Start `make dev` from the repository root. `/health` does not initialize models or contact services.

```sh
curl -N http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Which gallery is the Temple of Dendur in?"}'
```

The SSE stream sends a `session` event with the session UUID, zero or more waiting comments, `token` events containing `{"text": "..."}`, and one `answer` event containing the full `AgentAnswer`. The text chunks are emitted only after verification. This deliberately trades early draft display for preventing rejected claims from reaching the client. A failed model request emits an `error` event instead of an answer. HTTP 200 indicates that the stream opened, not that a valid answer was produced.

Send `session_id` on subsequent requests to retain conversation context. `language` is an optional EN/FR/ES/ZH hint. The model classifies language from the request and history. A final draft in a different language is rejected.

`AgentAnswer` includes text, citations, language, optional handoff, route, estimated cost, latency, grounding score, policy-refusal flag, session/turn IDs, and actual model-call records. A citation contains exactly one object ID or source URL and a nonblank verbatim quote. The route records the routing decision; `model_calls` records actual calls, including lite guardrails and a Groq fallback if used.

`GET /objects/{object_id}` proxies the Met API with an identifying User-Agent. Its bounded cache stores up to 512 objects for at most five minutes and returns independent copies. It includes GalleryNumber, source URL, image links, and fetch time. A missing object is HTTP 404; unavailable upstream service is HTTP 503. Non-public-domain objects can be looked up live, including golden row `col-008`, without adding them to the public-domain index.

`GET /sessions/{session_id}/events?after=0&limit=200` returns ordered audit events. `after` is an exclusive sequence cursor; the maximum page size is 500. Unknown sessions are HTTP 404. Events are stored in `DATA_DIR/sessions.sqlite3`, with append-only database triggers and known credentials redacted. The audit records user/model messages, prompts and hashes, tool calls/results, validation errors, guard decisions, usage, and the final answer. Concurrent turns for one session are serialized. The last twelve user/final-answer events provide conversational context; earlier citations do not authorize citations in a new turn.

## Guardrails and routing

The lite model classifies intent. Simple collection questions use the lite answer route; visitor and multi-step questions use main. Interpretive questions receive the localized non-interpretive policy and factual alternatives. Out-of-scope questions produce a structured terminal contact suggestion. The service never sends a message to the contact.

Every proposed tool call consumes the six-call budget, including invalid arguments and unknown tools. Calls execute sequentially. A terminal handoff prevents remaining proposed calls from executing. Pydantic validates strict inputs before any handler runs. Failed tool operations return sanitized errors for the model to handle.

Every citation must match this turn's tool evidence, including its quote. The lite grounding check extracts atomic claims and identifies supporting evidence keys. Unsupported claims or invalid citations trigger one regeneration. A second failure returns an honest inability to verify, with no citations and grounding score zero. Grounding uses a model judgment and can be wrong; the deterministic citation gate does not prove semantic correctness.

Versioned runtime prompts live in `api/prompts/`, are packaged in wheels, and have hashes in the audit. Their changelog distinguishes deterministic contract tests from measured model evaluation. The private build instructions remain outside Git.

## Providers, cost, and observability

Chat uses explicit credentials and exact configured Google model identifiers through LiteLLM. With AI Gateway enabled, Google calls use its provider-native gateway path and gateway authorization header. Embeddings continue to run locally unless Gemini is explicitly configured. Authentication and model-not-found failures do not trigger fallback. Only exhausted rate-limit or timeout failures may use configured Groq, directly, with its separate credential.

`cost_usd` sums estimates for successful model responses, including routing, drafting, verification, and repair calls. Unknown model pricing produces `null`, not a fabricated zero. Estimates use LiteLLM's available price catalog and reported token usage; they are not invoices and exclude infrastructure, gateway charges, and any unreported charges for failed requests. Latency is measured wall time and includes retrieval and guardrails.

Langfuse activates only with all three configured values. The official `@observe` decorator creates a chat root, tool observations describe tool operations, and a request-scoped LiteLLM `CustomLogger` callback records model attempts, parent trace context, usage, latency, and estimated cost. Input/output capture on the root decorator is disabled; explicitly selected, redacted fields are recorded instead. Grounding score is attached to the trace. No global SDK credential variables are populated. Langfuse is currently unconfigured, so remote trace export has not been verified.

Session audit content and configured Langfuse traces contain conversations and source excerpts. The current service has no user authentication, audit authorization, retention job, or public rate limit. Keep these Phase 2 interfaces on loopback or behind existing access controls. CORS is not authorization. Shared SQLite audit storage and process-local session locks are intended for one API process at this phase.

## Demo and verification status

```sh
make demo-check
```

This calls the real FastAPI `/chat` route through an ASGI HTTP transport for `col-003`, `vis-001`, and `ref-001`. It prints answer, citations, route, latency, estimated cost, and actual model paths. It saves a local `data/demo-check.json` report and exits nonzero on a provider error, failed verification, absent factual citations, or missing policy refusal. It is a smoke test, not the Phase 3 evaluator.

On September 4, 2026, live collection and visitor searches returned results, and image similarity returned published-vector neighbors. The Met API returned gallery 131 for object 547802 and gallery 851 for object 488978. The three-question live demo stopped on HTTP 403 from Google through AI Gateway before drafting. Google's upstream response described a project-level access denial. Restore access for the configured Google project or supply a key from an authorized working project in the local `.env`, then rerun the command. Creating a `workers.dev` domain does not resolve that denial.

Offline tests use controlled responses for routing, multilingual policy, unseen citations, one unsupported claim, the six-call budget, terminal handoff, SSE buffering, audit redaction, and MCP validation. They do not claim that the live classifier passes the golden set. Phase 3 will measure retrieval and answer quality once live model access works.
