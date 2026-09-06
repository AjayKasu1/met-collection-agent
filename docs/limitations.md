# Limitations and production controls

## Provider data policy

The active default route is Groq and Gemini fallback is disabled. If Gemini is enabled without an associated billing account, Google's current [Gemini API terms](https://ai.google.dev/gemini-api/terms) classify it as an unpaid service: submitted content and generated responses may be used to improve products, and human reviewers may process them. Do not send personal, confidential, or sensitive information through that route. Google's [pricing table](https://ai.google.dev/gemini-api/docs/pricing) marks free-tier content as used for product improvement and paid-tier content as excluded from that use. Recheck the current terms before enabling Gemini because provider policies can change.

Session audits and optional Langfuse traces also contain conversation content and retrieved excerpts. They need documented retention, deletion, access, and incident-response controls. Secret redaction reduces credential exposure but does not anonymize user questions.

## Source freshness

Gallery numbers and `on_view` values in `search_collection` reflect ingestion time. Displays move. A gallery claim is current only when the same turn calls `get_object`, which reads The Met's live object endpoint and caches the result for at most five minutes. If that endpoint is unavailable, the system must state that it cannot verify the current gallery.

Visitor answers come from saved public pages and include canonical URL plus capture time. Hours, admission, exhibitions, access guidance, maps, and policies can change after capture. The weekly workflow crawls the curated source list at a minimum three-second interval, builds a versioned collection, checks its schema, point count, and source coverage, then atomically moves the production alias. Any failure retains the previous release and opens a failed workflow run for review. The source site can still publish late or present dynamic content that requires manual inspection.

## Public demo boundary

The production profile has Turnstile bot verification, a Cloudflare rate-limit binding set to five chat submissions per source IP per minute, strict origin allowlisting, and a shared Worker-to-API credential. The rate-limit binding is fast and distributed by Cloudflare location, but intentionally eventually consistent. Shared networks can put many visitors behind one IP, while determined attackers can distribute traffic across many IPs. The public demo has no user identity, tenant isolation, or identity-based quota.

Production audit events use managed PostgreSQL, while SQLite remains a local fallback. Model cooldown and rate pacing remain process-local, so multiple Cloud Run instances do not coordinate provider budgets or circuit state. A larger institutional deployment should add identity-aware authentication, per-user quotas, distributed provider budgets, a staff review queue for unresolved or high-impact questions, and contractual data controls. The current public demo can meet a bounded launch profile only after every control in [operations](operations.md) is activated and its smoke test passes.

## Measured latency

Initial L6 measurement, September 4, 2026, using the saved configuration without model overrides: Groq GPT-OSS 120B main, GPT-OSS 20B lite, Gemini fallback, local multilingual E5 embeddings and AI Gateway. All three demo answers passed citation and grounding checks. No retry or fallback occurred.

| Question | Total | Quota pacing | Provider | Tools and service | Query embedding | Vector search | Rerank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dendur gallery | 76.156 s | 58.209 s | 2.602 s | 15.342 s | 0.854 s | 0.104 s | 0.776 s |
| Fifth Avenue Wednesday hours | 7.168 s | 0.010 s | 2.375 s | 4.781 s | 2.780 s | 0.081 s | 1.353 s |
| Interpretive refusal | 0.147 s | 0.001 s | 0.135 s | 0.011 s | N/A | N/A | N/A |

Embedding, vector search and reranking are components of tools and service time, not additional durations. Dendur includes cold model initialization and a live object lookup. These three observations are a smoke measurement, not a latency percentile benchmark. Local session events retain provider and tool timings.

Retrieval takes 40 dense and 40 sparse candidates, fuses them, then reranks only the top 20. Both reranks in that initial demo met the 1.5-second target. A later evaluation exposed slower batches, prompting the smaller-model comparison below. Model-facing tool results contain at most eight citable evidence records, omit null fields and image URLs, and exclude duplicate full records and ranking scores. Full tool responses remain available to API and MCP clients.

The 58-second pacing wait was an artifact of Groq's free-tier 8,000-token-per-minute budget for these models. Interactive pacing is now disabled, so ordinary questions return quickly and bursts can receive a provider rate-limit error. Paid capacity removes the free-tier bottleneck only after account limits are verified and `LLM_RATE_LIMITS` is set to the purchased allocation. Other account limits and concurrent clients can still cause throttling. See [Groq rate limits](https://console.groq.com/docs/rate-limits). Evaluation pacing remains enabled for repeatable batch runs.

Visitor answers reflect saved pages and their capture timestamps, not live opening-hour guarantees. The index is a selected public-domain subset; live object lookup can cover objects outside it. The English cross-encoder requires English query rewrites for multilingual questions. Retrieval and answer quality require the separate golden evaluation, and demo success alone does not establish either.

## Smaller reranker follow-up

The final configuration uses the pinned two-layer `Xenova/ms-marco-MiniLM-L-2-v2` at revision `b84c4fa7efd7b4801931e75773c940f002a494f5`, with batches of two to reduce peak memory. It is explicitly registered through FastEmbed's custom-model API; the text embedding model and index identity are unchanged. The [upstream model](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L2-v2) is Apache-2.0 and its [ONNX conversion](https://huggingface.co/Xenova/ms-marco-MiniLM-L-2-v2) is pinned rather than downloaded from a moving branch.

Three repeated reranks of identical top-20 candidate batches produced these local measurements:

| Query | Median rerank | Maximum rerank |
| --- | ---: | ---: |
| Dendur | 0.116 s | 0.124 s |
| Wednesday hours | 0.327 s | 1.562 s |
| German armor | 0.488 s | 0.524 s |
| Dog paintings | 0.531 s | 1.726 s |

All medians meet 1.5 seconds, but host memory pressure still produces occasional outliers. This is not a hard latency SLO. Both compared models retained labeled Hit@5 for Dendur and dog paintings; both missed the non-exhaustive armor seeds. The smaller model kept Plan Your Visit as the first visitor source. These four queries are a bounded model comparison, not a held-out relevance benchmark.

The first Phase 3 quick run overlapped briefly with a separate model benchmark and its latency is contaminated by local resource contention. That benchmark was stopped; the final measurements above used a single retrieval service without a simultaneous live evaluation.

## End-to-end demo before excerpt selection

The subsequent three-question demo at commit `a1a8e2f` passed with exact citations and grounding score 1 for both factual answers. Settings were loaded without overrides: main GPT-OSS 120B, lite GPT-OSS 20B, Gemini fallback. All calls used AI Gateway; no retries or fallback occurred.

| Question | Total | Pacing | Provider | Tools/service | Rerank |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dendur | 63.232 s | 37.508 s | 5.698 s | 20.019 s | 2.278 s |
| Wednesday hours | 62.789 s | 55.208 s | 4.507 s | 3.069 s | 0.536 s |
| Interpretive refusal | 0.454 s | 0.001 s | 0.435 s | 0.018 s | N/A |

Both factual questions selected the main route in this run, so they shared its per-minute budget. The refusal only called the lite classifier, although its selected workspace-route label is main. Dendur includes cold model initialization and showed another memory-sensitive rerank outlier. The warm visitor query took approximately 7.6 seconds excluding pacing. Paying for sufficient throughput and updating the configured limiter addresses pacing; it does not remove local CPU or memory overhead.

## Final demo with native excerpt selection

At commit `1e61742`, all three questions passed again. Final citations now select server-provided verbatim excerpts, preventing model-generated quote formatting errors. The same saved provider settings and verification rules applied; no retries or fallback occurred.

| Question | Total | Pacing | Provider | Tools/service | Estimated USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dendur | 15.721 s | 0.013 s | 2.996 s | 12.710 s | 0.001331925 |
| Wednesday hours | 105.909 s | 100.136 s | 2.432 s | 3.337 s | 0.001464075 |
| Interpretive refusal | 0.204 s | 0.000 s | 0.187 s | 0.016 s | 0.000080175 |

The visitor total includes more than one pacing wait because its sequential calls share the same model budget with the preceding question. This is an actual end-to-end observation, not a claim of constant query latency. Removing pacing alone would leave approximately 5.8 seconds for that warm visitor question and 15.7 seconds for cold Dendur initialization plus answering.

## Bounded recovery follow-up

The interrupted full evaluation reached 17 rows before provider and context failures made continued calls unproductive. The checkpoint is retained as an incomplete diagnostic run, not promoted as a baseline. Four recorded fallback events originated from the lite grounding route after successful main-model work. The previous fallback audit omitted the original rate-limit or timeout classification, so it cannot establish which condition occurred. Gemini then returned HTTP 403. A 403 establishes denied access; payment is not established as the cause.

Fallbacks are now disabled by default even when identifiers and credentials remain stored. Explicit opt-in, safe original-error telemetry, per-model cooldown and access quarantine prevent recurring calls to a known failing dependency within one runtime. The 120-second chat deadline covers the complete verified-answer path. It returns a temporary failure instead of an unchecked answer. Provider-specific notification and distributed circuit-breaker state are not implemented.

The same run showed that limiting each tool result independently did not limit accumulated turn context. Current packing deduplicates evidence keys and caps the complete turn at eight records and 6,000 source-text characters. The answer generator, verifier and evaluation judge see the same retained set. This can omit a relevant fact and therefore requires measured evaluation before a new baseline is accepted.

In that final demo, Dendur query embedding, vector search and reranking took 0.596, 0.178 and 0.121 seconds respectively. The visitor query took 2.259, 0.111 and 0.717 seconds. These components are already included in tools/service time.

## Phase 3 baseline measurement

The final ten-case batch run at commit `21bf3af` passed 9 of 10 rows with fallback disabled. The explicit evaluation deadline was 300 seconds; the saved interactive deadline remained 30 seconds. Mean end-to-end latency was 26.348 seconds and p95 was 98.623 seconds. Mean answer-path model cost was $0.0002823075 across ten priced rows, and the independent judge cost $0.0008523750 in total. All ten judge results had faithfulness 1.00.

The two factual collection rows each took about 60.9 seconds. The confirmed missing-ID trap took 34.1 seconds, almost entirely from lite-model classification, tool selection, and evaluation-judge pacing; its production answer generation is deterministic after the cited HTTP 404. The failed Mona Lisa trap took 98.6 seconds across seven lite calls, including 89.9 seconds of pacing and about 5.0 seconds of provider work. Its production grounding guard rejected the unsupported claim that absence from retrieved results proves absence from the collection. The final fail-closed response had no unchecked fact or citation.

Immediately following that run, the configured Cloudflare AI Gateway returned HTTP 500 on the first five rows of a full-suite attempt. No HTTP 429 or quota headers were recorded. Those failures are an upstream availability incident, not evidence of a Groq quota problem and not a retrieval or answer-quality measurement. The run was stopped and retained as incomplete. The application recorded the sanitized exception type, status, model, and route without logging provider response bodies or credentials.
