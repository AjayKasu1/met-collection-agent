# Measured latency and operating limits

Measured September 4, 2026 using the saved configuration without model overrides: Groq GPT-OSS 120B main, GPT-OSS 20B lite, Gemini fallback, local multilingual E5 embeddings and AI Gateway. All three demo answers passed citation and grounding checks. No retry or fallback occurred.

| Question | Total | Quota pacing | Provider | Tools and service | Query embedding | Vector search | Rerank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dendur gallery | 76.156 s | 58.209 s | 2.602 s | 15.342 s | 0.854 s | 0.104 s | 0.776 s |
| Fifth Avenue Wednesday hours | 7.168 s | 0.010 s | 2.375 s | 4.781 s | 2.780 s | 0.081 s | 1.353 s |
| Interpretive refusal | 0.147 s | 0.001 s | 0.135 s | 0.011 s | N/A | N/A | N/A |

Embedding, vector search and reranking are components of tools and service time, not additional durations. Dendur includes cold model initialization and a live object lookup. These three observations are a smoke measurement, not a latency percentile benchmark. Local session events retain provider and tool timings.

Retrieval takes 40 dense and 40 sparse candidates, fuses them, then reranks only the top 20. Both measured reranks meet the 1.5-second target, at that point. A later evaluation exposed slower batches, prompting the smaller-model comparison below. Model-facing tool results contain at most eight citable evidence records, omit null fields and image URLs, and exclude duplicate full records and ranking scores. Full tool responses remain available to API and MCP clients.

The 58-second pacing wait is an artifact of Groq's free-tier 8,000-token-per-minute budget for these models. A paid provider with sufficient allocated throughput removes this free-tier bottleneck when the application's per-model token budget is updated to that provider's verified limit. Merely changing billing does not change the application's configured limiter. Other account limits and concurrent clients can still cause throttling. See [Groq rate limits](https://console.groq.com/docs/rate-limits). The process-local limiter does not coordinate multiple API workers.

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
