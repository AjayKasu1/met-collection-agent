# Provider routing and minute budgets

Chat credentials are selected by the configured model prefix: `groq/`, `cerebras/`, or Gemini (including existing unprefixed Google identifiers). Main and lite keys are required. Fallbacks are disabled by default: stored identifiers and credentials are inactive unless `LLM_FALLBACK_ENABLED=true` is explicitly configured. When enabled, each fallback is independently omitted if its provider key is missing. Startup logs list active fallback aliases and model identifiers without credentials. Availability still requires a successful provider request.

With AI Gateway enabled, chat uses the native `groq`, `cerebras`, or `google-ai-studio/v1beta` path under the configured gateway root. Direct mode uses the provider endpoint. Local E5 embeddings remain the default.

Only rate limits and timeouts enter the fallback chain after bounded retries. Authentication failures, HTTP 403, missing models, malformed responses, and grounding failures never trigger provider fallback. Every retry reserves its own token allowance. Retry-After delays up to 60 seconds are respected; longer delays exhaust that route rather than retrying early.

`CHAT_DEADLINE_SECONDS` defaults to 30 for Runtime chat, including quota waits, tool execution, generation and grounding. Expiry cancels the awaiting turn and returns `verification_unavailable`; no unchecked answer is streamed. Cancellation does not forcibly terminate a local inference thread that has already started. HTTP tools retain their own timeouts. The evaluation runner uses the same deadline and records expiry as an execution error. A higher deadline is an explicit configuration choice, not a hidden benchmark override.

Rate limits, timeouts, HTTP 408, and HTTP 5xx responses receive the same bounded retry treatment. Retry exhaustion puts that model on cooldown for at least 30 seconds, or the numeric Retry-After interval if longer. HTTP 401/403 quarantines that model for the lifetime of the runtime; correct the configuration and restart before using it again. Client errors such as HTTP 400 and 404 are not retried. These states are shared by aliases using the same model within one process. They are not a distributed circuit breaker. Retry jitter spreads attempts without shortening a provider's Retry-After. `provider_attempt_failed` records the original exception class, status, attempt and allowlisted quota values before any fallback. Provider response bodies, arbitrary headers and credentials are excluded. Operator notification is not implemented; the event log does not claim to have contacted a human.

Evidence is deduplicated across the whole turn and limited to eight records, 6,000 source-text characters total and 1,400 characters per record. Latest copies take precedence; earlier duplicate tool evidence is removed from model messages. Excerpts remain literal source substrings. The generator, citation validator, grounding check and independent evaluation judge receive the same retained evidence, also recorded as `evidence_context`. Full source results remain in tool audits. Truncation can omit relevant facts, so retrieval quality still needs measured evaluation. Citation and grounding acceptance rules are unchanged.

A confirmed Met API 404 is a typed terminal result. The server returns a localized statement that the requested object was not found and no object record was returned, citing the exact lookup-status evidence. It validates that citation with the same identity and verbatim-quote function before release and records a deterministic guardrail event. This path needs the intent and tool-selection calls, but skips model answer generation and model grounding. Other errors cannot enter this path.

The shared per-process rolling limiter reserves estimated prompt tokens (including tools, a tokenizer margin, and message overhead) plus `LLM_MAX_OUTPUT_TOKENS`. It reconciles successful calls with reported usage. Failed calls retain their reservation. Aliases using the same model share a budget. Oversized requests fail explicitly rather than waiting indefinitely. Other processes and daily quotas can still produce provider 429 responses. Model-facing tool messages contain exact citation evidence once, without duplicate full collection records or ranking scores; full tool responses remain available through MCP and session audits.

As checked on September 4, 2026, [Groq's published free limits](https://console.groq.com/docs/rate-limits) list both GPT-OSS models at 30 RPM, 8,000 TPM, 1,000 RPD, and 200,000 TPD. The authenticated model list included `openai/gpt-oss-120b` and `openai/gpt-oss-20b`, but omitted `llama-3.3-70b-versatile` and `llama-3.1-8b-instant`. A configured Llama fallback is therefore not verified as available. Account-specific limits take precedence.

[Cerebras's current catalog](https://inference-docs.cerebras.ai/models/overview) lists `gpt-oss-120b` and `qwen-3.8-27b`. Its [rate-limit documentation](https://inference-docs.cerebras.ai/support/rate-limits) describes a payment-method-required, 30-day trial rather than permanent free access. Cerebras is optional and was not activated for the live Phase 2 demo.

The built-in minute budgets cover the two Groq GPT-OSS models and Cerebras GPT-OSS 120B (30,000 uncached TPM and 5 RPM). Override them with `LLM_RATE_LIMITS`, a JSON object keyed by the exact configured model, with `tokens_per_minute` and `requests_per_minute` positive integers. A supplied object replaces the built-in map. Unlisted models use the conservative configurable defaults of 6,000 TPM and 5 RPM. `LLM_PACING_ENABLED=false` is intended for isolated tests, not quota bypass.

## Initial live Phase 2 demo

The September 4 demo used temporary process settings, leaving the local dotenv file unchanged:

```sh
LLM_MODEL=groq/openai/gpt-oss-120b \
LLM_MODEL_LITE=groq/openai/gpt-oss-20b \
LLM_MODEL_FALLBACK=groq/llama-3.3-70b-versatile \
CEREBRAS_API_KEY= make demo-check
```

Groq chat through AI Gateway succeeded. The full three-question demo did not pass: the collection and visitor questions returned HTTP 400 before a verified final answer. A diagnostic replay identified `tool_use_failed`: the model attempted an unregistered tool named `json`. The interpretive question correctly returned the policy refusal. No grounding or citation rules were relaxed, and no provider fallback was used. Unknown pricing is reported as unavailable rather than zero. Local reports and session events are excluded from Git.

## Native final answers, costs, and timing

Groq's [structured-output contract](https://console.groq.com/docs/structured-outputs) supports strict JSON Schema on GPT-OSS 120B and 20B, but excludes simultaneous tool use. Tool selection therefore uses a separate prompt that does not ask for a final JSON answer. Once selection ends, a no-tool call composes the final answer with `response_format.type=json_schema` and `strict=true`. Pydantic schemas are closed recursively and all properties are required. For final calls with tool evidence, citation values are constrained to an enum of server-provided excerpt keys. Each key resolves to an unchanged, contiguous source excerpt and its known Object ID or URL. This prevents the provider from joining separate fields or escaping newlines incorrectly while constructing a quote. Unknown keys are rejected. The adapter produces the unchanged public citation type before verification; each excerpt is bounded to 1,200 characters. Legacy final calls without tool context retain the disjoint nested source schema. Server-side citation identity, verbatim quotes, language, and atomic grounding checks remain unchanged. The same native format is used for intent and grounding on supported Groq models.

The checked-in price table uses [Groq's standard inference prices](https://console.groq.com/docs/models), verified September 4, 2026: GPT-OSS 120B costs $0.15 input and $0.60 output per million tokens; GPT-OSS 20B costs $0.075 input and $0.30 output per million tokens. Each query sums reported input and completion usage across every model call, including guardrails. These are standard-price estimates, not a claim of charges on a free account. Unknown model prices remain unavailable. No unverified cached-token discount is assumed.

Each model-call record separates `pacing_ms`, `provider_ms`, and `retry_ms`. Tool durations are separate `tool_timing` events. Demo output reports those components plus remaining tool and service overhead. Cold local model loading and free-tier minute-budget waits are included in end-to-end latency; neither is hidden as provider inference time.

## Verified rerun

After fixing the provider wire schema and preserving saved hours tables, all three demo questions passed without relaxing citation or grounding checks. The main model remains GPT-OSS 120B; no model substitution was required. The earlier Dendur demo used the lite route and cited Object 547802; the latest demo selected main. The visitor answer used main and cited the captured Plan Your Visit table, and the interpretive request returned the configured refusal. See the [README measurement](../README.md#phase-2-demo-measurement) for actual prices and latency. Earlier failed event logs remain local and are not published.
