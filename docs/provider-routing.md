# Provider routing and minute budgets

Chat credentials are selected by the configured model prefix: `groq/`, `cerebras/`, or Gemini (including existing unprefixed Google identifiers). Main and lite keys are required. Each fallback is independently omitted when its provider key is missing. Startup logs list the active fallback aliases and model identifiers without credentials. Availability still requires a successful provider request.

With AI Gateway enabled, chat uses the native `groq`, `cerebras`, or `google-ai-studio/v1beta` path under the configured gateway root. Direct mode uses the provider endpoint. Local E5 embeddings remain the default.

Only rate limits and timeouts enter the fallback chain after bounded retries. Authentication failures, HTTP 403, missing models, malformed responses, and grounding failures never trigger provider fallback. Every retry reserves its own token allowance. Retry-After delays up to 60 seconds are respected; longer delays exhaust that route rather than retrying early.

The shared per-process rolling limiter reserves estimated prompt tokens (including tools, a tokenizer margin, and message overhead) plus `LLM_MAX_OUTPUT_TOKENS`. It reconciles successful calls with reported usage. Failed calls retain their reservation. Aliases using the same model share a budget. Oversized requests fail explicitly rather than waiting indefinitely. Other processes and daily quotas can still produce provider 429 responses. Model-facing tool messages contain exact citation evidence once, without duplicate full collection records or ranking scores; full tool responses remain available through MCP and session audits.

As checked on September 4, 2026, [Groq's published free limits](https://console.groq.com/docs/rate-limits) list both GPT-OSS models at 30 RPM, 8,000 TPM, 1,000 RPD, and 200,000 TPD. The authenticated model list included `openai/gpt-oss-120b` and `openai/gpt-oss-20b`, but omitted `llama-3.3-70b-versatile` and `llama-3.1-8b-instant`. A configured Llama fallback is therefore not verified as available. Account-specific limits take precedence.

[Cerebras's current catalog](https://inference-docs.cerebras.ai/models/overview) lists `gpt-oss-120b` and `qwen-3.8-27b`. Its [rate-limit documentation](https://inference-docs.cerebras.ai/support/rate-limits) describes a payment-method-required, 30-day trial rather than permanent free access. Cerebras is optional and was not activated for the live Phase 2 demo.

The built-in minute budgets cover the two Groq GPT-OSS models and Cerebras GPT-OSS 120B (30,000 uncached TPM and 5 RPM). Override them with `LLM_RATE_LIMITS`, a JSON object keyed by the exact configured model, with `tokens_per_minute` and `requests_per_minute` positive integers. A supplied object replaces the built-in map. Unlisted models use the conservative configurable defaults of 6,000 TPM and 5 RPM. `LLM_PACING_ENABLED=false` is intended for isolated tests, not quota bypass.

## Live Phase 2 demo

The September 4 demo used temporary process settings, leaving the local dotenv file unchanged:

```sh
LLM_MODEL=groq/openai/gpt-oss-120b \
LLM_MODEL_LITE=groq/openai/gpt-oss-20b \
LLM_MODEL_FALLBACK=groq/llama-3.3-70b-versatile \
CEREBRAS_API_KEY= make demo-check
```

Groq chat through AI Gateway succeeded. The full three-question demo did not pass: the collection and visitor questions returned HTTP 400 before a verified final answer. A diagnostic replay identified `tool_use_failed`: the model attempted an unregistered tool named `json`. The interpretive question correctly returned the policy refusal. No grounding or citation rules were relaxed, and no provider fallback was used. Unknown pricing is reported as unavailable rather than zero. Local reports and session events are excluded from Git.
