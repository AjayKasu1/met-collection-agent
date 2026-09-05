# Golden evaluations

Run `make evals-quick` for the ten `verify: false` rows or `make evals` for all 32. Both use the saved settings without provider overrides. The agent and independent lite judge share one per-model pacing instance; evaluations execute sequentially. The full run can take many minutes on free-tier quotas.

The latest complete reviewed report is `2026-09-05T031955-21bf3af5-quick` at commit `21bf3af5`. It is the checked-in `baseline.json` used by the pull-request gate.

| Metric | Measured result |
| --- | ---: |
| Completed | 10/10 |
| Overall pass | 90% |
| Collection lookup | 100% |
| Refusal and handoff | 100% |
| Hallucination traps | 50% |
| Mean independent faithfulness | 1.00 across 10 judgments |
| Mean agent latency | 26.348 s |
| Nearest-rank p95 | 98.623 s |
| Mean answer-path cost | $0.0002823075 |
| Independent judge cost | $0.0008523750 total |

The later `2026-09-05T032640-21bf3af5-full` report contains five upstream HTTP 500 failures and is incomplete. It remains an incident record and is not promoted as an evaluation result.

The default command preserves the interactive chat deadline. For a batch quality measurement that permits longer quota waits, use `make evals-quick ARGS="--chat-deadline-seconds 300"` or the equivalent full command. This explicit override is recorded alongside the saved interactive deadline, provider path and pacing configuration. It does not change `.env`, prompt content, providers or verification rules, and it does not demonstrate that the same questions meet the interactive deadline. Resume rejects a changed execution configuration; baseline comparison rejects differing chat deadlines. The complete 30-second run at `155fdfc` scored 5/10, with five deadline expirations. Preserve it as the interactive result rather than replacing it with a batch score.

Every row starts a fresh production Runtime chat session. `contains` requires every expected substring, case-insensitively. Retrieval passes on Hit@5 and also reports Hit@1/10. Refusal requires the interpretive guardrail event, the policy-refusal flag and an independent no-opinion judgment. Handoff requires a successful handoff and its tool-call event. A failed grounding check fails the row. Faithfulness is additionally judged from model-visible evidence, without expected answers or world knowledge. It is reported separately and is not a replacement for the production grounding check.

Retrieval metrics use an explicit `search_collection` probe with the original golden question and `k=10`, after the agent answer. This evaluates search relevance independently of the agent's choice of tools or query rewrite. The probe does not feed the answer and its time is reported separately. Chat context remains capped at eight records. Seed labels are non-exhaustive: an unlisted relevant result can count as a miss. The two previously unlabeled rows now use independently selected source CSV records, not search ranks. The Met API returned 403 during the label audit; see each row's notes. The index was selected partly to include golden IDs, so this is a regression suite, not a held-out benchmark.

Reports are written after each completed row to `evals/reports/<date-time>-<git-sha>-<suite>.json` and `.md`. They include answers, citations, session IDs, failure reasons, agent latency and model-call timings, independent judge calls/cost/time, retrieval probe time and run identity. Session events remain in the ignored data directory. Unknown costs and absent judgments remain null, with sample counts alongside averages. p95 uses the nearest-rank definition. Overall pass rate uses the full expected suite denominator, including failures and incomplete rows.

Resume with `make evals-quick ARGS="--resume evals/reports/<run>.json"`. Completed rows, including failures, are retained. Resume rejects changes to the commit, golden file, prompt hashes, models or selected IDs. A new run is required after changing code or labels. Reports reference their execution commit; the later report-only commit naturally has a different hash.

For regression comparison, use `make evals-quick ARGS="--baseline evals/reports/baseline.json"`. Complete, matching suites and golden hashes are required. A drop greater than five percentage points fails; exactly five is allowed. With ten rows, one additional failure is a ten-point drop. Normal evaluation commands return nonzero on execution errors; scored factual misses are reported without aborting the remaining questions. Baselines must come from measured runs and be reviewed explicitly.

## Reviewed Phase 3 baseline

The reviewed quick baseline is the complete run `2026-09-05T031955-21bf3af5-quick`. It used the explicit 300-second batch deadline while retaining the saved 30-second interactive deadline. Fallback was disabled. It passed 9 of 10 cases, with 100% for collection lookup, refusal, and out-of-scope handoff, plus one of two hallucination traps. All ten independent evidence judgments reported faithfulness 1.00. Mean end-to-end latency was 26.348 seconds and nearest-rank p95 was 98.623 seconds. Mean answer-path model cost was $0.0002823075; the ten independent judge calls cost $0.0008523750 in total at the checked-in standard rates.

`hal-001` asks where the Mona Lisa is displayed at the Met. Search returned unrelated collection records, so the production grounding policy rejected the model's unsupported absence claim and returned the configured verification-unavailable response. It therefore failed the strict expected strings `Louvre` and `not`. This is the desired fail-closed runtime behavior, but it is an evaluation miss because the workspace lacks an authoritative external-location source. The redacted 28-event session audit is retained locally at `data/eval-hal-001-final-events.json`.

`hal-002` passed through the confirmed-404 path. The Met API returned HTTP 404 for Object ID 999999999, and the answer cited that lookup status. The runtime used two lite calls for intent and tool selection, then created the localized not-found result without another answer-generation or grounding-model call. The evaluation's separate judge call still ran and confirmed faithfulness.

An attempted 32-row run immediately after the baseline received HTTP 500 `InternalServerError` from the configured Cloudflare AI Gateway on the first five rows. No HTTP 429 or quota headers were recorded, so this must not be described as a Groq rate-limit failure. It was stopped rather than spending the remaining free-tier budget on an unavailable upstream. Its five-row incomplete report is retained as an incident artifact and is not a quality baseline. The earlier 17-row incomplete run remains diagnostic for the same reason. A complete full-suite score requires a healthy provider window and a fresh run.

When Langfuse is configured, the selected versioned golden set is uploaded as a Dataset and measured outputs are imported as an Experiment with pass and faithfulness scores. This does not repeat inference. Experiment metadata explicitly identifies a recorded-results import; use output `latency_ms` instead of import-span duration for performance comparisons. Local reports are saved before publication so an upload failure does not discard a run. The integration follows the [Langfuse experiment SDK](https://langfuse.com/docs/evaluation/experiments/experiments-via-sdk).

## Pull request gate setup

The live regression job runs for pull requests to main and uses the `evaluations` GitHub environment. Configure required reviewers on that environment before making its secrets available to untrusted pull requests. It needs `GROQ_API_KEY`, `QDRANT_URL`, and a **read-only** `QDRANT_API_KEY` for the prepared collection and visitor index. Gemini credentials are optional; an unavailable fallback key is skipped by the router. The job uses direct Groq access, the same main/lite model IDs and local embedding identity as the measured baseline. It does not change developer `.env` files or copy their secrets.

The comparison reads the baseline from the pull request target commit, so a pull request cannot weaken its own reference score. Golden-label changes require an explicit baseline review.

Missing credentials fail the live job with the missing setting name; they do not silently skip it or substitute mocked scores. Fork pull requests do not receive repository secrets automatically. Review them before running a credentialed evaluation in a trusted context. Do not use `pull_request_target` to execute unreviewed pull request code. Make the live job a required check only after configuring the protected environment and baseline. Offline lint, typing and tests remain a separate job on pushes and pull requests.

## Interpreting full-suite misses

The full suite retains manually reviewed (`verify: true`) labels as written, including strict wording and known ambiguities. `col-004` rejects the supported paraphrase `oil painting on wood` because it requires `Oil on wood`. `col-009` says in its notes that either maker is acceptable, but its expected field contains only Giuliano da Maiano. `col-010` asks for culture while also requiring the artist name Carpeaux. These are label-contract misses, not evidence that every answer is factually wrong. No expected strings were relaxed after observing the answers.

The Pollock row exposes a separate functional gap: the 20k public-domain index excludes Autumn Rhythm, and the current tools require an already-known ID for a live lookup. Repeated subset searches cannot establish that ID. A bounded authoritative title resolver and repeated-search detection are candidates for a separately measured improvement; an invented ID is never an acceptable substitute.
