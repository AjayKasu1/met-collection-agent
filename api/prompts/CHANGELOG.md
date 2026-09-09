# Runtime prompt history

## v1, September 4, 2026

Introduces bounded tool use, source-only factual answers, multilingual intent classification, non-interpretive policy, and atomic-claim grounding. Prompts are packaged with the API and their SHA-256 hashes appear in the audit trail. Offline fixtures exercise the safety contracts. Live evaluation deltas are not yet measured; the Phase 3 evaluation harness is outside this change.

## Evaluation judge v1, September 4, 2026

Added `evaluation_v1` for an independent, evidence-only faithfulness score and an explicit no-opinion judgment. It does not change the answer, routing, citation, or grounding prompts. Golden expected strings are withheld from the judge. Evaluation deltas are recorded in the dated reports rather than inferred from deterministic tests.

## Scope and exact quoting v2, September 4, 2026

The first measured quick run scored 5/10. `intent_v2` makes the workspace boundary explicit for external businesses and retail policies and selects the appropriate bounded handoff contact. `system_v2` requires contiguous quotes, preserving intervening fields, and explains how to cite confirmed HTTP 404 lookup evidence by API URL. Existing citation and atomic grounding validation remain unchanged. The original failed run and subsequent measured results are retained in `evals/reports`; score deltas are not claimed before rerunning.

## Verbatim excerpt selection v1, September 4, 2026

The second measured quick run scored 7/10 versus 5/10 initially. All out-of-scope handoffs passed, but quote reconstruction still failed for Van Gogh and the confirmed 404. That comparison includes reranker and evidence changes and does not isolate a prompt-only effect.

`citations_v1` changes the provider-facing final contract to select an enum of server-provided excerpt keys. Excerpts are contiguous substrings of this turn's tool evidence. The server resolves keys to the original source and quote before the unchanged public citation and atomic grounding checks. Unknown keys are rejected. Answer prose normalizes nonbreaking presentation spaces; quotes retain their exact source characters. A live GPT-OSS 20B canary accepted the native schema and returned a valid Van Gogh citation in 0.95 seconds. End-to-end score deltas require the subsequent measured run.

## Recovery controls, September 4, 2026

Runtime behavior changed without changing prompt text. Fallback providers now require explicit opt-in. A complete interactive chat has a configurable deadline, and model cooldown or access quarantine prevents repeated calls to a dependency that is expected to fail. Retry and failure events record only allowlisted diagnostics.

Evidence is now deduplicated and bounded across the complete turn. The final-answer contract, citation validator and grounding check remain unchanged. The generator, grounding check and evaluation judge receive the same retained evidence. The interrupted full run remains a diagnostic artifact; a new measured run is required before promoting a baseline.

## Wayfinding default v4, September 8, 2026

`intent_v4` recognizes a simple route request to one explicit numbered gallery when the entrance is omitted. The classifier preserves the origin as unspecified. The server then discloses its Fifth Avenue assumption and uses the same live-map handler, citation identity, and atomic grounding check as an explicitly stated Fifth Avenue request. Other origins, accessibility routes, multiple stops, constraints, and ambiguous numbers remain on the general path.
