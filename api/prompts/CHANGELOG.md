# Runtime prompt history

## v1, September 4, 2026

Introduces bounded tool use, source-only factual answers, multilingual intent classification, non-interpretive policy, and atomic-claim grounding. Prompts are packaged with the API and their SHA-256 hashes appear in the audit trail. Offline fixtures exercise the safety contracts. Live evaluation deltas are not yet measured; the Phase 3 evaluation harness is outside this change.

## Evaluation judge v1, September 4, 2026

Added `evaluation_v1` for an independent, evidence-only faithfulness score and an explicit no-opinion judgment. It does not change the answer, routing, citation, or grounding prompts. Golden expected strings are withheld from the judge. Evaluation deltas are recorded in the dated reports rather than inferred from deterministic tests.

## Scope and exact quoting v2, September 4, 2026

The first measured quick run scored 5/10. `intent_v2` makes the workspace boundary explicit for external businesses and retail policies and selects the appropriate bounded handoff contact. `system_v2` requires contiguous quotes, preserving intervening fields, and explains how to cite confirmed HTTP 404 lookup evidence by API URL. Existing citation and atomic grounding validation remain unchanged. The original failed run and subsequent measured results are retained in `evals/reports`; score deltas are not claimed before rerunning.
