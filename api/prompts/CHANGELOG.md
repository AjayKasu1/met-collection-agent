# Runtime prompt history

## v1, September 4, 2026

Introduces bounded tool use, source-only factual answers, multilingual intent classification, non-interpretive policy, and atomic-claim grounding. Prompts are packaged with the API and their SHA-256 hashes appear in the audit trail. Offline fixtures exercise the safety contracts. Live evaluation deltas are not yet measured; the Phase 3 evaluation harness is outside this change.

## Evaluation judge v1, September 4, 2026

Added `evaluation_v1` for an independent, evidence-only faithfulness score and an explicit no-opinion judgment. It does not change the answer, routing, citation, or grounding prompts. Golden expected strings are withheld from the judge. Evaluation deltas are recorded in the dated reports rather than inferred from deterministic tests.
