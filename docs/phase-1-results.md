# Phase 1 collection build

The local build completed on September 4, 2026. The published release is [AJAYKASU/met-collection-index at 6fdd37c042d069250b8ed5030946b84ff8a3e555](https://huggingface.co/datasets/AJAYKASU/met-collection-index/tree/6fdd37c042d069250b8ed5030946b84ff8a3e555).

## Index contents

| Property | Verified result |
| --- | --- |
| Collection | `met_objects` |
| Public-domain records | 20,000 |
| Text provider | `local`, FastEmbed 0.8.0 on CPU |
| Text model | `intfloat/multilingual-e5-large` |
| Text dimensions | 1,024 |
| Sparse model | `Qdrant/bm25`, IDF enabled |
| Image vectors | 19,964 of 20,000, or 99.82% |
| Missing published images | 36 objects retain text and sparse vectors only |
| Image model | `google/siglip2-so400m-patch14-384` |
| Image dimensions | 1,152 |
| Image source revision | `45fe67456ef1cffbe5ba801d5cd341fcf31cd806` |
| Collection schema / bundle format | 3 / 3 |
| Qdrant version | 1.19.0 |

Image vectors come exclusively from The Met's published [SigLIP 2 dataset](https://huggingface.co/datasets/metmuseum/openaccess-embeddings-siglip2). Its coverage made a fallback unnecessary. The [README data section](../README.md#data-and-attribution) records CC0 attribution, transformations, and this project's independence from The Met.

## Measured duration

The first worker started at `2026-09-04T06:18:31.807279Z` and the completed index was acknowledged at `2026-09-04T07:58:29.176832Z`: **1 hour 39 minutes 57 seconds elapsed**. This includes a brief stop and restart to group similarly sized texts. Checkpointed processing and verification accumulated **5,879.02 seconds**, or **1 hour 37 minutes 59 seconds** of active time across both attempts.

The host had 8 GB of memory and eight logical CPUs; local inference used four threads and batches of 32. Development checks ran concurrently for part of the build, so this is an operational measurement, not an isolated model benchmark. Source and model downloads preceded the timed worker and are excluded. No Gemini embedding or token-counting calls were made by the local run.

## Verification and preservation

The completed worker checked every Object ID and payload, every stored published image vector, and all 36 declared image absences. Export repeated these checks. The exact release snapshot was then restored into a temporary local Qdrant collection and verified without embedding calls. Public Hub verification checked the exact file allowlist, small-file contents, and large-file SHA-256 hashes. The test collection was removed afterward.

`verify_golden.py --collection met_objects` passed against the actual index and live Met API: all nine unique referenced object IDs exist, and all eight eligible IDs are indexed. `col-008` references non-public-domain Pollock object `488978`, intentionally outside the search index and documented for `get_object`. `col-009` resolves to Studiolo `198556`; `col-010` resolves to Ugolino and His Sons `204812`.

These checks establish identity, membership, provenance, and artifact integrity. They do not measure retrieval relevance, multilingual answer quality, or faithfulness. Those evaluations belong to later phases.

The stopped Gemini run remains in `met_objects_gemini_798` with all 798 vectors and payloads, using 768 dimensions. Its original checkpoint, log, and local snapshot are preserved under ignored `data/`. The original 200-object, 3,072-dimensional pilot remains in `met_objects_pilot_200`. Neither index contributes vectors to the local collection.

Phase 1 ends here. `find_similar_objects(object_id, k)` has a typed image-only tool schema; its runtime implementation and the remaining retrieval/agent tools belong to Phase 2.
