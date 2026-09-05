# ADR 0001: Use hybrid retrieval with local reranking

- Status: Accepted
- Date: 2026-09-05

## Context

Collection questions mix names, accession numbers, multilingual descriptions, materials, dates, and broad concepts. Dense retrieval handles semantic paraphrases, while lexical retrieval remains stronger for identifiers and exact names. Either method alone creates avoidable misses.

## Decision

Store named `dense` vectors from `intfloat/multilingual-e5-large` at 1,024 dimensions and Qdrant's BM25 sparse vectors. Retrieve 40 candidates from each channel, fuse them, and rerank only the top 20 with the pinned two-layer `Xenova/ms-marco-MiniLM-L-2-v2` cross-encoder. Return at most eight records to the model. Keep The Met's published SigLIP 2 vectors in the separate named `image` space.

## Consequences

Exact identifiers and semantic queries both have a retrieval path. Query embedding and reranking run locally, so they add CPU, memory, model-download, and cold-start costs. The English cross-encoder requires an English query rewrite for multilingual input. Every embedding identity and dimension must match collection metadata before a query runs.
