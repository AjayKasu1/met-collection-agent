# ADR 0004: Verify factual output before streaming

- Status: Accepted
- Date: 2026-09-05

## Context

Prompt instructions alone cannot guarantee that a model will quote a source exactly, use a source seen in the current turn, or keep every generated claim supported. Streaming an unchecked draft makes later correction ineffective.

## Decision

Derive citation candidates only from typed tool evidence. Final generation uses provider-native JSON Schema and source keys that the server maps to canonical URLs and excerpts. Require every citation excerpt to match same-turn evidence, then run an atomic-claim grounding check against the identical evidence pack. Permit one generation repair without new tools. On another failure, return a safe verification-unavailable response. Stream text only after the final answer passes.

## Consequences

Users never receive a partially streamed factual draft that the server later rejects. Verification adds a model call and latency. The system can fail closed when its sources are incomplete, as the Mona Lisa trap demonstrates. Deterministic, cited 404 answers avoid an unnecessary generation call.
