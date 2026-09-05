# ADR 0003: Enforce a non-interpretive policy

- Status: Accepted
- Date: 2026-09-05

## Context

An institutional collection assistant can safely report documented facts and attributed curatorial text. Generating its own artistic meaning, quality ranking, or value judgment would blur the line between source material and machine opinion.

## Decision

Classify interpretive requests before tool planning and return a fixed, translated policy response. The assistant may offer documented artist, date, medium, provenance, and curatorial information. It must not generate its own interpretation, judge quality, or rank artworks.

## Consequences

Policy refusals are stable, testable, and recorded as guardrail events. Some mixed questions may require the user to restate the factual part. Curatorial interpretation can be quoted only when the retrieved source clearly attributes it.
