# ADR 0005: Constrain tool execution and evidence context

- Status: Accepted
- Date: 2026-09-05

## Context

Unbounded agents can repeat searches, accumulate excessive context, increase provider spend, and make failure diagnosis difficult. Tool arguments and results also cross trust boundaries.

## Decision

Register six tools with strict Pydantic input and output schemas. Reject extra or oversized arguments, execute calls sequentially, cap each turn at six tool calls, deduplicate evidence, and cap model-visible context at eight records and 6,000 source-text characters. Treat `handoff` as terminal. Preserve full validated tool output in the API and audit record while sending only citable fields to the model.

## Consequences

Each turn has bounded work, cost, and model context. Sequential execution simplifies event ordering and avoids concurrent pressure on local models. Complex research questions may fail closed or need another user turn. Expanding limits requires a measured evaluation rather than an ad hoc prompt change.
