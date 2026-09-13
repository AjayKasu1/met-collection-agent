import type { AnswerProvenance } from "@/lib/types";

function label(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function cost(value: number | null): string {
  if (value === null) return "Unavailable";
  if (value === 0) return "$0.00";
  if (value < 0.000001) return "<$0.000001";
  return `$${value.toFixed(6)}`;
}

export function ProvenancePanel({ provenance }: { provenance: AnswerProvenance }): React.ReactNode {
  const { answer, tools } = provenance;
  const isSocial = answer.answer_kind === "social";
  const displayKind = answer.answer_kind ? label(answer.answer_kind) : label(answer.route);

  if (isSocial) {
    return (
      <details className="provenance provenance-subtle">
        <summary>
          <span className="subtle-link">Details</span>
        </summary>
        <div className="provenance-grid">
          <div>
            <span className="metric-label">Type</span>
            <strong>Social</strong>
          </div>
          <div>
            <span className="metric-label">Verification</span>
            <strong>Not applicable</strong>
          </div>
          <div>
            <span className="metric-label">Latency</span>
            <strong>{(answer.latency_ms / 1000).toFixed(2)} s</strong>
          </div>
          <div>
            <span className="metric-label">Estimated cost</span>
            <strong>$0.00</strong>
          </div>
        </div>
      </details>
    );
  }

  return (
    <details className="provenance">
      <summary>
        <span>How this was answered</span>
        <span className="summary-stats">
          {displayKind} · {(answer.latency_ms / 1000).toFixed(1)} s
        </span>
      </summary>
      <div className="provenance-grid">
        <div>
          <span className="metric-label">Type</span>
          <strong>{displayKind}</strong>
        </div>
        <div>
          <span className="metric-label">{answer.grounding_score !== null ? "Grounding" : "Verification"}</span>
          <strong>
            {answer.grounding_score !== null
              ? `${Math.round(answer.grounding_score * 100)}%`
              : answer.verification_status
                ? label(answer.verification_status)
                : "Not applicable"}
          </strong>
        </div>
        <div>
          <span className="metric-label">Latency</span>
          <strong>{(answer.latency_ms / 1000).toFixed(2)} s</strong>
        </div>
        <div>
          <span className="metric-label">Estimated cost</span>
          <strong>{cost(answer.cost_usd)}</strong>
        </div>
      </div>
      <div className="tool-ledger">
        <p className="metric-label">Evidence path</p>
        {tools.length ? (
          <ol>
            {tools.map((tool, index) => (
              <li key={`${tool.name}-${index}`}>
                <span>{label(tool.name)}</span>
                <span>{tool.latency_ms === null ? "Recorded" : `${tool.latency_ms.toFixed(0)} ms`}</span>
              </li>
            ))}
          </ol>
        ) : (
          <p className="muted-copy">No collection tool was needed for this answer.</p>
        )}
      </div>
      {answer.model_calls.length > 0 && (
        <p className="model-note">
          {answer.model_calls.length} model {answer.model_calls.length === 1 ? "call" : "calls"} ·{" "}
          {answer.model_calls.map((call) => call.model).join(", ")}
        </p>
      )}
    </details>
  );
}
