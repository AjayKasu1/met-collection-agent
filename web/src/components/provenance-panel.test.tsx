import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ProvenancePanel } from "@/components/provenance-panel";
import { validAnswer } from "@/test/fixtures";

describe("ProvenancePanel", () => {
  it("renders answer metrics and tool order", () => {
    render(
      <ProvenancePanel
        provenance={{
          answer: validAnswer,
          tools: [
            { name: "search_collection", latency_ms: 120 },
            { name: "get_object", latency_ms: 80 },
          ],
        }}
      />,
    );

    expect(screen.getByText("How this was answered")).toBeInTheDocument();
    expect(screen.getByText("100%")).toBeInTheDocument();
    expect(screen.getByText("$0.000310")).toBeInTheDocument();
    expect(screen.getByText("Search Collection")).toBeInTheDocument();
    expect(screen.getByText("Get Object")).toBeInTheDocument();
  });

  it("hides grounding percentage and sources for social answers", () => {
    render(
      <ProvenancePanel
        provenance={{
          answer: {
            ...validAnswer,
            answer_kind: "social",
            verification_status: "not_applicable",
            grounding_score: null,
            cost_usd: 0,
          },
          tools: [],
        }}
      />,
    );

    expect(screen.getByText("Social")).toBeInTheDocument();
    expect(screen.queryByText("Grounding")).not.toBeInTheDocument();
    expect(screen.queryByText("100%")).not.toBeInTheDocument();
    expect(screen.queryByText("Evidence path")).not.toBeInTheDocument();
  });
});
