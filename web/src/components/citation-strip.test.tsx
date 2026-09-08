import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { CitationStrip } from "@/components/citation-strip";

describe("CitationStrip", () => {
  it("labels a live wayfinding citation without exposing map feature identifiers", () => {
    render(
      <CitationStrip
        citations={[
          {
            object_id: null,
            source_url:
              "https://maps.metmuseum.org/navigate/a29e5baa0c56824cc94c18c53ad5ec05/a5d0882ad40744d55990d58da2d511c4?floor=1&lang=en-GB",
            quote: "The Met Interactive Map route",
          },
        ]}
      />,
    );

    expect(screen.getByRole("heading", { name: "The Met Interactive Map" })).toBeInTheDocument();
    expect(screen.queryByText(/a5d0882/i)).not.toBeInTheDocument();
  });
});
