import { describe, expect, it } from "vitest";

import { parseAgentAnswer, parseLiveObject, safeHttpUrl } from "@/lib/validation";
import { validAnswer } from "@/test/fixtures";

describe("public response validation", () => {
  it("accepts a complete verified answer", () => {
    expect(parseAgentAnswer(validAnswer)).toEqual(validAnswer);
  });

  it("rejects ambiguous and unsafe citations", () => {
    expect(() =>
      parseAgentAnswer({
        ...validAnswer,
        citations: [{ object_id: 547802, source_url: "https://example.com", quote: "x" }],
      }),
    ).toThrow("exactly one source");
    expect(() =>
      parseAgentAnswer({
        ...validAnswer,
        citations: [{ object_id: null, source_url: "javascript:alert(1)", quote: "x" }],
      }),
    ).toThrow("source URL");
  });

  it("rejects invalid grounding and object identities", () => {
    expect(() => parseAgentAnswer({ ...validAnswer, grounding_score: 1.1 })).toThrow(
      "verification state",
    );
    expect(() =>
      parseLiveObject({
        object_id: 0,
        title: "Object",
        artist: "",
        object_date: "",
        gallery_number: "",
        is_on_view: false,
        primary_image_small: "",
        source_url: "https://www.metmuseum.org/object",
      }),
    ).toThrow("identity");
  });

  it("allows only HTTP protocols", () => {
    expect(safeHttpUrl("https://www.metmuseum.org/visit")).toBe(
      "https://www.metmuseum.org/visit",
    );
    expect(safeHttpUrl("data:text/html,unsafe")).toBeNull();
  });
});
