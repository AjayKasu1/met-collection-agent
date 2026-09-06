import type { UIMessage } from "ai";
import { describe, expect, it } from "vitest";

import { latestUserText, parseApiError, parseChatRequest, toolTrace } from "@/lib/api";

const sessionId = "123e4567-e89b-42d3-a456-426614174000";

describe("chat request boundary", () => {
  it("uses the latest user text and validates the session and language", () => {
    const messages = [
      { id: "1", role: "user", parts: [{ type: "text", text: "First" }] },
      { id: "2", role: "assistant", parts: [{ type: "text", text: "Answer" }] },
      { id: "3", role: "user", parts: [{ type: "text", text: "  Latest question  " }] },
    ] as UIMessage[];

    expect(latestUserText(messages)).toBe("Latest question");
    expect(
      parseChatRequest({
        messages,
        session_id: sessionId,
        language: "fr",
        turnstile_token: "unit-test-token",
      }),
    ).toEqual({
      messages,
      session_id: sessionId,
      language: "fr",
      turnstile_token: "unit-test-token",
    });
  });

  it("rejects malformed sessions and empty questions", () => {
    expect(() =>
      parseChatRequest({
        messages: [],
        session_id: "not-a-uuid",
        language: "en",
        turnstile_token: "unit-test-token",
      }),
    ).toThrow("session");
    expect(() =>
      parseChatRequest({ messages: [], session_id: sessionId, language: "en" }),
    ).toThrow("security check");
    expect(() => latestUserText([])).toThrow("Enter a question");
  });

  it("maps provider errors to bounded public messages", () => {
    expect(parseApiError({ code: "provider_unavailable", message: "secret detail" }).message).toBe(
      "The model provider is temporarily unavailable. Try again shortly.",
    );
    expect(parseApiError({ code: "unknown", message: "secret detail" }).message).toBe(
      "The museum assistant is unavailable.",
    );
  });
});

describe("audit event projection", () => {
  it("retains tool order and pairs each timing once", () => {
    expect(
      toolTrace([
        { sequence: 1, kind: "tool_call", data: { name: "search_collection" } },
        { sequence: 2, kind: "tool_timing", data: { name: "search_collection", latency_ms: 24 } },
        { sequence: 3, kind: "tool_call", data: { name: "get_object" } },
        { sequence: 4, kind: "tool_timing", data: { name: "get_object", latency_ms: 81 } },
      ]),
    ).toEqual([
      { name: "search_collection", latency_ms: 24 },
      { name: "get_object", latency_ms: 81 },
    ]);
  });
});
