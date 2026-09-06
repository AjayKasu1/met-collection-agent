import type { UIMessage } from "ai";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/server-security", () => ({
  protectChat: vi.fn().mockResolvedValue({ "X-Origin-Auth": "unit-test-origin-token" }),
}));

import { POST } from "@/app/api/chat/route";
import { validAnswer } from "@/test/fixtures";

const sessionId = validAnswer.session_id;

function response(body: string, contentType = "text/event-stream"): Response {
  return new Response(body, { status: 200, headers: { "Content-Type": contentType } });
}

function request(messages: UIMessage[]): Request {
  return new Request("http://localhost/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages,
      session_id: sessionId,
      language: "en",
      turnstile_token: "unit-test-turnstile-token",
    }),
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("AI SDK chat adapter", () => {
  it("forwards only the latest question and attaches verified provenance", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_URL", "https://api.example.test");
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        response(
          `event: session\ndata: {"session_id":"${sessionId}"}\n\n` +
            "event: token\ndata: {\"text\":\"The Temple of Dendur is in Gallery 131.\"}\n\n" +
            `event: answer\ndata: ${JSON.stringify(validAnswer)}\n\n`,
        ),
      )
      .mockResolvedValueOnce(
        response(
          JSON.stringify([
            { sequence: 1, kind: "tool_call", data: { name: "get_object" } },
            { sequence: 2, kind: "tool_timing", data: { name: "get_object", latency_ms: 18 } },
          ]),
          "application/json",
        ),
      );
    vi.stubGlobal("fetch", fetchMock);

    const result = await POST(
      request([
        { id: "u1", role: "user", parts: [{ type: "text", text: "Earlier" }] },
        { id: "a1", role: "assistant", parts: [{ type: "text", text: "Earlier answer" }] },
        { id: "u2", role: "user", parts: [{ type: "text", text: "Where is Dendur?" }] },
      ] as UIMessage[]),
    );
    const stream = await result.text();

    const upstreamBody = JSON.parse(fetchMock.mock.calls[0]?.[1]?.body as string) as Record<string, unknown>;
    expect(upstreamBody).toEqual({
      message: "Where is Dendur?",
      session_id: sessionId,
      language: "en",
    });
    expect(fetchMock.mock.calls[0]?.[1]?.headers).toMatchObject({
      "X-Origin-Auth": "unit-test-origin-token",
    });
    expect(stream).toContain("Gallery 131");
    expect(stream).toContain("data-provenance");
    expect(stream).toContain("get_object");
    expect(stream).toContain("[DONE]");
  });

  it("fails closed when streamed text differs from the verified answer", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_URL", "https://api.example.test");
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        response(
          `event: session\ndata: {"session_id":"${sessionId}"}\n\n` +
            "event: token\ndata: {\"text\":\"Unverified text\"}\n\n" +
            `event: answer\ndata: ${JSON.stringify(validAnswer)}\n\n`,
        ),
      ),
    );

    const result = await POST(
      request([{ id: "u1", role: "user", parts: [{ type: "text", text: "Question" }] }] as UIMessage[]),
    );
    const stream = await result.text();
    expect(stream).toContain("The verified response changed while streaming.");
    expect(stream).not.toContain("data-provenance");
  });
});
