import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/server-security", () => ({
  originHeaders: vi.fn().mockResolvedValue({ "X-Origin-Auth": "unit-test-origin-token" }),
}));

import { GET } from "@/app/api/sessions/[sessionId]/events/route";

const validSessionId = "123e4567-e89b-42d3-a456-426614174000";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("session events proxy route", () => {
  it("rejects non-UUID session IDs with 400", async () => {
    const request = new Request("http://localhost/api/sessions/invalid-id/events");
    const response = await GET(request, { params: Promise.resolve({ sessionId: "invalid-id" }) });
    expect(response.status).toBe(400);
    const body = await response.json();
    expect(body).toEqual({ error: "Invalid session ID" });
  });

  it("forwards token exclusively via X-Session-Token header and omits it from query params", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_URL", "https://api.example.test");
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify([{ sequence: 1, kind: "user_turn" }]), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const request = new Request(`http://localhost/api/sessions/${validSessionId}/events?limit=50`, {
      headers: { "x-session-token": "secret-token" },
    });
    const response = await GET(request, { params: Promise.resolve({ sessionId: validSessionId }) });

    expect(response.status).toBe(200);
    expect(response.headers.get("Cache-Control")).toBe("no-store, no-cache");

    // Upstream URL must NOT contain token parameter
    const upstreamUrl = fetchMock.mock.calls[0]?.[0] as string;
    expect(upstreamUrl).not.toContain("token=");
    expect(upstreamUrl).toContain("limit=50");
    expect(upstreamUrl).toContain(`/sessions/${validSessionId}/events`);

    // Headers must include X-Session-Token and X-Origin-Auth
    const headers = fetchMock.mock.calls[0]?.[1]?.headers as Record<string, string>;
    expect(headers["X-Session-Token"]).toBe("secret-token");
    expect(headers["X-Origin-Auth"]).toBe("unit-test-origin-token");
  });

  it("ignores token in query string if not provided in header", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_URL", "https://api.example.test");
    const fetchMock = vi.fn().mockResolvedValue(new Response("Unauthorized", { status: 401 }));
    vi.stubGlobal("fetch", fetchMock);

    const request = new Request(`http://localhost/api/sessions/${validSessionId}/events?limit=50&token=secret-token`);
    const response = await GET(request, { params: Promise.resolve({ sessionId: validSessionId }) });

    expect(response.status).toBe(401);
    const upstreamUrl = fetchMock.mock.calls[0]?.[0] as string;
    expect(upstreamUrl).not.toContain("token=");

    const headers = fetchMock.mock.calls[0]?.[1]?.headers as Record<string, string>;
    expect(headers["X-Session-Token"]).toBeUndefined();
  });

  it("propagates 401 unauthorized and 404 not found cleanly", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_URL", "https://api.example.test");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("Unauthorized", { status: 401 })));

    const req401 = new Request(`http://localhost/api/sessions/${validSessionId}/events`);
    const res401 = await GET(req401, { params: Promise.resolve({ sessionId: validSessionId }) });
    expect(res401.status).toBe(401);

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("Not Found", { status: 404 })));
    const req404 = new Request(`http://localhost/api/sessions/${validSessionId}/events`);
    const res404 = await GET(req404, { params: Promise.resolve({ sessionId: validSessionId }) });
    expect(res404.status).toBe(404);
  });
});
