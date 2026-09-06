import { beforeEach, describe, expect, it, vi } from "vitest";

const { limit } = vi.hoisted(() => ({ limit: vi.fn() }));

vi.mock("@opennextjs/cloudflare", () => ({
  getCloudflareContext: vi.fn().mockResolvedValue({
    env: { CHAT_RATE_LIMITER: { limit } },
  }),
}));

import { originHeaders, protectChat } from "@/lib/server-security";

function request(): Request {
  return new Request("https://museum.example/api/chat", {
    method: "POST",
    headers: { "cf-connecting-ip": "192.0.2.10", "cf-ray": "unit-test-ray" },
  });
}

beforeEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.stubEnv("ORIGIN_AUTH_TOKEN", "unit-test-origin-token");
  vi.stubEnv("TURNSTILE_SECRET", "unit-test-turnstile-secret");
  vi.stubEnv("TURNSTILE_HOSTNAMES", "museum.example");
  limit.mockReset();
  limit.mockResolvedValue({ success: true });
});

describe("Worker request protection", () => {
  it("rate limits, validates the action and hostname, then returns origin authentication", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      Response.json({ success: true, action: "chat", hostname: "museum.example" }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(protectChat(request(), "single-use-token")).resolves.toEqual({
      "X-Origin-Auth": "unit-test-origin-token",
    });
    expect(limit).toHaveBeenCalledWith({ key: "192.0.2.10" });
    const body = fetchMock.mock.calls[0]?.[1]?.body as URLSearchParams;
    expect(body.get("response")).toBe("single-use-token");
    expect(body.get("remoteip")).toBe("192.0.2.10");
  });

  it("rejects over-limit requests before calling Siteverify", async () => {
    limit.mockResolvedValue({ success: false });
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(protectChat(request(), "token")).rejects.toMatchObject({
      code: "rate_limited",
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects mismatched actions and fails closed on Siteverify errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        Response.json({ success: true, action: "signup", hostname: "museum.example" }),
      ),
    );
    await expect(protectChat(request(), "token")).rejects.toMatchObject({
      code: "verification_required",
    });

    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network")));
    await expect(protectChat(request(), "token")).rejects.toMatchObject({
      code: "verification_unavailable",
    });
  });

  it("requires every security setting and never emits an empty origin header", async () => {
    vi.stubEnv("ORIGIN_AUTH_TOKEN", "");
    await expect(originHeaders()).rejects.toMatchObject({ code: "service_unavailable" });
    await expect(protectChat(request(), "token")).rejects.toMatchObject({
      code: "service_unavailable",
    });
  });
});
