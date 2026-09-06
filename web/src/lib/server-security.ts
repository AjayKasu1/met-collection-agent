import { getCloudflareContext } from "@opennextjs/cloudflare";

import { PublicApiError } from "@/lib/api";
import { isRecord } from "@/lib/validation";

const TURNSTILE_ACTION = "chat";
const SITEVERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify";

function hostnames(value: string | undefined): Set<string> {
  return new Set(
    (value ?? "")
      .split(",")
      .map((hostname) => hostname.trim().toLowerCase())
      .filter(Boolean),
  );
}

async function workerEnv(): Promise<CloudflareEnv> {
  return (await getCloudflareContext({ async: true })).env;
}

export async function originHeaders(): Promise<Record<string, string>> {
  const token = (await workerEnv()).ORIGIN_AUTH_TOKEN?.trim();
  if (!token) {
    throw new PublicApiError("service_unavailable", "The museum assistant is not configured.");
  }
  return { "X-Origin-Auth": token };
}

export async function protectChat(request: Request, token: string): Promise<Record<string, string>> {
  const env = await workerEnv();
  const clientIp = request.headers.get("cf-connecting-ip")?.trim();
  const secret = env.TURNSTILE_SECRET?.trim();
  const expectedHostnames = hostnames(env.TURNSTILE_HOSTNAMES);
  const originToken = env.ORIGIN_AUTH_TOKEN?.trim();

  if (!clientIp || !secret || expectedHostnames.size === 0 || !originToken) {
    throw new PublicApiError("service_unavailable", "The museum assistant is not configured.");
  }

  const { success: withinLimit } = await env.CHAT_RATE_LIMITER.limit({ key: clientIp });
  if (!withinLimit) {
    console.warn(JSON.stringify({ event: "chat_rate_limited", request_id: request.headers.get("cf-ray") }));
    throw new PublicApiError("rate_limited", "Too many questions. Try again in a minute.");
  }

  let response: Response;
  try {
    response = await fetch(SITEVERIFY_URL, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ secret, response: token, remoteip: clientIp }),
      signal: AbortSignal.timeout(10_000),
    });
  } catch {
    throw new PublicApiError("verification_unavailable", "The security check is unavailable.");
  }

  let result: unknown;
  try {
    result = await response.json();
  } catch {
    result = null;
  }
  const valid =
    response.ok &&
    isRecord(result) &&
    result.success === true &&
    result.action === TURNSTILE_ACTION &&
    typeof result.hostname === "string" &&
    expectedHostnames.has(result.hostname.toLowerCase());
  if (!valid) {
    console.warn(JSON.stringify({ event: "turnstile_rejected", request_id: request.headers.get("cf-ray") }));
    throw new PublicApiError("verification_required", "Complete the security check and try again.");
  }

  return { "X-Origin-Auth": originToken };
}
