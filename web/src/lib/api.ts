import type { UIMessage } from "ai";

import type { AuditEvent, Language, ToolTrace } from "@/lib/types";
import { isRecord, parseAuditEvents, safeHttpUrl } from "@/lib/validation";

export class PublicApiError extends Error {
  constructor(
    public readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

export function apiBase(): string {
  const envUrl = process.env.NEXT_PUBLIC_API_URL;
  const safe = safeHttpUrl(envUrl ?? "");
  if (!safe) throw new Error("NEXT_PUBLIC_API_URL must be an HTTP URL");
  return safe.replace(/\/$/, "");
}

export function latestUserText(messages: UIMessage[]): string {
  const message = messages.findLast((item) => item.role === "user");
  const text = message?.parts
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("")
    .trim();
  if (!text || text.length > 4000) throw new PublicApiError("invalid_request", "Enter a question.");
  return text;
}

export function parseChatRequest(value: unknown): {
  messages: UIMessage[];
  session_id: string | null;
  session_token: string | null;
  language: Language;
  turnstile_token: string;
} {
  if (!isRecord(value) || !Array.isArray(value.messages)) {
    throw new PublicApiError("invalid_request", "The chat request is invalid.");
  }
  let sessionId: string | null = null;
  let sessionToken: string | null = null;
  if (value.session_id !== null && value.session_id !== undefined && value.session_id !== "") {
    if (
      typeof value.session_id !== "string" ||
      !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(
        value.session_id,
      )
    ) {
      throw new PublicApiError("invalid_request", "The chat session is invalid.");
    }
    sessionId = value.session_id;
    if (typeof value.session_token === "string" && value.session_token.length > 0) {
      sessionToken = value.session_token;
    }
  }
  if (!(["en", "fr", "es", "zh"] as unknown[]).includes(value.language)) {
    throw new PublicApiError("invalid_request", "The language selection is invalid.");
  }
  if (
    typeof value.turnstile_token !== "string" ||
    value.turnstile_token.length === 0 ||
    value.turnstile_token.length > 2048
  ) {
    throw new PublicApiError("verification_required", "Complete the security check and try again.");
  }
  return {
    messages: value.messages as UIMessage[],
    session_id: sessionId,
    session_token: sessionToken,
    language: value.language as Language,
    turnstile_token: value.turnstile_token,
  };
}

export function parseApiError(value: unknown): PublicApiError {
  if (isRecord(value) && typeof value.code === "string" && typeof value.message === "string") {
    const messages: Record<string, string> = {
      service_unavailable: "The museum assistant is temporarily unavailable.",
      request_timeout: "The assistant took too long to answer. Try asking about a specific work.",
      verification_required: "Complete the security check and try again.",
      invalid_request: "Check your question and try again.",
      invalid_response: "The assistant returned an unexpected response. Try again.",
    };
    return new PublicApiError(value.code, messages[value.code] ?? "The museum assistant is temporarily unavailable.");
  }
  return new PublicApiError("service_unavailable", "The museum assistant is temporarily unavailable.");
}

export function toolTrace(events: AuditEvent[]): ToolTrace[] {
  const calls: ToolTrace[] = [];
  for (const event of events) {
    if (event.kind === "tool_call" && isRecord(event.data) && typeof event.data.name === "string") {
      calls.push({ name: event.data.name, latency_ms: null });
    }
    if (
      event.kind === "tool_timing" &&
      isRecord(event.data) &&
      typeof event.data.name === "string" &&
      typeof event.data.latency_ms === "number"
    ) {
      const { name, latency_ms: latencyMs } = event.data;
      const match = calls.find((call) => call.name === name && call.latency_ms === null);
      if (match) match.latency_ms = latencyMs;
    }
  }
  return calls;
}

export async function fetchToolTrace(
  sessionId: string,
  signal: AbortSignal,
  headers: HeadersInit = {},
  sessionToken?: string | null,
): Promise<ToolTrace[]> {
  try {
    const authHeaders: Record<string, string> = sessionToken ? { "X-Session-Token": sessionToken } : {};
    const response = await fetch(`${apiBase()}/sessions/${encodeURIComponent(sessionId)}/events?limit=200`, {
      headers: { Accept: "application/json", ...headers, ...authHeaders },
      cache: "no-store",
      signal,
    });
    if (!response.ok) return [];
    return toolTrace(parseAuditEvents(await response.json()));
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    return [];
  }
}
