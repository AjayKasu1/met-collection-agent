import type {
  AgentAnswer,
  AuditEvent,
  Citation,
  Language,
  LiveObject,
  ModelCall,
} from "@/lib/types";

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringValue(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`Invalid ${key}`);
  }
  return value;
}

function nullableString(record: Record<string, unknown>, key: string): string | null {
  const value = record[key];
  if (value === null) return null;
  if (typeof value !== "string") throw new Error(`Invalid ${key}`);
  return value;
}

function httpUrlValue(record: Record<string, unknown>, key: string): string {
  const value = safeHttpUrl(stringValue(record, key));
  if (!value) throw new Error(`Invalid ${key}`);
  return value;
}

function finiteNumber(record: Record<string, unknown>, key: string): number {
  const value = record[key];
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`Invalid ${key}`);
  }
  return value;
}

function nullableNumber(record: Record<string, unknown>, key: string): number | null {
  const value = record[key];
  if (value === null) return null;
  return finiteNumber(record, key);
}

function parseCitation(value: unknown): Citation {
  if (!isRecord(value)) throw new Error("Invalid citation");
  const objectId = value.object_id;
  if (objectId !== null && (!Number.isInteger(objectId) || Number(objectId) <= 0)) {
    throw new Error("Invalid citation object_id");
  }
  const sourceUrl = nullableString(value, "source_url");
  if ((objectId === null) === (sourceUrl === null)) {
    throw new Error("Citation requires exactly one source");
  }
  if (sourceUrl !== null && safeHttpUrl(sourceUrl) === null) {
    throw new Error("Invalid citation source URL");
  }
  return {
    object_id: objectId === null ? null : Number(objectId),
    source_url: sourceUrl,
    quote: stringValue(value, "quote"),
  };
}

function parseModelCall(value: unknown): ModelCall {
  if (!isRecord(value)) throw new Error("Invalid model call");
  const provider = stringValue(value, "provider");
  const path = stringValue(value, "path");
  if (!(["gemini", "groq", "cerebras"] as string[]).includes(provider)) {
    throw new Error("Invalid model provider");
  }
  if (
    !(["ai_gateway", "direct_google", "direct_groq", "direct_cerebras"] as string[]).includes(
      path,
    )
  ) {
    throw new Error("Invalid model path");
  }
  return {
    model: stringValue(value, "model"),
    route: stringValue(value, "route"),
    provider: provider as ModelCall["provider"],
    path: path as ModelCall["path"],
    pacing_ms: finiteNumber(value, "pacing_ms"),
    provider_ms: finiteNumber(value, "provider_ms"),
    retry_ms: finiteNumber(value, "retry_ms"),
    input_tokens: finiteNumber(value, "input_tokens"),
    output_tokens: finiteNumber(value, "output_tokens"),
    cost_usd: nullableNumber(value, "cost_usd"),
    latency_ms: finiteNumber(value, "latency_ms"),
  };
}

export function parseAgentAnswer(value: unknown): AgentAnswer {
  if (!isRecord(value)) throw new Error("Invalid final answer");
  const language = stringValue(value, "language");
  const route = stringValue(value, "route");
  if (!(["en", "fr", "es", "zh"] as string[]).includes(language)) {
    throw new Error("Invalid answer language");
  }
  if (route !== "lite" && route !== "main") throw new Error("Invalid answer route");
  if (!Array.isArray(value.citations) || !Array.isArray(value.model_calls)) {
    throw new Error("Invalid answer evidence");
  }
  const handoff = value.handoff;
  let parsedHandoff: AgentAnswer["handoff"] = null;
  if (handoff !== null) {
    if (!isRecord(handoff)) throw new Error("Invalid handoff");
    const contact = stringValue(handoff, "suggested_contact");
    if (contact !== "info@metmuseum.org" && contact !== "store.support@metmuseum.org") {
      throw new Error("Invalid handoff contact");
    }
    parsedHandoff = { reason: stringValue(handoff, "reason"), suggested_contact: contact };
  }
  const grounding = finiteNumber(value, "grounding_score");
  if (grounding < 0 || grounding > 1 || typeof value.policy_refusal !== "boolean") {
    throw new Error("Invalid verification state");
  }
  return {
    text: stringValue(value, "text"),
    citations: value.citations.map(parseCitation),
    language: language as Language,
    session_id: stringValue(value, "session_id"),
    turn_id: stringValue(value, "turn_id"),
    handoff: parsedHandoff,
    route,
    cost_usd: nullableNumber(value, "cost_usd"),
    latency_ms: finiteNumber(value, "latency_ms"),
    grounding_score: grounding,
    policy_refusal: value.policy_refusal,
    model_calls: value.model_calls.map(parseModelCall),
  };
}

export function parseAuditEvents(value: unknown): AuditEvent[] {
  if (!Array.isArray(value)) throw new Error("Invalid audit response");
  return value.map((entry) => {
    if (!isRecord(entry)) throw new Error("Invalid audit event");
    return {
      sequence: finiteNumber(entry, "sequence"),
      kind: stringValue(entry, "kind"),
      data: entry.data,
    };
  });
}

export function parseLiveObject(value: unknown): LiveObject {
  if (!isRecord(value)) throw new Error("Invalid object response");
  const objectId = finiteNumber(value, "object_id");
  if (!Number.isInteger(objectId) || objectId <= 0 || typeof value.is_on_view !== "boolean") {
    throw new Error("Invalid object identity");
  }
  return {
    object_id: objectId,
    title: stringValue(value, "title"),
    artist: typeof value.artist === "string" ? value.artist : "",
    object_date: typeof value.object_date === "string" ? value.object_date : "",
    gallery_number: typeof value.gallery_number === "string" ? value.gallery_number : "",
    is_on_view: value.is_on_view,
    primary_image_small:
      typeof value.primary_image_small === "string" ? value.primary_image_small : "",
    source_url: httpUrlValue(value, "source_url"),
  };
}

export function safeHttpUrl(value: string): string | null {
  try {
    const url = new URL(value);
    return url.protocol === "https:" || url.protocol === "http:" ? url.href : null;
  } catch {
    return null;
  }
}
