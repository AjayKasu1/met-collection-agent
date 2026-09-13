import type { UIMessage } from "ai";

export type Language = "en" | "fr" | "es" | "zh";

export type Citation = {
  object_id: number | null;
  source_url: string | null;
  quote: string;
};

export type ModelCall = {
  model: string;
  route: string;
  provider: "gemini" | "groq" | "cerebras";
  path: "ai_gateway" | "direct_google" | "direct_groq" | "direct_cerebras";
  pacing_ms: number;
  provider_ms: number;
  retry_ms: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number | null;
  latency_ms: number;
};

export type Handoff = {
  reason: string;
  suggested_contact: "info@metmuseum.org" | "store.support@metmuseum.org";
};

export type AnswerKind =
  | "social"
  | "clarification"
  | "factual"
  | "policy_refusal"
  | "unavailable"
  | "handoff";

export type VerificationStatus =
  | "verified"
  | "not_applicable"
  | "unverified";

export type AgentAnswer = {
  text: string;
  citations: Citation[];
  language: Language;
  session_id: string;
  turn_id: string;
  answer_kind?: AnswerKind;
  verification_status?: VerificationStatus;
  handoff: Handoff | null;
  route: "lite" | "main";
  cost_usd: number | null;
  latency_ms: number;
  grounding_score: number | null;
  policy_refusal: boolean;
  model_calls: ModelCall[];
};

export type ToolTrace = {
  name: string;
  latency_ms: number | null;
};

export type AnswerProvenance = {
  answer: AgentAnswer;
  tools: ToolTrace[];
  session_token?: string | null;
};

export type MuseumDataParts = {
  provenance: AnswerProvenance;
};

export type MuseumMessage = UIMessage<never, MuseumDataParts>;

export type AuditEvent = {
  sequence: number;
  kind: string;
  data: unknown;
};

export type LiveObject = {
  object_id: number;
  title: string;
  artist: string;
  object_date: string;
  gallery_number: string;
  is_on_view: boolean;
  primary_image_small: string;
  source_url: string;
};
