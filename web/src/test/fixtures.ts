import type { AgentAnswer } from "@/lib/types";

export const validAnswer: AgentAnswer = {
  text: "The Temple of Dendur is in Gallery 131.",
  citations: [{ object_id: 547802, source_url: null, quote: "Gallery 131" }],
  language: "en",
  session_id: "123e4567-e89b-42d3-a456-426614174000",
  turn_id: "123e4567-e89b-42d3-a456-426614174001",
  handoff: null,
  route: "main",
  cost_usd: 0.00031,
  latency_ms: 2345,
  grounding_score: 1,
  policy_refusal: false,
  model_calls: [
    {
      model: "openai/gpt-oss-120b",
      route: "main",
      provider: "groq",
      path: "direct_groq",
      pacing_ms: 0,
      provider_ms: 500,
      retry_ms: 0,
      input_tokens: 100,
      output_tokens: 30,
      cost_usd: 0.00031,
      latency_ms: 500,
    },
  ],
};
