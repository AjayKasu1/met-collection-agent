import {
  createUIMessageStream,
  createUIMessageStreamResponse,
} from "ai";

import {
  apiBase,
  fetchToolTrace,
  latestUserText,
  parseApiError,
  parseChatRequest,
  PublicApiError,
} from "@/lib/api";
import { readServerEvents } from "@/lib/sse";
import type { AgentAnswer, MuseumMessage } from "@/lib/types";
import { isRecord, parseAgentAnswer } from "@/lib/validation";

export const dynamic = "force-dynamic";

function publicMessage(error: unknown): string {
  return error instanceof PublicApiError
    ? error.message
    : "The museum assistant is temporarily unavailable.";
}

export async function POST(request: Request): Promise<Response> {
  let chatRequest: ReturnType<typeof parseChatRequest>;
  let question: string;

  try {
    chatRequest = parseChatRequest(await request.json());
    question = latestUserText(chatRequest.messages);
  } catch (error) {
    return Response.json(
      { error: publicMessage(error) },
      { status: 400, headers: { "Cache-Control": "no-store" } },
    );
  }

  const stream = createUIMessageStream<MuseumMessage>({
    originalMessages: chatRequest.messages as MuseumMessage[],
    execute: async ({ writer }) => {
      const upstream = await fetch(`${apiBase()}/chat`, {
        method: "POST",
        headers: {
          Accept: "text/event-stream",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: question,
          session_id: chatRequest.session_id,
          language: chatRequest.language,
        }),
        cache: "no-store",
        signal: request.signal,
      });

      if (!upstream.ok) {
        throw new PublicApiError(
          "service_unavailable",
          upstream.status === 429
            ? "The assistant is at capacity. Try again shortly."
            : "The museum assistant is temporarily unavailable.",
        );
      }
      if (!upstream.body) {
        throw new PublicApiError("service_unavailable", "The assistant returned no response.");
      }

      const textId = crypto.randomUUID();
      let started = false;
      let streamedText = "";
      let finalAnswer: AgentAnswer | null = null;

      for await (const event of readServerEvents(upstream.body)) {
        if (event.event === "session") {
          if (
            !isRecord(event.data) ||
            event.data.session_id !== chatRequest.session_id
          ) {
            throw new PublicApiError("invalid_response", "The assistant returned an invalid session.");
          }
          continue;
        }

        if (event.event === "token") {
          if (!isRecord(event.data) || typeof event.data.text !== "string") {
            throw new PublicApiError("invalid_response", "The assistant returned invalid text.");
          }
          if (!started) {
            writer.write({ type: "text-start", id: textId });
            started = true;
          }
          streamedText += event.data.text;
          writer.write({ type: "text-delta", id: textId, delta: event.data.text });
          continue;
        }

        if (event.event === "answer") {
          finalAnswer = parseAgentAnswer(event.data);
          continue;
        }

        if (event.event === "error") throw parseApiError(event.data);
      }

      if (!finalAnswer || finalAnswer.session_id !== chatRequest.session_id) {
        throw new PublicApiError("invalid_response", "The assistant response was incomplete.");
      }

      if (!started) {
        writer.write({ type: "text-start", id: textId });
        writer.write({ type: "text-delta", id: textId, delta: finalAnswer.text });
        streamedText = finalAnswer.text;
        started = true;
      }
      if (streamedText !== finalAnswer.text) {
        throw new PublicApiError("invalid_response", "The verified response changed while streaming.");
      }

      const tools = await fetchToolTrace(finalAnswer.session_id, request.signal);
      writer.write({
        type: "data-provenance",
        data: { answer: finalAnswer, tools },
      });
      writer.write({ type: "text-end", id: textId });
    },
    onError: publicMessage,
  });

  return createUIMessageStreamResponse({
    stream,
    headers: {
      "Cache-Control": "no-cache, no-store",
      "X-Accel-Buffering": "no",
      "X-Content-Type-Options": "nosniff",
    },
  });
}
