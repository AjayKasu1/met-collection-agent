import type { AnswerProvenance, MuseumMessage } from "@/lib/types";

export function messageText(message: MuseumMessage): string {
  return message.parts
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("");
}

export function messageProvenance(message: MuseumMessage): AnswerProvenance | null {
  const part = message.parts.findLast((item) => item.type === "data-provenance");
  return part?.data ?? null;
}
