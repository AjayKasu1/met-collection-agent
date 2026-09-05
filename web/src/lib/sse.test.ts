import { describe, expect, it } from "vitest";

import { readServerEvents } from "@/lib/sse";

function body(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

describe("readServerEvents", () => {
  it("preserves events split across transport chunks", async () => {
    const events = [];
    for await (const event of readServerEvents(
      body([
        "event: session\r\ndata: {\"session_id\":\"abc\"}\r\n\r\nevent: to",
        "ken\ndata: {\"text\":\"Gallery 131\"}\n\n",
      ]),
    )) {
      events.push(event);
    }

    expect(events).toEqual([
      { event: "session", data: { session_id: "abc" } },
      { event: "token", data: { text: "Gallery 131" } },
    ]);
  });

  it("ignores keepalives and combines multiline JSON data", async () => {
    const events = [];
    for await (const event of readServerEvents(
      body([": waiting\n\nevent: answer\ndata: {\"text\":\ndata: \"verified\"}\n\n"]),
    )) {
      events.push(event);
    }
    expect(events).toEqual([{ event: "answer", data: { text: "verified" } }]);
  });

  it("fails closed on malformed event data", async () => {
    const consume = async () => {
      for await (const event of readServerEvents(body(["event: answer\ndata: {nope}\n\n"]))) {
        void event;
      }
    };
    await expect(consume()).rejects.toThrow("malformed stream data");
  });
});
