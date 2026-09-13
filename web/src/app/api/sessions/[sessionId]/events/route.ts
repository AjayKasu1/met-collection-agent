import { apiBase } from "@/lib/api";
import { originHeaders } from "@/lib/server-security";

type RouteContext = { params: Promise<{ sessionId: string }> };

export async function GET(request: Request, context: RouteContext): Promise<Response> {
  const { sessionId } = await context.params;
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(sessionId)) {
    return Response.json({ error: "Invalid session ID" }, { status: 400 });
  }

  const url = new URL(request.url);
  const token = request.headers.get("x-session-token") ?? "";
  const limit = url.searchParams.get("limit") ?? "200";
  const after = url.searchParams.get("after");

  const queryParams = new URLSearchParams({ limit });
  if (after) queryParams.set("after", after);

  try {
    const upstream = await fetch(
      `${apiBase()}/sessions/${encodeURIComponent(sessionId)}/events?${queryParams.toString()}`,
      {
        headers: {
          Accept: "application/json",
          ...(token ? { "X-Session-Token": token } : {}),
          ...(await originHeaders()),
        },
        cache: "no-store",
        signal: request.signal,
      },
    );

    if (upstream.status === 401) {
      return Response.json({ error: "Unauthorized session access" }, { status: 401 });
    }
    if (upstream.status === 404) {
      return Response.json({ error: "Session not found" }, { status: 404 });
    }
    if (!upstream.ok) {
      return Response.json({ error: "History temporarily unavailable" }, { status: 503 });
    }

    return Response.json(await upstream.json(), {
      headers: { "Cache-Control": "no-store, no-cache" },
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    return Response.json({ error: "Failed to retrieve session events" }, { status: 503 });
  }
}
