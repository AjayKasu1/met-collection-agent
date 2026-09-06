import { apiBase } from "@/lib/api";
import { originHeaders } from "@/lib/server-security";
import { parseLiveObject } from "@/lib/validation";

type RouteContext = { params: Promise<{ objectId: string }> };

export async function GET(request: Request, context: RouteContext): Promise<Response> {
  const { objectId } = await context.params;
  if (!/^\d{1,10}$/.test(objectId) || Number(objectId) < 1) {
    return Response.json({ error: "Invalid object ID" }, { status: 400 });
  }

  try {
    const upstream = await fetch(`${apiBase()}/objects/${objectId}`, {
      headers: { Accept: "application/json", ...(await originHeaders()) },
      cache: "no-store",
      signal: request.signal,
    });
    if (upstream.status === 404) {
      return Response.json({ error: "Object not found" }, { status: 404 });
    }
    if (!upstream.ok) throw new Error("Object service unavailable");
    return Response.json(parseLiveObject(await upstream.json()), {
      headers: { "Cache-Control": "public, max-age=60, stale-while-revalidate=300" },
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    return Response.json({ error: "Object details are temporarily unavailable" }, { status: 503 });
  }
}
