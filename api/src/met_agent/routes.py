"""Expose verified chat SSE, live objects, and bounded session audit reads."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Annotated, cast
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Path, Query, Request
from starlette.responses import StreamingResponse

from met_agent.agent.events import Event
from met_agent.agent.models import ChatRequest
from met_agent.llm.chat import ModelError
from met_agent.runtime import Runtime
from met_agent.tools.get_object import ObjectNotFound
from met_agent.tools.models import GetObjectArguments, LiveObject

router = APIRouter()


def runtime(request: Request) -> Runtime:
    if getattr(request.app.state, "runtime", None) is None:
        request.app.state.runtime = Runtime(request.app.state.settings)
    return cast(Runtime, request.app.state.runtime)


def event(name: str, payload: object) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
    service = runtime(request)
    body = body.model_copy(update={"session_id": body.session_id or uuid4()})

    async def stream() -> AsyncIterator[str]:
        task = asyncio.create_task(service.chat(body))
        try:
            yield event("session", {"session_id": str(body.session_id)})
            while not task.done():
                done, _ = await asyncio.wait({task}, timeout=15)
                if not done:
                    yield ": waiting for verified answer\n\n"
            answer = await task
            # Buffer the model draft until all citation and claim checks finish.
            for offset in range(0, len(answer.text), 48):
                yield event("token", {"text": answer.text[offset : offset + 48]})
            yield event("answer", answer.model_dump(mode="json"))
        except ModelError as error:
            yield event("error", {"code": error.code, "message": str(error)})
        except Exception:
            yield event(
                "error", {"code": "service_unavailable", "message": "Chat service unavailable"}
            )
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/objects/{object_id}", response_model=LiveObject)
async def get_object(request: Request, object_id: Annotated[int, Path(gt=0)]) -> LiveObject:
    try:
        return await asyncio.to_thread(
            runtime(request).live.get, GetObjectArguments(object_id=object_id)
        )
    except ObjectNotFound:
        raise HTTPException(404, "Object not found") from None
    except Exception:
        raise HTTPException(503, "Met object service unavailable") from None


@router.get("/sessions/{session_id}/events", response_model=list[Event])
async def session_events(
    request: Request,
    session_id: UUID,
    after: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> list[Event]:
    store = runtime(request).events
    if not await asyncio.to_thread(store.read, session_id, limit=1):
        raise HTTPException(404, "Session not found")
    return await asyncio.to_thread(store.read, session_id, after=after, limit=limit)
