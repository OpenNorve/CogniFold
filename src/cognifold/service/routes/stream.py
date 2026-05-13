"""SSE streaming endpoint for real-time graph updates."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from cognifold.service.sse import SSEBroker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions/{session_id}", tags=["stream"])

_HEARTBEAT_INTERVAL = 30  # seconds


@router.get("/stream")
async def stream_events(
    session_id: str,
    request: Request,
) -> StreamingResponse:
    """Subscribe to real-time SSE updates for a session."""
    mgr = request.app.state.session_manager
    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    broker: SSEBroker = request.app.state.sse_broker

    async def event_generator() -> AsyncIterator[str]:
        queue = await broker.subscribe(session_id)
        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_INTERVAL)
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield "event: heartbeat\ndata: {}\n\n"
                    continue

                if event is None:
                    # Session closed sentinel
                    break

                yield event.format_sse()
        finally:
            await broker.unsubscribe(session_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
