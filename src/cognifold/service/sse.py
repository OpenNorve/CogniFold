"""Server-Sent Events (SSE) broker for real-time graph updates."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Maximum queued events per subscriber before dropping
_MAX_QUEUE_SIZE = 256


@dataclass
class SSEEvent:
    """A single SSE event ready to be sent to subscribers."""

    event_type: str
    data: dict[str, Any]
    session_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def format_sse(self) -> str:
        """Format as an SSE text block (``event:`` + ``data:`` + blank line)."""
        payload = json.dumps(self.data, ensure_ascii=False)
        return f"event: {self.event_type}\ndata: {payload}\n\n"


class SSEBroker:
    """Per-session publish/subscribe broker for SSE events.

    Each session can have multiple subscribers (e.g. multiple browser tabs).
    Publishing is non-blocking: if a subscriber's queue is full the event
    is silently dropped (slow consumer protection).
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[SSEEvent | None]]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, session_id: str) -> asyncio.Queue[SSEEvent | None]:
        """Register a new subscriber for a session.

        Returns an asyncio.Queue that will receive SSEEvent objects.
        A ``None`` sentinel signals the session stream has been closed.
        """
        queue: asyncio.Queue[SSEEvent | None] = asyncio.Queue(maxsize=_MAX_QUEUE_SIZE)
        async with self._lock:
            if session_id not in self._subscribers:
                self._subscribers[session_id] = []
            self._subscribers[session_id].append(queue)
        logger.debug("SSE subscriber added for session %s", session_id)
        return queue

    async def unsubscribe(self, session_id: str, queue: asyncio.Queue[SSEEvent | None]) -> None:
        """Remove a subscriber."""
        async with self._lock:
            subs = self._subscribers.get(session_id, [])
            with contextlib.suppress(ValueError):
                subs.remove(queue)
            if not subs:
                self._subscribers.pop(session_id, None)
        logger.debug("SSE subscriber removed for session %s", session_id)

    async def publish(self, event: SSEEvent) -> None:
        """Publish an event to all subscribers of its session.

        Non-blocking: full queues are skipped with a warning.
        """
        async with self._lock:
            subs = list(self._subscribers.get(event.session_id, []))

        for queue in subs:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "SSE queue full for session %s, dropping event %s",
                    event.session_id,
                    event.event_type,
                )

    async def close_session(self, session_id: str) -> None:
        """Send a ``None`` sentinel to all subscribers and remove the session."""
        async with self._lock:
            subs = self._subscribers.pop(session_id, [])

        for queue in subs:
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(None)
