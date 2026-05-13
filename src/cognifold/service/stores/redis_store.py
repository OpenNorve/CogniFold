"""Redis-based session persistence store."""

from __future__ import annotations

import json
import logging
from typing import Any

from cognifold.service.stores.base import SessionStore

logger = logging.getLogger(__name__)

# Key prefix for all session data
_KEY_PREFIX = "cognifold:session:"


class RedisSessionStore(SessionStore):
    """Persists sessions in Redis as JSON strings.

    Requires the ``redis`` package (``pip install cognifold[redis]``).
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 86400,
    ) -> None:
        """Initialize Redis store.

        Args:
            redis_url: Redis connection URL.
            ttl_seconds: TTL for session keys (default: 24 hours).
        """
        try:
            import redis.asyncio as aioredis  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "Redis support requires the redis package. "
                "Install with: pip install cognifold[redis]"
            ) from exc

        self._client: Any = aioredis.from_url(redis_url, decode_responses=True)
        self._ttl = ttl_seconds

    async def save_session(self, session_id: str, data: dict[str, Any]) -> None:
        """Save graph data to Redis with TTL."""
        key = f"{_KEY_PREFIX}{session_id}"
        payload = json.dumps(data, ensure_ascii=False)
        await self._client.set(key, payload, ex=self._ttl)
        logger.info("Persisted session %s to Redis", session_id)

    async def load_session(self, session_id: str) -> dict[str, Any] | None:
        """Load graph data from Redis."""
        key = f"{_KEY_PREFIX}{session_id}"
        payload = await self._client.get(key)
        if payload is None:
            return None
        result: dict[str, Any] = json.loads(payload)
        return result

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session key from Redis."""
        key = f"{_KEY_PREFIX}{session_id}"
        deleted: int = await self._client.delete(key)
        return deleted > 0

    async def list_sessions(self) -> list[str]:
        """List session IDs by scanning Redis keys."""
        prefix = _KEY_PREFIX
        sessions: list[str] = []
        async for key in self._client.scan_iter(match=f"{prefix}*"):
            session_id = key[len(prefix) :]
            sessions.append(session_id)
        return sessions

    async def ping(self) -> bool:
        """Check if Redis is reachable."""
        try:
            result: bool = await self._client.ping()
            return result
        except Exception:
            return False

    async def close(self) -> None:
        """Close the Redis connection."""
        await self._client.aclose()
