"""Supabase-based session persistence store."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from cognifold.service.stores.base import SessionStore

logger = logging.getLogger(__name__)


class SupabaseSessionStore(SessionStore):
    """Persists sessions in Supabase Postgres as JSONB.

    Requires the ``supabase`` package (``pip install cognifold[supabase]``).
    Uses the ``sessions`` table with ``graph_snapshot`` JSONB column.
    """

    def __init__(self, supabase_url: str, supabase_key: str) -> None:
        try:
            from supabase import create_client  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "Supabase support requires the supabase package. "
                "Install with: pip install cognifold[supabase]"
            ) from exc

        self._client: Any = create_client(supabase_url, supabase_key)

    async def save_session(self, session_id: str, data: dict[str, Any]) -> None:
        """Upsert session data into the sessions table."""
        config = data.get("config", {})
        row = {
            "session_id": session_id,
            "user_id": data.get("user_id"),
            "config": config,
            "graph_snapshot": data,
            "domain": config.get("domain", "personal-timeline"),
        }
        await asyncio.to_thread(
            lambda: self._client.table("sessions").upsert(row, on_conflict="session_id").execute()
        )
        logger.info("Persisted session %s to Supabase", session_id)

    async def load_session(self, session_id: str) -> dict[str, Any] | None:
        """Load session data from the sessions table."""
        resp = await asyncio.to_thread(
            lambda: (
                self._client.table("sessions")
                .select("graph_snapshot")
                .eq("session_id", session_id)
                .maybe_single()
                .execute()
            )
        )
        # supabase-py maybe_single() returns None (not a response) when no row
        if resp is None or resp.data is None:
            return None
        snapshot: dict[str, Any] = resp.data["graph_snapshot"]
        # Handle case where snapshot is stored as a JSON string
        if isinstance(snapshot, str):
            snapshot = json.loads(snapshot)
        return snapshot

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session (cascades to graph_nodes/edges)."""
        resp = await asyncio.to_thread(
            lambda: self._client.table("sessions").delete().eq("session_id", session_id).execute()
        )
        return len(resp.data) > 0 if resp.data else False

    async def list_sessions(self) -> list[str]:
        """List all session IDs."""
        resp = await asyncio.to_thread(
            lambda: self._client.table("sessions").select("session_id").execute()
        )
        return [row["session_id"] for row in (resp.data or [])]

    async def ping(self) -> bool:
        """Check Supabase connectivity."""
        try:
            await asyncio.to_thread(
                lambda: self._client.table("sessions").select("session_id").limit(1).execute()
            )
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """No-op: supabase-py uses HTTP, no persistent connection."""
