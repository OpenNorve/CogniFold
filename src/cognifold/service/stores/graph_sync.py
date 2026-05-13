"""Write-through observer that mirrors graph mutations to Supabase relational tables.

All methods are fire-and-forget: exceptions are caught and logged so that
graph operations are never blocked by persistence failures.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class GraphSyncWriter:
    """Mirrors individual graph mutations to ``graph_nodes`` / ``graph_edges`` tables.

    Nodes and edges are queued during plan execution and flushed in bulk
    after the session row has been persisted to Supabase.  This avoids FK
    violations (``graph_nodes.session_id`` references ``sessions.session_id``).
    """

    def __init__(self, supabase_client: Any, session_id: str) -> None:
        self._client = supabase_client
        self._session_id = session_id
        # Pending operations queued during execute(), flushed by flush()
        self._pending_nodes: list[dict[str, Any]] = []
        self._pending_edges: list[dict[str, Any]] = []
        self._pending_deletes: list[tuple[str, str, dict[str, Any]]] = []  # (table, col, match)

    # ------------------------------------------------------------------
    # Flush — call after session row is persisted
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Write all queued mutations to Supabase.

        Call this *after* the session row has been saved so FK constraints
        on ``session_id`` are satisfied.
        """
        # Upsert nodes first (edges reference them)
        for row in self._pending_nodes:
            try:
                self._client.table("graph_nodes").upsert(row, on_conflict="session_id,id").execute()
            except Exception:
                logger.warning("GraphSync: failed to flush node %s", row.get("id"), exc_info=True)
        # Upsert edges
        for row in self._pending_edges:
            try:
                self._client.table("graph_edges").upsert(
                    row, on_conflict="session_id,source_id,target_id,edge_type"
                ).execute()
            except Exception:
                logger.warning(
                    "GraphSync: failed to flush edge %s->%s",
                    row.get("source_id"),
                    row.get("target_id"),
                    exc_info=True,
                )
        # Process deletes
        for table, _col, match in self._pending_deletes:
            try:
                q = self._client.table(table).delete()
                for k, v in match.items():
                    q = q.eq(k, v)
                q.execute()
            except Exception:
                logger.warning("GraphSync: failed to flush delete on %s", table, exc_info=True)
        # Clear queues
        self._pending_nodes.clear()
        self._pending_edges.clear()
        self._pending_deletes.clear()

    # ------------------------------------------------------------------
    # Node operations — queue for later flush
    # ------------------------------------------------------------------

    def on_node_added(
        self,
        node_id: str,
        node_type: str,
        data: dict[str, Any],
        embedding: list[float] | None = None,
    ) -> None:
        """Queue a node upsert."""
        row: dict[str, Any] = {
            "id": node_id,
            "session_id": self._session_id,
            "node_type": node_type,
            "data": data,
        }
        # Only include embedding if it's a non-empty vector
        if embedding:
            row["embedding"] = embedding
        self._pending_nodes.append(row)

    def on_node_updated(self, node_id: str, data: dict[str, Any], node_type: str = "event") -> None:
        """Queue a node data update (upsert with updated data)."""
        self._pending_nodes.append(
            {
                "id": node_id,
                "session_id": self._session_id,
                "node_type": node_type,
                "data": data,
            }
        )

    def on_node_removed(self, node_id: str) -> None:
        """Queue a node deletion (edges cascade via FK)."""
        self._pending_deletes.append(
            (
                "graph_nodes",
                "id",
                {"session_id": self._session_id, "id": node_id},
            )
        )

    # ------------------------------------------------------------------
    # Edge operations — queue for later flush
    # ------------------------------------------------------------------

    def on_edge_added(
        self,
        source_id: str,
        target_id: str,
        edge_type: str | None = None,
        weight: float = 1.0,
    ) -> None:
        """Queue an edge upsert."""
        resolved_type = edge_type or "__legacy__"
        self._pending_edges.append(
            {
                "session_id": self._session_id,
                "source_id": source_id,
                "target_id": target_id,
                "edge_type": resolved_type,
                "weight": weight,
            }
        )

    def on_edge_removed(
        self,
        source_id: str,
        target_id: str,
        edge_type: str | None = None,
    ) -> None:
        """Queue an edge deletion."""
        resolved_type = edge_type or "__legacy__"
        self._pending_deletes.append(
            (
                "graph_edges",
                "source_id",
                {
                    "session_id": self._session_id,
                    "source_id": source_id,
                    "target_id": target_id,
                    "edge_type": resolved_type,
                },
            )
        )
