"""In-process memory wrapper shared by the MCP tools.

This module mirrors the minimal flow the HTTP service performs
(``service/processor.py`` + ``service/session.py``) but in a single in-process
object suitable for a stdio MCP server: one persistent graph, ingested via the
existing ``Pipeline``, queried via the existing ``MemoryQueryAgent``, and saved
to disk through the existing ``graph/persistence`` helpers so memory survives
across tool calls and restarts.

Nothing here changes algorithm behaviour — it only wires the existing classes
together and adds load/save around them.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from cognifold.config import CognifoldConfig
from cognifold.graph.persistence import load_graph, save_graph
from cognifold.graph.store import ConceptGraph
from cognifold.models.event import Event
from cognifold.models.node import NodeType
from cognifold.pipeline.classic import Pipeline

DEFAULT_GRAPH_PATH = Path.home() / ".cognifold" / "mcp_graph.json"


def _redirect_logging_to_stderr() -> None:
    """Point CogniFold's log handlers at stderr.

    The MCP stdio transport owns stdout for JSON-RPC framing — any log line on
    stdout corrupts the protocol. CogniFold's ``setup_logging`` (invoked inside
    ``Pipeline.__init__``) attaches a ``StreamHandler(sys.stdout)`` to the root
    logger. We retarget those handlers to stderr after the pipeline is built,
    without otherwise changing logging behaviour.
    """
    for logger in (logging.getLogger(), logging.getLogger("cognifold")):
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
                handler.setStream(sys.stderr)


def _resolve_graph_path() -> Path:
    """Resolve the on-disk graph path from env or the default location."""
    raw = os.environ.get("COGNIFOLD_MCP_GRAPH")
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_GRAPH_PATH


class CognifoldMemory:
    """A single persistent CogniFold memory backing the MCP tools.

    Holds one ``Pipeline`` (for ingestion) over a ``ConceptGraph`` that is
    loaded from / saved to ``graph_path``. A ``MemoryQueryAgent`` is lazily
    constructed against the same graph for queries.

    Args:
        config: Optional pre-built config. Defaults to ``CognifoldConfig.load()``
            which reads model name + API keys from the environment.
        graph_path: Where the graph JSON is persisted. Defaults to
            ``$COGNIFOLD_MCP_GRAPH`` or ``~/.cognifold/mcp_graph.json``.
    """

    def __init__(
        self,
        config: CognifoldConfig | None = None,
        graph_path: str | Path | None = None,
    ) -> None:
        self._config = config or CognifoldConfig.load()
        self._graph_path = Path(graph_path).expanduser() if graph_path else _resolve_graph_path()

        # Build the pipeline (creates its own empty ConceptGraph), then swap in
        # the persisted graph if one exists on disk. The pipeline reads its
        # graph through the private `_graph` attribute; we replace it before any
        # event is processed so the ranker/agent operate on the loaded graph.
        self._pipeline = Pipeline(self._config)
        # Pipeline.__init__ wired logging to stdout; reclaim stdout for MCP.
        _redirect_logging_to_stderr()
        if self._graph_path.exists():
            loaded = load_graph(self._graph_path)
            # Pipeline has no public graph setter; swap the persisted graph in
            # before any event is processed. Deliberate internal write.
            self._pipeline._graph = loaded  # pyright: ignore[reportPrivateUsage]

        self._query_agent: Any = None

    @property
    def graph(self) -> ConceptGraph:
        """The live concept graph (same object the pipeline mutates)."""
        return self._pipeline.graph

    @property
    def graph_path(self) -> Path:
        """Path the graph is persisted to."""
        return self._graph_path

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def remember(self, text: str, timestamp: datetime | None = None) -> dict[str, Any]:
        """Ingest a free-text observation into the persistent graph.

        Mirrors ``service/processor.process_event_sync``: build an ``Event``,
        run it through the pipeline (LLM agent if an API key is configured,
        otherwise a default add-node plan), persist, and report deltas.

        Args:
            text: The observation / fact / event to remember.
            timestamp: When it happened. Defaults to now.

        Returns:
            A dict summarising the graph deltas and resulting totals.
        """
        before_nodes = self.graph.node_count
        before_edges = self.graph.edge_count

        event = Event(
            event_id=f"evt-{uuid.uuid4().hex[:12]}",
            timestamp=timestamp or datetime.now(),
            source="mcp",
            event_type="note",
            title=text[:120],
            description=text,
        )

        result = self._pipeline.process_event(event)

        # Persist so memory survives across tool calls / restarts.
        save_graph(self.graph, self._graph_path, backup=False)

        # The query agent's cached indexes are now stale.
        if self._query_agent is not None:
            self._query_agent.invalidate_search_cache()

        return {
            "event_id": event.event_id,
            "success": result.success,
            "error": result.error,
            "nodes_added": self.graph.node_count - before_nodes,
            "edges_added": self.graph.edge_count - before_edges,
            "concepts_created": result.concepts_created,
            "intents_created": result.actions_created,
            "total_nodes": self.graph.node_count,
            "total_edges": self.graph.edge_count,
            "graph_path": str(self._graph_path),
        }

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def _get_query_agent(self) -> Any:
        if self._query_agent is None:
            from cognifold.query import MemoryQueryAgent

            self._query_agent = MemoryQueryAgent(self.graph)
        return self._query_agent

    def query(self, question: str, max_nodes: int = 10) -> dict[str, Any]:
        """Answer a question from memory via ``MemoryQueryAgent``.

        Returns the assembled context plus brief supporting-node summaries.
        Note: CogniFold's query layer returns retrieved context (not an
        LLM-generated answer); the MCP client's model reads the context to
        answer.

        Args:
            question: Natural-language question.
            max_nodes: Max supporting nodes to retrieve.

        Returns:
            A dict with ``context`` and ``supporting_nodes``.
        """
        agent = self._get_query_agent()
        result = agent.query(question, max_nodes=max_nodes)

        supporting = [
            {
                "node_id": n.node_id,
                "type": n.node_type,
                "title": n.title,
                "relevance": round(n.relevance_score, 4),
                "description": (n.description or "")[:280] or None,
            }
            for n in result.nodes
        ]

        return {
            "question": question,
            "context": result.context,
            "supporting_nodes": supporting,
            "nodes_scanned": result.total_nodes_scanned,
            "query_time_ms": round(result.query_time_ms, 2),
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def graph_stats(self) -> dict[str, Any]:
        """Node/edge counts by type (mirrors ``Session.get_graph_stats``)."""
        graph = self.graph
        return {
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "events": len(graph.get_nodes_by_type(NodeType.EVENT)),
            "concepts": len(graph.get_nodes_by_type(NodeType.CONCEPT)),
            "intents": len(graph.get_nodes_by_type(NodeType.INTENT)),
            "time_nodes": len(graph.get_nodes_by_type(NodeType.TIME)),
            "graph_path": str(self._graph_path),
        }

    # ------------------------------------------------------------------
    # Intents
    # ------------------------------------------------------------------
    def list_intents(self) -> list[dict[str, Any]]:
        """List intent nodes (goals/desires) currently in the graph.

        Intents are stored as ``NodeType.INTENT`` nodes; status lives in
        ``node.data['status']`` when present.
        """
        intents: list[dict[str, Any]] = []
        for node in self.graph.get_nodes_by_type(NodeType.INTENT):
            data = node.data or {}
            intents.append(
                {
                    "id": node.id,
                    "status": data.get("status", "pending"),
                    "title": data.get("title") or data.get("name") or node.id,
                    "description": data.get("description"),
                    "reasoning": node.reasoning,
                    "last_accessed": node.last_accessed.isoformat(),
                }
            )
        return intents
