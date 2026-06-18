"""FastMCP stdio server exposing CogniFold memory tools.

Tools:
  - ``cognifold_remember``    ingest an observation into the persistent graph
  - ``cognifold_query``       retrieve context from memory for a question
  - ``cognifold_graph_stats`` node/edge counts by type
  - ``cognifold_list_intents``current intents (goals/desires) in the graph

The MCP Python SDK (``mcp``) is an optional dependency. If it is missing we
raise a clear, actionable error rather than a bare ``ImportError`` so users
know to ``pip install 'cognifold[mcp]'``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from cognifold.mcp.memory import CognifoldMemory

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP  # pyright: ignore[reportMissingImports]

_MCP_INSTALL_HINT = (
    "The MCP Python SDK is required to run the CogniFold MCP server but is not "
    "installed. Install it with:\n\n    pip install 'cognifold[mcp]'\n\n"
    '(or `pip install "mcp>=1.0"`).'
)


def _require_fastmcp() -> type[FastMCP]:
    """Import FastMCP, raising a helpful message if the SDK is absent."""
    try:
        from mcp.server.fastmcp import FastMCP  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - exercised only without SDK
        raise RuntimeError(_MCP_INSTALL_HINT) from exc
    return FastMCP


def build_server(memory: CognifoldMemory | None = None) -> FastMCP:
    """Construct the FastMCP server with CogniFold tools registered.

    Args:
        memory: Optional pre-built memory (useful for tests). Defaults to a
            fresh ``CognifoldMemory`` reading config + graph path from env.

    Returns:
        A configured ``FastMCP`` instance ready to ``.run()``.
    """
    fast_mcp_cls = _require_fastmcp()
    mem = memory or CognifoldMemory()

    server = fast_mcp_cls(
        name="cognifold",
        instructions=(
            "CogniFold is a persistent concept-graph memory. Use "
            "cognifold_remember to store observations/facts as they happen, "
            "cognifold_query to recall relevant context before answering, "
            "cognifold_graph_stats to inspect memory size, and "
            "cognifold_list_intents to see tracked goals."
        ),
    )

    @server.tool()
    def cognifold_remember(text: str, timestamp: str | None = None) -> str:  # pyright: ignore[reportUnusedFunction]
        """Store an observation, fact, or event in CogniFold's persistent memory.

        The text is ingested through CogniFold's pipeline, which extracts
        concepts/intents and links them into the concept graph. The graph is
        persisted to disk so it survives across calls and restarts.

        Args:
            text: The observation to remember (a fact, event, or note).
            timestamp: Optional ISO-8601 timestamp of when it happened
                (e.g. "2026-06-18T14:30:00"). Defaults to now.

        Returns:
            JSON summary of graph deltas (nodes/edges added, concepts created).
        """
        from datetime import datetime

        ts = datetime.fromisoformat(timestamp) if timestamp else None
        result = mem.remember(text, timestamp=ts)
        return json.dumps(result, ensure_ascii=False, indent=2)

    @server.tool()
    def cognifold_query(question: str, max_nodes: int = 10) -> str:  # pyright: ignore[reportUnusedFunction]
        """Recall relevant context from CogniFold memory for a question.

        Runs CogniFold's MemoryQueryAgent over the persistent graph and returns
        the assembled context plus the supporting nodes. Read the returned
        context to answer the user's question.

        Args:
            question: The natural-language question to recall context for.
            max_nodes: Maximum number of supporting nodes to retrieve (default 10).

        Returns:
            JSON with the assembled ``context`` and brief ``supporting_nodes``.
        """
        result = mem.query(question, max_nodes=max_nodes)
        return json.dumps(result, ensure_ascii=False, indent=2)

    @server.tool()
    def cognifold_graph_stats() -> str:  # pyright: ignore[reportUnusedFunction]
        """Report CogniFold memory size: node and edge counts by type.

        Returns:
            JSON with total node/edge counts and per-type counts (events,
            concepts, intents, time nodes) plus the on-disk graph path.
        """
        return json.dumps(mem.graph_stats(), ensure_ascii=False, indent=2)

    @server.tool()
    def cognifold_list_intents() -> str:  # pyright: ignore[reportUnusedFunction]
        """List the goals/desires (intents) CogniFold is currently tracking.

        Returns:
            JSON array of intents, each with id, status, title, and description.
        """
        return json.dumps(mem.list_intents(), ensure_ascii=False, indent=2)

    return server


def main(argv: list[str] | None = None) -> None:
    """Entry point: build and run the stdio MCP server.

    Args:
        argv: Optional CLI args. Supports ``--help`` and ``--version``; any
            other args are ignored (the server reads config from env).
    """
    import sys

    args = list(sys.argv[1:] if argv is None else argv)

    if "--help" in args or "-h" in args:
        print(
            "cognifold-mcp — CogniFold Model Context Protocol (stdio) server\n\n"
            "Usage: cognifold-mcp\n"
            "       python -m cognifold.mcp\n\n"
            "Environment:\n"
            "  COGNIFOLD_MCP_GRAPH   path to persist the graph "
            "(default ~/.cognifold/mcp_graph.json)\n"
            "  COGNIFOLD_MODEL__NAME LLM model name (default gemini-2.5-flash)\n"
            "  GOOGLE_API_KEY / OPENAI_API_KEY   API key for LLM-based folding\n\n"
            "Exposes tools: cognifold_remember, cognifold_query, "
            "cognifold_graph_stats, cognifold_list_intents.",
        )
        return

    if "--version" in args:
        from cognifold import __version__

        print(f"cognifold-mcp {__version__}")
        return

    server = build_server()
    server.run()
