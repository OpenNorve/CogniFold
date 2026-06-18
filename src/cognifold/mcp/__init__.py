"""Model Context Protocol (MCP) server for CogniFold.

Exposes CogniFold's persistent concept-graph memory as an MCP server so it
plugs into Claude Code, Claude Desktop, Cursor, and any other MCP client.

The server wraps the existing in-process logic (``Pipeline`` for ingestion,
``MemoryQueryAgent`` for retrieval, ``ConceptGraph`` metrics for stats). It does
not reimplement memory — it is a thin adapter over the same code paths the
HTTP service uses.

Run it as a stdio server::

    python -m cognifold.mcp
    # or, if installed with the console script:
    cognifold-mcp

Install the optional dependency first::

    pip install 'cognifold[mcp]'
"""

from __future__ import annotations

from cognifold.mcp.server import build_server, main

__all__ = ["build_server", "main"]
