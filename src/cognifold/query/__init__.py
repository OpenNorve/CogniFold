"""Memory Query Interface for Cognifold.

This module provides the read/query capability for the memory system,
allowing agents to retrieve relevant context from the concept graph.

Key components:
- QueryType: Enum for query modes (semantic, temporal, structural, hybrid)
- QueryResult: Result container with context, nodes, and traversal path
- NodeSummary: Summary of a retrieved node with relevance score
- QueryConfig: Configuration for query processing
- MemoryQueryAgent: Main agent for processing queries

Example:
    >>> from cognifold.graph.store import ConceptGraph
    >>> from cognifold.query import MemoryQueryAgent, QueryType
    >>>
    >>> graph = ConceptGraph()
    >>> # ... populate graph ...
    >>> agent = MemoryQueryAgent(graph)
    >>> result = agent.query("What patterns exist?", QueryType.HYBRID)
    >>> print(result.context)
"""

from __future__ import annotations

from cognifold.query.agent import MemoryQueryAgent
from cognifold.query.models import (
    NodeSummary,
    QueryConfig,
    QueryResult,
    QueryType,
    RetrievalMode,
)

__all__ = [
    "MemoryQueryAgent",
    "NodeSummary",
    "QueryConfig",
    "QueryResult",
    "QueryType",
    "RetrievalMode",
]
