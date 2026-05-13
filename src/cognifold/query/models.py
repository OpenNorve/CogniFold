"""Data models for the Memory Query Interface.

This module defines the core data structures for querying the concept graph:
- QueryType: Enum for different query strategies
- RetrievalMode: Enum for retrieval backend selection
- QueryConfig: Configuration for query processing
- NodeSummary: Summary of a retrieved node
- QueryResult: Complete result of a query
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class QueryType(str, Enum):
    """Types of queries for different retrieval strategies.

    Attributes:
        SEMANTIC: Find nodes related to query meaning (uses embeddings)
        TEMPORAL: Find nodes from specific time periods (recent, date ranges)
        STRUCTURAL: Find highly connected/important nodes (PageRank-based)
        HYBRID: Combine multiple strategies for best results (default)
    """

    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    STRUCTURAL = "structural"
    HYBRID = "hybrid"


class RetrievalMode(str, Enum):
    """Retrieval backend mode for entry point selection.

    Attributes:
        LEGACY: Use original keyword matching (no embeddings required)
        BM25: Use BM25 inverted index for keyword retrieval
        SEMANTIC: Use embedding-based semantic search
        HYBRID: Combine BM25 + semantic with RRF fusion (best quality)
    """

    LEGACY = "legacy"
    BM25 = "bm25"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class QueryConfig:
    """Configuration for query processing.

    Attributes:
        max_nodes: Maximum number of nodes to return. Default 20.
        max_context_chars: Maximum characters in assembled context. Default 6000.
        max_description_chars: Maximum characters per node description before truncation. Default 500.
        max_traversal_depth: Maximum graph traversal depth from entry points. Default 3.
        min_relevance_score: Minimum relevance score to include a node. Default 0.1.
        prefer_concepts: Weight concepts higher than raw events. Default True.
        include_reasoning: Include node reasoning in results. Default True.
        include_grounding: Include grounding references in results. Default True.
        domain: Domain hint for domain-specific query processing. Default None.
        speaker_aware: Enable speaker-aware retrieval for conversation benchmarks. Default False.
        use_llm_rerank: Use LLM to re-rank retrieved candidates. Default False.
        use_query_expansion: Expand query with synonyms and related terms. Default False.
        retrieval_mode: Backend for entry point selection. Default HYBRID (auto-degrades to BM25 if no embedder).
        semantic_weight: Weight for semantic scores in hybrid mode. Default 0.5.
        keyword_weight: Weight for BM25 scores in hybrid mode. Default 0.5.
        use_ppr: Use Personalized PageRank for query-aware scoring. Default True.
        adaptive_depth: Auto-adjust traversal depth based on graph density. Default True.
        adaptive_depth_max: Maximum traversal depth when adaptive. Default 5.
    """

    max_nodes: int = 20
    max_context_chars: int = 6000
    max_description_chars: int = 500
    max_traversal_depth: int = 3
    min_relevance_score: float = 0.1
    prefer_concepts: bool = True
    include_reasoning: bool = True
    include_grounding: bool = True
    domain: str | None = None
    speaker_aware: bool = False
    use_llm_rerank: bool = False
    use_query_expansion: bool = False
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    semantic_weight: float = 0.5
    keyword_weight: float = 0.5
    use_ppr: bool = True
    adaptive_depth: bool = True
    adaptive_depth_max: int = 5
    use_intent_routing: bool = True


@dataclass
class QueryIntent:
    """Parsed intent from a natural language query.

    Attributes:
        query_type: Inferred query type (semantic, temporal, structural, hybrid).
        key_topics: Main topics/concepts being queried.
        time_context: Temporal reference if present (e.g., "yesterday", "last week").
        scope: Query scope (broad, focused, deep).
        alternative_queries: Suggested alternative phrasings.
        speaker_filter: If query targets a specific speaker (for conversation benchmarks).
    """

    query_type: QueryType = QueryType.HYBRID
    key_topics: list[str] = field(default_factory=list)
    time_context: str | None = None
    scope: str = "focused"
    alternative_queries: list[str] = field(default_factory=list)
    speaker_filter: str | None = None


@dataclass
class NodeSummary:
    """Summary of a node retrieved by a query.

    Contains the essential information about a node along with
    its relevance score for the specific query.

    Attributes:
        node_id: Unique identifier of the node.
        node_type: Type of the node (event, concept, action, time).
        title: Title or short description of the node.
        description: Full description if available.
        relevance_score: Query-specific relevance score (0.0 to 1.0).
        reasoning: Why this node exists (for explainability).
        grounded_in: List of event IDs that justify this node.
        created_at: When the node was created.
        data: Additional node data.
    """

    node_id: str
    node_type: str
    title: str
    relevance_score: float
    description: str | None = None
    reasoning: str | None = None
    grounded_in: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_text(self, include_metadata: bool = False) -> str:
        """Convert to human-readable text format.

        Args:
            include_metadata: Include reasoning and grounding info.

        Returns:
            Formatted text representation.
        """
        lines = [f"[{self.node_type.upper()}] {self.title}"]

        if self.description:
            lines.append(f"  {self.description}")

        if include_metadata:
            if self.reasoning:
                lines.append(f"  Reasoning: {self.reasoning}")
            if self.grounded_in:
                lines.append(f"  Grounded in: {', '.join(self.grounded_in)}")

        return "\n".join(lines)


@dataclass
class QueryResult:
    """Result of a memory query.

    Contains the assembled context, retrieved nodes, and metadata
    about how the query was processed.

    Attributes:
        context: Formatted context text ready for LLM consumption.
        nodes: List of retrieved nodes with relevance scores.
        traversal_path: Node IDs in the order they were visited.
        query_metadata: Information about query processing.
        total_nodes_scanned: Total nodes examined during query.
        query_time_ms: Time taken to process query in milliseconds.
    """

    context: str
    nodes: list[NodeSummary]
    traversal_path: list[str] = field(default_factory=list)
    query_metadata: dict[str, Any] = field(default_factory=dict)
    total_nodes_scanned: int = 0
    query_time_ms: float = 0.0

    @property
    def node_count(self) -> int:
        """Number of nodes in the result."""
        return len(self.nodes)

    @property
    def context_length(self) -> int:
        """Length of the context string in characters."""
        return len(self.context)

    def get_node_ids(self) -> list[str]:
        """Get list of node IDs in the result."""
        return [node.node_id for node in self.nodes]

    def get_nodes_by_type(self, node_type: str) -> list[NodeSummary]:
        """Filter nodes by type.

        Args:
            node_type: Type to filter by (event, concept, action, time).

        Returns:
            List of nodes matching the type.
        """
        return [node for node in self.nodes if node.node_type == node_type]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "context": self.context,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type,
                    "title": n.title,
                    "description": n.description,
                    "relevance_score": n.relevance_score,
                    "reasoning": n.reasoning,
                    "grounded_in": n.grounded_in,
                }
                for n in self.nodes
            ],
            "traversal_path": self.traversal_path,
            "query_metadata": self.query_metadata,
            "total_nodes_scanned": self.total_nodes_scanned,
            "query_time_ms": self.query_time_ms,
        }
