"""Query-specific node relevance scoring.

This module provides scoring functions that consider both graph structure
and query intent when ranking nodes for retrieval.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from cognifold.query.config import (
    DEPTH_PENALTY_FACTOR,
    NON_MATCH_PENALTY,
    RELEVANCE_BOOSTS,
    apply_type_boost,
)
from cognifold.query.models import NodeSummary, QueryConfig, QueryType
from cognifold.query.strategies import TraversalResult
from cognifold.query.text_utils import compute_text_similarity

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph
    from cognifold.models.node import Node


class QueryScorer:
    """Scores nodes for relevance to a specific query."""

    def __init__(
        self,
        graph: ConceptGraph,
        config: QueryConfig | None = None,
    ) -> None:
        """Initialize the scorer.

        Args:
            graph: The concept graph being queried.
            config: Query configuration.
        """
        self.graph = graph
        self.config = config or QueryConfig()

    def score_traversal_results(
        self,
        traversal: TraversalResult,
        query_type: QueryType,
        reference_time: datetime | None = None,
        query_text: str | None = None,
    ) -> list[NodeSummary]:
        """Score and convert traversal results to NodeSummary objects.

        Args:
            traversal: Result from graph traversal.
            query_type: Type of query for scoring adjustments.
            reference_time: Reference time for temporal scoring.
            query_text: Original query text for keyword matching.

        Returns:
            List of NodeSummary objects sorted by relevance.
        """
        if reference_time is None:
            reference_time = datetime.now()

        summaries: list[NodeSummary] = []

        for node_id, depth, base_score in traversal.visited_nodes:
            node = self.graph.get_node_or_none(node_id)
            if node is None:
                continue

            # Compute final relevance score
            relevance = self._compute_relevance(
                node, depth, base_score, query_type, reference_time, query_text
            )

            # Skip nodes below threshold
            if relevance < self.config.min_relevance_score:
                continue

            # Create summary
            summary = self._node_to_summary(node, relevance)
            summaries.append(summary)

        # Sort by relevance descending
        summaries.sort(key=lambda s: s.relevance_score, reverse=True)

        # Limit to max_nodes
        return summaries[: self.config.max_nodes]

    def _compute_relevance(
        self,
        node: Node,
        depth: int,
        base_score: float,
        query_type: QueryType,
        reference_time: datetime,
        query_text: str | None = None,
    ) -> float:
        """Compute final relevance score for a node.

        Args:
            node: The node to score.
            depth: Traversal depth from entry point.
            base_score: Initial score from traversal.
            query_type: Type of query.
            reference_time: Reference time.
            query_text: Original query for text matching.

        Returns:
            Final relevance score between 0.0 and 1.0.
        """
        relevance = base_score

        # Text matching boost - reinforces relevance for matching nodes
        # Entry points are now selected by text search, so this is a secondary boost
        if query_text:
            text_score = self._compute_text_match_score(node, query_text)
            if text_score > 0:
                # Boost for matching nodes (up to 2x)
                relevance *= 1.0 + text_score
            elif query_type in (QueryType.SEMANTIC, QueryType.HYBRID):
                # Slight penalty for non-matching nodes discovered via traversal
                # These are connected to matching nodes, so still somewhat relevant
                relevance *= NON_MATCH_PENALTY

        # Apply type-based adjustments
        if self.config.prefer_concepts:
            relevance = apply_type_boost(relevance, node.type, RELEVANCE_BOOSTS)

        # Depth penalty (already applied in traversal, but add slight additional)
        depth_penalty = DEPTH_PENALTY_FACTOR**depth
        relevance *= depth_penalty

        # Query-type specific adjustments
        if query_type == QueryType.STRUCTURAL:
            # Structural queries favor highly connected nodes
            # Boost nodes with more edges
            neighbor_count = len(list(self.graph.get_neighbors(node.id)))
            predecessor_count = len(list(self.graph.get_predecessors(node.id)))
            connection_count = neighbor_count + predecessor_count
            if connection_count > 0:
                connection_boost = min(1.5, 1.0 + connection_count * 0.1)
                relevance *= connection_boost

        elif query_type == QueryType.TEMPORAL:
            # Temporal queries favor recent nodes
            time_diff = (reference_time - node.last_accessed).total_seconds() / 3600
            time_diff = max(0, time_diff)  # Handle future timestamps
            recency_factor = 1.0 / (1.0 + time_diff * 0.1)
            relevance *= recency_factor

        elif query_type == QueryType.SEMANTIC:
            # Semantic queries favor nodes with reasoning/explanation
            if node.reasoning:
                relevance *= 1.2
            if node.grounded_in:
                relevance *= 1.1

        # Normalize to 0-1 range
        relevance = min(1.0, max(0.0, relevance))

        return relevance

    def _compute_text_match_score(self, node: Node, query_text: str) -> float:
        """Compute text matching score between query and node content.

        Args:
            node: The node to match against.
            query_text: The query string.

        Returns:
            Match score between 0.0 and 1.0.
        """
        # Combine node text fields
        text_parts = []

        title = node.data.get("title", "")
        if title:
            text_parts.append(str(title))

        description = node.data.get("description", "")
        if description:
            text_parts.append(str(description))

        if node.reasoning:
            text_parts.append(node.reasoning)

        node_text = " ".join(text_parts)
        if not node_text:
            return 0.0

        return compute_text_similarity(query_text, node_text)

    def node_to_summary(self, node: Node, relevance: float) -> NodeSummary:
        """Convert a Node to a NodeSummary.

        Public API for external callers (e.g. MemoryQueryAgent).

        Args:
            node: The node to convert.
            relevance: Computed relevance score.

        Returns:
            NodeSummary object.
        """
        return self._node_to_summary(node, relevance)

    def _node_to_summary(self, node: Node, relevance: float) -> NodeSummary:
        """Convert a Node to a NodeSummary (internal).

        Args:
            node: The node to convert.
            relevance: Computed relevance score.

        Returns:
            NodeSummary object.
        """
        # Extract title and description from data
        title = node.data.get("title", node.data.get("event_type", f"Node {node.id}"))
        description = node.data.get("description")

        return NodeSummary(
            node_id=node.id,
            node_type=node.type.value,
            title=str(title),
            description=str(description) if description else None,
            relevance_score=relevance,
            reasoning=node.reasoning if self.config.include_reasoning else None,
            grounded_in=node.grounded_in if self.config.include_grounding else [],
            created_at=node.created_at,
            data=node.data,
        )

    def rank_nodes_for_query(
        self,
        nodes: list[Node],
        query_type: QueryType,
        reference_time: datetime | None = None,
    ) -> list[NodeSummary]:
        """Rank a list of nodes for a query.

        Convenience method for ranking pre-selected nodes.

        Args:
            nodes: Nodes to rank.
            query_type: Type of query.
            reference_time: Reference time.

        Returns:
            Sorted list of NodeSummary objects.
        """
        if reference_time is None:
            reference_time = datetime.now()

        summaries: list[NodeSummary] = []
        for i, node in enumerate(nodes):
            # Use position as base score
            base_score = 1.0 - (i * 0.1)
            base_score = max(0.1, base_score)

            relevance = self._compute_relevance(node, 0, base_score, query_type, reference_time)

            if relevance >= self.config.min_relevance_score:
                summary = self._node_to_summary(node, relevance)
                summaries.append(summary)

        summaries.sort(key=lambda s: s.relevance_score, reverse=True)
        return summaries[: self.config.max_nodes]
