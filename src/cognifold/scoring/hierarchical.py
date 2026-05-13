"""Hierarchical context window selection for Phase 9.2.

Provides multi-level context with different granularities:
- Immediate: Recent events, high-urgency intents (focus here)
- Working: Active concepts, related patterns (broader context)
- Background: Historical context, weak signals (reference)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from cognifold.scoring.ranker import ContextRanker, NodeScore, ScoringConfig

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph
    from cognifold.models.node import Edge, Node


@dataclass
class HierarchicalContextConfig:
    """Configuration for hierarchical context selection.

    Level sizes can be specified as proportions (0.0-1.0) of total context window,
    or as explicit sizes. Explicit sizes take precedence if set.
    """

    # Total context window size
    total_size: int = 100

    # Level proportions (sum should leave ~10% buffer for threshold filtering)
    immediate_proportion: float = 0.10  # 10% of context window
    working_proportion: float = 0.30  # 30% of context window
    background_proportion: float = 0.50  # 50% of context window

    # Explicit sizes (override proportions if set)
    immediate_size: int | None = None
    working_size: int | None = None
    background_size: int | None = None

    # Relevance threshold (don't include nodes below this)
    relevance_threshold: float = 0.1

    # Immediate level weights (favor recency and urgency)
    immediate_recency_weight: float = 0.7
    immediate_urgency_weight: float = 0.3

    # Working level weights (balance PageRank, recency, and type)
    working_pagerank_weight: float = 0.5
    working_recency_weight: float = 0.3
    working_type_weight: float = 0.2  # Favor concepts

    # Background level weights (favor PageRank and diversity)
    background_pagerank_weight: float = 0.8
    background_diversity_weight: float = 0.2

    def get_immediate_size(self) -> int:
        """Get the immediate level size."""
        if self.immediate_size is not None:
            return self.immediate_size
        return int(self.total_size * self.immediate_proportion)

    def get_working_size(self) -> int:
        """Get the working level size."""
        if self.working_size is not None:
            return self.working_size
        return int(self.total_size * self.working_proportion)

    def get_background_size(self) -> int:
        """Get the background level size."""
        if self.background_size is not None:
            return self.background_size
        return int(self.total_size * self.background_proportion)


@dataclass
class ContextLevel:
    """A single level of the hierarchical context.

    Contains nodes and all edges connected to those nodes.
    """

    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    node_scores: dict[str, float] = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        """Number of nodes in this level."""
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges in this level."""
        return len(self.edges)

    @property
    def node_ids(self) -> set[str]:
        """Set of node IDs in this level."""
        return {n.id for n in self.nodes}


@dataclass
class HierarchicalContext:
    """Multi-level hierarchical context.

    Contains three levels with deduplication (each node appears in at most one level).
    """

    immediate: ContextLevel = field(default_factory=ContextLevel)
    working: ContextLevel = field(default_factory=ContextLevel)
    background: ContextLevel = field(default_factory=ContextLevel)

    @property
    def total_nodes(self) -> int:
        """Total nodes across all levels."""
        return self.immediate.node_count + self.working.node_count + self.background.node_count

    @property
    def total_edges(self) -> int:
        """Total edges across all levels."""
        return self.immediate.edge_count + self.working.edge_count + self.background.edge_count

    @property
    def all_node_ids(self) -> set[str]:
        """All node IDs across all levels."""
        return self.immediate.node_ids | self.working.node_ids | self.background.node_ids

    def get_level(self, level_name: str) -> ContextLevel:
        """Get a level by name."""
        if level_name == "immediate":
            return self.immediate
        elif level_name == "working":
            return self.working
        elif level_name == "background":
            return self.background
        else:
            raise ValueError(f"Unknown level: {level_name}")


@dataclass
class ContextMetrics:
    """Metrics for tracking hierarchical context performance."""

    # Selection metrics
    immediate_selected: int = 0
    working_selected: int = 0
    background_selected: int = 0
    nodes_below_threshold: int = 0

    # Contribution tracking (updated after plan execution)
    immediate_contributed: int = 0  # Nodes from immediate level used in plan
    working_contributed: int = 0
    background_contributed: int = 0

    def record_selection(self, context: HierarchicalContext, total_candidates: int) -> None:
        """Record selection metrics from a hierarchical context."""
        self.immediate_selected = context.immediate.node_count
        self.working_selected = context.working.node_count
        self.background_selected = context.background.node_count
        self.nodes_below_threshold = (
            total_candidates
            - self.immediate_selected
            - self.working_selected
            - self.background_selected
        )

    def record_contribution(self, plan_node_ids: set[str], context: HierarchicalContext) -> None:
        """Record which levels contributed to the plan."""
        self.immediate_contributed = len(plan_node_ids & context.immediate.node_ids)
        self.working_contributed = len(plan_node_ids & context.working.node_ids)
        self.background_contributed = len(plan_node_ids & context.background.node_ids)


class HierarchicalContextSelector:
    """Selects nodes for hierarchical context with level-specific scoring.

    Uses different scoring weights for each level:
    - Immediate: Favor recency and urgency
    - Working: Balance PageRank, recency, and node type (favor concepts)
    - Background: Favor PageRank and diversity
    """

    def __init__(
        self,
        config: HierarchicalContextConfig | None = None,
        scoring_config: ScoringConfig | None = None,
    ) -> None:
        """Initialize with configuration.

        Args:
            config: Hierarchical context configuration.
            scoring_config: Base scoring configuration for the ranker.
        """
        self.config = config or HierarchicalContextConfig()
        self.scoring_config = scoring_config or ScoringConfig()
        self._ranker = ContextRanker(self.scoring_config)
        self._metrics = ContextMetrics()

    @property
    def metrics(self) -> ContextMetrics:
        """Get the current metrics."""
        return self._metrics

    def select_context(
        self,
        graph: ConceptGraph,
        reference_time: datetime | None = None,
    ) -> HierarchicalContext:
        """Select hierarchical context from the graph.

        Nodes are deduplicated to the highest priority level they qualify for.

        Args:
            graph: The concept graph.
            reference_time: Reference time for recency/urgency calculations.

        Returns:
            HierarchicalContext with immediate, working, and background levels.
        """
        if graph.node_count == 0:
            return HierarchicalContext()

        # Get base scores for all nodes
        all_scores = self._ranker.score_nodes(graph, reference_time)
        all_nodes = graph.get_all_nodes()
        node_map = {n.id: n for n in all_nodes}

        selected_ids: set[str] = set()

        # 1. Select Immediate (highest priority)
        immediate_candidates = self._score_for_immediate(all_scores, reference_time)
        immediate_nodes, immediate_node_scores = self._select_above_threshold(
            immediate_candidates,
            node_map,
            max_size=self.config.get_immediate_size(),
        )
        selected_ids.update(n.id for n in immediate_nodes)

        # 2. Select Working (exclude Immediate)
        remaining_scores = [s for s in all_scores if s.node_id not in selected_ids]
        working_candidates = self._score_for_working(remaining_scores, node_map)
        working_nodes, working_node_scores = self._select_above_threshold(
            working_candidates,
            node_map,
            max_size=self.config.get_working_size(),
        )
        selected_ids.update(n.id for n in working_nodes)

        # 3. Select Background (exclude Immediate + Working)
        remaining_scores = [s for s in all_scores if s.node_id not in selected_ids]
        background_candidates = self._score_for_background(remaining_scores)
        background_nodes, background_node_scores = self._select_above_threshold(
            background_candidates,
            node_map,
            max_size=self.config.get_background_size(),
        )

        # 4. Collect edges for each level
        immediate_edges = self._collect_edges_for_nodes(graph, immediate_nodes)
        working_edges = self._collect_edges_for_nodes(graph, working_nodes)
        background_edges = self._collect_edges_for_nodes(graph, background_nodes)

        context = HierarchicalContext(
            immediate=ContextLevel(
                nodes=immediate_nodes,
                edges=immediate_edges,
                node_scores=immediate_node_scores,
            ),
            working=ContextLevel(
                nodes=working_nodes,
                edges=working_edges,
                node_scores=working_node_scores,
            ),
            background=ContextLevel(
                nodes=background_nodes,
                edges=background_edges,
                node_scores=background_node_scores,
            ),
        )

        # Record metrics
        self._metrics.record_selection(context, len(all_nodes))

        return context

    def _score_for_immediate(
        self,
        scores: list[NodeScore],
        reference_time: datetime | None,
    ) -> list[tuple[str, float]]:
        """Score nodes for immediate level (favor recency and urgency).

        Args:
            scores: Base scores for all nodes.
            reference_time: Reference time for calculations.

        Returns:
            List of (node_id, immediate_score) tuples, sorted descending.
        """
        results: list[tuple[str, float]] = []

        for score in scores:
            # Immediate score = recency_weight * recency + urgency_weight * (urgency - 1)
            # Note: urgency_score is 1.0 when no urgency, so we subtract 1 to get boost
            immediate_score = (
                self.config.immediate_recency_weight * score.recency_score
                + self.config.immediate_urgency_weight * (score.urgency_score - 1.0)
            )
            results.append((score.node_id, immediate_score))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def _score_for_working(
        self,
        scores: list[NodeScore],
        node_map: dict[str, Node],
    ) -> list[tuple[str, float]]:
        """Score nodes for working level (balance PageRank, recency, type).

        Favors concepts over events for working memory.

        Args:
            scores: Base scores for remaining nodes.
            node_map: Map of node ID to Node.

        Returns:
            List of (node_id, working_score) tuples, sorted descending.
        """
        from cognifold.models.node import NodeType

        results: list[tuple[str, float]] = []

        for score in scores:
            node = node_map.get(score.node_id)
            if node is None:
                continue

            # Type bonus: concepts get higher score for working memory
            type_bonus = 0.0
            if node.type == NodeType.CONCEPT:
                type_bonus = 1.0
            elif node.type == NodeType.INTENT:
                type_bonus = 0.8

            working_score = (
                self.config.working_pagerank_weight * score.structural_rank
                + self.config.working_recency_weight * score.recency_score
                + self.config.working_type_weight * type_bonus
            )
            results.append((score.node_id, working_score))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def _score_for_background(
        self,
        scores: list[NodeScore],
    ) -> list[tuple[str, float]]:
        """Score nodes for background level (favor PageRank).

        Args:
            scores: Base scores for remaining nodes.

        Returns:
            List of (node_id, background_score) tuples, sorted descending.
        """
        results: list[tuple[str, float]] = []

        for score in scores:
            # Background score primarily based on structural importance
            # Diversity weight could be used for type distribution in future
            background_score = (
                self.config.background_pagerank_weight * score.structural_rank
                + self.config.background_diversity_weight * score.access_score
            )
            results.append((score.node_id, background_score))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def _select_above_threshold(
        self,
        candidates: list[tuple[str, float]],
        node_map: dict[str, Node],
        max_size: int,
    ) -> tuple[list[Node], dict[str, float]]:
        """Select nodes above relevance threshold, up to max size.

        Args:
            candidates: List of (node_id, score) tuples, sorted descending.
            node_map: Map of node ID to Node.
            max_size: Maximum number of nodes to select.

        Returns:
            Tuple of (selected nodes, node scores dict).
        """
        nodes: list[Node] = []
        node_scores: dict[str, float] = {}

        for node_id, score in candidates:
            if score < self.config.relevance_threshold:
                break  # Sorted descending, so all remaining are below threshold

            if len(nodes) >= max_size:
                break

            node = node_map.get(node_id)
            if node is not None:
                nodes.append(node)
                node_scores[node_id] = score

        return nodes, node_scores

    def _collect_edges_for_nodes(
        self,
        graph: ConceptGraph,
        nodes: list[Node],
    ) -> list[Edge]:
        """Collect all edges connected to the given nodes.

        Includes both incoming and outgoing edges. Edges are "free" and
        don't count against level size limits.

        Args:
            graph: The concept graph.
            nodes: Nodes to collect edges for.

        Returns:
            List of edges connected to the nodes.
        """
        edges: list[Edge] = []
        seen_edge_keys: set[tuple[str, str, str | None]] = set()

        for node in nodes:
            # Outgoing edges
            for neighbor_id in graph.get_neighbors(node.id):
                for edge in graph.get_edges_between(node.id, neighbor_id):
                    if edge.edge_key not in seen_edge_keys:
                        edges.append(edge)
                        seen_edge_keys.add(edge.edge_key)

            # Incoming edges
            for predecessor_id in graph.get_predecessors(node.id):
                for edge in graph.get_edges_between(predecessor_id, node.id):
                    if edge.edge_key not in seen_edge_keys:
                        edges.append(edge)
                        seen_edge_keys.add(edge.edge_key)

        return edges
