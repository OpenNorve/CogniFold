"""Relevance scoring and context window selection."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import networkx as nx

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph
    from cognifold.models.node import Node


@dataclass(frozen=True)
class ScoringConfig:
    """Configuration for relevance scoring.

    Attributes:
        alpha: Weight for structural rank (PageRank). Default 0.4.
        beta: Weight for recency score. Default 0.4.
        gamma: Weight for access frequency score. Default 0.2.
        decay_rate: Exponential decay rate per hour for recency. Default 0.01.
        edge_decay_rate: Exponential decay rate per hour for edge recency. Default 0.005.
        context_window_size: Maximum nodes in context window. Default 50.
        min_score_threshold: Minimum score to be included. Default 0.01.
        urgency_boost: Max multiplier for nodes connected to approaching TIME nodes. Default 2.0.
        urgency_window_hours: Hours before deadline when urgency starts increasing. Default 24.
        use_weighted_pagerank: Whether to use edge weights in PageRank. Default True.
    """

    alpha: float = 0.4
    beta: float = 0.4
    gamma: float = 0.2
    decay_rate: float = 0.01
    edge_decay_rate: float = 0.005  # Slower decay for edges
    context_window_size: int = 50
    min_score_threshold: float = 0.01
    urgency_boost: float = 2.0
    urgency_window_hours: float = 24.0
    use_weighted_pagerank: bool = True

    def __post_init__(self) -> None:
        """Validate that weights sum to 1.0."""
        total = self.alpha + self.beta + self.gamma
        if not math.isclose(total, 1.0, rel_tol=1e-9):
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class NodeScore:
    """Score components for a single node.

    Attributes:
        node_id: The node's ID.
        structural_rank: PageRank-based importance.
        recency_score: Time-decay based freshness.
        access_score: Normalized access frequency.
        urgency_score: Boost from connected TIME nodes (1.0 = no boost).
        composite_score: Weighted combination of all scores.
    """

    node_id: str
    structural_rank: float
    recency_score: float
    access_score: float
    urgency_score: float
    composite_score: float


class ContextRanker:
    """Computes relevance scores and selects context window nodes.

    Uses a composite scoring formula:
        Score = alpha * StructuralRank + beta * RecencyScore + gamma * AccessScore

    Where:
        - StructuralRank: PageRank on graph topology
        - RecencyScore: exp(-decay_rate * hours_since_last_update)
        - AccessScore: access_count / max_access_count (normalized)
    """

    def __init__(self, config: ScoringConfig | None = None) -> None:
        """Initialize the ranker with optional configuration."""
        from cognifold.scoring.cache import PageRankCache

        self.config = config or ScoringConfig()
        self._pagerank_cache = PageRankCache()

    def invalidate_cache(self) -> None:
        """Invalidate the PageRank cache (e.g. after graph replacement)."""
        self._pagerank_cache.invalidate()

    @staticmethod
    def compute_adaptive_alpha(graph: ConceptGraph) -> float:
        """Compute PPR damping factor.

        Uses fixed alpha=0.85 (standard PageRank default) for stability.
        Adaptive alpha (0.75 for sparse, 0.92 for dense) was tested but
        caused regressions on sparse-graph benchmarks like BABILong.

        Args:
            graph: The concept graph (unused, kept for API compat).

        Returns:
            Damping factor of 0.85.
        """
        return 0.85

    def compute_pagerank(
        self,
        graph: ConceptGraph,
        reference_time: datetime | None = None,
    ) -> dict[str, float]:
        """Compute PageRank scores for all nodes.

        When use_weighted_pagerank is True (default), computes effective edge weights:
            effective_weight(e) = e.weight * recency_factor(e.created_at)

        This allows strongly-typed edges (high weight) to contribute more to
        PageRank, while also considering how recently the edge was created.

        Uses adaptive damping factor based on graph density:
        - Sparse (edge/node < 2.0): alpha = 0.75 (wider diffusion)
        - Dense (edge/node > 5.0): alpha = 0.92 (tighter focus)
        - Default: alpha = 0.85

        Args:
            graph: The concept graph to analyze.
            reference_time: Reference time for edge recency calculation.

        Returns:
            Dictionary mapping node IDs to PageRank scores.
        """
        if graph.node_count == 0:
            return {}

        if reference_time is None:
            reference_time = datetime.now()

        alpha = self.compute_adaptive_alpha(graph)
        nx_graph = graph.internal_graph

        if not self.config.use_weighted_pagerank or graph.edge_count == 0:
            # Standard unweighted PageRank with adaptive alpha
            return nx.pagerank(nx_graph, alpha=alpha)

        # Compute effective weights for each edge
        # effective_weight = edge.weight * recency_factor(edge.created_at)
        self._compute_effective_edge_weights(nx_graph, reference_time)

        # Use weighted PageRank with adaptive alpha
        return nx.pagerank(nx_graph, alpha=alpha, weight="effective_weight")

    def _compute_effective_edge_weights(
        self,
        nx_graph: nx.MultiDiGraph[Any],
        reference_time: datetime,
    ) -> None:
        """Compute effective weights for all edges in the graph.

        For each edge:
            effective_weight = base_weight * recency_factor

        Where:
            - base_weight: The edge's semantic weight (0.0 to 1.0)
            - recency_factor: exp(-edge_decay_rate * hours_since_created)

        Args:
            nx_graph: The NetworkX graph to update.
            reference_time: Reference time for recency calculation.
        """
        # Handle timezone for reference time
        ref_time = reference_time
        if ref_time.tzinfo is not None:
            ref_time = ref_time.replace(tzinfo=None)

        for source, target, key, attrs in nx_graph.edges(keys=True, data=True):
            # Get base weight (default 1.0 for legacy edges)
            base_weight = attrs.get("weight", 1.0)

            # Compute recency factor from edge creation time
            created_at_str = attrs.get("created_at")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                    if created_at.tzinfo is not None:
                        created_at = created_at.replace(tzinfo=None)

                    hours_elapsed = (ref_time - created_at).total_seconds() / 3600.0
                    hours_elapsed = max(0.0, hours_elapsed)
                    # Ablation hook: COGNIFOLD_ABLATE_DECAY=1 disables edge decay
                    # (recency_factor=1.0 regardless of age). Used by ablation
                    # benchmark runs; production/normal runs leave it unset.
                    if os.environ.get("COGNIFOLD_ABLATE_DECAY") == "1":
                        recency_factor = 1.0
                    else:
                        recency_factor = math.exp(-self.config.edge_decay_rate * hours_elapsed)
                except (ValueError, TypeError):
                    recency_factor = 1.0
            else:
                recency_factor = 1.0

            # Compute and store effective weight
            effective_weight = base_weight * recency_factor
            nx_graph[source][target][key]["effective_weight"] = effective_weight

    def compute_recency_score(self, node: Node, reference_time: datetime | None = None) -> float:
        """Compute recency score using exponential decay.

        Score = exp(-decay_rate * hours_since_last_access)

        Args:
            node: The node to score.
            reference_time: The reference time (defaults to now).

        Returns:
            Recency score between 0 and 1.
        """
        if reference_time is None:
            reference_time = datetime.now()

        # Handle timezone-aware vs naive datetime comparison
        ref_time = reference_time
        node_time = node.last_accessed

        # If one is aware and the other is naive, convert to naive
        if ref_time.tzinfo is not None and node_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=None)
        elif ref_time.tzinfo is None and node_time.tzinfo is not None:
            node_time = node_time.replace(tzinfo=None)

        time_delta = ref_time - node_time
        hours_elapsed = time_delta.total_seconds() / 3600.0

        # Clamp to non-negative (in case of clock skew)
        hours_elapsed = max(0.0, hours_elapsed)

        return math.exp(-self.config.decay_rate * hours_elapsed)

    def compute_access_score(self, node: Node, max_access_count: int) -> float:
        """Compute normalized access frequency score.

        Score = access_count / max_access_count

        Args:
            node: The node to score.
            max_access_count: Maximum access count in the graph.

        Returns:
            Access score between 0 and 1.
        """
        if max_access_count == 0:
            return 0.0
        return node.access_count / max_access_count

    def compute_urgency_score(
        self,
        node: Node,
        graph: ConceptGraph,
        reference_time: datetime | None = None,
    ) -> float:
        """Compute urgency boost from connected TIME nodes.

        Finds all TIME nodes connected to this node (via edges in either direction)
        and computes urgency based on proximity to their scheduled times.

        Urgency increases as the deadline approaches:
        - Outside urgency_window_hours: urgency = 1.0 (no boost)
        - At deadline: urgency = urgency_boost (max boost)
        - Linear interpolation in between

        Args:
            node: The node to score.
            graph: The concept graph for traversal.
            reference_time: The reference time (defaults to now).

        Returns:
            Urgency multiplier (1.0 = no boost, up to urgency_boost).
        """
        from cognifold.models.node import NodeType

        if reference_time is None:
            reference_time = datetime.now()

        # Find connected TIME nodes (neighbors and predecessors)
        connected_ids = set(graph.get_neighbors(node.id)) | set(graph.get_predecessors(node.id))

        max_urgency = 1.0  # No boost by default

        for connected_id in connected_ids:
            connected_node = graph.get_node_or_none(connected_id)
            if connected_node is None or connected_node.type != NodeType.TIME:
                continue

            # Get the scheduled time from node data
            scheduled_time_str = connected_node.data.get("scheduled_time")
            if not scheduled_time_str:
                continue

            try:
                # Parse the scheduled time
                if isinstance(scheduled_time_str, str):
                    scheduled_time = datetime.fromisoformat(
                        scheduled_time_str.replace("Z", "+00:00")
                    )
                else:
                    continue

                # Handle timezone-aware vs naive datetime comparison
                ref_time = reference_time
                sched_time = scheduled_time
                if ref_time.tzinfo is not None and sched_time.tzinfo is None:
                    ref_time = ref_time.replace(tzinfo=None)
                elif ref_time.tzinfo is None and sched_time.tzinfo is not None:
                    sched_time = sched_time.replace(tzinfo=None)

                # Calculate hours until deadline
                time_delta = sched_time - ref_time
                hours_until = time_delta.total_seconds() / 3600.0

                # If deadline has passed, no urgency boost
                if hours_until < 0:
                    continue

                # Calculate urgency based on proximity
                if hours_until >= self.config.urgency_window_hours:
                    urgency = 1.0  # Outside window, no boost
                else:
                    # Linear interpolation from 1.0 to urgency_boost as deadline approaches
                    progress = 1.0 - (hours_until / self.config.urgency_window_hours)
                    urgency = 1.0 + progress * (self.config.urgency_boost - 1.0)

                max_urgency = max(max_urgency, urgency)

            except (ValueError, TypeError):
                # Skip if scheduled_time is malformed
                continue

        return max_urgency

    def score_nodes(
        self,
        graph: ConceptGraph,
        reference_time: datetime | None = None,
    ) -> list[NodeScore]:
        """Compute composite scores for all nodes.

        The composite score is:
            base_score = alpha * structural + beta * recency + gamma * access
            composite = base_score * urgency

        Where urgency is a multiplier (1.0 to urgency_boost) based on
        proximity to connected TIME nodes.

        Args:
            graph: The concept graph to analyze.
            reference_time: Reference time for recency calculation.

        Returns:
            List of NodeScore objects sorted by composite score (descending).
        """
        if graph.node_count == 0:
            return []

        # Use cached PageRank (avoids duplicate computation when
        # get_context_node_ids and score_nodes are called back-to-back)
        pagerank_scores = self._pagerank_cache.get_or_compute(graph, self, reference_time)

        # Get all nodes and find max access count
        nodes = graph.get_all_nodes()
        max_access = max((n.access_count for n in nodes), default=0)

        # Compute scores for each node
        scores: list[NodeScore] = []
        for node in nodes:
            structural = pagerank_scores.get(node.id, 0.0)
            recency = self.compute_recency_score(node, reference_time)
            access = self.compute_access_score(node, max_access)
            urgency = self.compute_urgency_score(node, graph, reference_time)

            # Base score from standard weights
            base_score = (
                self.config.alpha * structural
                + self.config.beta * recency
                + self.config.gamma * access
            )

            # Apply urgency as a multiplier
            composite = base_score * urgency

            scores.append(
                NodeScore(
                    node_id=node.id,
                    structural_rank=structural,
                    recency_score=recency,
                    access_score=access,
                    urgency_score=urgency,
                    composite_score=composite,
                )
            )

        # Sort by composite score descending
        scores.sort(key=lambda s: s.composite_score, reverse=True)
        return scores

    def select_context_window(
        self,
        graph: ConceptGraph,
        reference_time: datetime | None = None,
    ) -> list[NodeScore]:
        """Select top-k nodes for the context window.

        Applies both the size limit and minimum score threshold.

        Args:
            graph: The concept graph to analyze.
            reference_time: Reference time for recency calculation.

        Returns:
            List of NodeScore objects for context window nodes.
        """
        all_scores = self.score_nodes(graph, reference_time)

        # Filter by minimum threshold
        filtered = [s for s in all_scores if s.composite_score >= self.config.min_score_threshold]

        # Take top-k
        return filtered[: self.config.context_window_size]

    def get_context_node_ids(
        self,
        graph: ConceptGraph,
        reference_time: datetime | None = None,
    ) -> list[str]:
        """Get just the node IDs for the context window.

        Convenience method that returns only IDs.

        Args:
            graph: The concept graph to analyze.
            reference_time: Reference time for recency calculation.

        Returns:
            List of node IDs in the context window.
        """
        window = self.select_context_window(graph, reference_time)
        return [s.node_id for s in window]

    def compute_personalized_pagerank(
        self,
        graph: ConceptGraph,
        entry_points: list[str],
        reference_time: datetime | None = None,
    ) -> dict[str, float]:
        """Compute Personalized PageRank seeded from query-relevant entry points.

        Unlike standard PageRank which distributes uniformly, PPR biases the
        random walk toward the seed nodes. This makes query-relevant parts of
        the graph score higher — critical for multi-hop QA (MuSiQue, BABILong).

        Args:
            graph: The concept graph to analyze.
            entry_points: Node IDs to seed the personalization vector.
            reference_time: Reference time for edge recency weighting.

        Returns:
            Dictionary mapping node IDs to PPR scores.
        """
        if graph.node_count == 0 or not entry_points:
            return {}

        if reference_time is None:
            reference_time = datetime.now()

        nx_graph = graph.internal_graph

        # Build personalization vector: seed on entry points
        all_nodes = list(nx_graph.nodes())
        personalization = dict.fromkeys(all_nodes, 0.0)
        valid_seeds = [ep for ep in entry_points if ep in personalization]
        if not valid_seeds:
            return nx.pagerank(nx_graph, alpha=0.85)

        seed_weight = 1.0 / len(valid_seeds)
        for ep in valid_seeds:
            personalization[ep] = seed_weight

        # Use weighted edges if available
        if self.config.use_weighted_pagerank and graph.edge_count > 0:
            self._compute_effective_edge_weights(nx_graph, reference_time)
            return nx.pagerank(
                nx_graph,
                alpha=0.85,
                personalization=personalization,
                weight="effective_weight",
            )

        return nx.pagerank(nx_graph, alpha=0.85, personalization=personalization)
