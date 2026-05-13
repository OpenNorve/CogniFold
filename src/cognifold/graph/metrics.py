"""Quality metrics for graph integrity.

This module tracks quality metrics for the concept graph,
helping measure and improve graph health over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph
    from cognifold.graph.validator import ValidationReport


@dataclass
class QualityMetrics:
    """Quality metrics snapshot for a graph.

    Tracks various quality indicators that should be monitored:
    - orphan_rate: Percentage of non-event nodes without edges (target: 0%)
    - ungrounded_rate: Percentage of non-event nodes without grounding (target: 0%)
    - missing_reasoning_rate: Percentage of non-event nodes without reasoning
    - connectivity_violation_rate: Percentage of nodes violating connectivity rules
    - avg_reasoning_length: Average length of reasoning strings
    - concept_count: Number of concept nodes
    - action_count: Number of action nodes
    - event_count: Number of event nodes
    - time_count: Number of time nodes
    - edge_density: Edges per node ratio
    """

    total_nodes: int = 0
    event_count: int = 0
    concept_count: int = 0
    action_count: int = 0
    time_count: int = 0
    total_edges: int = 0

    orphan_count: int = 0
    ungrounded_count: int = 0
    missing_reasoning_count: int = 0
    connectivity_violations: int = 0

    avg_reasoning_length: float = 0.0
    min_reasoning_length: int = 0
    max_reasoning_length: int = 0

    @property
    def non_event_count(self) -> int:
        """Count of non-event nodes (concept + action + time)."""
        return self.concept_count + self.action_count + self.time_count

    @property
    def orphan_rate(self) -> float:
        """Percentage of non-event nodes that are orphans."""
        if self.non_event_count == 0:
            return 0.0
        return (self.orphan_count / self.non_event_count) * 100

    @property
    def ungrounded_rate(self) -> float:
        """Percentage of non-event nodes without grounding."""
        if self.non_event_count == 0:
            return 0.0
        return (self.ungrounded_count / self.non_event_count) * 100

    @property
    def missing_reasoning_rate(self) -> float:
        """Percentage of non-event nodes without reasoning."""
        if self.non_event_count == 0:
            return 0.0
        return (self.missing_reasoning_count / self.non_event_count) * 100

    @property
    def connectivity_violation_rate(self) -> float:
        """Percentage of nodes with connectivity violations."""
        if self.non_event_count == 0:
            return 0.0
        return (self.connectivity_violations / self.non_event_count) * 100

    @property
    def edge_density(self) -> float:
        """Edges per node ratio."""
        if self.total_nodes == 0:
            return 0.0
        return self.total_edges / self.total_nodes

    @property
    def is_healthy(self) -> bool:
        """Check if graph meets quality thresholds.

        A healthy graph has:
        - 0% orphan rate
        - 0% connectivity violation rate
        - < 20% ungrounded rate (warning level)
        - < 20% missing reasoning rate (warning level)
        """
        return self.orphan_rate == 0.0 and self.connectivity_violation_rate == 0.0

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "HEALTHY" if self.is_healthy else "NEEDS ATTENTION"
        lines = [
            f"Graph Quality: {status}",
            "",
            "Node Counts:",
            f"  Events: {self.event_count}",
            f"  Concepts: {self.concept_count}",
            f"  Actions: {self.action_count}",
            f"  Time nodes: {self.time_count}",
            f"  Total: {self.total_nodes}",
            "",
            f"Edge Count: {self.total_edges} (density: {self.edge_density:.2f})",
            "",
            "Quality Rates:",
            f"  Orphan rate: {self.orphan_rate:.1f}% ({self.orphan_count} nodes)",
            f"  Connectivity violations: {self.connectivity_violation_rate:.1f}%",
            f"  Ungrounded rate: {self.ungrounded_rate:.1f}%",
            f"  Missing reasoning: {self.missing_reasoning_rate:.1f}%",
            "",
            "Reasoning Quality:",
            f"  Average length: {self.avg_reasoning_length:.1f} chars",
            f"  Range: {self.min_reasoning_length}-{self.max_reasoning_length} chars",
        ]
        return "\n".join(lines)


@dataclass
class MetricsCollector:
    """Collects and tracks quality metrics over time."""

    history: list[QualityMetrics] = field(default_factory=list)

    def collect(
        self,
        graph: ConceptGraph,
        report: ValidationReport | None = None,
    ) -> QualityMetrics:
        """Collect metrics from current graph state.

        Args:
            graph: The graph to measure.
            report: Optional pre-computed validation report.

        Returns:
            QualityMetrics snapshot.
        """
        from cognifold.graph.validator import GraphValidator

        # Get validation report if not provided
        if report is None:
            validator = GraphValidator(graph)
            report = validator.validate_all()

        # Count nodes by type
        metrics = QualityMetrics()
        reasoning_lengths: list[int] = []

        for node in graph.get_all_nodes():
            metrics.total_nodes += 1
            node_type = node.type.value

            if node_type == "event":
                metrics.event_count += 1
            elif node_type == "concept":
                metrics.concept_count += 1
            elif node_type == "intent":
                metrics.action_count += 1
            elif node_type == "time":
                metrics.time_count += 1

            # Track reasoning length for non-events
            if node_type != "event" and node.reasoning:
                reasoning_lengths.append(len(node.reasoning))

        # Get edge count
        metrics.total_edges = graph.edge_count

        # Pull from validation report
        metrics.orphan_count = len(report.orphan_nodes)
        metrics.ungrounded_count = len(report.ungrounded_nodes)
        metrics.missing_reasoning_count = len(report.nodes_missing_reasoning)
        metrics.connectivity_violations = len(report.connectivity_violations)

        # Calculate reasoning stats
        if reasoning_lengths:
            metrics.avg_reasoning_length = sum(reasoning_lengths) / len(reasoning_lengths)
            metrics.min_reasoning_length = min(reasoning_lengths)
            metrics.max_reasoning_length = max(reasoning_lengths)

        # Store in history
        self.history.append(metrics)

        return metrics

    def get_trend(self, metric_name: str, last_n: int = 10) -> list[float]:
        """Get trend for a specific metric.

        Args:
            metric_name: Name of the metric property.
            last_n: Number of recent samples to include.

        Returns:
            List of metric values over time.
        """
        recent = self.history[-last_n:]
        return [getattr(m, metric_name, 0.0) for m in recent]

    def is_improving(self, metric_name: str, lower_is_better: bool = True) -> bool:
        """Check if a metric is improving over recent history.

        Args:
            metric_name: Name of the metric to check.
            lower_is_better: If True, decreasing values are improvement.

        Returns:
            True if metric is trending in the right direction.
        """
        trend = self.get_trend(metric_name, 5)
        if len(trend) < 2:
            return True  # Not enough data

        if lower_is_better:
            return trend[-1] <= trend[0]
        else:
            return trend[-1] >= trend[0]

    def clear_history(self) -> None:
        """Clear metrics history."""
        self.history.clear()
