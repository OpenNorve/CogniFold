"""CogEval-Bench Track C: Temporal Compression Evaluator.

Evaluates cognitive compression quality:
1. Rate-Distortion curve: graph complexity vs downstream QA accuracy
2. Schema Acceleration: decreasing operations per event over time
3. Emergence Trajectory: metrics over time as events are ingested
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CompressionEvalResult:
    """Results from temporal compression evaluation."""

    acceleration: float = 0.0
    early_ops_avg: float = 0.0
    late_ops_avg: float = 0.0
    ops_per_event: list[float] = field(default_factory=list)

    rd_knee_concepts: int = 0
    rd_knee_accuracy: float = 0.0

    trajectory: list[dict[str, Any]] = field(default_factory=list)
    convergence_event: int = 0

    pagerank_gini: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "acceleration": round(self.acceleration, 4),
            "early_ops_avg": round(self.early_ops_avg, 4),
            "late_ops_avg": round(self.late_ops_avg, 4),
            "ops_per_event_summary": {
                "min": round(min(self.ops_per_event), 2) if self.ops_per_event else 0,
                "max": round(max(self.ops_per_event), 2) if self.ops_per_event else 0,
                "mean": round(float(np.mean(self.ops_per_event)), 2) if self.ops_per_event else 0,
            },
            "rd_knee_concepts": self.rd_knee_concepts,
            "rd_knee_accuracy": round(self.rd_knee_accuracy, 4),
            "convergence_event": self.convergence_event,
            "pagerank_gini": round(self.pagerank_gini, 4),
            "trajectory_length": len(self.trajectory),
        }


def compute_schema_acceleration(
    ops_per_event: list[float],
) -> tuple[float, float, float]:
    """Compute schema acceleration from operations-per-event time series.

    Compares average operations in first 25% vs last 25% of events.
    Positive acceleration means later events require fewer operations —
    the system has formed schemas that accelerate processing.

    Returns:
        (acceleration, early_avg, late_avg)
    """
    if len(ops_per_event) < 4:
        return (0.0, 0.0, 0.0)

    quarter = max(1, len(ops_per_event) // 4)
    early = ops_per_event[:quarter]
    late = ops_per_event[-quarter:]

    early_avg = float(np.mean(early))
    late_avg = float(np.mean(late))

    if early_avg == 0:
        return (0.0, early_avg, late_avg)

    acceleration = 1.0 - (late_avg / early_avg)
    return (acceleration, early_avg, late_avg)


def compute_pagerank_gini(pagerank_scores: list[float]) -> float:
    """Compute Gini coefficient of PageRank distribution.

    Higher Gini = more concentrated PageRank = more hierarchical structure.
    Gini = 0 means all nodes equally important (flat).
    Gini → 1 means one node dominates (highly hierarchical).
    """
    if not pagerank_scores or len(pagerank_scores) < 2:
        return 0.0

    scores = sorted(pagerank_scores)
    n = len(scores)
    total = sum(scores)

    if total == 0:
        return 0.0

    arr = np.array(scores)
    gini = (2.0 * np.sum(np.arange(1, n + 1) * arr)) / (n * total) - (n + 1) / n
    return max(0.0, float(gini))


def find_convergence_point(
    trajectory: list[dict[str, Any]],
    metric_key: str = "concept_count",
    window: int = 5,
    threshold: float = 0.05,
) -> int:
    """Find the event index where a metric stabilizes.

    Looks for the first point where the metric's relative change over
    a sliding window stays below the threshold.

    Returns:
        Event index of convergence (0 if never converges).
    """
    if len(trajectory) < window * 2:
        return 0

    values = [t.get(metric_key, 0) for t in trajectory]

    for i in range(window, len(values) - window):
        recent = values[i - window : i + 1]
        if not any(recent):
            continue
        max_val = max(recent)
        min_val = min(recent)
        if max_val == 0:
            continue
        relative_change = (max_val - min_val) / max_val
        if relative_change < threshold:
            return trajectory[i].get("event_idx", i)

    return 0


def build_emergence_trajectory(
    snapshots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build emergence trajectory from GraphEvolutionTracker snapshots.

    Each snapshot should have: event_idx, node_count, edge_count,
    concept_count, compression_ratio, pagerank_gini, etc.
    """
    trajectory = []
    for snap in snapshots:
        trajectory.append(
            {
                "event_idx": snap.get("event_idx", 0),
                "node_count": snap.get("node_count", 0),
                "edge_count": snap.get("edge_count", 0),
                "concept_count": snap.get("concept_count", 0),
                "intent_count": snap.get("intent_count", 0),
                "compression_ratio": snap.get("compression_ratio", 0.0),
                "edge_density": snap.get("edge_density", 0.0),
                "pagerank_gini": snap.get("pagerank_gini", 0.0),
            }
        )
    return trajectory


def evaluate_compression(
    ops_per_event: list[float],
    pagerank_scores: list[float],
    trajectory_snapshots: list[dict[str, Any]],
) -> CompressionEvalResult:
    """Run full compression evaluation.

    Args:
        ops_per_event: Number of graph operations per ingested event.
        pagerank_scores: PageRank values for all nodes.
        trajectory_snapshots: GraphEvolutionTracker snapshots.

    Returns:
        CompressionEvalResult with all metrics.
    """
    result = CompressionEvalResult()

    acceleration, early_avg, late_avg = compute_schema_acceleration(ops_per_event)
    result.acceleration = acceleration
    result.early_ops_avg = early_avg
    result.late_ops_avg = late_avg
    result.ops_per_event = ops_per_event

    result.pagerank_gini = compute_pagerank_gini(pagerank_scores)

    result.trajectory = build_emergence_trajectory(trajectory_snapshots)
    result.convergence_event = find_convergence_point(result.trajectory)

    return result
