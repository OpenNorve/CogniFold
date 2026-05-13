"""Track graph evolution metrics across event processing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx

from cognifold.graph.store import ConceptGraph
from cognifold.models.node import NodeType


@dataclass
class GraphSnapshot:
    """A single snapshot of graph metrics at a point in time."""

    event_idx: int
    node_count: int
    edge_count: int
    concept_count: int
    intent_count: int
    event_node_count: int
    compression_ratio: float
    edge_density: float
    pagerank_gini: float


@dataclass
class GraphEvolutionTracker:
    """Records graph metrics after each event is processed.

    Usage::

        tracker = GraphEvolutionTracker()
        for idx, event in enumerate(events):
            process(event, graph)
            tracker.record(idx, graph)
        tracker.save("evolution.json")

    Later, load and plot::

        tracker = GraphEvolutionTracker.load("evolution.json")
    """

    snapshots: list[GraphSnapshot] = field(default_factory=list)
    benchmark_name: str = ""
    metadata: dict = field(default_factory=dict)

    def record(self, event_idx: int, graph: ConceptGraph) -> GraphSnapshot:
        """Record a snapshot of graph metrics after processing an event.

        Args:
            event_idx: Zero-based index of the event just processed.
            graph: The ConceptGraph to measure.

        Returns:
            The recorded snapshot.
        """
        node_count = graph.node_count
        edge_count = graph.edge_count

        # Count nodes by type
        concept_count = len(graph.get_nodes_by_type(NodeType.CONCEPT))
        intent_count = len(graph.get_nodes_by_type(NodeType.INTENT))
        event_node_count = len(graph.get_nodes_by_type(NodeType.EVENT))

        # Compression ratio: how many concepts per event node
        compression_ratio = (
            concept_count / event_node_count if event_node_count > 0 else 0.0
        )

        # Edge density: edges per node
        edge_density = edge_count / node_count if node_count > 0 else 0.0

        # PageRank Gini coefficient
        pagerank_gini = self._compute_pagerank_gini(graph)

        snapshot = GraphSnapshot(
            event_idx=event_idx,
            node_count=node_count,
            edge_count=edge_count,
            concept_count=concept_count,
            intent_count=intent_count,
            event_node_count=event_node_count,
            compression_ratio=compression_ratio,
            edge_density=edge_density,
            pagerank_gini=pagerank_gini,
        )
        self.snapshots.append(snapshot)
        return snapshot

    @staticmethod
    def _compute_pagerank_gini(graph: ConceptGraph) -> float:
        """Compute the Gini coefficient of the PageRank distribution.

        A Gini of 0 means all nodes have equal importance.
        A Gini approaching 1 means importance is highly concentrated.

        Returns 0.0 if the graph has fewer than 2 nodes.
        """
        g = graph.internal_graph
        if g.number_of_nodes() < 2:
            return 0.0

        try:
            pr = nx.pagerank(g)
        except nx.PowerIterationFailedConvergence:
            return 0.0

        values = sorted(pr.values())
        n = len(values)
        if n == 0:
            return 0.0

        # Gini coefficient via the relative mean absolute difference formula:
        # G = (2 * sum_i( (i+1) * x_i )) / (n * sum(x)) - (n + 1) / n
        cumsum = 0.0
        total = 0.0
        for i, v in enumerate(values):
            cumsum += (i + 1) * v
            total += v

        if total == 0.0:
            return 0.0

        gini = (2.0 * cumsum) / (n * total) - (n + 1) / n
        return max(0.0, min(1.0, gini))  # clamp to [0, 1]

    def save(self, path: str | Path) -> None:
        """Save snapshots to a JSON file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "benchmark_name": self.benchmark_name,
            "metadata": self.metadata,
            "snapshots": [asdict(s) for s in self.snapshots],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> GraphEvolutionTracker:
        """Load a tracker from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            A populated GraphEvolutionTracker.
        """
        with open(path) as f:
            data = json.load(f)

        snapshots = [GraphSnapshot(**s) for s in data.get("snapshots", [])]
        return cls(
            snapshots=snapshots,
            benchmark_name=data.get("benchmark_name", ""),
            metadata=data.get("metadata", {}),
        )
