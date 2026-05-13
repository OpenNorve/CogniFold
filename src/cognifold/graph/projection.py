"""Read-only graph projection interface for downstream consumers.

Provides a clean abstraction layer for graph access that any future direction
(neural network, symbolic tracker, consolidation engine) can use without
depending on ConceptGraph internals.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from cognifold.models.node import Edge, Node, NodeType

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph
    from cognifold.scoring.ranker import ContextRanker


@runtime_checkable
class GraphProjection(Protocol):
    """Read-only projection of the concept graph for downstream consumers.

    Any component that needs to read graph state (neural consolidation,
    symbolic tracking, snapshot logging, etc.) should depend on this
    protocol rather than on ConceptGraph directly.
    """

    def get_active_concepts(self, limit: int = 50) -> list[Node]:
        """Get concepts sorted by composite score, excluding archived/forgotten.

        Args:
            limit: Maximum number of concepts to return.

        Returns:
            List of concept nodes sorted by score descending.
        """
        ...

    def get_concept_connections(
        self, node_id: str, max_depth: int = 2
    ) -> list[tuple[Node, Edge, Node]]:
        """Get edges connected to a concept up to max_depth hops.

        Args:
            node_id: The starting node ID.
            max_depth: Maximum number of hops to traverse.

        Returns:
            List of (source_node, edge, target_node) triples.
        """
        ...

    def get_cluster_summary(self) -> list[dict[str, object]]:
        """Get community/cluster summary using connected components.

        Returns:
            List of cluster dicts with keys: cluster_id, size, top_nodes.
        """
        ...

    def get_lifecycle_state(self, node_id: str) -> str:
        """Get lifecycle state of a node.

        Returns one of: active, consolidated, archived, forgotten.
        Reads from node.data["lifecycle_state"] if present, defaults to "active".

        Args:
            node_id: The node ID to check.

        Returns:
            Lifecycle state string.

        Raises:
            KeyError: If the node does not exist.
        """
        ...

    def get_node_count_by_type(self) -> dict[str, int]:
        """Count nodes grouped by type.

        Returns:
            Dict mapping node type name to count.
        """
        ...


class NetworkXProjection:
    """Default implementation of GraphProjection wrapping ConceptGraph.

    Uses ConceptGraph for node/edge access and optionally ContextRanker
    for scoring-based ordering.
    """

    def __init__(
        self,
        graph: ConceptGraph,
        ranker: ContextRanker | None = None,
    ) -> None:
        self._graph = graph
        self._ranker = ranker

    def get_active_concepts(self, limit: int = 50) -> list[Node]:
        """Get concepts sorted by composite score, excluding archived/forgotten."""
        concepts = self._graph.get_nodes_by_type(NodeType.CONCEPT)

        # Filter out archived and forgotten nodes
        active = [
            c
            for c in concepts
            if c.data.get("lifecycle_state", "active") in ("active", "consolidated")
        ]

        if self._ranker is not None:
            # Score all nodes and build a lookup
            scores = self._ranker.score_nodes(self._graph)
            score_map: dict[str, float] = {s.node_id: s.composite_score for s in scores}
            active.sort(key=lambda n: score_map.get(n.id, 0.0), reverse=True)
        else:
            # Fallback: sort by access_count descending, then created_at descending
            active.sort(key=lambda n: (n.access_count, n.created_at), reverse=True)

        return active[:limit]

    def get_concept_connections(
        self, node_id: str, max_depth: int = 2
    ) -> list[tuple[Node, Edge, Node]]:
        """Get edges connected to a concept up to max_depth hops."""
        if not self._graph.has_node(node_id):
            raise KeyError(f"Node '{node_id}' not found")

        # BFS to collect node IDs within max_depth
        visited: set[str] = {node_id}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            neighbors: set[str] = set()
            neighbors.update(self._graph.get_neighbors(current))
            neighbors.update(self._graph.get_predecessors(current))

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        # Collect all edges between visited nodes as (source_node, edge, target_node)
        result: list[tuple[Node, Edge, Node]] = []
        all_edges = self._graph.get_all_edges()
        for edge in all_edges:
            if edge.source in visited and edge.target in visited:
                source_node = self._graph.get_node(edge.source)
                target_node = self._graph.get_node(edge.target)
                result.append((source_node, edge, target_node))

        return result

    def get_cluster_summary(self) -> list[dict[str, object]]:
        """Get community/cluster summary using weakly connected components."""
        import networkx as nx

        nx_graph = self._graph.internal_graph

        if nx_graph.number_of_nodes() == 0:
            return []

        # Use weakly connected components (directed graph)
        components = list(nx.weakly_connected_components(nx_graph))

        clusters: list[dict[str, object]] = []
        for i, component in enumerate(components):
            # Get top nodes by access count
            nodes_in_cluster: list[Node] = []
            for nid in component:
                node = self._graph.get_node_or_none(nid)
                if node is not None:
                    nodes_in_cluster.append(node)

            nodes_in_cluster.sort(key=lambda n: n.access_count, reverse=True)
            top_nodes = [
                {"id": n.id, "title": n.data.get("title", n.id), "type": n.type.value}
                for n in nodes_in_cluster[:5]
            ]

            clusters.append(
                {
                    "cluster_id": i,
                    "size": len(component),
                    "top_nodes": top_nodes,
                }
            )

        # Sort by size descending
        clusters.sort(key=lambda c: c["size"], reverse=True)  # type: ignore[arg-type]
        return clusters

    def get_lifecycle_state(self, node_id: str) -> str:
        """Get lifecycle state of a node."""
        node = self._graph.get_node(node_id)  # raises KeyError if missing
        return str(node.data.get("lifecycle_state", "active"))

    def get_node_count_by_type(self) -> dict[str, int]:
        """Count nodes grouped by type."""
        counts: dict[str, int] = {}
        for node_type in NodeType:
            nodes = self._graph.get_nodes_by_type(node_type)
            if nodes:
                counts[node_type.value] = len(nodes)
        return counts


@dataclass
class GraphSnapshot:
    """Serializable point-in-time graph state for trace logging.

    Captures a lightweight summary of the graph at a specific moment,
    suitable for logging, debugging, and comparing graph evolution over time.
    """

    timestamp: datetime
    node_count: int
    edge_count: int
    nodes_by_type: dict[str, int]
    top_concepts: list[dict[str, object]] = field(default_factory=list)
    active_intents: list[dict[str, object]] = field(default_factory=list)
    cluster_count: int = 0

    def to_dict(self) -> dict[str, object]:
        """Serialize snapshot to a plain dict.

        Returns:
            Dictionary representation with ISO-formatted timestamp.
        """
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> GraphSnapshot:
        """Deserialize snapshot from a plain dict.

        Args:
            data: Dictionary with snapshot data.

        Returns:
            Reconstructed GraphSnapshot.
        """
        ts_raw = data["timestamp"]
        assert isinstance(ts_raw, str)
        timestamp = datetime.fromisoformat(ts_raw)

        node_count_raw = data["node_count"]
        assert isinstance(node_count_raw, int)

        edge_count_raw = data["edge_count"]
        assert isinstance(edge_count_raw, int)

        nodes_by_type_raw = data["nodes_by_type"]
        assert isinstance(nodes_by_type_raw, dict)

        top_concepts_raw = data.get("top_concepts", [])
        assert isinstance(top_concepts_raw, list)

        active_intents_raw = data.get("active_intents", [])
        assert isinstance(active_intents_raw, list)

        cluster_count_raw = data.get("cluster_count", 0)
        assert isinstance(cluster_count_raw, int)

        return cls(
            timestamp=timestamp,
            node_count=node_count_raw,
            edge_count=edge_count_raw,
            nodes_by_type=nodes_by_type_raw,
            top_concepts=top_concepts_raw,
            active_intents=active_intents_raw,
            cluster_count=cluster_count_raw,
        )


def graph_to_snapshot(
    graph: ConceptGraph,
    ranker: ContextRanker | None = None,
) -> GraphSnapshot:
    """Create a serializable snapshot of current graph state.

    Args:
        graph: The concept graph to snapshot.
        ranker: Optional ranker for scoring concepts.

    Returns:
        A GraphSnapshot capturing the current state.
    """
    projection = NetworkXProjection(graph, ranker)

    # Node counts by type
    nodes_by_type = projection.get_node_count_by_type()

    # Top concepts (up to 10)
    top_concepts_nodes = projection.get_active_concepts(limit=10)
    top_concepts: list[dict[str, object]] = []

    if ranker is not None:
        scores = ranker.score_nodes(graph)
        score_map: dict[str, float] = {s.node_id: s.composite_score for s in scores}
    else:
        score_map = {}

    for node in top_concepts_nodes:
        top_concepts.append(
            {
                "id": node.id,
                "title": node.data.get("title", node.id),
                "score": score_map.get(node.id, 0.0),
                "type": node.type.value,
            }
        )

    # Active intents (pending status)
    intent_nodes = graph.get_nodes_by_type(NodeType.INTENT)
    active_intents: list[dict[str, object]] = [
        {
            "id": n.id,
            "title": n.data.get("title", n.id),
            "status": n.data.get("status", "pending"),
        }
        for n in intent_nodes
        if n.data.get("status", "pending") == "pending"
    ]

    # Cluster count
    clusters = projection.get_cluster_summary()

    return GraphSnapshot(
        timestamp=datetime.now(),
        node_count=graph.node_count,
        edge_count=graph.edge_count,
        nodes_by_type=nodes_by_type,
        top_concepts=top_concepts,
        active_intents=active_intents,
        cluster_count=len(clusters),
    )
