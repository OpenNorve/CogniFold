"""Graph storage using NetworkX."""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Any

import networkx as nx

from cognifold.graph.entity_index import EntityIndex
from cognifold.models.node import Edge, Node, NodeType, validate_edge_type_constraints

logger = logging.getLogger(__name__)


class ConceptGraph:
    """A concept graph backed by NetworkX MultiDiGraph.

    Provides CRUD operations for nodes and edges, with metadata tracking
    for relevance scoring.

    Supports multiple edges between the same node pair with different types.
    Each edge is keyed by (source, target, edge_type) to allow:
    - Event A → Concept B with GROUNDS (0.9)
    - Event A → Concept B with REINFORCES (0.7)
    """

    def __init__(self) -> None:
        """Initialize an empty graph."""
        # Use MultiDiGraph to support multiple edges between same node pair
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._entity_index: EntityIndex | None = None
        # Monotonic revision counter for in-place graph mutations.
        #
        # Retrieval components cache indexes based on the graph instance. In benchmark
        # runners we mutate the same graph object over time, so we need a cheap signal
        # that the node set/content has changed.
        self._revision: int = 0

    @property
    def entity_index(self) -> EntityIndex | None:
        """Access the entity index, if built."""
        return self._entity_index

    @entity_index.setter
    def entity_index(self, index: EntityIndex | None) -> None:
        self._entity_index = index

    @property
    def revision(self) -> int:
        """Monotonic counter incremented on node-level mutations.

        This is used by retrieval backends (BM25 / semantic / hybrid) to detect
        that the graph was mutated in place and cached indexes must be refreshed.
        """
        return self._revision

    def _bump_revision(self) -> None:
        """Increment the internal revision counter."""
        self._revision += 1

    @property
    def internal_graph(self) -> nx.MultiDiGraph:
        """Access the underlying NetworkX graph for advanced operations.

        Prefer using ConceptGraph's public methods when possible.
        This property exists for cases where direct graph manipulation
        is needed (e.g., in-place node data updates in the executor).
        """
        return self._graph

    @property
    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Return the number of edges in the graph."""
        return self._graph.number_of_edges()

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return self._graph.has_node(node_id)

    def has_edge(self, source_id: str, target_id: str, edge_type: str | None = None) -> bool:
        """Check if an edge exists between two nodes.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            edge_type: Optional edge type to check for. If None, checks for
                       any edge between the nodes.

        Returns:
            True if the edge exists.
        """
        if not self._graph.has_edge(source_id, target_id):
            return False

        if edge_type is None:
            return True

        # Check if specific edge type exists
        edge_key = self._get_edge_key(edge_type)
        return edge_key in self._graph[source_id][target_id]

    def _get_edge_key(self, edge_type: str | None) -> str:
        """Get the edge key for storing in MultiDiGraph.

        Args:
            edge_type: The edge type (None for legacy edges).

        Returns:
            A string key for the edge.
        """
        return edge_type if edge_type is not None else "__legacy__"

    def add_node(self, node: Node) -> None:
        """Add a node to the graph.

        Raises:
            ValueError: If a node with the same ID already exists.
        """
        if self.has_node(node.id):
            raise ValueError(f"Node with id '{node.id}' already exists")

        # Serialize update_history for storage
        update_history_serialized = [
            {
                "timestamp": entry.timestamp.isoformat(),
                "update_reasoning": entry.update_reasoning,
                "changes": entry.changes,
            }
            for entry in node.update_history
        ]

        self._graph.add_node(
            node.id,
            type=node.type.value,
            data=node.data,
            created_at=node.created_at.isoformat(),
            last_accessed=node.last_accessed.isoformat(),
            access_count=node.access_count,
            # Explainability fields (Phase 5.5)
            reasoning=node.reasoning,
            grounded_in=node.grounded_in,
            update_history=update_history_serialized,
            embedding=node.embedding,
        )
        self._bump_revision()

    def get_node_or_none(self, node_id: str) -> Node | None:
        """Retrieve a node by ID, returning None if not found.

        Unlike ``get_node``, this does **not** raise on missing IDs, making
        it safe for defensive lookups where the caller already handles the
        ``None`` case (e.g. stale index entries).
        """
        if not self.has_node(node_id):
            return None
        return self.get_node(node_id)

    def get_node(self, node_id: str) -> Node:
        """Retrieve a node by ID.

        Raises:
            KeyError: If the node does not exist.
        """
        from cognifold.models.node import UpdateHistoryEntry

        if not self.has_node(node_id):
            raise KeyError(f"Node '{node_id}' not found")

        attrs = self._graph.nodes[node_id]

        # Deserialize update_history
        update_history = [
            UpdateHistoryEntry(
                timestamp=datetime.fromisoformat(entry["timestamp"]),
                update_reasoning=entry["update_reasoning"],
                changes=entry["changes"],
            )
            for entry in attrs.get("update_history", [])
        ]

        return Node(
            id=node_id,
            type=NodeType(attrs["type"]),
            data=attrs["data"],
            created_at=datetime.fromisoformat(attrs["created_at"]),
            last_accessed=datetime.fromisoformat(attrs["last_accessed"]),
            access_count=attrs["access_count"],
            # Explainability fields (Phase 5.5)
            reasoning=attrs.get("reasoning"),
            grounded_in=attrs.get("grounded_in", []),
            update_history=update_history,
            embedding=attrs.get("embedding"),
        )

    def update_node(self, node_id: str, data: dict[str, Any]) -> None:
        """Update a node's data (partial update).

        Raises:
            KeyError: If the node does not exist.
        """
        if not self.has_node(node_id):
            raise KeyError(f"Node '{node_id}' not found")

        current_data = self._graph.nodes[node_id]["data"]
        self._graph.nodes[node_id]["data"] = {**current_data, **data}
        self._graph.nodes[node_id]["last_accessed"] = datetime.now().isoformat()
        self._graph.nodes[node_id]["access_count"] += 1
        self._bump_revision()

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges.

        Raises:
            KeyError: If the node does not exist.
        """
        if not self.has_node(node_id):
            raise KeyError(f"Node '{node_id}' not found")

        self._graph.remove_node(node_id)
        self._bump_revision()

    def add_edge(self, edge: Edge) -> None:
        """Add an edge between two nodes.

        Multiple edges between the same node pair are allowed if they have
        different edge types. Soft validation warnings are logged for
        edge type constraint violations.

        Raises:
            KeyError: If either node does not exist.
            ValueError: If an edge with the same type already exists.
        """
        if not self.has_node(edge.source):
            raise KeyError(f"Source node '{edge.source}' not found")
        if not self.has_node(edge.target):
            raise KeyError(f"Target node '{edge.target}' not found")

        # Check for duplicate edge with same type
        if self.has_edge(edge.source, edge.target, edge.edge_type):
            type_desc = edge.edge_type or "untyped"
            raise ValueError(
                f"Edge from '{edge.source}' to '{edge.target}' "
                f"with type '{type_desc}' already exists"
            )

        # Soft validation: log warnings for type constraint violations
        source_node = self.get_node(edge.source)
        target_node = self.get_node(edge.target)
        warnings = validate_edge_type_constraints(edge, source_node, target_node)
        for warning in warnings:
            logger.warning(f"Edge constraint violation: {warning}")

        edge_key = self._get_edge_key(edge.edge_type)
        self._graph.add_edge(
            edge.source,
            edge.target,
            key=edge_key,
            edge_type=edge.edge_type,
            weight=edge.weight,
            created_at=edge.created_at.isoformat(),
            metadata=edge.metadata,
        )

    def get_edge(self, source_id: str, target_id: str, edge_type: str | None = None) -> Edge:
        """Retrieve an edge by source, target, and optionally type.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            edge_type: Optional edge type. If None and multiple edges exist,
                       returns the first one (legacy behavior).

        Raises:
            KeyError: If the edge does not exist.
        """
        if not self.has_edge(source_id, target_id, edge_type):
            type_desc = f" with type '{edge_type}'" if edge_type else ""
            raise KeyError(f"Edge from '{source_id}' to '{target_id}'{type_desc} not found")

        edge_key = self._get_edge_key(edge_type)
        if edge_key in self._graph[source_id][target_id]:
            attrs = self._graph[source_id][target_id][edge_key]
        else:
            # If specific type not found but edge exists, get first available
            first_key = next(iter(self._graph[source_id][target_id]))
            attrs = self._graph[source_id][target_id][first_key]

        return Edge(
            source=source_id,
            target=target_id,
            edge_type=attrs.get("edge_type"),
            weight=attrs.get("weight", 1.0),
            created_at=datetime.fromisoformat(attrs["created_at"]),
            metadata=attrs.get("metadata", {}),
        )

    def get_edges_between(self, source_id: str, target_id: str) -> list[Edge]:
        """Get all edges between two nodes.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.

        Returns:
            List of all edges between the nodes (may be empty).
        """
        if not self._graph.has_edge(source_id, target_id):
            return []

        edges = []
        for _edge_key, attrs in self._graph[source_id][target_id].items():
            edges.append(
                Edge(
                    source=source_id,
                    target=target_id,
                    edge_type=attrs.get("edge_type"),
                    weight=attrs.get("weight", 1.0),
                    created_at=datetime.fromisoformat(attrs["created_at"]),
                    metadata=attrs.get("metadata", {}),
                )
            )
        return edges

    def update_edge_attrs(
        self,
        source_id: str,
        target_id: str,
        edge_type: str | None,
        attrs: dict[str, Any],
    ) -> None:
        """Update attributes on an existing edge in-place.

        Merges *attrs* into the edge's stored attributes (shallow update).
        Useful for updating weight, metadata, or other edge properties
        without removing and re-adding the edge.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            edge_type: The edge type.
            attrs: Key-value pairs to merge into edge attributes.

        Raises:
            KeyError: If the edge does not exist.
        """
        if not self.has_edge(source_id, target_id, edge_type):
            type_desc = f" with type '{edge_type}'" if edge_type else ""
            raise KeyError(f"Edge from '{source_id}' to '{target_id}'{type_desc} not found")

        edge_key = self._get_edge_key(edge_type)
        stored = self._graph[source_id][target_id][edge_key]
        stored.update(attrs)

    def iter_edges_raw(self) -> list[tuple[str, str, str | None, float, dict[str, Any]]]:
        """Iterate over all edges returning raw attributes.

        Returns a snapshot list of (source, target, edge_type, weight, metadata)
        tuples.  Safe to iterate while modifying edge attributes.
        """
        result: list[tuple[str, str, str | None, float, dict[str, Any]]] = []
        for source, target, _key, attrs in self._graph.edges(keys=True, data=True):
            result.append(
                (
                    source,
                    target,
                    attrs.get("edge_type"),
                    attrs.get("weight", 1.0),
                    attrs.get("metadata", {}),
                )
            )
        return result

    def remove_edge(self, source_id: str, target_id: str, edge_type: str | None = None) -> None:
        """Remove an edge between two nodes.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            edge_type: Optional edge type. If None and multiple edges exist,
                       removes all edges between the nodes.

        Raises:
            KeyError: If the edge does not exist.
        """
        if not self.has_edge(source_id, target_id, edge_type):
            type_desc = f" with type '{edge_type}'" if edge_type else ""
            raise KeyError(f"Edge from '{source_id}' to '{target_id}'{type_desc} not found")

        if edge_type is not None:
            # Remove specific edge type
            edge_key = self._get_edge_key(edge_type)
            self._graph.remove_edge(source_id, target_id, key=edge_key)
        else:
            # Remove all edges between nodes (legacy behavior)
            # We need to get all keys first, then remove
            keys = list(self._graph[source_id][target_id].keys())
            for key in keys:
                self._graph.remove_edge(source_id, target_id, key=key)

    def get_neighbors(self, node_id: str) -> list[str]:
        """Get IDs of nodes connected by outgoing edges.

        Raises:
            KeyError: If the node does not exist.
        """
        if not self.has_node(node_id):
            raise KeyError(f"Node '{node_id}' not found")

        return list(self._graph.successors(node_id))

    def get_predecessors(self, node_id: str) -> list[str]:
        """Get IDs of nodes with edges pointing to this node.

        Raises:
            KeyError: If the node does not exist.
        """
        if not self.has_node(node_id):
            raise KeyError(f"Node '{node_id}' not found")

        return list(self._graph.predecessors(node_id))

    def get_all_nodes(self) -> list[Node]:
        """Get all nodes in the graph."""
        return [self.get_node(node_id) for node_id in self._graph.nodes()]

    def get_all_edges(self) -> list[Edge]:
        """Get all edges in the graph (including multiple edges between same nodes)."""
        edges = []
        for source, target, _key, attrs in self._graph.edges(keys=True, data=True):
            edges.append(
                Edge(
                    source=source,
                    target=target,
                    edge_type=attrs.get("edge_type"),
                    weight=attrs.get("weight", 1.0),
                    created_at=datetime.fromisoformat(attrs["created_at"]),
                    metadata=attrs.get("metadata", {}),
                )
            )
        return edges

    def get_nodes_by_type(self, node_type: NodeType) -> list[Node]:
        """Get all nodes of a specific type."""
        return [
            self.get_node(node_id)
            for node_id in self._graph.nodes()
            if self._graph.nodes[node_id]["type"] == node_type.value
        ]

    def clear(self) -> None:
        """Remove all nodes and edges from the graph."""
        self._graph.clear()
        self._bump_revision()

    def expand_from_node(
        self,
        node_id: str,
        max_depth: int = 1,
        direction: str = "both",
        max_nodes: int = 50,
    ) -> tuple[list[tuple[str, int]], list[tuple[str, str, str | None, float]]]:
        """BFS expansion from a node.

        Args:
            node_id: The starting node.
            max_depth: Maximum number of hops (1-5).
            direction: "outgoing", "incoming", or "both".
            max_nodes: Maximum nodes to return.

        Returns:
            (nodes_with_depth, edges) where:
            - nodes_with_depth: list of (node_id, depth) tuples
            - edges: list of (source, target, edge_type, weight) tuples

        Raises:
            KeyError: If node_id does not exist.
        """
        if not self.has_node(node_id):
            raise KeyError(f"Node '{node_id}' not found")

        visited: dict[str, int] = {node_id: 0}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            neighbors: set[str] = set()
            if direction in ("outgoing", "both"):
                neighbors.update(self._graph.successors(current))
            if direction in ("incoming", "both"):
                neighbors.update(self._graph.predecessors(current))

            for neighbor in neighbors:
                if neighbor not in visited:
                    if len(visited) >= max_nodes:
                        break
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1))

        # Collect edges between visited nodes
        edges: list[tuple[str, str, str | None, float]] = []
        visited_set = set(visited)
        for src in visited_set:
            for tgt in visited_set:
                if self._graph.has_edge(src, tgt):
                    for _key, attrs in self._graph[src][tgt].items():
                        edges.append(
                            (
                                src,
                                tgt,
                                attrs.get("edge_type"),
                                attrs.get("weight", 1.0),
                            )
                        )

        nodes_with_depth = list(visited.items())
        return nodes_with_depth, edges

    def find_nearest_nodes(
        self, query_embedding: list[float], k: int = 10
    ) -> list[tuple[str, float]]:
        """Find nodes with embeddings closest to the query embedding.

        Args:
            query_embedding: The query vector.
            k: Number of results to return.

        Returns:
            List of (node_id, similarity_score) tuples, sorted by score descending.
        """
        import numpy as np

        if not query_embedding:
            return []

        query_vec = np.array(query_embedding)
        norm_query = np.linalg.norm(query_vec)
        if norm_query == 0:
            return []

        candidates: list[tuple[str, float]] = []

        # Iterate over all nodes that have embeddings
        # TODO: Optimize with a vector index (FAISS or Annoy) if scaling is needed
        for node_id, attrs in self._graph.nodes(data=True):
            embedding = attrs.get("embedding")
            if embedding:
                vec = np.array(embedding)
                norm_vec = np.linalg.norm(vec)
                if norm_vec > 0:
                    similarity = np.dot(query_vec, vec) / (norm_query * norm_vec)
                    candidates.append((node_id, float(similarity)))

        # Sort by similarity descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]

    def infer_edges(
        self,
        similarity_threshold: float = 0.30,
        max_edges_per_node: int = 3,
    ) -> int:
        """Run post-ingestion edge inference using stored node embeddings.

        Creates RELATED_TO edges between nodes with similar embeddings.
        Uses kNN approach: for each node, connect to top-k most similar
        neighbors above the similarity floor.

        Args:
            similarity_threshold: Minimum cosine similarity to create an edge.
            max_edges_per_node: Maximum edges to create per node.

        Returns:
            Number of edges created.
        """
        from cognifold.graph.edge_inference import EdgeInferenceEngine

        engine = EdgeInferenceEngine(
            self,
            similarity_threshold=similarity_threshold,
            max_edges_per_node=max_edges_per_node,
        )
        edges = engine.infer_edges()
        return len(edges)

    def get_concept_quality_stats(self) -> dict[str, object]:
        """Return concept extraction quality metrics.

        Returns a dict with total_concepts, total_events, concepts_per_event,
        orphan_concepts, and orphan_rate.
        """
        from cognifold.models.node import NodeType

        concepts = self.get_nodes_by_type(NodeType.CONCEPT)
        events = self.get_nodes_by_type(NodeType.EVENT)

        orphan_count = sum(1 for c in concepts if self._graph.degree(c.id) == 0)

        return {
            "total_concepts": len(concepts),
            "total_events": len(events),
            "concepts_per_event": round(len(concepts) / max(len(events), 1), 2),
            "orphan_concepts": orphan_count,
            "orphan_rate": round(orphan_count / max(len(concepts), 1), 3),
        }
