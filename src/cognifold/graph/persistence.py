"""JSON persistence for the concept graph."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from cognifold.graph.store import ConceptGraph
from cognifold.models.node import Edge, Node, NodeType


def save_graph(graph: ConceptGraph, path: str | Path, backup: bool = True) -> None:
    """Save a graph to a JSON file.

    Args:
        graph: The graph to save.
        path: Path to the output file.
        backup: If True, create a backup of existing file before overwriting.
    """
    path = Path(path)

    if backup and path.exists():
        backup_path = path.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak")
        shutil.copy2(path, backup_path)

    data = graph_to_dict(graph)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_graph(path: str | Path) -> ConceptGraph:
    """Load a graph from a JSON file.

    Args:
        path: Path to the input file.

    Returns:
        A new ConceptGraph populated with the saved data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid data.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return dict_to_graph(data)


def graph_to_dict(graph: ConceptGraph) -> dict[str, Any]:
    """Convert a graph to a dictionary for JSON serialization."""
    nodes = []
    for node in graph.get_all_nodes():
        nodes.append(
            {
                "id": node.id,
                "type": node.type.value,
                "data": node.data,
                "created_at": node.created_at.isoformat(),
                "last_accessed": node.last_accessed.isoformat(),
                "access_count": node.access_count,
            }
        )

    edges = []
    for edge in graph.get_all_edges():
        edge_dict: dict[str, Any] = {
            "source": edge.source,
            "target": edge.target,
            "created_at": edge.created_at.isoformat(),
        }
        # Include edge_type and weight only if set (Phase 9.1)
        if edge.edge_type is not None:
            edge_dict["edge_type"] = edge.edge_type
        if edge.weight != 1.0:  # Only store non-default weights
            edge_dict["weight"] = edge.weight
        if edge.metadata:  # Only store if non-empty
            edge_dict["metadata"] = edge.metadata
        edges.append(edge_dict)

    return {
        "version": "1.1",  # Bumped for Phase 9.1 edge types
        "saved_at": datetime.now().isoformat(),
        "nodes": nodes,
        "edges": edges,
    }


def dict_to_graph(data: dict[str, Any]) -> ConceptGraph:
    """Convert a dictionary to a ConceptGraph."""
    if "nodes" not in data or "edges" not in data:
        raise ValueError("Invalid graph data: missing 'nodes' or 'edges' key")

    graph = ConceptGraph()

    for node_data in data["nodes"]:
        node = Node(
            id=node_data["id"],
            type=NodeType(node_data["type"]),
            data=node_data.get("data", {}),
            created_at=datetime.fromisoformat(node_data["created_at"]),
            last_accessed=datetime.fromisoformat(node_data["last_accessed"]),
            access_count=node_data.get("access_count", 0),
        )
        graph.add_node(node)

    for edge_data in data["edges"]:
        # Use Edge.create for proper default weight handling
        edge = Edge.create(
            source=edge_data["source"],
            target=edge_data["target"],
            edge_type=edge_data.get("edge_type"),  # None for legacy edges
            weight=edge_data.get("weight"),  # None uses type-based default
            metadata=edge_data.get("metadata"),
        )
        # Override created_at from file
        edge = Edge(
            source=edge.source,
            target=edge.target,
            edge_type=edge.edge_type,
            weight=edge.weight,
            created_at=datetime.fromisoformat(edge_data["created_at"]),
            metadata=edge.metadata,
        )
        graph.add_edge(edge)

    return graph
