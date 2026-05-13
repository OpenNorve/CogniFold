"""Graph traversal tools for the LLM agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph


class GraphTools:
    """Tools for LLM to explore the concept graph.

    These tools are called by the LangGraph agent to retrieve
    information from the graph beyond the initial context window.
    """

    def __init__(self, graph: ConceptGraph):
        """Initialize with a reference to the graph.

        Args:
            graph: The concept graph to query.
        """
        self._graph = graph

    def get_node(self, node_id: str) -> dict[str, Any]:
        """Retrieve full details of a node by ID.

        Args:
            node_id: The ID of the node to retrieve.

        Returns:
            Node data including type, data payload, and metadata.
            Returns error dict if node not found.
        """
        try:
            node = self._graph.get_node(node_id)
            return {
                "id": node.id,
                "type": node.type.value,
                "data": node.data,
                "created_at": node.created_at.isoformat(),
                "last_accessed": node.last_accessed.isoformat(),
                "access_count": node.access_count,
            }
        except KeyError:
            return {"error": f"Node '{node_id}' not found"}

    def get_neighbors(self, node_id: str, direction: str = "both") -> list[dict[str, Any]]:
        """Get nodes connected to the specified node.

        Args:
            node_id: The node to find neighbors for.
            direction: One of "outgoing", "incoming", or "both".

        Returns:
            List of neighbor node summaries. Returns error list if node not found.
        """
        try:
            neighbors: list[str] = []

            if direction in ("outgoing", "both"):
                neighbors.extend(self._graph.get_neighbors(node_id))
            if direction in ("incoming", "both"):
                neighbors.extend(self._graph.get_predecessors(node_id))

            # Remove duplicates while preserving order
            seen: set[str] = set()
            unique_neighbors = []
            for n in neighbors:
                if n not in seen:
                    seen.add(n)
                    unique_neighbors.append(n)

            # Return summaries, not full nodes
            result = []
            for neighbor_id in unique_neighbors[:20]:  # Limit results
                try:
                    node = self._graph.get_node(neighbor_id)
                    result.append(
                        {
                            "id": node.id,
                            "type": node.type.value,
                            "title": node.data.get("title", node.id),
                        }
                    )
                except KeyError:
                    continue

            return result

        except KeyError:
            return [{"error": f"Node '{node_id}' not found"}]

    def find_nodes_by_type(self, node_type: str) -> list[dict[str, Any]]:
        """Find all nodes of a specific type.

        Args:
            node_type: One of "event", "concept", or "action".

        Returns:
            List of node summaries (limited to 20 most recent).
        """
        from cognifold.models.node import NodeType

        try:
            node_type_enum = NodeType.from_string(node_type.lower())
        except ValueError:
            return [
                {
                    "error": f"Invalid node type: {node_type}. Use 'event', 'concept', 'intent', or 'action'"
                }
            ]

        nodes = self._graph.get_nodes_by_type(node_type_enum)

        # Sort by creation time (most recent first) and limit
        nodes.sort(key=lambda n: n.created_at, reverse=True)
        nodes = nodes[:20]

        return [
            {
                "id": node.id,
                "type": node.type.value,
                "title": node.data.get("title", node.id),
                "created_at": node.created_at.isoformat(),
            }
            for node in nodes
        ]

    def search_nodes(self, keyword: str) -> list[dict[str, Any]]:
        """Search nodes by keyword in title and data.

        Args:
            keyword: Search term to match (case-insensitive).

        Returns:
            Matching nodes (limited to 10).
        """
        keyword_lower = keyword.lower()
        matches = []

        for node in self._graph.get_all_nodes():
            # Check title
            title = node.data.get("title", "")
            if keyword_lower in title.lower():
                matches.append(node)
                continue

            # Check description
            desc = node.data.get("description", "")
            if keyword_lower in desc.lower():
                matches.append(node)
                continue

            # Check other string fields in data
            for value in node.data.values():
                if isinstance(value, str) and keyword_lower in value.lower():
                    matches.append(node)
                    break

        # Limit and format results
        return [
            {
                "id": node.id,
                "type": node.type.value,
                "title": node.data.get("title", node.id),
                "data_preview": str(node.data)[:100],
            }
            for node in matches[:10]
        ]

    def get_graph_stats(self) -> dict[str, Any]:
        """Get overview statistics of the graph.

        Returns:
            Dictionary with node counts, edge count, etc.
        """
        from cognifold.models.node import NodeType

        return {
            "total_nodes": self._graph.node_count,
            "total_edges": self._graph.edge_count,
            "event_count": len(self._graph.get_nodes_by_type(NodeType.EVENT)),
            "concept_count": len(self._graph.get_nodes_by_type(NodeType.CONCEPT)),
            "intent_count": len(self._graph.get_nodes_by_type(NodeType.INTENT)),
        }

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions for LangChain/LangGraph.

        Returns:
            List of tool definition dictionaries.
        """
        return [
            {
                "name": "get_node",
                "description": "Retrieve full details of a node by ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_id": {
                            "type": "string",
                            "description": "The ID of the node to retrieve",
                        }
                    },
                    "required": ["node_id"],
                },
            },
            {
                "name": "get_neighbors",
                "description": "Get nodes connected to the specified node",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_id": {
                            "type": "string",
                            "description": "The node to find neighbors for",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["outgoing", "incoming", "both"],
                            "description": "Direction of edges to follow",
                            "default": "both",
                        },
                    },
                    "required": ["node_id"],
                },
            },
            {
                "name": "find_nodes_by_type",
                "description": "Find all nodes of a specific type (event, concept, or action)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_type": {
                            "type": "string",
                            "enum": ["event", "concept", "action"],
                            "description": "The type of nodes to find",
                        }
                    },
                    "required": ["node_type"],
                },
            },
            {
                "name": "search_nodes",
                "description": "Search nodes by keyword in title and data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "Search term to match",
                        }
                    },
                    "required": ["keyword"],
                },
            },
            {
                "name": "get_graph_stats",
                "description": "Get overview statistics of the graph",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool by name with the given arguments.

        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            Tool result.

        Raises:
            ValueError: If tool name is unknown.
        """
        tools = {
            "get_node": self.get_node,
            "get_neighbors": self.get_neighbors,
            "find_nodes_by_type": self.find_nodes_by_type,
            "search_nodes": self.search_nodes,
            "get_graph_stats": self.get_graph_stats,
        }

        if tool_name not in tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        return tools[tool_name](**arguments)
