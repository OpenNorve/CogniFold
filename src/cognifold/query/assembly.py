"""Context assembly from retrieved nodes.

This module assembles human-readable context from query results,
formatting nodes into text suitable for LLM consumption.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cognifold.query.models import NodeSummary, QueryConfig, QueryResult

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph


class ContextAssembler:
    """Assembles context text from retrieved nodes."""

    def __init__(
        self,
        config: QueryConfig | None = None,
        graph: ConceptGraph | None = None,
    ) -> None:
        """Initialize the assembler.

        Args:
            config: Query configuration.
            graph: Optional graph reference for edge lookups.
        """
        self.config = config or QueryConfig()
        self.graph = graph

    def assemble(
        self,
        nodes: list[NodeSummary],
        traversal_path: list[str],
        query_metadata: dict[str, Any] | None = None,
        total_nodes_scanned: int = 0,
        query_time_ms: float = 0.0,
    ) -> QueryResult:
        """Assemble a QueryResult from retrieved nodes.

        Args:
            nodes: Sorted list of NodeSummary objects.
            traversal_path: Order of node traversal.
            query_metadata: Additional query information.
            total_nodes_scanned: Total nodes examined.
            query_time_ms: Query processing time.

        Returns:
            Complete QueryResult with formatted context.
        """
        # Build context text respecting max_context_chars
        context = self._build_context_text(nodes)

        return QueryResult(
            context=context,
            nodes=nodes,
            traversal_path=traversal_path,
            query_metadata=query_metadata or {},
            total_nodes_scanned=total_nodes_scanned,
            query_time_ms=query_time_ms,
        )

    def build_context_text(
        self, nodes: list[NodeSummary], type_order: list[str] | None = None
    ) -> str:
        """Build formatted context text from nodes.

        Public API for external callers (e.g. MemoryQueryAgent).
        Respects max_context_chars by truncating if necessary.

        Args:
            nodes: Nodes to format.
            type_order: Optional override for the section ordering. Default
                puts concepts first; pass ["event", "concept", ...] to put
                raw events first (used by LoCoMo iter1 to surface
                conversation turns ahead of folded concepts).

        Returns:
            Formatted context string.
        """
        return self._build_context_text(nodes, type_order=type_order)

    def _build_context_text(
        self,
        nodes: list[NodeSummary],
        type_order: list[str] | None = None,
    ) -> str:
        """Build formatted context text from nodes (internal).

        Respects max_context_chars by truncating if necessary.

        Args:
            nodes: Nodes to format.
            type_order: Optional override for the section ordering.

        Returns:
            Formatted context string.
        """
        if not nodes:
            return "No relevant context found."

        sections: list[str] = []
        current_length = 0
        max_chars = self.config.max_context_chars

        # Group nodes by type for better organization
        grouped = self._group_by_type(nodes, type_order=type_order)

        for node_type, type_nodes in grouped.items():
            if current_length >= max_chars:
                break

            # Add section header
            header = f"\n## {node_type.upper()}S\n"
            if current_length + len(header) > max_chars:
                break
            sections.append(header)
            current_length += len(header)

            for node in type_nodes:
                if current_length >= max_chars:
                    break

                # Format node
                node_text = self._format_node(node)

                # Check if we have room
                if current_length + len(node_text) > max_chars:
                    # Try truncated version
                    remaining = max_chars - current_length
                    if remaining > 50:  # Only include if we have reasonable space
                        truncated = node_text[: remaining - 3] + "..."
                        sections.append(truncated)
                        current_length += len(truncated)
                    break

                sections.append(node_text)
                current_length += len(node_text)

        return "".join(sections).strip()

    def _group_by_type(
        self,
        nodes: list[NodeSummary],
        type_order: list[str] | None = None,
    ) -> dict[str, list[NodeSummary]]:
        """Group nodes by their type.

        Args:
            nodes: Nodes to group.
            type_order: Optional override for ordering. Default puts concepts
                first; pass ["event", ...] to surface raw turns first.

        Returns:
            Dictionary mapping type to list of nodes.
        """
        grouped: dict[str, list[NodeSummary]] = {}

        if type_order is None:
            # Order: concepts first (highest level), then intents, events, time
            # "action" included for backward compatibility with legacy graphs
            type_order = ["concept", "intent", "action", "event", "time"]

        for node in nodes:
            node_type = node.node_type
            if node_type not in grouped:
                grouped[node_type] = []
            grouped[node_type].append(node)

        # Sort by type order
        result: dict[str, list[NodeSummary]] = {}
        for t in type_order:
            if t in grouped:
                result[t] = grouped[t]

        # Add any remaining types not in order
        for t in grouped:
            if t not in result:
                result[t] = grouped[t]

        return result

    def _format_node(self, node: NodeSummary) -> str:
        """Format a single node as text.

        Args:
            node: Node to format.

        Returns:
            Formatted text representation.
        """
        lines = []

        # Title with relevance indicator
        relevance_indicator = self._relevance_indicator(node.relevance_score)

        # Tag belief vs world-state nodes for theory-of-mind disambiguation
        belief_tag = ""
        if node.data:
            belief_type = node.data.get("belief_type")
            if belief_type == "agent_belief":
                holder = node.data.get("belief_holder", "unknown")
                belief_tag = f" [BELIEF: {holder}]"
            elif belief_type == "world_state":
                belief_tag = " [WORLD STATE]"

        lines.append(f"- {relevance_indicator} **{node.title}**{belief_tag}")

        # Description if available
        if node.description:
            # Truncate long descriptions
            desc = node.description
            max_desc = self.config.max_description_chars
            if len(desc) > max_desc:
                desc = desc[: max_desc - 3] + "..."
            lines.append(f"  {desc}")

        # Metadata based on config
        if self.config.include_reasoning and node.reasoning:
            lines.append(f"  _Reasoning: {node.reasoning}_")

        if self.config.include_grounding and node.grounded_in:
            grounding = ", ".join(node.grounded_in[:5])  # Limit to 5
            if len(node.grounded_in) > 5:
                grounding += f" (+{len(node.grounded_in) - 5} more)"
            lines.append(f"  _Based on: {grounding}_")

        # Append structured data fields that may contain answer-bearing info
        if node.data:
            structured_fields: list[str] = []
            for key in (
                "speaker",
                "entity",
                "location",
                "timestamp",
                "date",
                "subject",
                "object",
                "state",
                "value",
                "answer",
                "content",
                "role",
            ):
                val = node.data.get(key)
                if val and isinstance(val, str):
                    structured_fields.append(f"{key}: {val}")
            context = node.data.get("context")
            if isinstance(context, dict):
                for k, v in context.items():
                    if v is not None:
                        structured_fields.append(f"{k}: {v}")
            if structured_fields:
                lines.append("  Data: " + " | ".join(structured_fields))

        # Include edge connections if graph is available
        if self.graph:
            edges_text = self._format_node_edges(node.node_id)
            if edges_text:
                lines.append(edges_text)

        lines.append("")  # Blank line between nodes
        return "\n".join(lines)

    def _format_node_edges(self, node_id: str, max_edges: int = 3) -> str:
        """Format edges connected to a node.

        Args:
            node_id: The node ID to get edges for.
            max_edges: Maximum number of edges to show.

        Returns:
            Formatted edge text or empty string.
        """
        if not self.graph or not self.graph.has_node(node_id):
            return ""

        edges = []

        # Get outgoing edges
        try:
            for neighbor_id in list(self.graph.get_neighbors(node_id))[:max_edges]:
                neighbor = self.graph.get_node(neighbor_id)
                if neighbor:
                    neighbor_title = neighbor.data.get("title", neighbor_id)
                    edge = self.graph.get_edge(node_id, neighbor_id)
                    edge_type = getattr(edge, "edge_type", None) or "related_to"
                    edges.append(f"→ {edge_type}: {neighbor_title}")
        except Exception:
            pass  # Graph may not support edge lookups

        if not edges:
            return ""

        return "  _Connects: " + "; ".join(edges) + "_"

    def _relevance_indicator(self, score: float) -> str:
        """Get a visual indicator for relevance score.

        Args:
            score: Relevance score (0.0 to 1.0).

        Returns:
            Emoji or character indicating relevance.
        """
        if score >= 0.8:
            return "[HIGH]"
        elif score >= 0.5:
            return "[MED]"
        else:
            return "[LOW]"

    def format_for_llm(
        self,
        result: QueryResult,
        include_metadata: bool = True,
    ) -> str:
        """Format a QueryResult for LLM consumption.

        Creates a structured prompt-friendly format.

        Args:
            result: The query result to format.
            include_metadata: Include query metadata.

        Returns:
            LLM-ready context string.
        """
        parts = ["# Retrieved Context\n"]

        if include_metadata and result.query_metadata:
            parts.append("## Query Information\n")
            for key, value in result.query_metadata.items():
                parts.append(f"- {key}: {value}\n")
            parts.append("\n")

        parts.append(result.context)

        if include_metadata:
            parts.append(f"\n\n---\n_Retrieved {result.node_count} nodes")
            parts.append(f" from {result.total_nodes_scanned} scanned")
            parts.append(f" in {result.query_time_ms:.1f}ms_")

        return "".join(parts)
