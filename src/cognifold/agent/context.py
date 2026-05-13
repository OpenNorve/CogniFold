"""Context packaging for the agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph
    from cognifold.models.event import Event
    from cognifold.models.node import Node


# Phase 9.3: Configuration for concept expansion
OVERLOADED_CONCEPT_THRESHOLD = 5  # Concepts with 5+ connections need refinement
MAX_EXPANDED_CONNECTIONS = 10  # Show at most 10 connections per overloaded concept


@dataclass
class ContextNode:
    """Summarized node information for LLM context.

    Provides a condensed view of a node suitable for including
    in the LLM prompt without overwhelming the context window.
    """

    id: str
    type: str
    title: str
    score: float
    neighbor_count: int
    data_summary: str

    @classmethod
    def from_node(cls, node: Node, score: float, neighbor_count: int) -> ContextNode:
        """Create a ContextNode from a full Node."""
        title = node.data.get("title", node.id)

        # Create a brief summary of key data fields
        summary_parts = []
        for key in ("event_type", "location", "description", "strength", "status"):
            if node.data.get(key):
                summary_parts.append(f"{key}={node.data[key]}")
        data_summary = "; ".join(summary_parts[:3])  # Limit to 3 fields

        return cls(
            id=node.id,
            type=node.type.value,
            title=title,
            score=score,
            neighbor_count=neighbor_count,
            data_summary=data_summary,
        )


@dataclass
class AgentContext:
    """Packaged context for the agent.

    Contains the new event, summarized context window nodes,
    and a reference to the graph for tool calls.
    """

    event: Event
    context_nodes: list[ContextNode]
    graph: ConceptGraph  # For tool access, not direct LLM visibility
    calibration_context: str = ""  # Phase 14.1: intent personalization prompt snippet

    def format_event_for_prompt(self) -> str:
        """Format the event as a string for the LLM prompt."""
        parts = [
            f"Event ID: {self.event.event_id}",
            f"Timestamp: {self.event.timestamp.isoformat()}",
            f"Type: {self.event.event_type}",
            f"Title: {self.event.title}",
        ]
        if self.event.description:
            parts.append(f"Description: {self.event.description}")
        if self.event.location:
            parts.append(f"Location: {self.event.location}")
        if self.event.duration_minutes:
            parts.append(f"Duration: {self.event.duration_minutes} minutes")
        if self.event.metadata:
            parts.append(f"Metadata: {self.event.metadata}")

        return "\n".join(parts)

    def format_context_for_prompt(self) -> str:
        """Format the context window as a string for the LLM prompt.

        Includes both nodes and edges to show the LLM the expected edge type pattern.
        """
        if not self.context_nodes:
            return "(No existing nodes in context - this is the first event)"

        lines = ["### Nodes"]
        for node in self.context_nodes:
            line = f"- [{node.type.upper()}] {node.id}: {node.title}"
            if node.data_summary:
                line += f" ({node.data_summary})"
            line += f" [score={node.score:.3f}, neighbors={node.neighbor_count}]"
            lines.append(line)

        # Include edges to show the edge type pattern
        lines.append("")
        lines.append("### Relationships")
        context_node_ids = {n.id for n in self.context_nodes}
        edges_shown = set()
        for node in self.context_nodes:
            # Outgoing edges
            for neighbor_id in self.graph.get_neighbors(node.id):
                for edge in self.graph.get_edges_between(node.id, neighbor_id):
                    edge_key = (edge.source, edge.target, edge.edge_type)
                    if edge_key not in edges_shown:
                        edge_type = edge.edge_type or "related_to"
                        weight = edge.weight
                        lines.append(
                            f"- {edge.source} --[{edge_type} ({weight:.2f})]--> {edge.target}"
                        )
                        edges_shown.add(edge_key)
            # Incoming edges from context nodes
            for pred_id in self.graph.get_predecessors(node.id):
                if pred_id in context_node_ids:
                    for edge in self.graph.get_edges_between(pred_id, node.id):
                        edge_key = (edge.source, edge.target, edge.edge_type)
                        if edge_key not in edges_shown:
                            edge_type = edge.edge_type or "related_to"
                            weight = edge.weight
                            lines.append(
                                f"- {edge.source} --[{edge_type} ({weight:.2f})]--> {edge.target}"
                            )
                            edges_shown.add(edge_key)

        if len(edges_shown) == 0:
            lines.append("(No relationships yet)")

        # Phase 9.3: Add overloaded concepts section
        overloaded = self._get_overloaded_concepts()
        if overloaded:
            lines.append("")
            lines.append("### Concepts Needing Refinement")
            lines.append("These concepts have many connections. Consider creating sub-concepts.")
            lines.append("")
            for concept_id, connections in overloaded.items():
                concept_node = next((n for n in self.context_nodes if n.id == concept_id), None)
                title = concept_node.title if concept_node else concept_id
                lines.append(
                    f"**{title}** ({concept_id}) - {len(connections)} connections - NEEDS REFINEMENT"
                )
                # Group connections by event type if available
                for conn_id, conn_title, conn_type in connections[:MAX_EXPANDED_CONNECTIONS]:
                    lines.append(f"  - {conn_id}: {conn_title} ({conn_type})")
                if len(connections) > MAX_EXPANDED_CONNECTIONS:
                    lines.append(f"  - ... and {len(connections) - MAX_EXPANDED_CONNECTIONS} more")
                lines.append("")

        return "\n".join(lines)

    def _get_overloaded_concepts(self) -> dict[str, list[tuple[str, str, str]]]:
        """Find concepts with many connections that need refinement.

        Returns:
            Dict mapping concept_id to list of (node_id, title, type) tuples.
        """
        from cognifold.models.node import NodeType

        overloaded: dict[str, list[tuple[str, str, str]]] = {}

        for node in self.context_nodes:
            if node.type == NodeType.CONCEPT.value:
                # Get all incoming connections (events/concepts that ground this concept)
                predecessors = list(self.graph.get_predecessors(node.id))
                if len(predecessors) >= OVERLOADED_CONCEPT_THRESHOLD:
                    connections = []
                    for pred_id in predecessors:
                        try:
                            pred_node = self.graph.get_node(pred_id)
                            pred_title = pred_node.data.get("title", pred_id)
                            pred_type = pred_node.data.get("event_type", pred_node.type.value)
                            connections.append((pred_id, pred_title, pred_type))
                        except KeyError:
                            continue
                    overloaded[node.id] = connections

        return overloaded

    @classmethod
    def build(
        cls,
        event: Event,
        graph: ConceptGraph,
        context_node_ids: list[str],
        node_scores: dict[str, float],
        calibration_context: str = "",
    ) -> AgentContext:
        """Build an AgentContext from raw components.

        Args:
            event: The event being processed.
            graph: The concept graph.
            context_node_ids: IDs of nodes in the context window.
            node_scores: Composite scores for nodes.
            calibration_context: Optional intent personalization prompt snippet.

        Returns:
            A fully constructed AgentContext.
        """
        context_nodes = []
        for node_id in context_node_ids:
            try:
                node = graph.get_node(node_id)
                score = node_scores.get(node_id, 0.0)
                neighbor_count = len(graph.get_neighbors(node_id)) + len(
                    graph.get_predecessors(node_id)
                )
                context_nodes.append(ContextNode.from_node(node, score, neighbor_count))
            except KeyError:
                continue  # Skip nodes that no longer exist

        return cls(
            event=event,
            context_nodes=context_nodes,
            graph=graph,
            calibration_context=calibration_context,
        )
