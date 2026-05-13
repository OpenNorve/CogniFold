"""Graph visualization for the simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph


@dataclass(frozen=True)
class VisualizerConfig:
    """Configuration for graph visualization.

    Attributes:
        node_colors: Mapping of node types to colors.
        context_border_color: Border color for context window nodes.
        context_border_width: Border width for context window nodes.
        default_node_size: Default size for nodes.
        max_node_size: Maximum node size (for scaling by score).
        min_node_size: Minimum node size.
        physics_enabled: Whether to enable physics simulation.
        height: Height of the visualization in pixels.
        width: Width of the visualization in pixels.
    """

    node_colors: dict[str, str] | None = None
    context_border_color: str = "#FFD700"  # Gold
    context_border_width: int = 4
    default_node_size: int = 25
    max_node_size: int = 50
    min_node_size: int = 15
    physics_enabled: bool = True
    height: str = "700px"
    width: str = "100%"

    def get_node_color(self, node_type: str) -> str:
        """Get color for a node type."""
        default_colors = {
            "event": "#3B82F6",  # Blue 500
            "concept": "#059669",  # Emerald 600 (Green)
            "action": "#DC2626",  # Red 600 (Red)
            "intent": "#F97316",  # Orange 500 (Orange)
            "time": "#78716C",  # Stone 500
            "action_result": "#10B981",  # Emerald 500
        }
        if self.node_colors:
            return self.node_colors.get(node_type, "#A1A1AA")
        return default_colors.get(node_type, "#A1A1AA")


class GraphVisualizer:
    """Visualizes the concept graph using pyvis."""

    def __init__(self, config: VisualizerConfig | None = None) -> None:
        """Initialize the visualizer with optional configuration."""
        self.config = config or VisualizerConfig()

    def _get_top_nodes_by_type(
        self,
        graph: ConceptGraph,
        node_type: str,
        node_scores: dict[str, float],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get top nodes of a specific type sorted by score.

        Args:
            graph: The concept graph.
            node_type: Type of nodes to get ("concept", "intent", "time").
                       Also accepts "action" for backward compatibility.
            node_scores: Scores for ranking nodes.
            limit: Maximum number of nodes to return.

        Returns:
            List of node info dicts with id, title, score, and extra data.
        """
        # Handle backward compatibility: "action" -> "intent"
        types_to_match = {node_type}
        if node_type == "intent":
            types_to_match.add("action")  # backward compat
        elif node_type == "action":
            types_to_match.add("intent")  # forward compat

        nodes: list[dict[str, Any]] = []
        for node in graph.get_all_nodes():
            if node.type.value in types_to_match:
                score = node_scores.get(node.id, 0.0)
                title = node.data.get("title", node.id)
                strength = node.data.get("strength", None)
                status = node.data.get("status", None)
                priority = node.data.get("priority", None)

                nodes.append(
                    {
                        "id": node.id,
                        "title": title,
                        "score": score,
                        "strength": strength,
                        "status": status,
                        "priority": priority,
                    }
                )

        # Sort by score descending
        nodes.sort(key=lambda x: x["score"], reverse=True)
        return nodes[:limit]

    def _build_sidebar_html(
        self,
        top_concepts: list[dict[str, Any]],
        top_intents: list[dict[str, Any]],
        top_time_nodes: list[dict[str, Any]],
    ) -> str:
        """Build HTML for the sidebar with top nodes.

        Args:
            top_concepts: List of top concept nodes.
            top_intents: List of top intent nodes (formerly "actions").
            top_time_nodes: List of top time nodes.

        Returns:
            HTML string for the sidebar.
        """
        concept_color = self.config.get_node_color("concept")
        intent_color = self.config.get_node_color("intent")
        time_color = self.config.get_node_color("time")

        def format_node_list(nodes: list[dict[str, Any]], node_type: str) -> str:
            if not nodes:
                return '<li class="empty-state">None</li>'

            items = []
            for node in nodes:
                title = node["title"]
                score = node["score"]

                # Add extra info based on type
                extra = ""
                if node_type == "concept" and node.get("strength") is not None:
                    extra = f'<span class="meta">str: {node["strength"]:.2f}</span>'
                elif node_type == "intent":
                    status = node.get("status", "pending")
                    priority = node.get("priority", "medium")
                    extra = f'<span class="meta tag {priority}">{priority}</span>'
                    if status == "resolved":
                        extra += ' <span class="status-icon">✓</span>'

                items.append(
                    f'<li title="ID: {node["id"]}, Score: {score:.4f}">'
                    f'<div class="node-content">'
                    f'<span class="node-title">{title}</span>'
                    f'<div class="node-details">{extra}</div>'
                    f"</div>"
                    f"</li>"
                )
            return "\n".join(items)

        return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    #sidebar {{
        position: fixed;
        right: 0;
        top: 0;
        width: 300px;
        height: 100%;
        background: #ffffff;
        border-left: 1px solid #e2e8f0;
        padding: 20px;
        overflow-y: auto;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 13px;
        z-index: 1000;
        box-shadow: -4px 0 16px rgba(0,0,0,0.05);
    }}
    #sidebar h3 {{
        margin: 24px 0 12px 0;
        padding: 0;
        color: #64748b;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        background: none;
    }}
    #sidebar h3:first-child {{
        margin-top: 0;
    }}
    #sidebar ul {{
        list-style: none;
        padding: 0;
        margin: 0;
    }}
    #sidebar li {{
        padding: 10px 12px;
        margin: 6px 0;
        background: #f8fafc;
        border-radius: 6px;
        border-left: 3px solid #cbd5e1;
        cursor: pointer;
        transition: all 0.2s;
        border: 1px solid #e2e8f0;
        border-left-width: 3px;
    }}
    #sidebar li:hover {{
        background: #fff;
        border-color: #cbd5e1;
        transform: translateX(-2px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    #sidebar li.empty-state {{
        color: #94a3b8;
        background: none;
        border: 1px dashed #e2e8f0;
        text-align: center;
        cursor: default;
    }}
    #sidebar li.empty-state:hover {{
        transform: none;
        box-shadow: none;
        background: none;
    }}

    .node-content {{
        display: flex;
        flex-direction: column;
        gap: 4px;
    }}
    .node-title {{
        font-weight: 500;
        color: #1e293b;
        line-height: 1.4;
    }}
    .node-details {{
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        color: #64748b;
    }}
    .meta {{
        color: #64748b;
    }}
    .tag {{
        padding: 1px 6px;
        border-radius: 4px;
        text-transform: uppercase;
        font-size: 10px;
        font-weight: 600;
    }}
    .tag.high {{ background: #fee2e2; color: #ef4444; }}
    .tag.medium {{ background: #fef3c7; color: #d97706; }}
    .tag.low {{ background: #f1f5f9; color: #64748b; }}

    .status-icon {{
        color: #22c55e;
        font-weight: bold;
    }}

    .concept-header {{ color: {concept_color} !important; }}
    .concept-item {{ border-left-color: {concept_color} !important; }}
    .intent-header {{ color: {intent_color} !important; }}
    .intent-item {{ border-left-color: {intent_color} !important; }}
    .time-header {{ color: {time_color} !important; }}
    .time-item {{ border-left-color: {time_color} !important; }}

    #mynetwork {{
        width: calc(100% - 300px) !important;
    }}
    .card {{
        width: calc(100% - 300px) !important;
    }}
</style>
<div id="sidebar">
    <h3 class="concept-header">Top Concepts</h3>
    <ul>
        {format_node_list(top_concepts, "concept").replace("<li>", '<li class="concept-item">')}
    </ul>

    <h3 class="intent-header">Intents</h3>
    <ul>
        {format_node_list(top_intents, "intent").replace("<li>", '<li class="intent-item">')}
    </ul>

    <h3 class="time-header">Time Anchors</h3>
    <ul>
        {format_node_list(top_time_nodes, "time").replace("<li>", '<li class="time-item">')}
    </ul>
</div>
"""

    def render(
        self,
        graph: ConceptGraph,
        output_path: str | Path,
        context_node_ids: list[str] | None = None,
        node_scores: dict[str, float] | None = None,
        title: str = "Cognifold Graph",
    ) -> Path:
        """Render the graph to an HTML file.

        Args:
            graph: The concept graph to visualize.
            output_path: Path for the output HTML file.
            context_node_ids: Node IDs in the context window (highlighted).
            node_scores: Optional scores for node size scaling.
            title: Title for the visualization.

        Returns:
            Path to the generated HTML file.
        """
        try:
            from pyvis.network import Network  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "pyvis is required for visualization. Install with: pip install pyvis"
            ) from e

        output_path = Path(output_path)
        context_node_ids = context_node_ids or []
        node_scores = node_scores or {}

        # Create network
        net = Network(
            height=self.config.height,
            width=self.config.width,
            bgcolor="#ffffff",
            font_color="#000000",  # type: ignore[arg-type]
            heading=title,
        )

        # Configure physics
        if not self.config.physics_enabled:
            net.toggle_physics(False)

        # Calculate score range for size scaling
        if node_scores:
            max_score = max(node_scores.values()) if node_scores else 1.0
            min_score = min(node_scores.values()) if node_scores else 0.0
            score_range = max_score - min_score if max_score > min_score else 1.0
        else:
            max_score = min_score = score_range = 1.0

        # Add nodes
        for node in graph.get_all_nodes():
            node_id = node.id
            node_type = node.type.value
            color = self.config.get_node_color(node_type)

            # Calculate size based on score
            if node_id in node_scores and score_range > 0:
                normalized = (node_scores[node_id] - min_score) / score_range
                size = self.config.min_node_size + normalized * (
                    self.config.max_node_size - self.config.min_node_size
                )
            else:
                size = self.config.default_node_size

            # Get title from data
            title_text = node.data.get("title", node_id)

            # Build label
            label = f"{title_text}\n({node_type})"

            # Check if in context window
            is_context = node_id in context_node_ids
            border_width = self.config.context_border_width if is_context else 1
            border_color = self.config.context_border_color if is_context else color

            # Build tooltip with explainability info
            tooltip_parts = [
                f"ID: {node_id}",
                f"Type: {node_type}",
                f"Access: {node.access_count}",
            ]

            # Add reasoning if present
            if node.reasoning:
                tooltip_parts.append(f"\nReasoning: {node.reasoning}")

            # Add grounded_in if present
            if node.grounded_in:
                grounded_str = ", ".join(node.grounded_in[:5])
                if len(node.grounded_in) > 5:
                    grounded_str += f" (+{len(node.grounded_in) - 5} more)"
                tooltip_parts.append(f"Grounded in: {grounded_str}")

            # Add update history summary if present
            if node.update_history:
                latest = node.update_history[-1]
                tooltip_parts.append(f"\nLast update: {latest.update_reasoning}")
                tooltip_parts.append(f"Updates: {len(node.update_history)} total")

            tooltip = "\n".join(tooltip_parts)

            net.add_node(
                node_id,
                label=label,
                title=tooltip,
                color={  # type: ignore[arg-type]
                    "background": color,
                    "border": border_color,
                    "highlight": {"background": color, "border": "#FF0000"},
                },
                size=size,
                borderWidth=border_width,
            )

        # Add edges
        for edge in graph.get_all_edges():
            net.add_edge(edge.source, edge.target)

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(str(output_path))

        # Inject sidebar with top concepts and intents
        top_concepts = self._get_top_nodes_by_type(graph, "concept", node_scores, limit=10)
        top_intents = self._get_top_nodes_by_type(graph, "intent", node_scores, limit=10)
        top_time_nodes = self._get_top_nodes_by_type(graph, "time", node_scores, limit=5)

        sidebar_html = self._build_sidebar_html(top_concepts, top_intents, top_time_nodes)

        # Read the generated HTML and inject sidebar
        with open(output_path) as f:
            html_content = f.read()

        # Inject sidebar right after <body>
        html_content = html_content.replace("<body>", f"<body>\n{sidebar_html}")

        with open(output_path, "w") as f:
            f.write(html_content)

        return output_path

    def render_step(
        self,
        graph: ConceptGraph,
        output_dir: str | Path,
        step_number: int,
        context_node_ids: list[str] | None = None,
        node_scores: dict[str, float] | None = None,
        event_title: str = "",
    ) -> Path:
        """Render a single simulation step.

        Args:
            graph: The concept graph state.
            output_dir: Directory for output files.
            step_number: Current step number.
            context_node_ids: Nodes in context window.
            node_scores: Scores for size scaling.
            event_title: Title of the current event.

        Returns:
            Path to the generated HTML file.
        """
        output_dir = Path(output_dir)
        output_path = output_dir / f"step_{step_number:03d}.html"

        title = f"Step {step_number}"
        if event_title:
            title = f"{title}: {event_title}"

        return self.render(
            graph=graph,
            output_path=output_path,
            context_node_ids=context_node_ids,
            node_scores=node_scores,
            title=title,
        )
