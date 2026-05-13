"""HTML renderer for graph evolution replay.

This module generates an interactive HTML file with playback controls
for visualizing how the graph evolved during a simulation run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cognifold.replay.player import ReplayPlayer


@dataclass
class ReplayRendererConfig:
    """Configuration for the replay renderer.

    Attributes:
        node_colors: Mapping of node types to colors.
        context_border_color: Border color for context window nodes.
        default_speed: Default playback speed (1.0 = normal).
        height: Height of the visualization.
        width: Width of the visualization.
        animation_duration_ms: Duration of animations in milliseconds.
    """

    node_colors: dict[str, str] | None = None
    context_border_color: str = "#FFD700"
    default_speed: float = 1.0
    height: str = "600px"
    width: str = "100%"
    animation_duration_ms: int = 500

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


class ReplayRenderer:
    """Renders interactive HTML replay from keyframes."""

    def __init__(self, config: ReplayRendererConfig | None = None) -> None:
        """Initialize the renderer.

        Args:
            config: Optional configuration.
        """
        self.config = config or ReplayRendererConfig()

    def render(
        self,
        player: ReplayPlayer,
        output_path: str | Path,
        title: str | None = None,
    ) -> Path:
        """Render the replay to an HTML file.

        Args:
            player: ReplayPlayer with keyframes.
            output_path: Path for the output HTML file.
            title: Optional title for the visualization.

        Returns:
            Path to the generated HTML file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        title = title or f"Graph Evolution Replay - {player.timeline_path}"

        # Convert keyframes to JSON for JavaScript
        keyframes_json = json.dumps(
            [kf.to_dict() for kf in player.keyframes],
            indent=2,
        )

        # Generate node colors config
        colors_json = json.dumps(
            {
                "event": self.config.get_node_color("event"),
                "concept": self.config.get_node_color("concept"),
                "action": self.config.get_node_color("action"),
                "intent": self.config.get_node_color("intent"),
                "time": self.config.get_node_color("time"),
                "action_result": self.config.get_node_color("action_result"),
            }
        )

        html = self._build_html(
            title=title,
            keyframes_json=keyframes_json,
            colors_json=colors_json,
            total_steps=player.total_steps,
            metadata=player.metadata,
        )

        with open(output_path, "w") as f:
            f.write(html)

        return output_path

    def _build_html(
        self,
        title: str,
        keyframes_json: str,
        colors_json: str,
        total_steps: int,
        metadata: dict[str, Any],
    ) -> str:
        """Build the complete HTML document."""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700&family=Inter:wght@400;500;600&display=swap');

        :root {{
            --bg-color: #FAFAF9;
            --sidebar-bg: #F4F4F5;
            --text-main: #27272A;
            --text-muted: #71717A;
            --border-color: #E4E4E7;
            --primary: #3B82F6;
            --primary-hover: #2563EB;
            --card-bg: #FFFFFF;
            --success: #059669;
            --warning: #B45309;
            --error: #DC2626;
            --font-serif: 'Merriweather', 'Georgia', serif;
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: var(--font-sans);
            background: var(--bg-color);
            color: var(--text-main);
            height: 100vh;
            display: flex;
            flex-direction: column;
            line-height: 1.6;
        }}
        .header {{
            background: var(--card-bg);
            padding: 16px 32px;
            border-bottom: 1px solid var(--border-color);
            z-index: 10;
        }}
        .header h1 {{
            font-family: var(--font-serif);
            font-size: 20px;
            font-weight: 700;
            color: var(--text-main);
            letter-spacing: -0.01em;
        }}
        .main {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}
        .graph-container {{
            flex: 1;
            position: relative;
            background: var(--bg-color);
        }}
        #network {{
            width: 100%;
            height: 100%;
        }}
        .sidebar {{
            width: 380px;
            background: var(--sidebar-bg);
            border-left: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            z-index: 20;
        }}
        .sidebar-section {{
            padding: 24px;
            border-bottom: 1px solid var(--border-color);
        }}
        .sidebar-section h3 {{
            font-family: var(--font-serif);
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 16px;
            font-weight: 700;
        }}
        .event-info {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 20px;
            border: 1px solid var(--border-color);
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }}
        .event-info .event-type {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 500;
            margin-bottom: 12px;
            background: #E7E5E4;
            color: #57534E;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .event-info .event-title {{
            font-family: var(--font-serif);
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 8px;
            color: var(--text-main);
            line-height: 1.4;
        }}
        .event-info .event-id {{
            font-size: 12px;
            color: var(--text-muted);
            font-family: var(--font-mono);
        }}
        .operations-list {{
            max-height: 200px;
            overflow-y: auto;
            padding-right: 4px;
        }}
        /* Custom Scrollbar */
        ::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        ::-webkit-scrollbar-track {{
            background: transparent;
        }}
        ::-webkit-scrollbar-thumb {{
            background: #D4D4D8;
            border-radius: 3px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: #A1A1AA;
        }}

        .operation {{
            background: var(--card-bg);
            border-radius: 6px;
            padding: 12px 16px;
            margin-bottom: 10px;
            font-size: 13px;
            border: 1px solid var(--border-color);
            transition: all 0.2s ease;
        }}
        .operation:hover {{
            border-color: #D4D4D8;
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }}
        .operation .op-type {{
            font-weight: 600;
            color: var(--primary);
            margin-bottom: 4px;
            display: block;
            font-family: var(--font-mono);
            font-size: 11px;
            text-transform: uppercase;
        }}
        .reasoning {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 20px;
            font-size: 14px;
            line-height: 1.7;
            max-height: 150px;
            overflow-y: auto;
            color: var(--text-main);
            border: 1px solid var(--border-color);
            font-family: var(--font-serif);
        }}
        .stats {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }}
        .stat {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            border: 1px solid var(--border-color);
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: 700;
            color: var(--primary);
            line-height: 1;
            margin-bottom: 6px;
            font-family: var(--font-serif);
        }}
        .stat-label {{
            font-size: 11px;
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .controls {{
            background: var(--card-bg);
            padding: 20px 32px;
            border-top: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 24px;
            z-index: 20;
        }}
        .playback-buttons {{
            display: flex;
            gap: 10px;
        }}
        .btn {{
            background: var(--bg-color);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }}
        .btn:hover {{
            background: #E7E5E4;
            transform: translateY(-1px);
        }}
        .btn:active {{
            background: #D6D3D1;
            transform: translateY(0);
        }}
        .btn.primary {{
            background: var(--primary);
            color: #fff;
            border-color: var(--primary);
        }}
        .btn.primary:hover {{
            background: var(--primary-hover);
            border-color: var(--primary-hover);
        }}
        .timeline {{
            flex: 1;
            display: flex;
            align-items: center;
            gap: 16px;
        }}
        .timeline-slider {{
            flex: 1;
            -webkit-appearance: none;
            height: 4px;
            border-radius: 2px;
            background: #E7E5E4;
            outline: none;
        }}
        .timeline-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: var(--primary);
            cursor: pointer;
            border: 2px solid #fff;
            box-shadow: 0 0 0 1px rgba(0,0,0,0.1);
            transition: transform 0.1s;
        }}
        .timeline-slider::-webkit-slider-thumb:hover {{
            transform: scale(1.2);
        }}
        .step-indicator {{
            font-size: 13px;
            min-width: 80px;
            text-align: center;
            color: var(--text-muted);
            font-feature-settings: "tnum";
            font-variant-numeric: tabular-nums;
            font-family: var(--font-mono);
        }}
        .speed-control {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .speed-control span {{
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 500;
        }}
        .speed-control select {{
            background: var(--bg-color);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 13px;
            outline: none;
            cursor: pointer;
            font-family: var(--font-sans);
        }}
        .layout-controls {{
            display: flex;
            gap: 8px;
            margin-left: 16px;
            padding-left: 16px;
            border-left: 1px solid var(--border-color);
        }}
        .legend {{
            display: flex;
            gap: 12px;
            padding: 16px 32px;
            background: var(--bg-color);
            border-top: 1px solid var(--border-color);
            flex-wrap: wrap;
            align-items: center;
        }}
        .filter-label {{
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 600;
            margin-right: 8px;
        }}
        .filter-btn {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 500;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 6px 12px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .filter-btn:hover {{
            border-color: var(--filter-color, var(--border-color));
            background: #F5F5F4;
        }}
        .filter-btn.active {{
            background: var(--filter-color, var(--primary));
            color: white;
            border-color: var(--filter-color, var(--primary));
        }}
        .filter-btn.active .legend-color {{
            background: white !important;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 500;
        }}
        .legend-color {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }}
        .added-node {{
            animation: pulse 0.5s ease-out;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(1.5); opacity: 0.5; }}
            100% {{ transform: scale(1); opacity: 1; }}
        }}
        .context-list {{
            max-height: 120px;
            overflow-y: auto;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .context-node {{
            display: inline-flex;
            align-items: center;
            background: #FFFBEB;
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 11px;
            border: 1px solid #FCD34D;
            color: #B45309;
            font-weight: 500;
        }}
        .top-nodes-list {{
            max-height: 150px;
            overflow-y: auto;
            padding-right: 4px;
        }}
        .top-node-item {{
            background: var(--card-bg);
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 8px;
            font-size: 13px;
            border: 1px solid var(--border-color);
            transition: transform 0.2s;
        }}
        .top-node-item:hover {{
            transform: translateX(2px);
            border-color: #D4D4D8;
        }}
        .top-node-item.concept {{
            border-left: 3px solid #059669;
        }}
        .top-node-item.action {{
            border-left: 3px solid #F97316;
        }}
        .top-node-item .node-title {{
            font-weight: 600;
            margin-bottom: 4px;
            color: var(--text-main);
        }}
        .top-node-item .node-meta {{
            font-size: 11px;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .top-node-item .node-score {{
            color: var(--primary);
            font-weight: 600;
            background: #FFF7ED;
            padding: 2px 6px;
            border-radius: 4px;
        }}
        .top-node-item .node-priority {{
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.05em;
        }}
        .top-node-item .node-priority.high {{
            background: #FEF2F2;
            color: #DC2626;
        }}
        .top-node-item .node-priority.medium {{
            background: #FFFBEB;
            color: #B45309;
        }}
        .top-node-item .node-priority.low {{
            background: #F4F4F5;
            color: #71717A;
        }}
        .action-flow-list {{
            max-height: 180px;
            overflow-y: auto;
            padding-right: 4px;
        }}
        .action-flow-item {{
            background: var(--card-bg);
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 8px;
            font-size: 12px;
            border: 1px solid var(--border-color);
        }}
        .action-flow-item.intent-selected {{
            border-left: 3px solid #F97316;
        }}
        .action-flow-item.action-generated {{
            border-left: 3px solid #DC2626;
        }}
        .action-flow-item.action-executed {{
            border-left: 3px solid #DC2626;
        }}
        .action-flow-item.action-result {{
            border-left: 3px solid #10B981;
        }}
        .action-flow-item .flow-type {{
            font-weight: 700;
            font-size: 10px;
            text-transform: uppercase;
            margin-bottom: 6px;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }}
        .action-flow-item .flow-title {{
            color: var(--text-main);
            font-weight: 500;
            font-family: var(--font-serif);
        }}
        .action-flow-item .flow-meta {{
            color: var(--text-muted);
            font-size: 11px;
            margin-top: 6px;
            border-top: 1px solid var(--border-color);
            padding-top: 6px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
    </div>
    <div class="main">
        <div class="graph-container">
            <div id="network"></div>
        </div>
        <div class="sidebar">
            <div class="sidebar-section">
                <h3>Current Event</h3>
                <div class="event-info" id="event-info">
                    <span class="event-type" id="event-type">-</span>
                    <div class="event-title" id="event-title">Initial State</div>
                    <div class="event-id" id="event-id">Step 0</div>
                </div>
            </div>
            <div class="sidebar-section">
                <h3 onclick="const el = document.getElementById('event-desc-content'); el.style.display = el.style.display === 'none' ? 'block' : 'none'" style="cursor: pointer; display: flex; justify-content: space-between; align-items: center;">
                    Event Content <span style="font-size: 12px">▼</span>
                </h3>
                <div class="event-desc" id="event-desc-content" style="display: none; font-size: 13px; color: var(--text-muted); max-height: 200px; overflow-y: auto; white-space: pre-wrap; margin-top: 10px; font-family: var(--font-serif);">
                    -
                </div>
            </div>
            <div class="sidebar-section">
                <h3>Graph Stats</h3>
                <div class="stats">
                    <div class="stat">
                        <div class="stat-value" id="node-count">0</div>
                        <div class="stat-label">Nodes</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="edge-count">0</div>
                        <div class="stat-label">Edges</div>
                    </div>
                </div>
            </div>
            <div class="sidebar-section">
                <h3 style="color: #059669;">Top Concepts</h3>
                <div class="top-nodes-list" id="top-concepts-list">
                    <span style="color: var(--text-muted); font-size: 12px;">None yet</span>
                </div>
            </div>
            <div class="sidebar-section">
                <h3 style="color: #F97316;">Top Intents</h3>
                <div class="top-nodes-list" id="top-actions-list">
                    <span style="color: var(--text-muted); font-size: 12px;">None yet</span>
                </div>
            </div>
            <div class="sidebar-section" id="action-flow-section" style="display: none;">
                <h3 style="color: #DC2626;">Action Flow</h3>
                <div class="action-flow-list" id="action-flow-list">
                    <span style="color: var(--text-muted); font-size: 12px;">No action activity</span>
                </div>
            </div>
            <div class="sidebar-section" style="flex: 1; overflow: hidden;">
                <h3>Operations Applied</h3>
                <div class="operations-list" id="operations-list">
                    <div class="operation" style="color: var(--text-muted);">No operations yet</div>
                </div>
            </div>
            <div class="sidebar-section">
                <h3>Context Window</h3>
                <div class="context-list" id="context-list">
                    <span style="color: var(--text-muted); font-size: 12px;">Empty</span>
                </div>
            </div>
        </div>
    </div>
    <div class="legend">
        <div class="filter-label">Filter:</div>
        <button class="filter-btn active" data-type="event" style="--filter-color: #3B82F6;">
            <div class="legend-color" style="background: #3B82F6;"></div>
            <span>Event</span>
        </button>
        <button class="filter-btn active" data-type="concept" style="--filter-color: #059669;">
            <div class="legend-color" style="background: #059669;"></div>
            <span>Concept</span>
        </button>
        <button class="filter-btn active" data-type="intent" style="--filter-color: #F97316;">
            <div class="legend-color" style="background: #F97316;"></div>
            <span>Intent</span>
        </button>
        <button class="filter-btn active" data-type="action" style="--filter-color: #DC2626;">
            <div class="legend-color" style="background: #DC2626;"></div>
            <span>Action</span>
        </button>
        <button class="filter-btn active" data-type="time" style="--filter-color: #78716C;">
            <div class="legend-color" style="background: #78716C;"></div>
            <span>Time</span>
        </button>
        <div class="legend-item" style="margin-left: 20px;">
            <div class="legend-color" style="background: transparent; border: 2px solid #B45309;"></div>
            <span>In Context</span>
        </div>
    </div>
    <div class="controls">
        <div class="playback-buttons">
            <button class="btn" id="btn-start" title="Go to start">⏮</button>
            <button class="btn" id="btn-prev" title="Previous step">⏪</button>
            <button class="btn primary" id="btn-play" title="Play/Pause">▶</button>
            <button class="btn" id="btn-next" title="Next step">⏩</button>
            <button class="btn" id="btn-end" title="Go to end">⏭</button>
        </div>
        <div class="timeline">
            <input type="range" class="timeline-slider" id="timeline-slider" min="0" max="{total_steps}" value="0">
            <div class="step-indicator" id="step-indicator">0 / {total_steps}</div>
        </div>
        <div class="speed-control">
            <span>Speed:</span>
            <select id="speed-select">
                <option value="0.5">0.5x</option>
                <option value="1" selected>1x</option>
                <option value="2">2x</option>
                <option value="5">5x</option>
            </select>
        </div>
        <div class="layout-controls">
            <button class="btn" id="btn-reorganize" title="Reorganize graph layout">⟳</button>
            <button class="btn" id="btn-fit" title="Fit graph to view">⊡</button>
        </div>
    </div>

    <script>
        // Keyframes data
        const keyframes = {keyframes_json};
        const nodeColors = {colors_json};
        const totalSteps = {total_steps};

        // State
        let currentStep = 0;
        let isPlaying = false;
        let playInterval = null;
        let playSpeed = 1;

        // Network
        let network = null;
        let nodes = new vis.DataSet([]);
        let edges = new vis.DataSet([]);

        // Initialize network
        function initNetwork() {{
            const container = document.getElementById('network');
            const data = {{ nodes, edges }};
            const options = {{
                nodes: {{
                    shape: 'dot',
                    font: {{ color: '#57534E', size: 14, face: 'Inter' }},
                    borderWidth: 2,
                    shadow: {{ enabled: true, color: 'rgba(0,0,0,0.1)', size: 4, x: 0, y: 2 }}
                }},
                edges: {{
                    color: {{ color: '#A8A29E', highlight: '#78716C' }},
                    arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }},
                    smooth: {{ type: 'continuous' }}
                }},
                physics: {{
                    enabled: true,
                    stabilization: {{ iterations: 200 }},
                    barnesHut: {{
                        gravitationalConstant: -15000,
                        centralGravity: 0.1,
                        springConstant: 0.015,
                        springLength: 300,
                        damping: 0.2,
                        avoidOverlap: 0.5
                    }}
                }},
                interaction: {{
                    hover: true,
                    tooltipDelay: 200,
                }},
            }};
            network = new vis.Network(container, data, options);

            // Handle node selection - highlight connected nodes
            let selectedNode = null;

            network.on("click", function (params) {{
                if (params.nodes.length > 0) {{
                    const nodeId = params.nodes[0];

                    // If clicking the same node, deselect
                    if (selectedNode === nodeId) {{
                        selectedNode = null;
                        resetHighlight();
                        return;
                    }}

                    selectedNode = nodeId;
                    highlightConnected(nodeId);
                }} else {{
                    // Clicked on empty space - reset
                    selectedNode = null;
                    resetHighlight();
                }}
            }});
        }}

        // Highlight only nodes connected to the selected node
        function highlightConnected(nodeId) {{
            const kf = keyframes.find(k => k.step === currentStep);
            if (!kf) return;

            // Find all connected node IDs (both directions)
            const connectedIds = new Set([nodeId]);
            kf.edges.forEach(e => {{
                if (e.source === nodeId) connectedIds.add(e.target);
                if (e.target === nodeId) connectedIds.add(e.source);
            }});

            // Update all nodes - dim non-connected ones
            const allNodes = nodes.get();
            allNodes.forEach(node => {{
                const isConnected = connectedIds.has(node.id);
                const nodeData = kf.nodes.find(n => n.id === node.id);
                if (!nodeData) return;

                const nodeType = nodeData.type;
                const isActionResult = nodeType === 'event' && nodeData.data && nodeData.data.event_type === 'action_result';
                const baseColor = isActionResult ? nodeColors['action_result'] : (nodeColors[nodeType] || '#808080');

                if (isConnected) {{
                    // Highlight connected nodes
                    nodes.update({{
                        id: node.id,
                        opacity: 1,
                        color: {{
                            background: baseColor,
                            border: node.id === nodeId ? '#DC2626' : baseColor,
                        }},
                        borderWidth: node.id === nodeId ? 5 : 3,
                        font: {{ color: '#27272A' }}
                    }});
                }} else {{
                    // Dim non-connected nodes
                    nodes.update({{
                        id: node.id,
                        opacity: 0.15,
                        color: {{
                            background: '#E4E4E7',
                            border: '#E4E4E7',
                        }},
                        borderWidth: 1,
                        font: {{ color: '#D4D4D8' }}
                    }});
                }}
            }});

            // Update edges - dim non-connected ones
            const allEdges = edges.get();
            allEdges.forEach(edge => {{
                const isConnected = (edge.from === nodeId || edge.to === nodeId);
                if (isConnected) {{
                    edges.update({{
                        id: edge.id,
                        color: {{ opacity: 1 }},
                        width: 3
                    }});
                }} else {{
                    edges.update({{
                        id: edge.id,
                        color: {{ opacity: 0.1 }},
                        width: 0.5
                    }});
                }}
            }});
        }}

        // Reset all nodes to normal state
        function resetHighlight() {{
            selectedNode = null;
            // Re-render the current keyframe to restore original styling
            renderKeyframe(currentStep);
        }}

        // Clear selection when changing steps
        const originalGoToStep = goToStep;
        goToStep = function(step) {{
            selectedNode = null;
            originalGoToStep(step);
        }};

        // Render keyframe
        function renderKeyframe(step) {{
            const kf = keyframes.find(k => k.step === step);
            if (!kf) return;

            // Update nodes
            const newNodeIds = new Set(kf.nodes.map(n => n.id));
            const existingIds = new Set(nodes.getIds());

            // Remove nodes not in keyframe
            existingIds.forEach(id => {{
                if (!newNodeIds.has(id)) {{
                    nodes.remove(id);
                }}
            }});

            // Add/update nodes
            kf.nodes.forEach(node => {{
                const nodeType = node.type;
                // Check if this is an action_result event (special color)
                const isActionResult = nodeType === 'event' && node.data && node.data.event_type === 'action_result';
                const color = isActionResult ? nodeColors['action_result'] : (nodeColors[nodeType] || '#808080');
                const title = node.data.title || (node.data.description ? (node.data.description.length > 50 ? node.data.description.substring(0, 50) + '...' : node.data.description) : node.id);
                const isInContext = kf.context_node_ids.includes(node.id);
                const isNew = kf.added_nodes.includes(node.id);
                const score = kf.scores[node.id] || 0;
                const size = 15 + (score * 35);

                // Build tooltip with explainability info
                let tooltip = `ID: ${{node.id}}\\nType: ${{nodeType}}\\nScore: ${{score.toFixed(4)}}`;
                if (node.reasoning) {{
                    tooltip += `\\n\\nReasoning: ${{node.reasoning}}`;
                }}
                if (node.grounded_in && node.grounded_in.length > 0) {{
                    const groundedStr = node.grounded_in.slice(0, 5).join(', ') + (node.grounded_in.length > 5 ? '...' : '');
                    tooltip += `\\nGrounded in: ${{groundedStr}}`;
                }}

                const nodeData = {{
                    id: node.id,
                    label: title,
                    color: {{
                        background: color,
                        border: isInContext ? '#B45309' : color,
                        highlight: {{ background: color, border: '#DC2626' }},
                    }},
                    borderWidth: isInContext ? 4 : 2,
                    size: size,
                    title: tooltip,
                    opacity: 1,
                    font: {{ color: '#57534E', size: 14, face: 'Inter' }},
                    hidden: false,
                }};

                if (existingIds.has(node.id)) {{
                    nodes.update(nodeData);
                }} else {{
                    nodes.add(nodeData);
                }}
            }});

            // Edge type colors
            const edgeTypeColors = {{
                'grounds': '#4cc9f0',      // Cyan - evidence
                'causes': '#f72585',       // Pink - causal
                'reinforces': '#7209b7',   // Purple - supporting
                'triggers': '#ff8c00',     // Orange - activation
                'part_of': '#06d6a0',      // Teal - containment
                'derived_from': '#ffd166', // Yellow - derivation
                'deadline_for': '#ef476f', // Red - temporal
                'related_to': '#888888',   // Gray - generic
            }};

            // Update edges
            const newEdges = kf.edges.map(e => {{
                const edgeType = e.edge_type || 'related_to';
                // Handle null, undefined, and missing weight values
                const weight = (e.weight !== null && e.weight !== undefined) ? e.weight : 0.5;
                return {{
                    from: e.source,
                    to: e.target,
                    edgeType: edgeType,
                    weight: weight,
                    key: `${{e.source}}-${{e.target}}-${{edgeType}}`
                }};
            }});
            const newEdgeKeys = new Set(newEdges.map(e => e.key));
            const existingEdges = edges.get();
            const existingEdgeKeys = new Set(existingEdges.map(e => e.key || `${{e.from}}-${{e.to}}-related_to`));

            // Remove edges not in keyframe
            existingEdges.forEach(e => {{
                const key = e.key || `${{e.from}}-${{e.to}}-related_to`;
                if (!newEdgeKeys.has(key)) {{
                    edges.remove(e.id);
                }}
            }});

            // Add or update edges with type info
            newEdges.forEach(e => {{
                const color = edgeTypeColors[e.edgeType] || '#888888';
                const safeWeight = (e.weight !== null && e.weight !== undefined) ? e.weight : 0.5;
                const edgeData = {{
                    from: e.from,
                    to: e.to,
                    key: e.key,
                    label: e.edgeType,
                    title: `${{e.edgeType}} (weight: ${{safeWeight.toFixed(2)}})`,
                    color: {{ color: color, highlight: color, hover: color, opacity: 1 }},
                    font: {{ size: 9, color: '#aaa', strokeWidth: 0 }},
                    width: 1 + (safeWeight * 2),
                    hidden: false,
                }};
                if (!existingEdgeKeys.has(e.key)) {{
                    edges.add(edgeData);
                }} else {{
                    // Find and update existing edge
                    const existingEdge = existingEdges.find(edge => (edge.key || `${{edge.from}}-${{edge.to}}-related_to`) === e.key);
                    if (existingEdge) {{
                        edgeData.id = existingEdge.id;
                        edges.update(edgeData);
                    }}
                }}
            }});

            // Update UI
            updateUI(kf);
        }}

        // Update UI elements
        function updateUI(kf) {{
            // Event info
            const typeEl = document.getElementById('event-type');
            typeEl.textContent = kf.event_type || '-';
            typeEl.style.background = '#E7E5E4';
            typeEl.style.color = '#57534E';

            document.getElementById('event-title').textContent = kf.event_title;
            document.getElementById('event-id').textContent = kf.event_id ? `Step ${{kf.step}} - ${{kf.event_id}}` : 'Step 0';

            // Event Description
            const eventNode = kf.nodes.find(n => n.id === kf.event_id);
            const desc = (eventNode && eventNode.data && eventNode.data.description) ? eventNode.data.description : 'No description available';
            document.getElementById('event-desc-content').textContent = desc;

            // Stats
            document.getElementById('node-count').textContent = kf.nodes.length;
            document.getElementById('edge-count').textContent = kf.edges.length;

            // Operations
            const opsList = document.getElementById('operations-list');
            if (kf.operations.length === 0) {{
                opsList.innerHTML = '<div class="operation" style="color: var(--text-muted);">No operations</div>';
            }} else {{
                opsList.innerHTML = kf.operations.map(op => `
                    <div class="operation">
                        <span class="op-type">${{op.op}}</span>
                        ${{formatOpDetails(op)}}
                    </div>
                `).join('');
            }}

            // Context window
            const contextList = document.getElementById('context-list');
            if (kf.context_node_ids.length === 0) {{
                contextList.innerHTML = '<span style="color: var(--text-muted); font-size: 12px;">Empty</span>';
            }} else {{
                contextList.innerHTML = kf.context_node_ids.map(id => `
                    <span class="context-node">${{id}}</span>
                `).join('');
            }}

            // Top Concepts
            updateTopNodes(kf, 'concept', 'top-concepts-list', 5);

            // Top Intents (formerly actions)
            updateTopNodes(kf, 'intent', 'top-actions-list', 5);
            // Fallback to 'action' type for backward compatibility
            if (kf.nodes.filter(n => n.type === 'intent').length === 0) {{
                updateTopNodes(kf, 'action', 'top-actions-list', 5);
            }}

            // Action Flow (Phase 8)
            updateActionFlow(kf);

            // Timeline
            document.getElementById('timeline-slider').value = kf.step;
            document.getElementById('step-indicator').textContent = `${{kf.step}} / ${{totalSteps}}`;
        }}

        function formatOpDetails(op) {{
            let details = '';
            let reasoning = '';

            if (op.op === 'ADD_NODE') {{
                const data = op.data || {{}};
                const title = data.title || data.event_id || data.concept_id || data.action_id || 'node';
                details = `: ${{op.node_type}} "${{title}}"`;
                if (op.reasoning) {{
                    reasoning = `<div style="font-size: 11px; color: var(--text-muted); margin-top: 4px; font-style: italic;">Reasoning: ${{op.reasoning}}</div>`;
                }}
                if (op.grounded_in && op.grounded_in.length > 0) {{
                    const groundedStr = op.grounded_in.slice(0, 3).join(', ') + (op.grounded_in.length > 3 ? '...' : '');
                    reasoning += `<div style="font-size: 10px; color: var(--text-muted); margin-top: 2px;">Grounded in: ${{groundedStr}}</div>`;
                }}
            }} else if (op.op === 'ADD_EDGE') {{
                const edgeType = op.edge_type || 'related_to';
                const hasWeight = op.weight !== null && op.weight !== undefined;
                const weightStr = hasWeight ? ` (${{op.weight.toFixed(2)}})` : '';
                details = `: ${{op.source_id}} --[${{edgeType}}${{weightStr}}]--> ${{op.target_id}}`;
            }} else if (op.op === 'UPDATE_NODE') {{
                details = `: ${{op.node_id}}`;
                if (op.update_reasoning) {{
                    reasoning = `<div style="font-size: 11px; color: var(--text-muted); margin-top: 4px; font-style: italic;">Reason: ${{op.update_reasoning}}</div>`;
                }}
            }} else if (op.op === 'REMOVE_NODE') {{
                details = `: ${{op.node_id}}`;
            }}
            return details + reasoning;
        }}

        // Update top nodes list (concepts or actions)
        function updateTopNodes(kf, nodeType, elementId, maxItems) {{
            const container = document.getElementById(elementId);

            // Filter nodes by type and sort by score
            const nodesOfType = kf.nodes
                .filter(n => n.type === nodeType)
                .map(n => ({{
                    ...n,
                    score: kf.scores[n.id] || 0
                }}))
                .sort((a, b) => b.score - a.score)
                .slice(0, maxItems);

            if (nodesOfType.length === 0) {{
                container.innerHTML = '<span style="color: var(--text-muted); font-size: 12px;">None yet</span>';
                return;
            }}

            container.innerHTML = nodesOfType.map(node => {{
                const data = node.data || {{}};
                const title = data.title || node.id;
                const score = node.score.toFixed(4);

                // For intents/actions, show priority if available
                let priorityHtml = '';
                if ((nodeType === 'action' || nodeType === 'intent') && data.priority) {{
                    const priorityClass = data.priority.toLowerCase();
                    priorityHtml = `<span class="node-priority ${{priorityClass}}">${{data.priority}}</span>`;
                }}

                // For concepts, show strength if available
                let strengthHtml = '';
                if (nodeType === 'concept' && data.strength !== undefined) {{
                    const strengthPct = Math.round(data.strength * 100);
                    strengthHtml = `<span style="color: #059669;">Strength: ${{strengthPct}}%</span>`;
                }}

                // Show reasoning on hover
                const reasoningTooltip = node.reasoning ? `title="${{node.reasoning}}"` : '';

                return `
                    <div class="top-node-item ${{nodeType}}" ${{reasoningTooltip}}>
                        <div class="node-title">${{title}}</div>
                        <div class="node-meta">
                            <span class="node-score">Score: ${{score}}</span>
                            ${{priorityHtml}}
                            ${{strengthHtml}}
                        </div>
                    </div>
                `;
            }}).join('');
        }}

        // Update action flow section (Phase 8)
        function updateActionFlow(kf) {{
            const section = document.getElementById('action-flow-section');
            const container = document.getElementById('action-flow-list');

            const intentsSelected = kf.intents_selected || [];
            const actionsGenerated = kf.actions_generated || [];
            const actionsExecuted = kf.actions_executed || [];
            const actionResults = kf.action_results || [];

            const hasActivity = intentsSelected.length > 0 || actionsGenerated.length > 0 ||
                               actionsExecuted.length > 0 || actionResults.length > 0;

            if (!hasActivity) {{
                section.style.display = 'none';
                return;
            }}

            section.style.display = 'block';
            let html = '';

            // Intents selected
            intentsSelected.forEach(item => {{
                html += `
                    <div class="action-flow-item intent-selected">
                        <div class="flow-type" style="color: #F97316;">Intent Selected</div>
                        <div class="flow-title">${{item.intent_title || item.intent_id}}</div>
                        <div class="flow-meta">Urgency: ${{item.urgency_score}} | Status: ${{item.status}}</div>
                    </div>
                `;
            }});

            // Actions generated
            actionsGenerated.forEach(item => {{
                html += `
                    <div class="action-flow-item action-generated">
                        <div class="flow-type" style="color: #DC2626;">Action Generated</div>
                        <div class="flow-title">${{item.action_title}}</div>
                        <div class="flow-meta">Scheduled: ${{item.scheduled_time}} | Urgency: ${{item.urgency}}</div>
                    </div>
                `;
            }});

            // Actions executed
            actionsExecuted.forEach(item => {{
                html += `
                    <div class="action-flow-item action-executed">
                        <div class="flow-type" style="color: #DC2626;">Action Executed</div>
                        <div class="flow-title">${{item.action_title}}</div>
                        <div class="flow-meta">Result: ${{item.result_event_id}}</div>
                    </div>
                `;
            }});

            // Action results
            actionResults.forEach(item => {{
                const resolvedText = item.intent_resolved ? '✓ Intent Resolved' : '';
                html += `
                    <div class="action-flow-item action-result">
                        <div class="flow-type" style="color: #10B981;">Action Result</div>
                        <div class="flow-title">${{item.result_event_id}}</div>
                        <div class="flow-meta">Outcome: ${{item.outcome}} ${{resolvedText}}</div>
                    </div>
                `;
            }});

            container.innerHTML = html || '<span style="color: var(--text-muted); font-size: 12px;">No action activity</span>';
        }}

        // Playback controls
        function play() {{
            if (currentStep >= totalSteps) {{
                currentStep = 0;
            }}
            isPlaying = true;
            document.getElementById('btn-play').textContent = '⏸';
            playInterval = setInterval(() => {{
                if (currentStep < totalSteps) {{
                    currentStep++;
                    renderKeyframe(currentStep);
                }} else {{
                    pause();
                }}
            }}, 1000 / playSpeed);
        }}

        function pause() {{
            isPlaying = false;
            document.getElementById('btn-play').textContent = '▶';
            if (playInterval) {{
                clearInterval(playInterval);
                playInterval = null;
            }}
        }}

        function goToStep(step) {{
            pause();
            currentStep = Math.max(0, Math.min(step, totalSteps));
            renderKeyframe(currentStep);
        }}

        // Event listeners
        document.getElementById('btn-play').addEventListener('click', () => {{
            if (isPlaying) pause();
            else play();
        }});

        document.getElementById('btn-prev').addEventListener('click', () => {{
            goToStep(currentStep - 1);
        }});

        document.getElementById('btn-next').addEventListener('click', () => {{
            goToStep(currentStep + 1);
        }});

        document.getElementById('btn-start').addEventListener('click', () => {{
            goToStep(0);
        }});

        document.getElementById('btn-end').addEventListener('click', () => {{
            goToStep(totalSteps);
        }});

        document.getElementById('timeline-slider').addEventListener('input', (e) => {{
            goToStep(parseInt(e.target.value));
        }});

        document.getElementById('speed-select').addEventListener('change', (e) => {{
            playSpeed = parseFloat(e.target.value);
            if (isPlaying) {{
                pause();
                play();
            }}
        }});

        document.getElementById('btn-reorganize').addEventListener('click', () => {{
            // Re-run physics simulation with strong repulsion to spread nodes out
            network.setOptions({{
                physics: {{
                    enabled: true,
                    stabilization: {{
                        enabled: true,
                        iterations: 300,
                        updateInterval: 25
                    }},
                    barnesHut: {{
                        gravitationalConstant: -20000,
                        centralGravity: 0.08,
                        springConstant: 0.012,
                        springLength: 350,
                        damping: 0.15,
                        avoidOverlap: 0.6
                    }}
                }}
            }});
            network.stabilize(300);
            // Disable physics after stabilization to prevent constant movement
            network.once('stabilizationIterationsDone', () => {{
                network.setOptions({{ physics: {{ enabled: false }} }});
            }});
        }});

        document.getElementById('btn-fit').addEventListener('click', () => {{
            // Fit all nodes into the view
            network.fit({{
                animation: {{
                    duration: 500,
                    easingFunction: 'easeInOutQuad'
                }}
            }});
        }});

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.code === 'Space') {{
                e.preventDefault();
                if (isPlaying) pause();
                else play();
            }} else if (e.code === 'ArrowLeft') {{
                goToStep(currentStep - 1);
            }} else if (e.code === 'ArrowRight') {{
                goToStep(currentStep + 1);
            }} else if (e.code === 'Home') {{
                goToStep(0);
            }} else if (e.code === 'End') {{
                goToStep(totalSteps);
            }}
        }});

        // Filter state
        let visibleTypes = new Set(['event', 'concept', 'intent', 'action', 'time', 'action_result']);

        // Filter button handlers
        document.querySelectorAll('.filter-btn').forEach(btn => {{
            btn.addEventListener('click', () => {{
                const nodeType = btn.dataset.type;
                if (visibleTypes.has(nodeType)) {{
                    visibleTypes.delete(nodeType);
                    btn.classList.remove('active');
                }} else {{
                    visibleTypes.add(nodeType);
                    btn.classList.add('active');
                }}
                applyFilters();
            }});
        }});

        // Apply filters to the current graph
        function applyFilters() {{
            const allNodes = nodes.get();
            const kf = keyframes.find(k => k.step === currentStep);
            if (!kf) return;

            allNodes.forEach(node => {{
                const nodeData = kf.nodes.find(n => n.id === node.id);
                if (!nodeData) return;

                const nodeType = nodeData.type;
                const isActionResult = nodeType === 'event' && nodeData.data && nodeData.data.event_type === 'action_result';
                const effectiveType = isActionResult ? 'action_result' : nodeType;

                const isVisible = visibleTypes.has(effectiveType);
                nodes.update({{
                    id: node.id,
                    hidden: !isVisible
                }});
            }});

            // Also hide edges connected to hidden nodes
            const hiddenNodeIds = new Set(
                allNodes.filter(n => {{
                    const nodeData = kf.nodes.find(nd => nd.id === n.id);
                    if (!nodeData) return true;
                    const nodeType = nodeData.type;
                    const isActionResult = nodeType === 'event' && nodeData.data && nodeData.data.event_type === 'action_result';
                    const effectiveType = isActionResult ? 'action_result' : nodeType;
                    return !visibleTypes.has(effectiveType);
                }}).map(n => n.id)
            );

            const allEdges = edges.get();
            allEdges.forEach(edge => {{
                const shouldHide = hiddenNodeIds.has(edge.from) || hiddenNodeIds.has(edge.to);
                edges.update({{
                    id: edge.id,
                    hidden: shouldHide
                }});
            }});
        }}

        // Initialize
        initNetwork();
        renderKeyframe(0);
    </script>
</body>
</html>'''
