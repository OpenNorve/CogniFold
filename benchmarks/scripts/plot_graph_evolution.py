#!/usr/bin/env python3
"""Plot graph evolution metrics from tracker JSON files.

Usage:
    python benchmarks/scripts/plot_graph_evolution.py \\
        results/rgb_evolution.json \\
        results/mutual_evolution.json \\
        --output papers/cognifold-neurips2025/figures/graph_evolution.pdf

Produces a 2x2 figure:
    Top-left:     Node count by type vs events processed
    Top-right:    Compression ratio (concepts/events) vs events processed
    Bottom-left:  Edge density (edges/nodes) vs events processed
    Bottom-right: PageRank Gini coefficient vs events processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import the tracker
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))
    sys.path.insert(0, str(_project_root))

from benchmarks.shared.graph_evolution_tracker import GraphEvolutionTracker

# Colorblind-safe palette (Wong 2011, Nature Methods)
_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # pink
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
]


def _configure_style() -> None:
    """Set publication-quality matplotlib defaults."""
    import matplotlib

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def plot_graph_evolution(
    tracker_paths: list[str],
    output_path: str,
) -> None:
    """Load tracker JSONs and produce a 2x2 evolution figure.

    Args:
        tracker_paths: Paths to one or more tracker JSON files.
        output_path: Where to save the figure (PDF recommended).
    """
    import matplotlib.pyplot as plt

    _configure_style()

    trackers: list[GraphEvolutionTracker] = []
    for p in tracker_paths:
        trackers.append(GraphEvolutionTracker.load(p))

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))
    ax_nodes = axes[0, 0]
    ax_compression = axes[0, 1]
    ax_density = axes[1, 0]
    ax_gini = axes[1, 1]

    for idx, tracker in enumerate(trackers):
        color = _PALETTE[idx % len(_PALETTE)]
        label = tracker.benchmark_name or Path(tracker_paths[idx]).stem
        snaps = tracker.snapshots
        if not snaps:
            continue

        xs = [s.event_idx for s in snaps]

        # -- Top-left: Node counts by type --
        events = [s.event_node_count for s in snaps]
        concepts = [s.concept_count for s in snaps]
        intents = [s.intent_count for s in snaps]

        if len(trackers) == 1:
            # Show breakdown by type when there is a single benchmark
            ax_nodes.plot(xs, events, color="#0072B2", linewidth=1.2, label="event")
            ax_nodes.plot(xs, concepts, color="#D55E00", linewidth=1.2, label="concept")
            ax_nodes.plot(xs, intents, color="#009E73", linewidth=1.2, label="intent")
        else:
            # Multiple benchmarks: show total node count per benchmark
            totals = [s.node_count for s in snaps]
            ax_nodes.plot(xs, totals, color=color, linewidth=1.2, label=label)

        # -- Top-right: Compression ratio --
        cr = [s.compression_ratio for s in snaps]
        ax_compression.plot(xs, cr, color=color, linewidth=1.2, label=label)

        # -- Bottom-left: Edge density --
        ed = [s.edge_density for s in snaps]
        ax_density.plot(xs, ed, color=color, linewidth=1.2, label=label)

        # -- Bottom-right: PageRank Gini --
        gini = [s.pagerank_gini for s in snaps]
        ax_gini.plot(xs, gini, color=color, linewidth=1.2, label=label)

    # Labels
    ax_nodes.set_xlabel("Events processed")
    ax_nodes.set_ylabel("Node count")
    ax_nodes.set_title("Node count by type")
    ax_nodes.legend(frameon=False)

    ax_compression.set_xlabel("Events processed")
    ax_compression.set_ylabel("Concepts / Events")
    ax_compression.set_title("Compression ratio")
    if len(trackers) > 1:
        ax_compression.legend(frameon=False)

    ax_density.set_xlabel("Events processed")
    ax_density.set_ylabel("Edges / Nodes")
    ax_density.set_title("Edge density")
    if len(trackers) > 1:
        ax_density.legend(frameon=False)

    ax_gini.set_xlabel("Events processed")
    ax_gini.set_ylabel("Gini coefficient")
    ax_gini.set_title("PageRank Gini")
    if len(trackers) > 1:
        ax_gini.legend(frameon=False)

    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))
    plt.close(fig)
    print(f"Saved figure to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot graph evolution metrics from tracker JSON files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more tracker JSON files to plot.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="papers/cognifold-neurips2025/figures/graph_evolution.pdf",
        help="Output figure path (default: papers/.../graph_evolution.pdf).",
    )
    args = parser.parse_args()

    plot_graph_evolution(args.inputs, args.output)


if __name__ == "__main__":
    main()
