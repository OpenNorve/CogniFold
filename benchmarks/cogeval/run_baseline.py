"""CogEval-Bench RAG Baseline: flat document store, no graph structure.

Reports all structural metrics as zero — a flat RAG system builds no
cognitive structure. Used as the lower bound in CogEval-Bench comparisons.

Usage:
    PYTHONPATH=src python -m benchmarks.cogeval.run_baseline \
        --scenario software_engineer --scale small
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_dataset(scenario: str, scale: str) -> dict[str, Any]:
    path = Path(__file__).parent / "data" / "generated" / f"{scenario}_{scale}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path) as f:
        return json.load(f)


def run_baseline(
    scenario: str = "software_engineer", scale: str = "small"
) -> dict[str, Any]:
    """Report structural metrics for a flat RAG system (all zeros)."""
    dataset = load_dataset(scenario, scale)
    n_events = len(dataset["events"])

    print(f"=== RAG Baseline (flat store): {dataset['name']} ===")
    print(f"  Events: {n_events}")
    print(f"  Storage: {n_events} documents stored as-is (no graph)")

    metrics = {
        "concept_emergence": {
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "separation": 0,
            "compression_ratio": 1.0,
            "n_matches": 0,
        },
        "topology": {
            "chain_discovery_rate": 0,
            "modularity": 0,
            "clustering_coefficient": 0,
            "edge_type_entropy": 0,
            "small_world_sigma": 0,
            "n_communities": 0,
        },
        "compression": {
            "pagerank_gini": 0,
            "compression_ratio": 1.0,
            "concept_fraction": 0,
            "edge_density": 0,
            "node_count": n_events,
            "edge_count": 0,
            "concept_count": 0,
            "intent_count": 0,
            "event_node_count": n_events,
            "input_events": n_events,
        },
    }

    print("\n" + "=" * 60)
    print(f"RAG BASELINE RESULTS — {dataset['name']}")
    print("=" * 60)
    print("\n  Track A — Concept Emergence: all 0 (no concepts)")
    print("  Track B — Relationship Topology: all 0 (no graph)")
    print("  Track C — Temporal Compression: 1.0x (no compression)")
    print(f"    {n_events} events stored as-is, 0 concepts, 0 intents")

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"baseline_results_{scenario}.json"

    output = {
        "benchmark": "cogeval_baseline",
        "method": "RAG (flat store)",
        "scenario": scenario,
        "scale": scale,
        "structural_metrics": metrics,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="CogEval-Bench RAG Baseline")
    parser.add_argument(
        "--scenario",
        choices=[
            "software_engineer",
            "health_journey",
            "team_project",
            "news_stream",
            "academic_research",
            "customer_support",
        ],
        default="software_engineer",
    )
    parser.add_argument(
        "--scale", choices=["small", "medium", "large"], default="small"
    )
    args = parser.parse_args()
    run_baseline(scenario=args.scenario, scale=args.scale)


if __name__ == "__main__":
    main()
