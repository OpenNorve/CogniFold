"""CogEval-Bench OpenIE KG Baseline (HippoRAG-style).

Simulates a flat knowledge graph system: extracts (subject, predicate, object)
triples from each event via LLM, builds a NetworkX graph, and evaluates the
same Track A/B/C structural metrics as CogniFold.

This represents the HippoRAG / GraphRAG approach: has graph structure but no
concept folding, merging, or intent emergence.

Usage:
    OPENAI_API_KEY=... PYTHONPATH=src python -m benchmarks.cogeval.run_baseline_kg \
        --scenario software_engineer --scale small
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import networkx as nx


def extract_triples_llm(
    events: list[dict[str, Any]],
    model: str = "openai:gpt-4o-mini",
    batch_size: int = 5,
) -> list[dict[str, str]]:
    """Extract (subject, predicate, object) triples from events via LLM.

    Processes events in batches to reduce API calls.
    Returns list of {subject, predicate, object} dicts.
    """
    from benchmarks.shared.base_runner import _call_llm_text

    all_triples: list[dict[str, str]] = []
    system_prompt = (
        "You are a knowledge graph extraction system. Extract factual triples "
        "(subject, predicate, object) from the given events. Output ONLY a "
        "JSON array of objects with keys: subject, predicate, object. Extract "
        "ALL meaningful relationships. Normalize entity names (lowercase, no "
        'articles). Example: [{"subject": "alice", "predicate": "works_at", '
        '"object": "google"}]'
    )

    for i in range(0, len(events), batch_size):
        batch = events[i : i + batch_size]
        events_text = "\n\n".join(
            f"Event {j + 1}: {ev.get('title', '')}. {ev.get('description', '')}"
            for j, ev in enumerate(batch)
        )

        try:
            response = _call_llm_text(
                model=model,
                system_prompt=system_prompt,
                user_prompt=f"Extract triples from these events:\n\n{events_text}",
                temperature=0.0,
                max_tokens=2000,
            )
            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[1].rsplit("```", 1)[0]
            triples = json.loads(response)
            if isinstance(triples, list):
                for t in triples:
                    if not isinstance(t, dict):
                        continue
                    if "subject" not in t:
                        continue
                    if "object" not in t:
                        continue
                    all_triples.append(
                        {
                            "subject": str(t["subject"]).strip().lower(),
                            "predicate": str(
                                t.get("predicate", "related_to")
                            ).strip().lower(),
                            "object": str(t["object"]).strip().lower(),
                        }
                    )
        except Exception as e:
            print(f"    Triple extraction error (batch {i // batch_size}): {e}")

        time.sleep(0.3)

    return all_triples


def build_triple_graph(triples: list[dict[str, str]]) -> nx.Graph:
    """Build a NetworkX graph from extracted triples."""
    G = nx.Graph()
    for t in triples:
        subj = t["subject"]
        obj = t["object"]
        pred = t.get("predicate", "related_to")
        G.add_node(subj)
        G.add_node(obj)
        G.add_edge(subj, obj, predicate=pred)
    return G


def evaluate_structure(
    G: nx.Graph,
    triples: list[dict[str, str]],
    gold_graph: dict[str, Any],
    events: list[dict[str, Any]],
    n_input_events: int,
) -> dict[str, Any]:
    """Run Track A/B/C evaluators on the triple graph."""
    results: dict[str, Any] = {}

    try:
        from benchmarks.cogeval.concept_evaluator import (
            compute_concept_separation,
            compute_harmony_score,
            evaluate_concept_emergence,
            evaluate_concept_quality_llm,
        )

        system_concepts = [
            {"id": node, "title": node, "label": node} for node in G.nodes()
        ]
        gold_concepts = gold_graph.get("concepts", [])

        ce_result = evaluate_concept_emergence(system_concepts, gold_concepts, events)
        track_a = ce_result.to_dict()

        labels = list(G.nodes())
        if len(labels) >= 2:
            track_a["separation"] = round(compute_concept_separation(labels), 4)

        degree_sorted = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
        sampled_nodes = degree_sorted[:30]
        sampled_concepts = [
            {"id": node, "title": node, "label": node} for node in sampled_nodes
        ]
        llm_quality, _ = evaluate_concept_quality_llm(sampled_concepts, events)
        track_a["llm_quality"] = llm_quality
        track_a["harmony"] = compute_harmony_score(track_a.get("f1", 0), llm_quality)
        track_a["purity"] = 0.0

        results["concept_emergence"] = track_a
    except Exception as e:
        print(f"    Track A error: {e}")
        results["concept_emergence"] = {}

    try:
        from benchmarks.cogeval.topology_evaluator import evaluate_topology

        node_content_map = {node: node for node in G.nodes()}
        edge_types = [
            G.edges[e].get("predicate", "related_to") for e in G.edges()
        ]
        planted_chains = gold_graph.get("planted_chains", [])

        topo_result = evaluate_topology(
            G, planted_chains, node_content_map, edge_types
        )
        results["topology"] = topo_result.to_dict()
    except Exception as e:
        print(f"    Track B error: {e}")
        results["topology"] = {}

    try:
        from benchmarks.cogeval.compression_evaluator import compute_pagerank_gini

        if G.number_of_nodes() > 0:
            pr = nx.pagerank(G)
            gini = compute_pagerank_gini(list(pr.values()))
        else:
            gini = 0.0

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        compression_ratio = n_input_events / max(n_nodes, 1)

        results["compression"] = {
            "pagerank_gini": round(gini, 4),
            "compression_ratio": round(compression_ratio, 2),
            "concept_fraction": 0.0,
            "edge_density": round(n_edges / max(n_nodes, 1), 4),
            "node_count": n_nodes,
            "edge_count": n_edges,
            "concept_count": 0,
            "entity_count": n_nodes,
            "intent_count": 0,
            "input_events": n_input_events,
            "total_triples": len(triples),
        }
    except Exception as e:
        print(f"    Track C error: {e}")
        results["compression"] = {}

    return results


def load_dataset(scenario: str, scale: str) -> dict[str, Any]:
    path = Path(__file__).parent / "data" / "generated" / f"{scenario}_{scale}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path) as f:
        return json.load(f)


def run_baseline_kg(
    scenario: str = "software_engineer",
    scale: str = "small",
    model: str = "openai:gpt-4o-mini",
) -> dict[str, Any]:
    dataset = load_dataset(scenario, scale)
    events = dataset["events"]
    gold_graph = dataset["gold_graph"]
    n_input_events = len(events)

    print(f"=== OpenIE KG Baseline (HippoRAG-style): {dataset['name']} ===")
    print(f"  Events: {n_input_events}")

    print("\n  Extracting triples via LLM...")
    triples = extract_triples_llm(events, model=model)
    print(f"  Extracted {len(triples)} triples")

    G = build_triple_graph(triples)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\n  Evaluating structural metrics...")
    metrics = evaluate_structure(G, triples, gold_graph, events, n_input_events)

    print("\n" + "=" * 60)
    print(f"OpenIE KG BASELINE RESULTS — {dataset['name']}")
    print("=" * 60)

    ce = metrics.get("concept_emergence", {})
    if ce:
        print("\n  Track A — Concept Emergence:")
        print(f"    Precision:   {ce.get('precision', 0):.3f}")
        print(f"    Recall:      {ce.get('recall', 0):.3f}")
        print(f"    F1:          {ce.get('f1', 0):.3f}")
        print(f"    Separation:  {ce.get('separation', 0):.3f}")

    tp = metrics.get("topology", {})
    if tp:
        print("\n  Track B — Relationship Topology:")
        print(f"    Chain discovery: {tp.get('chain_discovery_rate', 0):.3f}")
        print(f"    Modularity:      {tp.get('modularity', 0):.3f}")
        print(f"    Clustering:      {tp.get('clustering_coefficient', 0):.3f}")
        print(f"    Edge entropy:    {tp.get('edge_type_entropy', 0):.3f}")

    cp = metrics.get("compression", {})
    if cp:
        print("\n  Track C — Temporal Compression:")
        print(f"    Compression:  {cp.get('compression_ratio', 0):.1f}x")
        print(f"    PR Gini:      {cp.get('pagerank_gini', 0):.3f}")
        print(f"    Edge Density: {cp.get('edge_density', 0):.2f}")
        print(
            f"    Nodes: {cp.get('node_count', 0)} entities, "
            f"{cp.get('total_triples', 0)} triples"
        )
        print("    Concepts: 0, Intents: 0 (no folding)")

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"baseline_kg_results_{scenario}.json"

    output = {
        "benchmark": "cogeval_baseline_kg",
        "method": "OpenIE KG (HippoRAG-style)",
        "scenario": scenario,
        "scale": scale,
        "model": model,
        "structural_metrics": metrics,
        "graph_stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "triples": len(triples),
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="CogEval-Bench OpenIE KG Baseline")
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
    parser.add_argument(
        "--model", default="openai:gpt-4o-mini", help="LLM model"
    )
    args = parser.parse_args()

    run_baseline_kg(scenario=args.scenario, scale=args.scale, model=args.model)


if __name__ == "__main__":
    main()
