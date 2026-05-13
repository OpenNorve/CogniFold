"""CogEval-Bench Cognee Baseline (ECL pipeline: Extract → Cognify → Load).

Cognee builds a knowledge graph via its ECL pipeline: extracts entities and
relationships from text, cognifies them into graph nodes with typed edges,
and loads into a Kuzu graph database.

This represents a concept-level graph system with LLM-driven extraction and
community structure, but no temporal folding, no intent emergence, no online
streaming.

Usage:
    OPENAI_API_KEY=... COGNEE_SKIP_CONNECTION_TEST=true \
        PYTHONPATH=src python -m benchmarks.cogeval.run_baseline_cognee \
        --scenario software_engineer --scale small
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import networkx as nx


async def build_cognee_graph(
    events: list[dict[str, Any]], api_key: str
) -> nx.Graph:
    """Feed events into Cognee and extract the resulting graph as NetworkX."""
    import cognee

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    cognee.config.set_llm_config(
        {
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "llm_api_key": api_key,
        }
    )
    cognee.config.set_embedding_config(
        {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_api_key": api_key,
            "embedding_dimensions": 1536,
        }
    )

    for ev in events:
        text = f"{ev.get('title', '')}. {ev.get('description', '')}"
        await cognee.add(text.strip())

    await cognee.cognify()

    from cognee.infrastructure.databases.graph import get_graph_engine

    engine = await get_graph_engine()
    graph_data = await engine.get_graph_data()

    raw_nodes = graph_data[0] if len(graph_data) > 0 else []
    raw_edges = graph_data[1] if len(graph_data) > 1 else []

    G = nx.Graph()
    for node_tuple in raw_nodes:
        node_id = node_tuple[0]
        node_attrs = node_tuple[1] if len(node_tuple) > 1 else {}
        node_name = node_attrs.get("name", str(node_id))
        node_type = node_attrs.get("type", "unknown")
        G.add_node(node_id, name=node_name, type=node_type)

    for edge_tuple in raw_edges:
        src = edge_tuple[0]
        tgt = edge_tuple[1]
        rel = edge_tuple[2] if len(edge_tuple) > 2 else "related_to"
        G.add_edge(src, tgt, relation=rel)
        if src not in G:
            G.add_node(src)
        if tgt not in G:
            G.add_node(tgt)

    return G


def evaluate_structure(
    G: nx.Graph,
    gold_graph: dict[str, Any],
    events: list[dict[str, Any]],
    n_input_events: int,
) -> dict[str, Any]:
    """Run Track A/B/C evaluators on the Cognee graph."""
    results: dict[str, Any] = {}

    all_nodes = list(G.nodes(data=True))
    entity_nodes = [
        n for n in all_nodes
        if n[1].get("type", "") not in ("Document", "DocumentChunk", "")
    ]

    try:
        from benchmarks.cogeval.concept_evaluator import (
            compute_concept_separation,
            compute_harmony_score,
            evaluate_concept_emergence,
            evaluate_concept_quality_llm,
        )

        system_concepts = [
            {
                "id": str(n[0]),
                "title": n[1].get("name", str(n[0])),
                "label": n[1].get("name", str(n[0])),
            }
            for n in entity_nodes
            if n[1].get("name", "")
        ]
        gold_concepts = gold_graph.get("concepts", [])

        ce_result = evaluate_concept_emergence(system_concepts, gold_concepts, events)
        track_a = ce_result.to_dict()

        labels = [c["label"] for c in system_concepts if c["label"]]
        if len(labels) >= 2:
            track_a["separation"] = round(
                compute_concept_separation(labels[:50]), 4
            )

        degree_sorted = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
        sampled = degree_sorted[:30]
        sampled_concepts = [
            {
                "id": str(n),
                "title": G.nodes[n].get("name", str(n)),
                "label": G.nodes[n].get("name", str(n)),
            }
            for n in sampled
            if G.nodes[n].get("name", "")
        ]

        if sampled_concepts:
            llm_quality, _ = evaluate_concept_quality_llm(sampled_concepts, events)
        else:
            llm_quality = 0.0

        track_a["llm_quality"] = llm_quality
        track_a["harmony"] = compute_harmony_score(track_a.get("f1", 0), llm_quality)
        track_a["purity"] = 0.0

        results["concept_emergence"] = track_a
    except Exception as e:
        print(f"    Track A error: {e}")
        results["concept_emergence"] = {}

    try:
        from benchmarks.cogeval.topology_evaluator import evaluate_topology

        node_content_map = {
            n: G.nodes[n].get("name", str(n)) for n in G.nodes()
        }
        edge_types = [
            G.edges[e].get("relation", "related_to") for e in G.edges()
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
        n_entity = len(entity_nodes)

        type_counts: dict[str, int] = {}
        for _, attrs in all_nodes:
            t = attrs.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        n_concepts = type_counts.get("TextSummary", 0) + type_counts.get(
            "EntityType", 0
        )
        if n_concepts > 0:
            compression_ratio = n_input_events / max(n_concepts, 1)
        else:
            compression_ratio = n_input_events / max(n_nodes, 1)

        results["compression"] = {
            "pagerank_gini": round(gini, 4),
            "compression_ratio": round(compression_ratio, 2),
            "concept_fraction": round(n_concepts / max(n_nodes, 1), 4),
            "edge_density": round(n_edges / max(n_nodes, 1), 4),
            "node_count": n_nodes,
            "edge_count": n_edges,
            "concept_count": n_concepts,
            "entity_count": n_entity,
            "intent_count": 0,
            "input_events": n_input_events,
            "proactivity": 0.0,
            "schema_acceleration": 0.0,
            "node_type_distribution": type_counts,
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


def run_baseline_cognee(
    scenario: str = "software_engineer", scale: str = "small"
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    os.environ.setdefault("LLM_API_KEY", api_key)
    os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")

    dataset = load_dataset(scenario, scale)
    events = dataset["events"]
    gold_graph = dataset["gold_graph"]
    n_input_events = len(events)

    print(f"=== Cognee Baseline (ECL pipeline): {dataset['name']} ===")
    print(f"  Events: {n_input_events}")

    print("\n  Building Cognee graph (ECL pipeline)...")
    t0 = time.time()
    G = asyncio.run(build_cognee_graph(events, api_key))
    elapsed = time.time() - t0
    print(
        f"  Graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges ({elapsed:.1f}s)"
    )

    type_counts: dict[str, int] = {}
    for _, attrs in G.nodes(data=True):
        t = attrs.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t}: {c}")

    print("\n  Evaluating structural metrics...")
    metrics = evaluate_structure(G, gold_graph, events, n_input_events)

    print("\n" + "=" * 60)
    print(f"COGNEE BASELINE RESULTS — {dataset['name']}")
    print("=" * 60)

    ce = metrics.get("concept_emergence", {})
    if ce:
        print("\n  Track A — Concept Emergence:")
        print(f"    Precision:   {ce.get('precision', 0):.3f}")
        print(f"    Recall:      {ce.get('recall', 0):.3f}")
        print(f"    F1:          {ce.get('f1', 0):.3f}")
        print(f"    Separation:  {ce.get('separation', 0):.3f}")
        print(f"    LLM Quality: {ce.get('llm_quality', 0):.3f}")
        print(f"    Harmony:     {ce.get('harmony', 0):.3f}")

    tp = metrics.get("topology", {})
    if tp:
        print("\n  Track B — Relationship Topology:")
        print(f"    Chain discovery: {tp.get('chain_discovery_rate', 0):.3f}")
        print(f"    Modularity:      {tp.get('modularity', 0):.3f}")
        print(f"    Clustering:      {tp.get('clustering_coefficient', 0):.3f}")
        print(f"    Edge entropy:    {tp.get('edge_type_entropy', 0):.3f}")

    cp = metrics.get("compression", {})
    if cp:
        print("\n  Track C — Compression:")
        print(f"    Compression:  {cp.get('compression_ratio', 0):.1f}x")
        print(f"    PR Gini:      {cp.get('pagerank_gini', 0):.3f}")
        print(f"    Nodes: {cp.get('node_count', 0)}")

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"baseline_cognee_results_{scenario}.json"

    output = {
        "benchmark": "cogeval_baseline_cognee",
        "method": "Cognee (ECL pipeline)",
        "scenario": scenario,
        "scale": scale,
        "model": "openai:gpt-4o-mini",
        "structural_metrics": metrics,
        "graph_stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "node_types": type_counts,
            "build_time_s": round(elapsed, 1),
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="CogEval-Bench Cognee Baseline")
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

    run_baseline_cognee(scenario=args.scenario, scale=args.scale)


if __name__ == "__main__":
    main()
