"""CogEval-Bench HippoRAG 2 Baseline (OpenIE + PPR graph retrieval).

HippoRAG 2 builds an entity-level knowledge graph via OpenIE triple extraction,
then retrieves using Personalized PageRank over the graph.

This represents an entity-level graph system with LLM-driven OpenIE, synonym
detection, and PPR-based retrieval — but no concept abstraction, no intent
emergence, no temporal folding.

Usage:
    OPENAI_API_KEY=... PYTHONPATH=src python -m benchmarks.cogeval.run_baseline_hipporag \
        --scenario software_engineer --scale small
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import networkx as nx


def build_hipporag_graph(events: list[dict[str, Any]]) -> nx.Graph:
    """Feed events into HippoRAG 2 and extract the resulting graph as NetworkX."""
    from hipporag import HippoRAG

    work_dir = tempfile.mkdtemp(prefix="hipporag_cogeval_")

    try:
        hr = HippoRAG(
            save_dir=work_dir,
            llm_model_name="gpt-4o-mini",
            embedding_model_name="text-embedding-3-small",
        )

        docs = []
        for ev in events:
            text = f"{ev.get('title', '')}. {ev.get('description', '')}"
            docs.append(text.strip())

        hr.index(docs)

        ig_graph = hr.graph
        G = _igraph_to_networkx(ig_graph)

        return G
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _igraph_to_networkx(ig_graph: Any) -> nx.Graph:
    """Convert an igraph.Graph to a NetworkX graph."""
    G = nx.Graph()

    for v in ig_graph.vs:
        attrs = {k: v[k] for k in v.attributes() if v[k] is not None}
        node_id = attrs.get("name", str(v.index))
        node_type = _classify_hipporag_node(attrs)
        G.add_node(node_id, **{"type": node_type, **attrs})

    for e in ig_graph.es:
        src = ig_graph.vs[e.source]
        tgt = ig_graph.vs[e.target]
        src_id = src["name"] if "name" in src.attributes() else str(e.source)
        tgt_id = tgt["name"] if "name" in tgt.attributes() else str(e.target)

        edge_attrs = {k: e[k] for k in e.attributes() if e[k] is not None}
        relation = edge_attrs.pop("relation", "related_to")
        G.add_edge(src_id, tgt_id, **{"relation": relation, **edge_attrs})

    return G


def _classify_hipporag_node(attrs: dict) -> str:
    """Classify a HippoRAG node as entity or passage based on its key prefix."""
    name = attrs.get("name", "")
    if isinstance(name, str):
        if name.startswith("entity-"):
            return "Entity"
        if name.startswith("chunk-") or name.startswith("passage-"):
            return "Passage"

    content = attrs.get("content", "")
    if isinstance(content, str) and len(content) > 100:
        return "Passage"

    return "Entity"


def evaluate_structure(
    G: nx.Graph,
    gold_graph: dict[str, Any],
    events: list[dict[str, Any]],
    n_input_events: int,
) -> dict[str, Any]:
    """Run Track A/B/C evaluators on the HippoRAG graph."""
    results: dict[str, Any] = {}

    all_nodes = list(G.nodes(data=True))
    entity_nodes = [n for n in all_nodes if n[1].get("type", "") == "Entity"]

    try:
        from benchmarks.cogeval.concept_evaluator import (
            compute_concept_separation,
            compute_harmony_score,
            evaluate_concept_emergence,
            evaluate_concept_quality_llm,
        )

        system_concepts = []
        for n_id, attrs in entity_nodes:
            label = attrs.get("content", attrs.get("name", str(n_id)))
            if isinstance(label, str) and label.startswith("entity-"):
                label = attrs.get("content", label)
            if not label:
                continue
            system_concepts.append(
                {
                    "id": str(n_id),
                    "title": str(label)[:200],
                    "label": str(label)[:200],
                }
            )

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
                "title": G.nodes[n].get(
                    "content", G.nodes[n].get("name", str(n))
                ),
                "label": G.nodes[n].get(
                    "content", G.nodes[n].get("name", str(n))
                ),
            }
            for n in sampled
            if G.nodes[n].get("type") == "Entity"
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
            n: G.nodes[n].get("content", G.nodes[n].get("name", str(n)))
            for n in G.nodes()
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
        n_passage = sum(1 for _, a in all_nodes if a.get("type") == "Passage")

        compression_ratio = n_input_events / max(n_entity, 1)

        type_counts: dict[str, int] = {}
        for _, attrs in all_nodes:
            t = attrs.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        results["compression"] = {
            "pagerank_gini": round(gini, 4),
            "compression_ratio": round(compression_ratio, 2),
            "concept_fraction": 0.0,
            "edge_density": round(n_edges / max(n_nodes, 1), 4),
            "node_count": n_nodes,
            "edge_count": n_edges,
            "entity_count": n_entity,
            "passage_count": n_passage,
            "concept_count": 0,
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


def run_baseline_hipporag(
    scenario: str = "software_engineer", scale: str = "small"
) -> dict[str, Any]:
    """Run HippoRAG 2 baseline on a CogEval scenario."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    dataset = load_dataset(scenario, scale)
    events = dataset["events"]
    gold_graph = dataset["gold_graph"]
    n_input_events = len(events)

    print(f"=== HippoRAG 2 Baseline (OpenIE + PPR): {dataset['name']} ===")
    print(f"  Events: {n_input_events}")

    print("\n  Building HippoRAG graph (OpenIE extraction)...")
    t0 = time.time()
    G = build_hipporag_graph(events)
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
    print(f"HIPPORAG 2 BASELINE RESULTS — {dataset['name']}")
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
    output_path = output_dir / f"baseline_hipporag_results_{scenario}.json"

    output = {
        "benchmark": "cogeval_baseline_hipporag",
        "method": "HippoRAG 2 (OpenIE + PPR)",
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
    parser = argparse.ArgumentParser(description="CogEval-Bench HippoRAG 2 Baseline")
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

    run_baseline_hipporag(scenario=args.scenario, scale=args.scale)


if __name__ == "__main__":
    main()
