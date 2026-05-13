"""CogEval-Bench Mem0 Baseline (LLM-driven memory rewrites, flat vector store).

Mem0 represents the "graph-as-list" / state-management paradigm: per-turn LLM
fact extraction stored in a vector database, no graph topology between memories.
For CogEval we materialise each extracted memory as a node and induce edges via
high-similarity vector neighbours, so the topology metrics fairly reflect the
flat-store design rather than the absence of a graph backend.

Usage:
    OPENAI_API_KEY=... PYTHONPATH=src python -m benchmarks.cogeval.run_baseline_mem0 \
        --scenario software_engineer --scale small
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import networkx as nx


def build_mem0_graph(events: list[dict[str, Any]], api_key: str) -> tuple[nx.Graph, dict[str, int]]:
    """Feed events into Mem0 and materialise its memory store as a NetworkX graph."""
    from mem0 import Memory
    from mem0.configs.base import (
        EmbedderConfig,
        LlmConfig,
        MemoryConfig,
        VectorStoreConfig,
    )

    work_dir = tempfile.mkdtemp(prefix="mem0_cogeval_")
    user_id = f"cogeval-{uuid.uuid4().hex[:8]}"

    try:
        config = MemoryConfig(
            vector_store=VectorStoreConfig(
                provider="qdrant",
                config={
                    "collection_name": f"mem0_cogeval_{uuid.uuid4().hex[:8]}",
                    "embedding_model_dims": 1536,
                    "path": str(Path(work_dir) / "qdrant"),
                    "on_disk": False,
                },
            ),
            llm=LlmConfig(
                provider="openai",
                config={"model": "gpt-4o-mini", "api_key": api_key},
            ),
            embedder=EmbedderConfig(
                provider="openai",
                config={
                    "model": "text-embedding-3-small",
                    "api_key": api_key,
                    "embedding_dims": 1536,
                },
            ),
            history_db_path=str(Path(work_dir) / "history.db"),
        )
        mem = Memory(config=config)

        for ev in events:
            text = f"{ev.get('title', '')}. {ev.get('description', '')}".strip()
            if not text:
                continue
            try:
                mem.add(text, user_id=user_id, infer=True)
            except Exception as e:
                print(f"    [warn] add failed: {e}")

        try:
            stored = mem.get_all(filters={"user_id": user_id}, top_k=10000)
        except Exception:
            stored = mem.get_all(top_k=10000)

        results = stored.get("results", []) if isinstance(stored, dict) else stored
        G = nx.Graph()
        for r in results:
            mid = r.get("id") or r.get("memory_id") or str(uuid.uuid4())
            text = r.get("memory") or r.get("text") or ""
            G.add_node(str(mid), type="Memory", name=text[:200], content=text)

        node_ids = list(G.nodes())
        added = 0
        sim_top_k = 4
        for nid in node_ids:
            text = G.nodes[nid].get("content", "")
            if not text:
                continue
            try:
                sim = mem.search(text, top_k=sim_top_k + 1, filters={"user_id": user_id})
            except Exception:
                continue
            sim_results = sim.get("results", []) if isinstance(sim, dict) else sim
            for s in sim_results:
                sid = str(s.get("id") or s.get("memory_id") or "")
                if not sid or sid == nid or sid not in G:
                    continue
                if G.has_edge(nid, sid):
                    continue
                score = s.get("score", 0.0)
                G.add_edge(nid, sid, relation="similar_to", weight=float(score))
                added += 1

        type_counts = {"Memory": len(node_ids), "induced_edges": added}
        return G, type_counts
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def evaluate_structure(
    G: nx.Graph,
    gold_graph: dict[str, Any],
    events: list[dict[str, Any]],
    n_input_events: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    all_nodes = list(G.nodes(data=True))
    memory_nodes = [n for n in all_nodes if n[1].get("type") == "Memory"]

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
            for n in memory_nodes
            if n[1].get("name", "")
        ]
        gold_concepts = gold_graph.get("concepts", [])
        ce_result = evaluate_concept_emergence(system_concepts, gold_concepts, events)
        track_a = ce_result.to_dict()

        labels = [c["label"] for c in system_concepts if c["label"]]
        if len(labels) >= 2:
            track_a["separation"] = round(compute_concept_separation(labels[:50]), 4)

        degree_sorted = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
        sampled = [
            {
                "id": str(n),
                "title": G.nodes[n].get("name", str(n)),
                "label": G.nodes[n].get("name", str(n)),
            }
            for n in degree_sorted[:30]
            if G.nodes[n].get("type") == "Memory"
        ]
        if sampled:
            llm_quality, _ = evaluate_concept_quality_llm(sampled, events)
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

        node_content_map = {n: G.nodes[n].get("name", str(n)) for n in G.nodes()}
        edge_types = [G.edges[e].get("relation", "similar_to") for e in G.edges()]
        planted_chains = gold_graph.get("planted_chains", [])
        topo_result = evaluate_topology(G, planted_chains, node_content_map, edge_types)
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
        n_memory = len(memory_nodes)
        compression_ratio = n_input_events / max(n_memory, 1)

        results["compression"] = {
            "pagerank_gini": round(gini, 4),
            "compression_ratio": round(compression_ratio, 2),
            "concept_fraction": 0.0,
            "edge_density": round(n_edges / max(n_nodes, 1), 4),
            "node_count": n_nodes,
            "edge_count": n_edges,
            "memory_count": n_memory,
            "concept_count": 0,
            "intent_count": 0,
            "input_events": n_input_events,
            "proactivity": 0.0,
            "schema_acceleration": 0.0,
            "node_type_distribution": {"Memory": n_memory},
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


def run_baseline_mem0(scenario: str = "software_engineer", scale: str = "small") -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    dataset = load_dataset(scenario, scale)
    events = dataset["events"]
    gold_graph = dataset["gold_graph"]
    n_input_events = len(events)

    print(f"=== Mem0 Baseline (LLM rewrite + vector store): {dataset['name']} ===")
    print(f"  Events: {n_input_events}")

    print("\n  Building Mem0 memory store...")
    t0 = time.time()
    G, type_counts = build_mem0_graph(events, api_key)
    elapsed = time.time() - t0
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ({elapsed:.1f}s)")

    print("\n  Evaluating structural metrics...")
    metrics = evaluate_structure(G, gold_graph, events, n_input_events)

    print("\n" + "=" * 60)
    print(f"MEM0 BASELINE RESULTS — {dataset['name']}")
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
    output_path = output_dir / f"baseline_mem0_results_{scenario}.json"

    output = {
        "benchmark": "cogeval_baseline_mem0",
        "method": "Mem0 (LLM rewrite + flat vector store)",
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
    parser = argparse.ArgumentParser(description="CogEval-Bench Mem0 Baseline")
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
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="small")
    args = parser.parse_args()
    run_baseline_mem0(scenario=args.scenario, scale=args.scale)


if __name__ == "__main__":
    main()
