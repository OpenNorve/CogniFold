"""CogEval-Bench Zep/Graphiti Baseline (bi-temporal entity graph).

Graphiti (the open-source engine behind Zep) builds a bi-temporal knowledge
graph: each ingested episode goes through entity extraction, relation extraction,
and edge invalidation against the existing graph. Edges carry valid/invalid time
stamps. We run the embedded Kuzu driver so no external Neo4j is required, then
materialise the resulting Entity/RelatesToNode_/Episodic graph as NetworkX for
the standard CogEval Track A/B/C evaluators.

Usage:
    OPENAI_API_KEY=... PYTHONPATH=src python -m benchmarks.cogeval.run_baseline_zep \
        --scenario software_engineer --scale small
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx


async def _build(events: list[dict[str, Any]], api_key: str) -> tuple[nx.Graph, dict[str, int]]:
    from graphiti_core import Graphiti
    from graphiti_core.driver.kuzu_driver import KuzuDriver
    from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.llm_client import LLMConfig, OpenAIClient
    from graphiti_core.nodes import EpisodeType

    work_dir = tempfile.mkdtemp(prefix="graphiti_cogeval_")
    db_path = str(Path(work_dir) / "graph.kz")

    driver = KuzuDriver(db=db_path)
    llm = OpenAIClient(
        config=LLMConfig(api_key=api_key, model="gpt-4o-mini", small_model="gpt-4o-mini")
    )
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=api_key,
            embedding_model="text-embedding-3-small",
            embedding_dim=1536,
        )
    )

    # Kuzu requires the FTS extension and explicit FTS index creation.
    # Graphiti's build_indices_and_constraints fires them concurrently which
    # races against itself — create them serially up front instead.
    fts_setup = [
        "INSTALL FTS;",
        "LOAD EXTENSION FTS;",
        "CALL CREATE_FTS_INDEX('Episodic', 'episode_content', "
        "['content', 'source', 'source_description']);",
        "CALL CREATE_FTS_INDEX('Entity', 'node_name_and_summary', ['name', 'summary']);",
        "CALL CREATE_FTS_INDEX('Community', 'community_name', ['name']);",
        "CALL CREATE_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact', ['name', 'fact']);",
    ]
    for q in fts_setup:
        try:
            await driver.execute_query(q)
        except Exception as e:
            msg = str(e)
            if "already" not in msg.lower():
                print(f"    [warn] FTS setup '{q[:40]}...': {e}")

    g = Graphiti(graph_driver=driver, llm_client=llm, embedder=embedder)

    try:
        for i, ev in enumerate(events):
            text = f"{ev.get('title', '')}. {ev.get('description', '')}".strip()
            if not text:
                continue
            ts_str = ev.get("timestamp")
            try:
                ts = (
                    datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if ts_str
                    else datetime.now(timezone.utc)
                )
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                ts = datetime.now(timezone.utc)
            try:
                await g.add_episode(
                    name=f"episode_{i}",
                    episode_body=text,
                    source_description="cogeval_event",
                    reference_time=ts,
                    source=EpisodeType.message,
                )
            except Exception as e:
                print(f"    [warn] add_episode {i} failed: {e}")

        entity_rows, _, _ = await driver.execute_query(
            "MATCH (n:Entity) RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary"
        )
        episodic_rows, _, _ = await driver.execute_query(
            "MATCH (n:Episodic) RETURN n.uuid AS uuid, n.name AS name, n.content AS content"
        )
        rel_rows, _, _ = await driver.execute_query(
            "MATCH (a:Entity)-[:RELATES_TO]->(r:RelatesToNode_)-[:RELATES_TO]->(b:Entity) "
            "RETURN a.uuid AS src, b.uuid AS tgt, r.name AS rel, r.fact AS fact"
        )
        mention_rows, _, _ = await driver.execute_query(
            "MATCH (e:Episodic)-[m:MENTIONS]->(n:Entity) RETURN e.uuid AS src, n.uuid AS tgt"
        )
    finally:
        try:
            await driver.close()
        except Exception:
            pass
        shutil.rmtree(work_dir, ignore_errors=True)

    G = nx.Graph()
    for row in entity_rows or []:
        uid = row.get("uuid")
        if not uid:
            continue
        G.add_node(
            str(uid),
            type="Entity",
            name=row.get("name") or "",
            summary=row.get("summary") or "",
        )
    for row in episodic_rows or []:
        uid = row.get("uuid")
        if not uid:
            continue
        content = row.get("content") or ""
        G.add_node(
            str(uid),
            type="Episodic",
            name=row.get("name") or content[:120],
            content=content,
        )
    for row in rel_rows or []:
        s, t = row.get("src"), row.get("tgt")
        if not s or not t or s == t:
            continue
        rel = row.get("rel") or "relates_to"
        fact = row.get("fact") or ""
        if G.has_edge(str(s), str(t)):
            continue
        G.add_edge(str(s), str(t), relation=str(rel).lower(), fact=fact)
    for row in mention_rows or []:
        s, t = row.get("src"), row.get("tgt")
        if not s or not t or s == t:
            continue
        if G.has_edge(str(s), str(t)):
            continue
        G.add_edge(str(s), str(t), relation="mentions")

    type_counts: dict[str, int] = {}
    for _, attrs in G.nodes(data=True):
        type_counts[attrs.get("type", "unknown")] = (
            type_counts.get(attrs.get("type", "unknown"), 0) + 1
        )
    return G, type_counts


def evaluate_structure(
    G: nx.Graph,
    gold_graph: dict[str, Any],
    events: list[dict[str, Any]],
    n_input_events: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    all_nodes = list(G.nodes(data=True))
    entity_nodes = [n for n in all_nodes if n[1].get("type") == "Entity"]

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
            track_a["separation"] = round(compute_concept_separation(labels[:50]), 4)

        degree_sorted = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
        sampled = [
            {
                "id": str(n),
                "title": G.nodes[n].get("name", str(n)),
                "label": G.nodes[n].get("name", str(n)),
            }
            for n in degree_sorted[:30]
            if G.nodes[n].get("type") == "Entity"
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
        edge_types = [G.edges[e].get("relation", "relates_to") for e in G.edges()]
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
        n_entity = len(entity_nodes)
        n_episodic = sum(1 for _, a in all_nodes if a.get("type") == "Episodic")

        type_counts: dict[str, int] = {}
        for _, attrs in all_nodes:
            t = attrs.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        compression_ratio = n_input_events / max(n_entity, 1)

        results["compression"] = {
            "pagerank_gini": round(gini, 4),
            "compression_ratio": round(compression_ratio, 2),
            "concept_fraction": round(n_entity / max(n_nodes, 1), 4),
            "edge_density": round(n_edges / max(n_nodes, 1), 4),
            "node_count": n_nodes,
            "edge_count": n_edges,
            "entity_count": n_entity,
            "episodic_count": n_episodic,
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


def run_baseline_zep(scenario: str = "software_engineer", scale: str = "small") -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    dataset = load_dataset(scenario, scale)
    events = dataset["events"]
    gold_graph = dataset["gold_graph"]
    n_input_events = len(events)

    print(f"=== Zep/Graphiti Baseline (bi-temporal entity graph): {dataset['name']} ===")
    print(f"  Events: {n_input_events}")

    print("\n  Building Graphiti graph (Kuzu embedded driver)...")
    t0 = time.time()
    G, type_counts = asyncio.run(_build(events, api_key))
    elapsed = time.time() - t0
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ({elapsed:.1f}s)")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t}: {c}")

    print("\n  Evaluating structural metrics...")
    metrics = evaluate_structure(G, gold_graph, events, n_input_events)

    print("\n" + "=" * 60)
    print(f"ZEP/GRAPHITI BASELINE RESULTS — {dataset['name']}")
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
    output_path = output_dir / f"baseline_zep_results_{scenario}.json"

    output = {
        "benchmark": "cogeval_baseline_zep",
        "method": "Zep/Graphiti (bi-temporal entity graph)",
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
    parser = argparse.ArgumentParser(description="CogEval-Bench Zep/Graphiti Baseline")
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
    run_baseline_zep(scenario=args.scenario, scale=args.scale)


if __name__ == "__main__":
    main()
