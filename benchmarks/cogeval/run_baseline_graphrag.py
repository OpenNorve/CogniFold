"""CogEval-Bench GraphRAG Baseline (community-detection + summarization).

Simulates a GraphRAG-style system: builds an entity graph, runs community
detection, then uses LLM to summarize each community into a "concept."

This represents the GraphRAG approach: has graph + community structure but
no temporal folding, no intent emergence, no online processing.

Usage:
    OPENAI_API_KEY=... PYTHONPATH=src python -m benchmarks.cogeval.run_baseline_graphrag \
        --scenario software_engineer --scale small
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import networkx as nx


def extract_entities_and_relations(
    events: list[dict[str, Any]],
    model: str = "openai:gpt-4o-mini",
    batch_size: int = 5,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Extract entities and relations from events via LLM.

    Returns (entities, relations) where:
    - entities: list of {name, type} dicts
    - relations: list of {source, target, relation} dicts
    """
    from benchmarks.shared.base_runner import _call_llm_text

    all_entities: list[dict[str, str]] = []
    all_relations: list[dict[str, str]] = []

    system_prompt = (
        "You are an entity and relation extraction system. From the given "
        "events, extract:\n"
        "1. Named entities with their type (person, place, organization, "
        "concept, event)\n"
        "2. Relations between entities\n\n"
        "Output ONLY a JSON object with two keys:\n"
        '- "entities": [{"name": "...", "type": "..."}]\n'
        '- "relations": [{"source": "...", "target": "...", "relation": "..."}]\n'
        "Normalize entity names to lowercase."
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
                user_prompt=f"Extract entities and relations:\n\n{events_text}",
                temperature=0.0,
                max_tokens=2000,
            )
            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(response)

            for e in data.get("entities", []):
                if not isinstance(e, dict):
                    continue
                if "name" not in e:
                    continue
                all_entities.append(
                    {
                        "name": str(e["name"]).strip().lower(),
                        "type": str(e.get("type", "concept")).strip().lower(),
                    }
                )

            for r in data.get("relations", []):
                if not isinstance(r, dict):
                    continue
                if "source" not in r:
                    continue
                if "target" not in r:
                    continue
                all_relations.append(
                    {
                        "source": str(r["source"]).strip().lower(),
                        "target": str(r["target"]).strip().lower(),
                        "relation": str(
                            r.get("relation", "related_to")
                        ).strip().lower(),
                    }
                )
        except Exception as e:
            print(f"    Extraction error (batch {i // batch_size}): {e}")

        time.sleep(0.3)

    return all_entities, all_relations


def build_entity_graph(
    entities: list[dict[str, str]],
    relations: list[dict[str, str]],
) -> nx.Graph:
    """Build a NetworkX graph from entities and relations."""
    G = nx.Graph()
    for e in entities:
        G.add_node(e["name"], type=e.get("type", "concept"))
    for r in relations:
        G.add_edge(
            r["source"], r["target"], relation=r.get("relation", "related_to")
        )
        if r["source"] not in G:
            G.add_node(r["source"])
        if r["target"] not in G:
            G.add_node(r["target"])
    return G


def detect_communities(G: nx.Graph) -> list[set[str]]:
    """Detect communities using Louvain-like greedy modularity."""
    if G.number_of_nodes() < 2:
        if G.number_of_nodes() > 0:
            return [set(G.nodes())]
        return []

    try:
        communities = list(nx.community.greedy_modularity_communities(G))
        return [set(c) for c in communities]
    except Exception:
        return [set(G.nodes())]


def summarize_communities(
    communities: list[set[str]],
    events: list[dict[str, Any]],
    model: str = "openai:gpt-4o-mini",
) -> list[dict[str, str]]:
    """Use LLM to summarize each community into a concept label."""
    from benchmarks.shared.base_runner import _call_llm_text

    summaries: list[dict[str, Any]] = []

    for i, community in enumerate(communities):
        if len(community) < 2:
            if community:
                summaries.append(
                    {
                        "id": f"community_{i}",
                        "label": list(community)[0],
                        "title": list(community)[0],
                        "members": list(community),
                    }
                )
            continue

        members = ", ".join(sorted(community)[:20])

        try:
            response = _call_llm_text(
                model=model,
                system_prompt=(
                    "You summarize entity clusters into concept labels. "
                    "Given a cluster of related entities, produce a short "
                    "(2-5 word) concept label that captures the theme. "
                    "Output ONLY the label, nothing else."
                ),
                user_prompt=f"Entities in this cluster: {members}",
                temperature=0.0,
                max_tokens=50,
            )
            label = response.strip().strip('"').strip("'")
        except Exception:
            label = f"Community {i}"

        summaries.append(
            {
                "id": f"community_{i}",
                "label": label,
                "title": label,
                "members": list(community),
            }
        )

        time.sleep(0.2)

    return summaries


def evaluate_structure(
    G: nx.Graph,
    communities: list[set[str]],
    community_concepts: list[dict[str, str]],
    gold_graph: dict[str, Any],
    events: list[dict[str, Any]],
    n_input_events: int,
) -> dict[str, Any]:
    """Run Track A/B/C evaluators on the GraphRAG graph."""
    results: dict[str, Any] = {}

    try:
        from benchmarks.cogeval.concept_evaluator import (
            compute_concept_separation,
            compute_harmony_score,
            evaluate_concept_emergence,
            evaluate_concept_quality_llm,
        )

        system_concepts = community_concepts
        gold_concepts = gold_graph.get("concepts", [])

        ce_result = evaluate_concept_emergence(system_concepts, gold_concepts, events)
        track_a = ce_result.to_dict()

        labels = [c["label"] for c in community_concepts]
        if len(labels) >= 2:
            track_a["separation"] = round(compute_concept_separation(labels), 4)

        llm_quality, _ = evaluate_concept_quality_llm(system_concepts, events)
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
        n_communities = len(communities)
        compression_ratio = n_input_events / max(n_communities, 1)

        results["compression"] = {
            "pagerank_gini": round(gini, 4),
            "compression_ratio": round(compression_ratio, 2),
            "concept_fraction": round(n_communities / max(n_nodes, 1), 4),
            "edge_density": round(n_edges / max(n_nodes, 1), 4),
            "node_count": n_nodes,
            "edge_count": n_edges,
            "concept_count": n_communities,
            "entity_count": n_nodes,
            "intent_count": 0,
            "input_events": n_input_events,
            "proactivity": 0.0,
            "schema_acceleration": 0.0,
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


def run_baseline_graphrag(
    scenario: str = "software_engineer",
    scale: str = "small",
    model: str = "openai:gpt-4o-mini",
) -> dict[str, Any]:
    """Run GraphRAG baseline on a CogEval scenario."""
    dataset = load_dataset(scenario, scale)
    events = dataset["events"]
    gold_graph = dataset["gold_graph"]
    n_input_events = len(events)

    print(f"=== GraphRAG Baseline (community-detection): {dataset['name']} ===")
    print(f"  Events: {n_input_events}")

    print("\n  Extracting entities and relations via LLM...")
    entities, relations = extract_entities_and_relations(events, model=model)
    print(
        f"  Extracted {len(entities)} entities, {len(relations)} relations"
    )

    G = build_entity_graph(entities, relations)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\n  Detecting communities...")
    communities = detect_communities(G)
    print(f"  Found {len(communities)} communities")

    for i, c in enumerate(communities[:5]):
        print(f"    Community {i}: {len(c)} members — {list(c)[:5]}...")

    print("\n  Summarizing communities into concepts...")
    community_concepts = summarize_communities(communities, events, model=model)

    for cc in community_concepts[:5]:
        print(f"    {cc['label']} ({len(cc.get('members', []))} members)")

    print("\n  Evaluating structural metrics...")
    metrics = evaluate_structure(
        G, communities, community_concepts, gold_graph, events, n_input_events
    )

    print("\n" + "=" * 60)
    print(f"GraphRAG BASELINE RESULTS — {dataset['name']}")
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
        print(f"    Communities:  {cp.get('concept_count', 0)}")
        print(f"    Nodes: {cp.get('node_count', 0)} entities")

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"baseline_graphrag_results_{scenario}.json"

    output = {
        "benchmark": "cogeval_baseline_graphrag",
        "method": "GraphRAG (community-detection + summarization)",
        "scenario": scenario,
        "scale": scale,
        "model": model,
        "structural_metrics": metrics,
        "graph_stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "communities": len(communities),
            "entities": len(entities),
            "relations": len(relations),
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="CogEval-Bench GraphRAG Baseline")
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

    run_baseline_graphrag(
        scenario=args.scenario, scale=args.scale, model=args.model
    )


if __name__ == "__main__":
    main()
