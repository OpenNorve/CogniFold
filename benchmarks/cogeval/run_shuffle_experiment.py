"""Conceptual Bootstrapping (Order-Dependence) Experiment on CogEval-Bench.

Tests whether CogniFold's emergent cognitive structure is order-dependent
(Zhao et al. 2023, Nature Human Behaviour). For each scenario we ingest the
same event set in N different orderings and compare the resulting graphs.

Usage:
    # Pilot (1 scenario, 3 orders)
    OPENAI_API_KEY=... PYTHONPATH=src python -m benchmarks.cogeval.run_shuffle_experiment \\
        --scenario software_engineer --scale small

    # Full sweep (6 scenarios × 3 orders = 18 runs)
    OPENAI_API_KEY=... PYTHONPATH=src python -m benchmarks.cogeval.run_shuffle_experiment \\
        --scenario all --scale small

Each run produces:
    output/shuffle_{scenario}_{order}.json    (per-order graph snapshot)
    output/shuffle_{scenario}_summary.json    (cross-order comparison metrics)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Path setup (must run before cognifold imports)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_project_root, "src"))
sys.path.append(_project_root)

import networkx as nx  # noqa: E402

from cognifold.agent.agent import CognifoldAgent  # noqa: E402
from cognifold.agent.config import AgentConfig  # noqa: E402
from cognifold.executor.runner import PlanExecutor  # noqa: E402
from cognifold.graph.consolidation import (  # noqa: E402
    merge_similar_concepts,
    prune_orphan_concepts,
)
from cognifold.graph.store import ConceptGraph  # noqa: E402
from cognifold.models.event import Event  # noqa: E402
from cognifold.models.node import NodeType  # noqa: E402
from cognifold.query.agent import MemoryQueryAgent  # noqa: E402
from cognifold.query.models import QueryConfig, RetrievalMode  # noqa: E402

logger = logging.getLogger(__name__)

SCENARIOS = [
    "software_engineer",
    "health_journey",
    "team_project",
    "news_stream",
    "academic_research",
    "customer_support",
]

# (order_name, seed) — seed=None means keep chronological order
DEFAULT_ORDERS: list[tuple[str, int | None]] = [
    ("original", None),
    ("shuffle_42", 42),
    ("shuffle_123", 123),
]


# ---------------------------------------------------------------------------
# Dataset loading + event building
# ---------------------------------------------------------------------------
def load_dataset(scenario: str, scale: str) -> dict[str, Any]:
    path = Path(__file__).parent / "data" / "generated" / f"{scenario}_{scale}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path) as f:
        return json.load(f)


def build_events(dataset: dict[str, Any]) -> list[Event]:
    """Build events in the dataset's original chronological order."""
    events: list[Event] = []
    base_time = datetime(2024, 6, 1, 0, 0, 0)
    for i, ev_data in enumerate(dataset["events"]):
        try:
            ts = datetime.fromisoformat(ev_data["timestamp"])
        except (ValueError, KeyError):
            ts = base_time + timedelta(seconds=i * 60)
        events.append(
            Event(
                event_id=ev_data.get("event_id", str(uuid.uuid4())),
                timestamp=ts,
                source=ev_data.get("source", "cogeval-bench"),
                event_type=ev_data.get("event_type", "life_event"),
                title=ev_data.get("title", f"Event {i + 1}"),
                description=ev_data["description"],
                context={
                    "benchmark": "cogeval",
                    "scenario": dataset.get("scenario_id", ""),
                    "gold_concept": ev_data.get("gold_concept", ""),
                    "event_index": i,
                },
            )
        )
    return events


def permute(events: list[Event], seed: int | None) -> list[Event]:
    if seed is None:
        return list(events)
    rng = random.Random(seed)
    shuffled = list(events)
    rng.shuffle(shuffled)
    return shuffled


# ---------------------------------------------------------------------------
# Ingestion (compact mirror of base_runner.run, only what we need)
# ---------------------------------------------------------------------------
def ingest_one_order(
    events: list[Event],
    model_name: str,
    embedder: Any | None,
    retrieval_mode: RetrievalMode,
) -> ConceptGraph:
    """Run a single ingestion pass and return the resulting graph."""
    graph = ConceptGraph()
    config = AgentConfig(model_name=model_name, temperature=0.0)
    agent = CognifoldAgent(config=config)
    executor = PlanExecutor(graph)

    qc = QueryConfig(
        domain="cogeval",
        max_nodes=20,
        include_reasoning=True,
        retrieval_mode=retrieval_mode,
    )
    query_agent = MemoryQueryAgent(graph, config=qc, embedder=embedder)

    for event in events:
        try:
            desc = (event.description or "")[:200]
            retrieval = query_agent.query_semantic(desc)
            context_node_ids = [n.node_id for n in retrieval.nodes[:10]]
            plan = agent.process_event(
                event=event,
                graph=graph,
                context_node_ids=context_node_ids,
                node_scores={},
            )
            executor.execute(plan)
            time.sleep(0.1)
        except Exception as e:
            msg = str(e)
            logger.warning("Event %s failed: %s", event.event_id, msg)
            if "429" in msg:
                time.sleep(10)

    # Post-ingestion consolidation (same as base_runner)
    try:
        merge_similar_concepts(graph)
        prune_orphan_concepts(graph)
    except Exception as e:
        logger.debug("Consolidation failed: %s", e)

    return graph


# ---------------------------------------------------------------------------
# Graph snapshot
# ---------------------------------------------------------------------------
@dataclass
class GraphSnapshot:
    order_name: str
    seed: int | None
    n_events: int
    event_order: list[str]
    concepts: list[dict[str, Any]] = field(default_factory=list)  # id, title, pagerank
    intents: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, str]] = field(default_factory=list)
    structural: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_name": self.order_name,
            "seed": self.seed,
            "n_events": self.n_events,
            "event_order": self.event_order,
            "concepts": self.concepts,
            "intents": self.intents,
            "edges": self.edges,
            "structural": self.structural,
        }


def snapshot_graph(
    graph: ConceptGraph,
    order_name: str,
    seed: int | None,
    events: list[Event],
) -> GraphSnapshot:
    # Build nx.DiGraph for PageRank
    G = nx.DiGraph()
    for node in graph.get_all_nodes():
        G.add_node(node.id)
    for edge in graph.get_all_edges():
        G.add_edge(edge.source, edge.target)

    pr = nx.pagerank(G) if G.number_of_nodes() > 0 else {}

    # Filter out symbolic (sym-) concepts — we want LLM-generated only
    concepts: list[dict[str, Any]] = []
    for node in graph.get_nodes_by_type(NodeType.CONCEPT):
        if node.id.startswith("sym-") or node.data.get("symbolic_type"):
            continue
        concepts.append(
            {
                "id": node.id,
                "title": str(node.data.get("title", node.data.get("content", ""))),
                "pagerank": float(pr.get(node.id, 0.0)),
            }
        )

    intents: list[dict[str, Any]] = []
    for node in graph.get_nodes_by_type(NodeType.INTENT):
        n_groundings = sum(
            1 for e in graph.get_all_edges() if e.target == node.id
        )
        intents.append(
            {
                "id": node.id,
                "title": str(node.data.get("title", "")),
                "pagerank": float(pr.get(node.id, 0.0)),
                "n_groundings": n_groundings,
            }
        )

    edges: list[dict[str, str]] = []
    for e in graph.get_all_edges():
        edges.append(
            {
                "source": e.source,
                "target": e.target,
                "type": str(getattr(e, "edge_type", "RELATED_TO")),
            }
        )

    # Structural scalar metrics for topology distance
    und = G.to_undirected() if G.number_of_nodes() > 0 else nx.Graph()
    structural = {
        "node_count": float(G.number_of_nodes()),
        "edge_count": float(G.number_of_edges()),
        "concept_count": float(len(concepts)),
        "intent_count": float(len(intents)),
        "avg_clustering": float(nx.average_clustering(und)) if und.number_of_nodes() else 0.0,
        "pr_gini": float(_gini([v for v in pr.values()])) if pr else 0.0,
    }

    return GraphSnapshot(
        order_name=order_name,
        seed=seed,
        n_events=len(events),
        event_order=[e.event_id for e in events],
        concepts=concepts,
        intents=intents,
        edges=edges,
        structural=structural,
    )


def _gini(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cumul = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    return (2 * cumul) / (n * total) - (n + 1) / n


# ---------------------------------------------------------------------------
# Cross-order comparison metrics
# ---------------------------------------------------------------------------
def _soft_jaccard(
    labels_a: list[str], labels_b: list[str], threshold: float = 0.75
) -> float:
    """Embedding-based Jaccard: concept-a matches concept-b if cosine >= threshold."""
    if not labels_a or not labels_b:
        return 0.0
    try:
        from benchmarks.cogeval import concept_evaluator as _ce

        sim = _ce._embedding_similarity(labels_a, labels_b)  # noqa: SLF001
        # Greedy 1-to-1 matching above threshold
        matched_a: set[int] = set()
        matched_b: set[int] = set()
        # Rank pairs by similarity desc and take in order
        pairs = [
            (sim[i, j], i, j)
            for i in range(len(labels_a))
            for j in range(len(labels_b))
            if sim[i, j] >= threshold
        ]
        pairs.sort(reverse=True)
        for _s, i, j in pairs:
            if i not in matched_a and j not in matched_b:
                matched_a.add(i)
                matched_b.add(j)
        intersection = len(matched_a)
        union = len(labels_a) + len(labels_b) - intersection
        return intersection / union if union > 0 else 0.0
    except Exception as e:
        logger.warning("Soft-jaccard embedding fallback (exact-set): %s", e)
        set_a = set(labels_a)
        set_b = set(labels_b)
        union = len(set_a | set_b)
        return len(set_a & set_b) / union if union > 0 else 0.0


def _top_k_pagerank_titles(snap: GraphSnapshot, k: int) -> list[str]:
    sorted_c = sorted(snap.concepts, key=lambda c: c["pagerank"], reverse=True)
    return [c["title"] for c in sorted_c[:k]]


def compare_snapshots(snaps: list[GraphSnapshot]) -> dict[str, Any]:
    """Compute cross-run comparison metrics across all snapshots."""
    pairs = [
        (a.order_name, b.order_name, a, b)
        for idx, a in enumerate(snaps)
        for b in snaps[idx + 1 :]
    ]

    # Pairwise concept Jaccard (soft-matched)
    pairwise_concept_jaccard: dict[str, float] = {}
    for name_a, name_b, a, b in pairs:
        labels_a = [c["title"] for c in a.concepts if c["title"]]
        labels_b = [c["title"] for c in b.concepts if c["title"]]
        pairwise_concept_jaccard[f"{name_a}_vs_{name_b}"] = round(
            _soft_jaccard(labels_a, labels_b), 4
        )

    # Intent Jaccard (titles, soft-matched — intents are usually phrases)
    pairwise_intent_jaccard: dict[str, float] = {}
    for name_a, name_b, a, b in pairs:
        labels_a = [i["title"] for i in a.intents if i["title"]]
        labels_b = [i["title"] for i in b.intents if i["title"]]
        pairwise_intent_jaccard[f"{name_a}_vs_{name_b}"] = round(
            _soft_jaccard(labels_a, labels_b) if labels_a and labels_b else 0.0,
            4,
        )

    # Core concept stability: fraction of top-k PageRank concepts present in ALL runs
    # Use soft-match (cos >= 0.75) — a core concept is stable if, for each other run,
    # there exists some concept within threshold.
    core_stability: dict[str, float] = {}
    for k in (3, 5, 10):
        if not snaps:
            core_stability[f"top_{k}"] = 0.0
            continue
        ref_top = _top_k_pagerank_titles(snaps[0], k)
        if not ref_top:
            core_stability[f"top_{k}"] = 0.0
            continue
        from benchmarks.cogeval import concept_evaluator as _ce

        stable_count = 0
        for ref_label in ref_top:
            # Must have a match in every other snapshot
            present_everywhere = True
            for other in snaps[1:]:
                other_labels = [c["title"] for c in other.concepts if c["title"]]
                if not other_labels:
                    present_everywhere = False
                    break
                try:
                    sim = _ce._embedding_similarity([ref_label], other_labels)  # noqa: SLF001
                    if sim.max() < 0.75:
                        present_everywhere = False
                        break
                except Exception:
                    if ref_label not in other_labels:
                        present_everywhere = False
                        break
            if present_everywhere:
                stable_count += 1
        core_stability[f"top_{k}"] = round(stable_count / len(ref_top), 4)

    # Topology distance (pairwise)
    pairwise_topology: dict[str, dict[str, float]] = {}
    for name_a, name_b, a, b in pairs:
        pairwise_topology[f"{name_a}_vs_{name_b}"] = {
            "delta_edges": round(
                abs(a.structural["edge_count"] - b.structural["edge_count"]),
                2,
            ),
            "delta_concepts": round(
                abs(a.structural["concept_count"] - b.structural["concept_count"]),
                2,
            ),
            "delta_clustering": round(
                abs(a.structural["avg_clustering"] - b.structural["avg_clustering"]),
                4,
            ),
            "delta_pr_gini": round(
                abs(a.structural["pr_gini"] - b.structural["pr_gini"]),
                4,
            ),
        }

    return {
        "pairwise_concept_jaccard": pairwise_concept_jaccard,
        "pairwise_intent_jaccard": pairwise_intent_jaccard,
        "core_concept_stability": core_stability,
        "pairwise_topology_distance": pairwise_topology,
        "per_order_summary": [
            {
                "order_name": s.order_name,
                "seed": s.seed,
                "n_concepts": len(s.concepts),
                "n_intents": len(s.intents),
                "n_edges": int(s.structural["edge_count"]),
                "avg_clustering": round(s.structural["avg_clustering"], 4),
                "pr_gini": round(s.structural["pr_gini"], 4),
            }
            for s in snaps
        ],
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_scenario(
    scenario: str,
    scale: str,
    orders: list[tuple[str, int | None]],
    output_dir: Path,
    model_name: str,
) -> dict[str, Any]:
    print(f"\n{'=' * 72}\nScenario: {scenario} ({scale})\n{'=' * 72}")
    dataset = load_dataset(scenario, scale)
    base_events = build_events(dataset)
    print(f"Loaded {len(base_events)} events")

    # Resolve embedder (same logic as base_runner)
    try:
        from benchmarks._utils import create_embedder, resolve_embedding

        resolved = resolve_embedding(None, Path("configs/prompt_profiles.yaml"), "cogeval")
        embedder, retrieval_mode = create_embedder(resolved)
    except Exception as e:
        print(f"Embedder init failed ({e}); BM25 fallback")
        embedder = None
        retrieval_mode = RetrievalMode.BM25

    snapshots: list[GraphSnapshot] = []
    for order_name, seed in orders:
        print(f"\n--- Order: {order_name} (seed={seed}) ---")
        events = permute(base_events, seed)
        start = time.time()
        graph = ingest_one_order(events, model_name, embedder, retrieval_mode)
        elapsed = time.time() - start
        snap = snapshot_graph(graph, order_name, seed, events)
        snapshots.append(snap)
        print(
            f"  {elapsed:.1f}s | {len(snap.concepts)} concepts, "
            f"{len(snap.intents)} intents, "
            f"{int(snap.structural['edge_count'])} edges"
        )
        snap_path = output_dir / f"shuffle_{scenario}_{order_name}.json"
        with open(snap_path, "w") as f:
            json.dump(snap.to_dict(), f, indent=2)

    summary = compare_snapshots(snapshots)
    summary["scenario"] = scenario
    summary["scale"] = scale
    summary["orders"] = [o[0] for o in orders]
    summary["model"] = model_name

    summary_path = output_dir / f"shuffle_{scenario}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")

    _print_summary(summary)
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("\n  === Cross-order comparison ===")
    print("  Concept Jaccard (soft, cos>=0.75):")
    for pair, val in summary["pairwise_concept_jaccard"].items():
        print(f"    {pair}: {val:.3f}")
    print("  Intent Jaccard:")
    for pair, val in summary["pairwise_intent_jaccard"].items():
        print(f"    {pair}: {val:.3f}")
    print("  Core concept stability (top-k PR present in all orders):")
    for k_label, val in summary["core_concept_stability"].items():
        print(f"    {k_label}: {val:.3f}")
    print("  Topology distance:")
    for pair, dists in summary["pairwise_topology_distance"].items():
        print(
            f"    {pair}: Δedges={dists['delta_edges']}, "
            f"Δconcepts={dists['delta_concepts']}, "
            f"Δclustering={dists['delta_clustering']:.3f}, "
            f"Δpr_gini={dists['delta_pr_gini']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="CogEval shuffle experiment")
    parser.add_argument(
        "--scenario",
        default="software_engineer",
        help="Scenario name, or 'all'",
    )
    parser.add_argument("--scale", default="small", choices=["small", "medium"])
    parser.add_argument(
        "--orders",
        nargs="*",
        default=None,
        help='Orderings as "name:seed" pairs (use seed=none for original). '
        'Default: original, shuffle_42, shuffle_123',
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model override (default: openai:gpt-4o-mini if OPENAI_API_KEY else gemini-2.5-flash)",
    )
    args = parser.parse_args()

    # Resolve model
    if args.model:
        model_name = args.model
    elif os.environ.get("OPENAI_API_KEY"):
        model_name = "openai:gpt-4o-mini"
    elif os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        model_name = "gemini-2.5-flash"
    else:
        print("ERROR: Set OPENAI_API_KEY or GOOGLE_API_KEY")
        sys.exit(1)
    print(f"Model: {model_name}")

    # Resolve orders
    orders: list[tuple[str, int | None]] = DEFAULT_ORDERS
    if args.orders:
        parsed: list[tuple[str, int | None]] = []
        for spec in args.orders:
            if ":" not in spec:
                raise ValueError(f"Invalid order spec '{spec}' (expected name:seed)")
            name, seed_str = spec.split(":", 1)
            seed = None if seed_str.lower() in ("none", "null", "") else int(seed_str)
            parsed.append((name, seed))
        orders = parsed

    # Resolve scenarios
    scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]
    for sc in scenarios:
        if sc not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{sc}'. Choose from {SCENARIOS} or 'all'")

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, Any]] = []
    for sc in scenarios:
        summary = run_scenario(sc, args.scale, orders, output_dir, model_name)
        all_summaries.append(summary)

    if len(all_summaries) > 1:
        # Aggregate summary
        agg_path = output_dir / f"shuffle_all_summary_{args.scale}.json"
        with open(agg_path, "w") as f:
            json.dump({"scenarios": all_summaries}, f, indent=2)
        print(f"\nAggregate summary: {agg_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
