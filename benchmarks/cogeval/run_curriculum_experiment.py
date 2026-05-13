"""Stage B: Curriculum vs Anti-Curriculum Experiment on CogEval-Bench.

Validates the directional claim of Zhao et al. (2023) conceptual bootstrapping:
foundational-first (curriculum) orderings should produce more compositional,
more generalizable concept graphs than derived-first (anti-curriculum) orderings.

Four orderings per scenario:
    - chronological: dataset's original order (baseline)
    - random_42: random shuffle with fixed seed (baseline)
    - curriculum: sort events by foundational-ness of their gold_concept
                  (high expected_events + strength="high" first)
    - anti_curriculum: reverse of curriculum (derived first)

Metrics per order:
    - Structural (reused from shuffle): concepts, edges, clustering, PR Gini
    - Compositional: compositional-edge ratio (PART_OF + DERIVED_FROM + CAUSES
      over total edges); mean concept depth via PART_OF chains
    - Generalization: held-out event classification accuracy
      (20% stratified hold-out; for each test event, check whether top-k
      returned concepts semantically match its gold_concept_label)

Usage:
    OPENAI_API_KEY=... PYTHONPATH=src python -u -m \\
        benchmarks.cogeval.run_curriculum_experiment \\
        --scenario software_engineer --scale small
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Path setup (must run before cognifold imports)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_project_root, "src"))
sys.path.append(_project_root)

import networkx as nx  # noqa: E402

from benchmarks.cogeval.run_shuffle_experiment import (  # noqa: E402
    SCENARIOS,
    build_events,
    ingest_one_order,
    load_dataset,
    snapshot_graph,
)
from cognifold.graph.store import ConceptGraph  # noqa: E402
from cognifold.models.event import Event  # noqa: E402
from cognifold.models.node import NodeType  # noqa: E402
from cognifold.query.agent import MemoryQueryAgent  # noqa: E402
from cognifold.query.models import QueryConfig, RetrievalMode  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Foundational-ness scoring (from gold_graph)
# ---------------------------------------------------------------------------
STRENGTH_BONUS = {"high": 2.0, "medium": 1.0, "low": 0.0}


def build_concept_score_map(gold_graph: dict[str, Any]) -> dict[str, float]:
    """Map gold_concept_id → foundational score.

    Higher = more foundational (frequent, high-strength, root-level).
    """
    score: dict[str, float] = {}
    concepts = gold_graph.get("concepts", [])
    # Detect which concept IDs are referenced as parents (→ they have children
    # → they are more foundational in the hierarchy).
    child_counts: dict[str, int] = defaultdict(int)
    for c in concepts:
        parent = c.get("parent")
        if parent:
            child_counts[parent] += 1

    for c in concepts:
        cid = c.get("id", "")
        expected = float(c.get("expected_events", 1))
        strength = STRENGTH_BONUS.get(c.get("expected_strength", "medium"), 1.0)
        child_bonus = float(child_counts.get(cid, 0))  # root concepts often have children
        score[cid] = expected + strength + child_bonus

    return score


def score_event(event: Event, concept_scores: dict[str, float]) -> float:
    """Per-event foundational score. Events without a gold_concept (distractor/
    chain) fall in the middle so they don't dominate either tail."""
    gold_id = ""
    ctx = event.context
    if ctx:
        gold_id = ctx.get("gold_concept", "") or ""
    if gold_id in concept_scores:
        return concept_scores[gold_id]
    # Distractor / chain / unknown — use median of observed scores
    if concept_scores:
        vals = sorted(concept_scores.values())
        return vals[len(vals) // 2]
    return 0.0


def order_events(
    events: list[Event],
    concept_scores: dict[str, float],
    mode: str,
    seed: int | None = None,
) -> list[Event]:
    """Produce an event ordering by mode.

    Modes:
        chronological: keep input order
        random_<seed>: random shuffle with seed
        curriculum: sort by score desc (foundational first), stable within score
        anti_curriculum: sort by score asc (derived first), stable within score
    """
    if mode == "chronological":
        return list(events)
    if mode.startswith("random"):
        rng = random.Random(seed if seed is not None else 42)
        out = list(events)
        rng.shuffle(out)
        return out
    scored = [(score_event(e, concept_scores), i, e) for i, e in enumerate(events)]
    # stable secondary key = original index so ties keep chronological order
    if mode == "curriculum":
        scored.sort(key=lambda t: (-t[0], t[1]))
    elif mode == "anti_curriculum":
        scored.sort(key=lambda t: (t[0], t[1]))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return [t[2] for t in scored]


# ---------------------------------------------------------------------------
# Stratified 80/20 split for held-out classification test
# ---------------------------------------------------------------------------
def stratified_split(
    events: list[Event], test_frac: float = 0.2, seed: int = 7
) -> tuple[list[Event], list[Event]]:
    """Stratified by gold_concept: hold out test_frac of each concept's events.

    Events without a gold_concept stay in the training set (we can't test
    generalization on unlabeled events).
    """
    rng = random.Random(seed)
    by_concept: dict[str, list[Event]] = defaultdict(list)
    no_label: list[Event] = []
    for e in events:
        gold_id = (e.context or {}).get("gold_concept", "") if e.context else ""
        if gold_id:
            by_concept[gold_id].append(e)
        else:
            no_label.append(e)

    train: list[Event] = list(no_label)
    test: list[Event] = []
    for _, evs in by_concept.items():
        evs = list(evs)
        rng.shuffle(evs)
        n_test = max(0, int(round(len(evs) * test_frac)))
        # Never hold out so many that training has < 1
        n_test = min(n_test, max(0, len(evs) - 1))
        test.extend(evs[:n_test])
        train.extend(evs[n_test:])

    return train, test


# ---------------------------------------------------------------------------
# Held-out classification (generalization metric)
# ---------------------------------------------------------------------------
def _get_concept_labels(graph: ConceptGraph) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for node in graph.get_nodes_by_type(NodeType.CONCEPT):
        if node.id.startswith("sym-") or node.data.get("symbolic_type"):
            continue
        title = str(node.data.get("title", node.data.get("content", "")))
        if title:
            out.append({"id": node.id, "title": title})
    return out


def _cosine_sim_matrix(texts_a: list[str], texts_b: list[str]) -> Any:
    """Cosine similarity matrix using the shared CogniFold embedding service.

    Dependency-free alternative to concept_evaluator._embedding_similarity,
    which requires sentence-transformers / sklearn (both broken under
    Python 3.14 in our venv). Reuses the OpenAI / local embedder that the
    rest of CogniFold already uses.
    """
    import numpy as np

    from cognifold.utils.embeddings import get_embedding_service

    svc = get_embedding_service()
    emb_a = np.asarray([svc.embed_text(t) for t in texts_a], dtype=np.float32)
    emb_b = np.asarray([svc.embed_text(t) for t in texts_b], dtype=np.float32)

    # Normalize rows for cosine
    def _normalize(m):
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms

    a = _normalize(emb_a)
    b = _normalize(emb_b)
    return np.dot(a, b.T)


def classify_held_out(
    graph: ConceptGraph,
    held_out: list[Event],
    embedder: Any | None,
    retrieval_mode: RetrievalMode,
    top_k: int = 3,
    match_threshold: float = 0.75,
) -> dict[str, Any]:
    """For each held-out event, query graph for top-k concepts; check whether
    any top-k concept title semantically matches the event's gold_concept_label.

    Returns {accuracy_top1, accuracy_topk, n_tested, details}.
    """
    qc = QueryConfig(
        domain="cogeval",
        max_nodes=top_k * 5,
        include_reasoning=False,
        retrieval_mode=retrieval_mode,
    )
    query_agent = MemoryQueryAgent(graph, config=qc, embedder=embedder)
    concept_nodes = _get_concept_labels(graph)
    concept_titles = [c["title"] for c in concept_nodes]

    if not held_out or not concept_titles:
        return {
            "accuracy_top1": 0.0,
            "accuracy_topk": 0.0,
            "n_tested": 0,
            "details": [],
        }

    n_top1 = 0
    n_topk = 0
    details: list[dict[str, Any]] = []

    for ev in held_out:
        gold_label = (ev.context or {}).get("gold_concept", "")
        gold_label_str = str(
            (ev.context or {}).get("gold_concept_label", "") or gold_label
        )
        if not gold_label_str:
            continue

        # Retrieve top-k concept nodes via semantic query
        desc = (ev.description or ev.title or "")[:200]
        retrieval = query_agent.query_semantic(desc)
        retrieved_titles: list[str] = []
        for n in retrieval.nodes:
            # Only keep concept-type results
            node = graph.get_node(n.node_id)
            if node and node.type == NodeType.CONCEPT:
                if node.id.startswith("sym-") or node.data.get("symbolic_type"):
                    continue
                title = str(node.data.get("title", ""))
                if title and title not in retrieved_titles:
                    retrieved_titles.append(title)
            if len(retrieved_titles) >= top_k:
                break

        if not retrieved_titles:
            details.append(
                {"event_id": ev.event_id, "gold": gold_label_str, "top": [], "match": False}
            )
            continue

        # Soft-match each retrieved title against gold_label_str
        sim_matrix = _cosine_sim_matrix(retrieved_titles, [gold_label_str])
        top1_match = bool(sim_matrix[0, 0] >= match_threshold)
        topk_match = bool(sim_matrix.max() >= match_threshold)
        if top1_match:
            n_top1 += 1
        if topk_match:
            n_topk += 1
        details.append(
            {
                "event_id": ev.event_id,
                "gold": gold_label_str,
                "top": retrieved_titles,
                "top1_sim": float(sim_matrix[0, 0]),
                "topk_max_sim": float(sim_matrix.max()),
                "match_top1": top1_match,
                "match_topk": topk_match,
            }
        )

    total = sum(1 for d in details if d.get("gold"))
    return {
        "accuracy_top1": round(n_top1 / total, 4) if total else 0.0,
        "accuracy_topk": round(n_topk / total, 4) if total else 0.0,
        "n_tested": total,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Compositional structure metrics
# ---------------------------------------------------------------------------
COMPOSITIONAL_EDGE_TYPES = {"part_of", "derived_from", "causes"}


def compute_compositional_metrics(graph: ConceptGraph) -> dict[str, float]:
    edges = list(graph.get_all_edges())
    total = len(edges)
    if total == 0:
        return {"compositional_edge_ratio": 0.0, "concept_depth": 0.0, "n_edges": 0}

    comp_count = sum(
        1 for e in edges if str(getattr(e, "edge_type", "")).lower() in COMPOSITIONAL_EDGE_TYPES
    )

    # Concept depth: longest path in the PART_OF / DERIVED_FROM subgraph
    G = nx.DiGraph()
    for node in graph.get_nodes_by_type(NodeType.CONCEPT):
        if node.id.startswith("sym-") or node.data.get("symbolic_type"):
            continue
        G.add_node(node.id)
    for e in edges:
        etype = str(getattr(e, "edge_type", "")).lower()
        if etype in ("part_of", "derived_from"):
            if G.has_node(e.source) and G.has_node(e.target):
                G.add_edge(e.source, e.target)

    depth = 0
    if G.number_of_nodes() > 0:
        try:
            depth = nx.dag_longest_path_length(G) if nx.is_directed_acyclic_graph(G) else 0
        except Exception:
            depth = 0

    return {
        "compositional_edge_ratio": round(comp_count / total, 4),
        "concept_depth": int(depth),
        "n_edges": total,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
@dataclass
class OrderResult:
    order_name: str
    n_train: int
    n_test: int
    structural: dict[str, Any] = field(default_factory=dict)
    compositional: dict[str, Any] = field(default_factory=dict)
    generalization: dict[str, Any] = field(default_factory=dict)
    top_concepts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_name": self.order_name,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "structural": self.structural,
            "compositional": self.compositional,
            "generalization": {
                "accuracy_top1": self.generalization.get("accuracy_top1", 0.0),
                "accuracy_topk": self.generalization.get("accuracy_topk", 0.0),
                "n_tested": self.generalization.get("n_tested", 0),
            },
            "top_concepts": self.top_concepts,
        }


def run_scenario_curriculum(
    scenario: str,
    scale: str,
    output_dir: Path,
    model_name: str,
    test_frac: float = 0.2,
) -> dict[str, Any]:
    print(f"\n{'=' * 72}\nCurriculum · Scenario: {scenario} ({scale})\n{'=' * 72}")
    dataset = load_dataset(scenario, scale)
    events = build_events(dataset)
    gold_graph = dataset["gold_graph"]
    concept_scores = build_concept_score_map(gold_graph)
    print(
        f"Loaded {len(events)} events | {len(concept_scores)} gold concepts"
    )

    # Stratified 80/20 split (same split for all orders for fairness)
    train_events, test_events = stratified_split(events, test_frac=test_frac, seed=7)
    print(
        f"Split: {len(train_events)} train / {len(test_events)} held-out"
    )

    # Resolve embedder (same as base_runner)
    try:
        from benchmarks._utils import create_embedder, resolve_embedding

        resolved = resolve_embedding(
            None, Path("configs/prompt_profiles.yaml"), "cogeval"
        )
        embedder, retrieval_mode = create_embedder(resolved)
    except Exception as e:
        print(f"Embedder init failed ({e}); BM25 fallback")
        embedder = None
        retrieval_mode = RetrievalMode.BM25

    ORDERS = [
        ("chronological", None),
        ("random_42", 42),
        ("curriculum", None),
        ("anti_curriculum", None),
    ]

    results: list[OrderResult] = []
    for mode, seed in ORDERS:
        print(f"\n--- Order: {mode} ---")
        ordered_train = order_events(train_events, concept_scores, mode, seed=seed)
        start = time.time()
        graph = ingest_one_order(ordered_train, model_name, embedder, retrieval_mode)
        ingest_s = time.time() - start

        snap = snapshot_graph(graph, mode, seed, ordered_train)
        comp = compute_compositional_metrics(graph)

        gen_start = time.time()
        gen = classify_held_out(
            graph, test_events, embedder, retrieval_mode, top_k=3
        )
        gen_s = time.time() - gen_start

        top5 = sorted(snap.concepts, key=lambda c: c["pagerank"], reverse=True)[:5]
        print(
            f"  ingest={ingest_s:.1f}s eval={gen_s:.1f}s | "
            f"concepts={len(snap.concepts)} edges={int(snap.structural['edge_count'])} "
            f"comp_edge_ratio={comp['compositional_edge_ratio']:.3f} "
            f"depth={comp['concept_depth']}"
        )
        print(
            f"  held-out classification (n={gen['n_tested']}): "
            f"top1={gen['accuracy_top1']:.3f} top3={gen['accuracy_topk']:.3f}"
        )
        print(f"  top-5 by PR: {[c['title'] for c in top5]}")

        results.append(
            OrderResult(
                order_name=mode,
                n_train=len(ordered_train),
                n_test=gen["n_tested"],
                structural=snap.structural,
                compositional=comp,
                generalization=gen,
                top_concepts=top5,
            )
        )

    summary = {
        "scenario": scenario,
        "scale": scale,
        "model": model_name,
        "test_frac": test_frac,
        "orders": [r.order_name for r in results],
        "results": [r.to_dict() for r in results],
        "comparison": _build_comparison(results),
    }

    out = output_dir / f"curriculum_{scenario}_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {out}")
    comp_dict: dict[str, Any] = summary["comparison"]
    _print_comparison(comp_dict)
    return summary


def _build_comparison(results: list[OrderResult]) -> dict[str, Any]:
    """Zhao prediction: curriculum > random > chronological > anti_curriculum
    on generalization; curriculum > anti_curriculum on compositional ratio."""
    by_name = {r.order_name: r for r in results}

    def metric(name: str, path: tuple[str, str]) -> float:
        r = by_name.get(name)
        if not r:
            return 0.0
        container = getattr(r, path[0], {}) or {}
        return float(container.get(path[1], 0.0))

    return {
        "generalization_top1": {
            name: metric(name, ("generalization", "accuracy_top1"))
            for name in by_name
        },
        "generalization_topk": {
            name: metric(name, ("generalization", "accuracy_topk"))
            for name in by_name
        },
        "compositional_edge_ratio": {
            name: metric(name, ("compositional", "compositional_edge_ratio"))
            for name in by_name
        },
        "concept_depth": {
            name: metric(name, ("compositional", "concept_depth"))
            for name in by_name
        },
        "n_concepts": {
            name: metric(name, ("structural", "concept_count"))
            for name in by_name
        },
        "curriculum_vs_anti": {
            "Δ_gen_top1": round(
                metric("curriculum", ("generalization", "accuracy_top1"))
                - metric("anti_curriculum", ("generalization", "accuracy_top1")),
                4,
            ),
            "Δ_gen_topk": round(
                metric("curriculum", ("generalization", "accuracy_topk"))
                - metric("anti_curriculum", ("generalization", "accuracy_topk")),
                4,
            ),
            "Δ_comp_ratio": round(
                metric("curriculum", ("compositional", "compositional_edge_ratio"))
                - metric("anti_curriculum", ("compositional", "compositional_edge_ratio")),
                4,
            ),
            "Δ_depth": round(
                metric("curriculum", ("compositional", "concept_depth"))
                - metric("anti_curriculum", ("compositional", "concept_depth")),
                4,
            ),
        },
    }


def _print_comparison(comp: dict[str, Any]) -> None:
    print("\n  === Order comparison ===")
    for metric_name in (
        "generalization_top1",
        "generalization_topk",
        "compositional_edge_ratio",
        "concept_depth",
        "n_concepts",
    ):
        print(f"  {metric_name}:")
        for order_name, val in comp[metric_name].items():
            print(f"    {order_name}: {val:.3f}")
    delta = comp["curriculum_vs_anti"]
    print("\n  curriculum − anti_curriculum:")
    for k, v in delta.items():
        print(f"    {k}: {v:+.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CogEval curriculum-vs-anti-curriculum experiment"
    )
    parser.add_argument(
        "--scenario",
        default="software_engineer",
        help="Scenario name, or 'all'",
    )
    parser.add_argument("--scale", default="small", choices=["small", "medium"])
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

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

    scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]
    for sc in scenarios:
        if sc not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {sc}")

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, Any]] = []
    for sc in scenarios:
        s = run_scenario_curriculum(
            sc, args.scale, output_dir, model_name, test_frac=args.test_frac
        )
        all_summaries.append(s)

    if len(all_summaries) > 1:
        agg = output_dir / f"curriculum_all_summary_{args.scale}.json"
        with open(agg, "w") as f:
            json.dump({"scenarios": all_summaries}, f, indent=2)
        print(f"\nAggregate → {agg}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
