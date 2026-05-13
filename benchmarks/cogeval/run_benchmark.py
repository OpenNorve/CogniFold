"""CogEval-Bench: Cognitive Structure Emergence Benchmark for CogniFold.

Evaluates cognitive structure emergence through three tracks:
  Track A: Concept Emergence (precision/recall/F1 against gold concepts)
  Track B: Relationship Topology (chain discovery, modularity, clustering)
  Track C: Temporal Compression (PageRank Gini, compression ratio)

No downstream QA — structural metrics only. QA is measured by existing
benchmarks (ToMi, MuTual, MuSiQue, etc.).

Usage:
    OPENAI_API_KEY=... PYTHONPATH=src python -m benchmarks.cogeval.run_benchmark \
        --scenario software_engineer --scale small
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import networkx as nx

from benchmarks.shared.base_runner import BenchmarkRunner
from cognifold.graph.store import ConceptGraph
from cognifold.models.event import Event
from cognifold.query.agent import MemoryQueryAgent


class CogEvalRunner(BenchmarkRunner):
    benchmark_name = "cogeval"
    default_data_path = Path(__file__).parent / "data" / "generated" / "software_engineer_small.json"

    def __init__(self) -> None:
        super().__init__()
        self._dataset: dict[str, Any] = {}

    def add_extra_args(self, parser: Any) -> None:
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
            help="Which scenario to evaluate",
        )
        parser.add_argument(
            "--scale",
            choices=["small", "medium", "large"],
            default="small",
            help="Dataset scale",
        )

    def _resolve_data_path(self, data_path: Path | None, **kwargs: Any) -> Path:
        if data_path and data_path.exists():
            return data_path
        scenario = kwargs.get("scenario", "software_engineer")
        scale = kwargs.get("scale", "small")
        resolved = Path(__file__).parent / "data" / "generated" / f"{scenario}_{scale}.json"
        if resolved.exists():
            return resolved
        return self.default_data_path

    def run(
        self,
        limit: int | None = None,
        visualize: bool = False,
        disable_concepts: bool = False,
        query_mode: str = "mergefold",
        use_llm_eval: bool = True,
        use_profile: bool = True,
        data_path: Path | None = None,
        embedding: str | None = None,
        model: str | None = None,
        **extra_kwargs: Any,
    ) -> None:
        resolved = self._resolve_data_path(data_path, **extra_kwargs)
        for k in ("scenario", "scale"):
            extra_kwargs.pop(k, None)
        super().run(
            limit=limit,
            visualize=visualize,
            disable_concepts=disable_concepts,
            query_mode=query_mode,
            use_llm_eval=use_llm_eval,
            use_profile=use_profile,
            data_path=resolved,
            embedding=embedding,
            model=model,
            **extra_kwargs,
        )

    def load_dataset(
        self, data_path: Path, limit: Optional[int] = None
    ) -> list[dict[str, Any]]:
        with open(data_path) as f:
            self._dataset = json.load(f)

        print(f"Loaded CogEval scenario: {self._dataset['name']}")
        print(f"  Events: {self._dataset['statistics']['total_events']}")

        return [
            {
                "id": self._dataset["scenario_id"],
                "events_data": self._dataset["events"],
                "gold_graph": self._dataset["gold_graph"],
            }
        ]

    def build_events(self, example: dict[str, Any], idx: int) -> list[Event]:
        events: list[Event] = []
        base_time = datetime(2024, 6, 1, 0, 0, 0)

        for i, ev_data in enumerate(example["events_data"]):
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
                        "scenario": self._dataset.get("scenario_id", ""),
                        "gold_concept": ev_data.get("gold_concept", ""),
                        "event_index": i,
                    },
                )
            )

        return events

    def evaluate_example(  # type: ignore[override]
        self,
        example: dict[str, Any],
        idx: int,
        graph: ConceptGraph,
        query_agent: MemoryQueryAgent,
        query_mode: str,
        use_llm_eval: bool,
        profile_templates: dict[str, str],
        llm_model: str,
    ) -> list[dict[str, Any]]:
        gold_graph = example.get("gold_graph", {})
        n_input_events = len(example.get("events_data", []))

        print("\n  === Track A: Concept Emergence ===")
        concept_result = self._evaluate_concepts(graph, gold_graph)
        print(f"    Precision:   {concept_result.get('precision', 0):.3f}")
        print(f"    Recall:      {concept_result.get('recall', 0):.3f}")
        print(f"    F1 (gold):   {concept_result.get('f1', 0):.3f}")
        print(f"    Purity:      {concept_result.get('purity', 0):.3f}")
        print(f"    Separation:  {concept_result.get('separation', 0):.3f}")
        print(f"    LLM Quality: {concept_result.get('llm_quality', 0):.3f}")
        print(f"    Harmony:     {concept_result.get('harmony', 0):.3f}")

        print("\n  === Track B: Relationship Topology ===")
        topo_result = self._evaluate_topology(graph, gold_graph)
        print(f"    Chain discovery: {topo_result.get('chain_discovery_rate', 0):.3f}")
        print(f"    Modularity:     {topo_result.get('modularity', 0):.3f}")
        print(f"    Clustering:     {topo_result.get('clustering_coefficient', 0):.3f}")
        print(f"    Edge entropy:   {topo_result.get('edge_type_entropy', 0):.3f}")

        print("\n  === Track C: Compression & Dynamics ===")
        comp_result = self._evaluate_compression(graph, n_input_events)
        print(f"    Compression:    {comp_result.get('compression_ratio', 0):.1f}x")
        print(f"    PR Gini:        {comp_result.get('pagerank_gini', 0):.3f}")
        print(f"    Schema Accel:   {comp_result.get('schema_acceleration', 0):.3f}")
        print(f"    Proactivity:    {comp_result.get('proactivity', 0):.3f}")

        sym_count = comp_result.get("symbolic_count", 0)
        sym_str = f", S={sym_count}" if sym_count else ""
        print(
            f"    Nodes: {comp_result.get('node_count', 0)} "
            f"(C={comp_result.get('concept_count', 0)}, "
            f"E={comp_result.get('event_node_count', 0)}, "
            f"I={comp_result.get('intent_count', 0)}{sym_str})"
        )

        return [
            {
                "example_id": example["id"],
                "_structural_metrics": {
                    "concept_emergence": concept_result,
                    "topology": topo_result,
                    "compression": comp_result,
                },
            }
        ]

    def _get_concept_nodes(self, graph: ConceptGraph) -> list[Any]:
        """Get LLM-generated concept nodes (exclude symbolic state nodes)."""
        from cognifold.models.node import NodeType

        all_concepts = graph.get_nodes_by_type(NodeType.CONCEPT)
        return [
            n
            for n in all_concepts
            if not n.id.startswith("sym-") and not n.data.get("symbolic_type")
        ]

    def _evaluate_concepts(
        self, graph: ConceptGraph, gold_graph: dict
    ) -> dict[str, Any]:
        try:
            from benchmarks.cogeval.concept_evaluator import (
                compute_concept_purity_from_graph,
                compute_concept_separation,
                compute_harmony_score,
                evaluate_concept_emergence,
                evaluate_concept_quality_llm,
            )

            concept_nodes = self._get_concept_nodes(graph)

            system_concepts = [
                {
                    "id": n.id,
                    "title": n.data.get("title", ""),
                    "label": n.data.get("title", n.data.get("content", "")),
                }
                for n in concept_nodes
            ]

            gold_concepts = gold_graph.get("concepts", [])
            events = self._dataset.get("events", [])

            result = evaluate_concept_emergence(system_concepts, gold_concepts, events)
            metrics = result.to_dict()

            if system_concepts:
                sys_labels = [c["label"] for c in system_concepts if c.get("label")]
                if len(sys_labels) >= 2:
                    metrics["separation"] = round(
                        compute_concept_separation(sys_labels), 4
                    )

            purity = compute_concept_purity_from_graph(concept_nodes, graph)
            metrics["purity"] = round(purity, 4)

            llm_quality, per_concept = evaluate_concept_quality_llm(
                system_concepts, events
            )
            metrics["llm_quality"] = llm_quality
            metrics["llm_per_concept"] = per_concept

            metrics["harmony"] = compute_harmony_score(
                metrics.get("f1", 0), llm_quality
            )

            return metrics
        except Exception as e:
            print(f"    Concept evaluation error: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def _evaluate_topology(
        self, graph: ConceptGraph, gold_graph: dict
    ) -> dict[str, Any]:
        try:
            from benchmarks.cogeval.topology_evaluator import evaluate_topology

            G = nx.Graph()
            node_content_map: dict[str, str] = {}
            edge_types: list[str] = []

            for node in graph.get_all_nodes():
                G.add_node(node.id)
                content = node.data.get(
                    "description",
                    node.data.get("title", node.data.get("content", "")),
                )
                node_content_map[node.id] = str(content)

            for edge in graph.get_all_edges():
                G.add_edge(edge.source, edge.target)
                edge_types.append(str(getattr(edge, "edge_type", "RELATED_TO")))

            planted_chains = gold_graph.get("planted_chains", [])
            result = evaluate_topology(G, planted_chains, node_content_map, edge_types)
            return result.to_dict()
        except Exception as e:
            print(f"    Topology evaluation error: {e}")
            return {}

    def _evaluate_compression(
        self, graph: ConceptGraph, n_input_events: int
    ) -> dict[str, Any]:
        try:
            from benchmarks.cogeval.compression_evaluator import (
                compute_pagerank_gini,
                compute_schema_acceleration,
            )
            from cognifold.models.node import NodeType

            G = nx.DiGraph()
            for node in graph.get_all_nodes():
                G.add_node(node.id)
            for edge in graph.get_all_edges():
                G.add_edge(edge.source, edge.target)

            if G.number_of_nodes() > 0:
                pr = nx.pagerank(G)
                gini = compute_pagerank_gini(list(pr.values()))
            else:
                gini = 0.0

            all_concept_nodes = graph.get_nodes_by_type(NodeType.CONCEPT)
            event_nodes = graph.get_nodes_by_type(NodeType.EVENT)
            intent_nodes = graph.get_nodes_by_type(NodeType.INTENT)

            llm_concepts = [
                n
                for n in all_concept_nodes
                if not n.id.startswith("sym-") and not n.data.get("symbolic_type")
            ]
            sym_nodes = [
                n
                for n in all_concept_nodes
                if n.id.startswith("sym-") or n.data.get("symbolic_type")
            ]

            n_concepts = len(llm_concepts)
            n_symbolic = len(sym_nodes)
            n_event_nodes = len(event_nodes)
            n_intents = len(intent_nodes)

            compression_ratio = n_input_events / max(n_concepts, 1)
            concept_fraction = n_concepts / max(graph.node_count, 1)
            edge_density = graph.edge_count / max(graph.node_count, 1)

            ops_per_event = self._estimate_ops_per_event(graph, event_nodes)
            accel, _, _ = compute_schema_acceleration(ops_per_event)

            proactivity = self._compute_proactivity(graph, intent_nodes)

            return {
                "pagerank_gini": round(gini, 4),
                "compression_ratio": round(compression_ratio, 2),
                "concept_fraction": round(concept_fraction, 4),
                "edge_density": round(edge_density, 4),
                "schema_acceleration": round(accel, 4),
                "proactivity": round(proactivity, 4),
                "node_count": graph.node_count,
                "edge_count": graph.edge_count,
                "concept_count": n_concepts,
                "symbolic_count": n_symbolic,
                "event_node_count": n_event_nodes,
                "intent_count": n_intents,
                "input_events": n_input_events,
            }
        except Exception as e:
            print(f"    Compression evaluation error: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def _estimate_ops_per_event(
        self, graph: ConceptGraph, event_nodes: list[Any]
    ) -> list[float]:
        """Estimate graph operations per event from edge counts.

        For each event node, count how many edges connect to it.
        More edges = more operations the system performed for that event.
        """
        ops: list[float] = []
        for ev in event_nodes:
            edge_count = sum(
                1
                for e in graph.get_all_edges()
                if e.source == ev.id or e.target == ev.id
            )
            ops.append(float(edge_count))
        return ops

    def _compute_proactivity(
        self, graph: ConceptGraph, intent_nodes: list[Any]
    ) -> float:
        """Compute proactivity score: fraction of intents with grounding evidence.

        An intent is "proactive" if it crystallized from accumulated concept
        patterns (has grounding edges to concepts/events).
        Score = fraction of intents that have >= 2 grounding connections.
        """
        if not intent_nodes:
            return 0.0

        proactive_count = 0
        all_edges = graph.get_all_edges()

        for intent in intent_nodes:
            grounding = sum(
                1
                for e in all_edges
                if (e.source == intent.id or e.target == intent.id)
                and getattr(e, "edge_type", "")
                in (
                    "TRIGGERS",
                    "GROUNDS",
                    "triggers",
                    "grounds",
                    "REINFORCES",
                    "reinforces",
                    "CAUSES",
                    "causes",
                )
            )
            if grounding >= 2:
                proactive_count += 1

        return proactive_count / len(intent_nodes)

    def print_summary(
        self, all_results: list[dict[str, Any]], config: dict[str, Any]
    ) -> None:
        if not all_results:
            print("\nNo results to summarize")
            return

        print(f"\n{'=' * 60}")
        print("COGEVAL-BENCH RESULTS")
        print(f"{'=' * 60}")

        structural = all_results[0].get("_structural_metrics", {})
        if not structural:
            return

        ce = structural.get("concept_emergence", {})
        if ce:
            print("\n  Track A — Concept Emergence:")
            print(f"    Precision:   {ce.get('precision', 0):.3f}")
            print(f"    Recall:      {ce.get('recall', 0):.3f}")
            print(f"    F1 (gold):   {ce.get('f1', 0):.3f}")
            print(f"    Purity:      {ce.get('purity', 0):.3f}")
            print(f"    Separation:  {ce.get('separation', 0):.3f}")
            print(f"    LLM Quality: {ce.get('llm_quality', 0):.3f}")
            print(f"    Harmony:     {ce.get('harmony', 0):.3f}")

        tp = structural.get("topology", {})
        if tp:
            print("\n  Track B — Relationship Topology:")
            print(f"    Chain discovery: {tp.get('chain_discovery_rate', 0):.3f}")
            print(f"    Modularity:      {tp.get('modularity', 0):.3f}")
            print(f"    Clustering:      {tp.get('clustering_coefficient', 0):.3f}")
            print(f"    Edge entropy:    {tp.get('edge_type_entropy', 0):.3f}")

        cp = structural.get("compression", {})
        if cp:
            print("\n  Track C — Temporal Compression:")
            print(f"    Compression:  {cp.get('compression_ratio', 0):.1f}x")
            print(f"    PR Gini:      {cp.get('pagerank_gini', 0):.3f}")
            print(f"    Schema Accel:  {cp.get('schema_acceleration', 0):.3f}")
            print(f"    Proactivity:   {cp.get('proactivity', 0):.3f}")

            s_cnt = cp.get("symbolic_count", 0)
            s_str = f", S={s_cnt}" if s_cnt else ""
            print(
                f"    Nodes: {cp.get('node_count', 0)} "
                f"(C={cp.get('concept_count', 0)}, "
                f"E={cp.get('event_node_count', 0)}, "
                f"I={cp.get('intent_count', 0)}{s_str})"
            )

    def save_results(
        self,
        all_results: list[dict[str, Any]],
        output_dir: str,
        config: dict[str, Any],
    ) -> None:
        scenario = self._dataset.get("scenario_id", "results")
        output_path = Path(output_dir) / f"benchmark_results_{scenario}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        structural = (
            all_results[0].get("_structural_metrics", {}) if all_results else {}
        )

        output = {
            "benchmark": "cogeval",
            "method": "CogniFold",
            "scenario": self._dataset.get("scenario_id", ""),
            "scale": self._dataset.get("scale", ""),
            "run_config": config,
            "structural_metrics": structural,
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str, ensure_ascii=False)

        print(f"\nResults saved to {output_path}")


def main() -> None:
    runner = CogEvalRunner()
    runner.main()


if __name__ == "__main__":
    main()
