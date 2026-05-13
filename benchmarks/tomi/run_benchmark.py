#!/usr/bin/env python3
"""ToMi (Theory of Mind Inventory) benchmark runner for Cognifold.

Evaluates theory-of-mind reasoning: given stories about characters moving
objects and entering/exiting rooms, answer questions about beliefs and locations.

Dataset: facebookresearch/ToMi (parsed from raw text into JSON).
Metrics: Exact match, contains match, LLM verdict, per question-type breakdown.
"""

import json
import os
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from benchmarks.shared.base_runner import (
    BenchmarkRunner,
    enrich_eval_result,
    evaluate_with_llm,
    generate_answer_with_llm,
    save_wrong_cases,
)
from cognifold.graph.store import ConceptGraph
from cognifold.models.event import Event
from cognifold.query.agent import MemoryQueryAgent
from cognifold.symbolic.belief_tracker import SymbolicBeliefTracker


class ToMiRunner(BenchmarkRunner):
    benchmark_name = "tomi"
    default_data_path = Path(__file__).parent / "data" / "tomi_test.json"

    def __init__(self) -> None:
        super().__init__()
        self._belief_tracker: SymbolicBeliefTracker | None = None

    def load_dataset(self, data_path: Path, limit: Optional[int] = None) -> list[dict[str, Any]]:
        with open(data_path) as f:
            data = json.load(f)
        if limit:
            data = data[:limit]
        print(f"Loaded {len(data)} examples from {data_path}")
        return data

    def build_events(self, example: dict[str, Any], idx: int) -> list[Event]:
        story = example.get("story", [])
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        story_id = example.get("id", idx)
        events = []
        for i, sentence in enumerate(story):
            events.append(
                Event(
                    event_id=str(uuid.uuid4()),
                    timestamp=base_time + timedelta(seconds=idx * 100 + i),
                    source="tomi-benchmark",
                    event_type="agent_action",
                    title=f"Action {i + 1}",
                    description=sentence,
                    context={
                        "story_id": story_id,
                        "sentence_index": i,
                        "benchmark": self.benchmark_name,
                    },
                )
            )
        return events

    def post_ingest(self, graph: ConceptGraph, events: list[Event]) -> None:
        """Complement the general symbolic tracker with ToMi regex fallback.

        The general SymbolicStateTracker (in base_runner) processes LLM-extracted
        symbolic_actions. This hook runs the regex-based SymbolicBeliefTracker
        as a fallback to ensure ToMi actions are always captured, even if the LLM
        doesn't emit perfect symbolic_actions yet.
        """
        # Run regex-based tracker as fallback/complement
        tracker = SymbolicBeliefTracker()
        for event in events:
            tracker.process_event(event)
        self._belief_tracker = tracker

        if not tracker.state.world_state:
            return

        # Merge regex tracker results into the general tracker if available
        if hasattr(self, "_sym_tracker") and self._sym_tracker is not None:
            gen = self._sym_tracker
            # Port regex-extracted state into general tracker
            for entity, loc in tracker.state.world_state.items():
                gen.state.entity_attributes.setdefault(entity, {})["location"] = loc
            for entity, loc in tracker.state.initial_locations.items():
                gen.state.initial_attributes.setdefault(entity, {}).setdefault("location", loc)
            for agent, beliefs in tracker.state.agent_beliefs.items():
                if agent not in gen.state.agent_beliefs:
                    gen.state.agent_beliefs[agent] = {}
                for entity, loc in beliefs.items():
                    gen.state.agent_beliefs[agent].setdefault(entity, {})["location"] = loc
            for agent, loc in tracker.state.agent_locations.items():
                gen.state.agent_locations[agent] = loc
            for loc, agents in tracker.state.observers.items():
                gen.state.observers[loc] = agents
            # Merge known agents/entities from regex tracker
            for agent in tracker.state.agent_beliefs:
                gen.known_agents.add(agent)
            for entity in tracker.state.world_state:
                gen.known_entities.add(entity)

            # Re-inject with merged state
            nodes_injected, corrections = gen.inject_into_graph(graph)
            print(
                f"    ToMi regex fallback merged: {len(tracker.state.world_state)} entities,"
                f" {len(tracker.state.agent_beliefs)} agents."
                f" After merge: {nodes_injected} nodes injected,"
                f" {corrections} corrections"
            )
        else:
            # Fallback: use old-style direct injection
            tracker_nodes = tracker.generate_belief_nodes()
            from cognifold.models.node import Edge, Node, NodeType

            for bdata in tracker_nodes:
                node_id = str(uuid.uuid4())
                node = Node(
                    id=node_id,
                    type=NodeType.CONCEPT,
                    data={
                        "title": str(bdata["title"]),
                        "description": str(bdata["description"]),
                        "symbolic_type": str(bdata.get("symbolic_type", "")),
                    },
                    created_at=datetime.now(),
                )
                graph.add_node(node)
                # Connect to related events
                for ev in graph.get_all_nodes():
                    if ev.type != "event":
                        continue
                    desc = (ev.data.get("description", "") or "").lower()
                    entity = str(bdata.get("entity", ""))
                    if entity and entity in desc:
                        graph.add_edge(
                            Edge(
                                source=ev.id,
                                target=node_id,
                                edge_type="GROUNDS",
                                weight=0.9,
                            )
                        )
            print(
                f"    ToMi regex fallback: {len(tracker_nodes)} belief nodes injected"
                f" ({len(tracker.state.world_state)} entities,"
                f" {len(tracker.state.agent_beliefs)} agents)"
            )

    def print_example_header(self, example: dict[str, Any], idx: int, total: int) -> None:
        example_id = self.get_example_id(example, idx)
        question_type = example.get("question_type", "unknown")
        print(f"\nProcessing Example {idx + 1}/{total} (ID: {example_id})")
        print(f"  Question type: {question_type}, Story: {len(example.get('story', []))} sentences")

    def evaluate_example(
        self,
        example: dict[str, Any],
        idx: int,
        graph: ConceptGraph,
        query_agent: MemoryQueryAgent,
        query_mode: str,
        use_llm_eval: bool,
        profile_templates: dict[str, str],
        llm_model: str,
    ) -> dict[str, Any]:
        example_id = self.get_example_id(example, idx)
        question_type = example.get("question_type", "unknown")
        question = example.get("question", "")
        target = example.get("answer", "")
        story = example.get("story", [])

        # Build query: move events + entity names + question
        move_events = [s for s in story if "moved" in s.lower()]
        entities = re.findall(r"\b[A-Z][a-z]+\b", question)
        entity_context = " ".join(entities) if entities else ""
        query_text = (
            f"{' '.join(move_events)} {entity_context} {question}"
            if move_events
            else f"{' '.join(story[-3:])} {question}"
        )
        _query_start = time.time()

        # Unified cognition: recognition → reconstruction → validation
        cognition = self.cognition_query(
            question=query_text,
            query_agent=query_agent,
            domain=self.benchmark_name,
            query_mode=query_mode,
        )
        context = cognition.context
        result = cognition.query_result

        # If symbolic answered directly, use it
        if cognition.direct_answer:
            print(f"    [symbolic direct] {cognition.direct_answer}")
            eval_result: dict[str, Any] = {
                "example_id": example_id,
                "question_type": question_type,
                "question": question,
                "target": target,
                "generated_answer": cognition.direct_answer,
                "verdict": "CORRECT" if target.lower().strip() == cognition.direct_answer.lower().strip() else "INCORRECT",
                "explanation": "Symbolic direct answer",
                "exact_match": target.lower().strip() == cognition.direct_answer.lower().strip(),
                "contains_match": target.lower().strip() in cognition.direct_answer.lower().strip(),
                "story_length": len(story),
                "context_length": 0,
            }
            print(f"    {eval_result['verdict']} (target={target}, got={cognition.direct_answer})")
            return eval_result

        # Fallback: if CognitionRouter didn't inject beliefs but regex tracker has them
        if (
            "VERIFIED FACTS" not in context
            and self._belief_tracker is not None
        ):
            belief_ctx = self._belief_tracker.get_belief_context()
            if belief_ctx.strip():
                context = (
                    "=== VERIFIED FACTS (deterministic, use as ground truth) ===\n"
                    + belief_ctx
                    + "\n\n"
                    + context
                )

        if use_llm_eval:
            answer = generate_answer_with_llm(
                question=question,
                context=context,
                profile_templates=profile_templates,
                model=llm_model,
                default_system=(
                    "Answer theory-of-mind questions about character beliefs and object locations. "
                    "Reply with ONLY the answer (a single word or short phrase)."
                ),
                default_user="Question: {question}\n\nContext:\n{context}\n\nAnswer with a single word or short phrase:",
                max_tokens=20,
            )
        else:
            answer = context.split("\n")[0].strip() if context else "unknown"

        target_normalized = target.lower().strip()
        answer_normalized = answer.lower().strip()
        exact_match = target_normalized == answer_normalized
        contains_match = target_normalized in answer_normalized

        if use_llm_eval and not exact_match:
            verdict, explanation = evaluate_with_llm(
                question=question,
                expected=target,
                generated=answer,
                context=context,
                profile_templates=profile_templates,
                model=llm_model,
            )
        else:
            verdict = "CORRECT" if exact_match else ("PARTIAL" if contains_match else "INCORRECT")
            explanation = ""

        eval_result: dict[str, Any] = {
            "example_id": example_id,
            "question_type": question_type,
            "question": question,
            "target": target,
            "generated_answer": answer,
            "verdict": verdict,
            "explanation": explanation,
            "exact_match": exact_match,
            "contains_match": contains_match,
            "story_length": len(story),
            "context_length": len(context),
        }
        if enrich_eval_result is not None:
            enrich_eval_result(
                eval_result, graph=graph, query_result=result, query_start_time=_query_start
            )

        print(f"    {verdict} (target={target}, got={answer})")
        return eval_result

    def save_results(
        self, all_results: list[dict[str, Any]], output_dir: str, config: dict[str, Any]
    ) -> None:
        results_path = os.path.join(output_dir, "benchmark_results.json")
        total = len(all_results)
        if total == 0:
            print("No results to report.")
            return

        exact_match_count = sum(1 for r in all_results if r.get("exact_match"))
        contains_match_count = sum(1 for r in all_results if r.get("contains_match"))
        correct = sum(1 for r in all_results if r.get("verdict") == "CORRECT")
        partial = sum(1 for r in all_results if r.get("verdict") == "PARTIAL")

        per_type: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "exact_match": 0, "correct": 0}
        )
        for r in all_results:
            qt = r.get("question_type", "unknown")
            per_type[qt]["total"] += 1
            if r.get("exact_match"):
                per_type[qt]["exact_match"] += 1
            if r.get("verdict") == "CORRECT":
                per_type[qt]["correct"] += 1

        with open(results_path, "w") as f:
            json.dump(
                {
                    "summary": {
                        "total": total,
                        "exact_match": exact_match_count,
                        "contains_match": contains_match_count,
                        "correct": correct,
                        "partial": partial,
                        "exact_match_rate": exact_match_count / total,
                        "verdict_rate": correct / total,
                    },
                    "per_question_type": dict(per_type),
                    "results": all_results,
                    "config": config,
                },
                f,
                indent=2,
            )
        print(f"\nDetailed results saved to {results_path}")

        if save_wrong_cases is not None:
            save_wrong_cases(all_results, output_dir)

    def print_summary(self, all_results: list[dict[str, Any]], config: dict[str, Any]) -> None:
        total = len(all_results)
        if total == 0:
            print("No results to report.")
            return

        exact_match_count = sum(1 for r in all_results if r.get("exact_match"))
        contains_match_count = sum(1 for r in all_results if r.get("contains_match"))
        correct = sum(1 for r in all_results if r.get("verdict") == "CORRECT")
        partial = sum(1 for r in all_results if r.get("verdict") == "PARTIAL")

        per_type: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "exact_match": 0, "correct": 0}
        )
        for r in all_results:
            qt = r.get("question_type", "unknown")
            per_type[qt]["total"] += 1
            if r.get("exact_match"):
                per_type[qt]["exact_match"] += 1
            if r.get("verdict") == "CORRECT":
                per_type[qt]["correct"] += 1

        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        print(f"  Query Mode: {config.get('query_mode')}")
        print(f"  Disable Concepts: {config.get('disable_concepts')}")
        print(f"  LLM Eval: {config.get('use_llm_eval')}")
        print(
            f"  Exact Match: {exact_match_count}/{total} ({exact_match_count / total * 100:.1f}%)"
        )
        print(
            f"  Contains Match: {contains_match_count}/{total} ({contains_match_count / total * 100:.1f}%)"
        )
        print(f"  Correct (verdict): {correct}/{total} ({correct / total * 100:.1f}%)")
        print(f"  Partial: {partial}/{total} ({partial / total * 100:.1f}%)")
        print()
        print("  Per question-type breakdown:")
        for qt in sorted(per_type.keys()):
            stats = per_type[qt]
            em_pct = stats["exact_match"] / stats["total"] * 100 if stats["total"] > 0 else 0
            v_pct = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(
                f"    {qt}: {stats['exact_match']}/{stats['total']} exact ({em_pct:.1f}%), "
                f"{stats['correct']}/{stats['total']} verdict ({v_pct:.1f}%)"
            )


if __name__ == "__main__":
    ToMiRunner().main()
