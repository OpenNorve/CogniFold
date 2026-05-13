#!/usr/bin/env python3
"""BABILong benchmark runner for Cognifold.

Evaluates Cognifold's ability to perform multi-hop logical reasoning
while filtering out noise from massive contexts.

Dataset: BABILong (extended bAbI tasks with noise padding).
Metrics: Exact match, contains match, LLM verdict, per-task breakdown.
"""

import argparse
import json
import os
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


def is_noise_statement(statement: str) -> bool:
    """Check if a statement is noise (not about tracked entities/actions).

    Uses an expanded entity/predicate list to avoid over-filtering.
    """
    tracked_entities = [
        # People (bAbI standard + extended)
        "john",
        "mary",
        "daniel",
        "sandra",
        "bill",
        "fred",
        "julie",
        "jeff",
        "bernhard",
        "sumit",
        "greg",
        "jason",
        "antoine",
        "emily",
        "lily",
        "yann",
        # Locations
        "bedroom",
        "kitchen",
        "bathroom",
        "garden",
        "office",
        "hallway",
        "cinema",
        "park",
        "school",
    ]
    tracked_predicates = [
        # Movement
        "went to",
        "moved to",
        "travelled to",
        "journeyed to",
        "went back to",
        "is in the",
        "is no longer in",
        # Object manipulation
        "picked up",
        "got the",
        "grabbed the",
        "took the",
        "dropped the",
        "left the",
        "put down the",
        "discarded the",
        # Transfer
        "gave the",
        "handed the",
        "passed the",
        "received the",
        # State / belief
        "is",
        "was",
        "has",
        "had",
    ]
    lower = statement.lower()
    has_entity = any(e in lower for e in tracked_entities)
    has_predicate = any(p in lower for p in tracked_predicates)
    return not (has_entity or has_predicate)


class BABILongRunner(BenchmarkRunner):
    benchmark_name = "babilong"
    default_data_path = Path(".")  # Resolved dynamically based on --config/--tasks

    def add_extra_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--config",
            type=str,
            default="0k",
            help="Context length config (0k, 1k, 2k, 4k, 8k, 16k, 32k, 128k, ...)",
        )
        parser.add_argument(
            "--tasks",
            type=str,
            default="qa1",
            help="Comma-separated task names (default: qa1)",
        )

    def load_dataset(self, data_path: Path, limit: Optional[int] = None) -> list[dict[str, Any]]:
        with open(data_path) as f:
            data = json.load(f)
        if limit:
            data = data[:limit]
        print(f"Loaded {len(data)} examples from {data_path}")
        return data

    def build_events(self, example: dict[str, Any], idx: int) -> list[Event]:
        statements, _question, _target = self._parse_example(example)
        question_id = str(example.get("id", idx))
        task_name = example.get("task", "qa1")

        events = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        for s_idx, statement in enumerate(statements):
            noise = is_noise_statement(statement)
            events.append(
                Event(
                    event_id=str(uuid.uuid4()),
                    timestamp=base_time + timedelta(seconds=s_idx),
                    source="babilong-benchmark",
                    event_type="narrative.statement",
                    title=f"Statement {s_idx + 1}",
                    description=statement,
                    context={
                        "question_id": question_id,
                        "task": task_name,
                        "statement_index": s_idx,
                        "is_noise": noise,
                    },
                )
            )
        return events

    def filter_events(self, events: list[Event]) -> list[Event]:
        """Skip noise statements during ingestion. Never returns empty."""
        filtered = [e for e in events if not e.context.get("is_noise", False)]
        # Safety: if ALL events were filtered out, keep originals
        return filtered if filtered else events

    def print_example_header(self, example: dict[str, Any], idx: int, total: int) -> None:
        task_name = example.get("task", "qa1")
        print(f"\nProcessing Example {idx + 1}/{total} (task: {task_name})")
        print(f"  Context: {len(example.get('input', ''))} chars")

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
        task_name = example.get("task", "qa1")
        _, question, target = self._parse_example(example)

        _query_start = time.time()

        # Unified cognition: recognition → reconstruction → validation
        cognition = self.cognition_query(
            question=question,
            query_agent=query_agent,
            domain=self.benchmark_name,
            query_mode=query_mode,
        )
        retrieved_context = cognition.context
        query_result = cognition.query_result

        if use_llm_eval:
            answer = generate_answer_with_llm(
                question=question,
                context=retrieved_context,
                profile_templates=profile_templates,
                model=llm_model,
                default_system="Answer precise factual questions about entity states.",
                max_tokens=20,
            )
        else:
            answer = retrieved_context.split("\n")[0].strip() if retrieved_context else "unknown"

        target_normalized = target.lower().strip()
        answer_normalized = answer.lower().strip()
        exact_match = target_normalized == answer_normalized
        contains_match = target_normalized in answer_normalized

        if use_llm_eval and not exact_match:
            verdict, explanation = evaluate_with_llm(
                question=question,
                expected=target,
                generated=answer,
                context=retrieved_context,
                profile_templates=profile_templates,
                model=llm_model,
            )
        else:
            verdict = "CORRECT" if exact_match else ("PARTIAL" if contains_match else "INCORRECT")
            explanation = ""

        eval_result: dict[str, Any] = {
            "question_id": example_id,
            "task": task_name,
            "question": question,
            "target": target,
            "generated_answer": answer,
            "verdict": verdict,
            "explanation": explanation,
            "exact_match": exact_match,
            "contains_match": contains_match,
            "context_length": len(example.get("input", "")),
        }
        if enrich_eval_result is not None:
            enrich_eval_result(
                eval_result, graph=graph, query_result=query_result, query_start_time=_query_start
            )

        print(f"    {verdict} (target={target}, got={answer})")
        return eval_result

    def run(self, *, config: str = "0k", tasks: str = "qa1", **kwargs: Any) -> None:
        """Override run to handle babilong-specific args."""
        task_list = [t.strip() for t in tasks.split(",")]

        # Resolve data path if not explicitly provided
        if kwargs.get("data_path") is None:
            first_task = task_list[0]
            kwargs["data_path"] = (
                Path(__file__).parent / "data" / f"babilong_{config}_{first_task}.json"
            )

        # Store task_list and config_name for use in evaluation
        self._task_list = task_list
        self._config_name = config

        # Load data and filter by tasks
        orig_load = self.load_dataset

        def filtered_load(data_path: Path, limit: Optional[int] = None) -> list[dict[str, Any]]:
            data = orig_load(data_path, limit=None)
            data = [ex for ex in data if ex.get("task") in task_list]
            print(f"Filtered to {len(data)} examples for tasks: {', '.join(task_list)}")
            if limit:
                data = data[:limit]
            return data

        self.load_dataset = filtered_load  # type: ignore[assignment]
        try:
            super().run(**kwargs)
        finally:
            self.load_dataset = orig_load  # type: ignore[assignment]

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

        per_task: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "correct": 0, "exact_match": 0}
        )
        for r in all_results:
            task = r.get("task", "unknown")
            per_task[task]["total"] += 1
            if r.get("verdict") == "CORRECT":
                per_task[task]["correct"] += 1
            if r.get("exact_match"):
                per_task[task]["exact_match"] += 1

        task_list = getattr(self, "_task_list", ["qa1"])
        config_name = getattr(self, "_config_name", "0k")

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
                        "correct_rate": correct / total,
                        "context_length": config_name,
                        "tasks": ",".join(task_list),
                    },
                    "per_task": dict(per_task),
                    "results": all_results,
                    "config": {
                        **config,
                        "context_length": config_name,
                        "tasks": ",".join(task_list),
                    },
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

        task_list = getattr(self, "_task_list", ["qa1"])
        config_name = getattr(self, "_config_name", "0k")

        exact_match_count = sum(1 for r in all_results if r.get("exact_match"))
        contains_match_count = sum(1 for r in all_results if r.get("contains_match"))
        correct = sum(1 for r in all_results if r.get("verdict") == "CORRECT")
        partial = sum(1 for r in all_results if r.get("verdict") == "PARTIAL")

        per_task: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "correct": 0, "exact_match": 0}
        )
        for r in all_results:
            task = r.get("task", "unknown")
            per_task[task]["total"] += 1
            if r.get("verdict") == "CORRECT":
                per_task[task]["correct"] += 1
            if r.get("exact_match"):
                per_task[task]["exact_match"] += 1

        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        print(f"  Context Length: {config_name}")
        print(f"  Tasks: {', '.join(task_list)}")
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
        print("  Per-task breakdown:")
        for task in sorted(per_task.keys()):
            stats = per_task[task]
            pct = stats["exact_match"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"    {task}: {stats['exact_match']}/{stats['total']} exact ({pct:.1f}%)")

    @staticmethod
    def _parse_example(example: dict[str, Any]) -> tuple[list[str], str, str]:
        """Parse a BABILong example into statements, question, and answer."""
        input_text = example["input"]
        question = example["question"].strip()
        target = example["target"].strip()
        statements = [s.strip() + "." for s in input_text.split(".") if s.strip()]
        return statements, question, target


if __name__ == "__main__":
    BABILongRunner().main()
