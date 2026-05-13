#!/usr/bin/env python3
"""MuTual benchmark runner for Cognifold.

Evaluates dialogue reasoning via multiple-choice next-response prediction.
Dataset: MuTual (multi-turn dialogue reasoning, 4 choices each).
"""

import json
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from benchmarks.shared.base_runner import (
    BenchmarkRunner,
    answer_mc_with_llm,
    enrich_eval_result,
)
from cognifold.graph.store import ConceptGraph
from cognifold.models.event import Event
from cognifold.query.agent import MemoryQueryAgent


class MuTualRunner(BenchmarkRunner):
    benchmark_name = "mutual"
    default_data_path = Path(__file__).parent / "data" / "mutual_dev.json"

    def load_dataset(self, data_path: Path, limit: Optional[int] = None) -> list[dict[str, Any]]:
        with open(data_path) as f:
            data = json.load(f)
        if limit:
            data = data[:limit]
        print(f"Loaded {len(data)} examples from {data_path}")
        return data

    def build_events(self, example: dict[str, Any], idx: int) -> list[Event]:
        article = example.get("article", [])
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        events = []
        for turn_idx, line in enumerate(article):
            events.append(
                Event(
                    event_id=str(uuid.uuid4()),
                    timestamp=base_time + timedelta(seconds=idx * 1000 + turn_idx),
                    source="mutual-benchmark",
                    event_type="dialogue_turn",
                    title=f"Dialogue {idx + 1}, Turn {turn_idx + 1}",
                    description=line,
                    context={
                        "example_id": idx,
                        "turn_index": turn_idx,
                        "benchmark": self.benchmark_name,
                    },
                )
            )
        return events

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
        article = example.get("article", [])
        dialogue_text = "\n".join(article)
        option_list = example.get("options", [])
        correct_letter = example.get("answers", "A")
        example_id = self.get_example_id(example, idx)

        options: dict[str, str] = {}
        for j, opt in enumerate(option_list):
            options[chr(ord("A") + j)] = opt

        question = f"Given the following dialogue, what is the best next response?\n\n{dialogue_text}"

        _query_start = time.time()
        # Unified cognition: recognition → reconstruction → validation
        cognition = self.cognition_query(
            question=dialogue_text,
            query_agent=query_agent,
            domain=self.benchmark_name,
            query_mode=query_mode,
        )
        context = cognition.context
        result = cognition.query_result

        if use_llm_eval:
            predicted = answer_mc_with_llm(
                question=question,
                context=context,
                options=options,
                profile_templates=profile_templates,
                model=llm_model,
            )
            is_correct = predicted == correct_letter
        else:
            answer_text = options.get(correct_letter, "")
            if answer_text.lower() in context.lower():
                predicted = correct_letter
                is_correct = True
            else:
                predicted = ""
                is_correct = False

        qa_result: dict[str, Any] = {
            "example_id": example_id,
            "question": question,
            "correct": correct_letter,
            "predicted": predicted,
            "is_correct": is_correct,
            "context_length": len(context),
        }
        if not is_correct:
            qa_result["verdict"] = "INCORRECT"
        if enrich_eval_result is not None:
            enrich_eval_result(
                qa_result,
                graph=graph,
                query_result=result,
                retrieval_mode=query_mode,
                query_start_time=_query_start,
            )

        status = "CORRECT" if is_correct else "INCORRECT"
        print(f"    {status} (predicted={predicted}, correct={correct_letter})")
        return qa_result

    def save_results(
        self, all_results: list[dict[str, Any]], output_dir: str, config: dict[str, Any]
    ) -> None:
        from benchmarks.shared.base_runner import save_wrong_cases

        results_path = os.path.join(output_dir, "benchmark_results.json")
        total = len(all_results)
        correct_count = sum(1 for r in all_results if r.get("is_correct"))
        accuracy = (correct_count / total * 100) if total > 0 else 0

        with open(results_path, "w") as f:
            json.dump(
                {
                    "summary": [
                        {
                            "example_id": r.get("example_id"),
                            "graph_nodes": r.get("graph_node_count", 0),
                            "is_correct": r.get("is_correct", False),
                        }
                        for r in all_results
                    ],
                    "qa_details": all_results,
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
        correct_count = sum(1 for r in all_results if r.get("is_correct"))
        accuracy = (correct_count / total * 100) if total > 0 else 0

        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        print(f"  Query Mode: {config.get('query_mode')}")
        print(f"  Disable Concepts: {config.get('disable_concepts')}")
        print(f"  LLM Eval: {config.get('use_llm_eval')}")
        print(f"  Correct: {correct_count}/{total} ({accuracy:.1f}%)")


if __name__ == "__main__":
    MuTualRunner().main()
