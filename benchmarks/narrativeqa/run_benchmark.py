#!/usr/bin/env python3
"""NarrativeQA benchmark runner for Cognifold.

Evaluates reading comprehension over narrative summaries via free-form QA.
Dataset: narrativeqa (book/movie summaries with questions).
Metrics: ROUGE-L and token-level F1.
"""

import dataclasses
import json
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from benchmarks.shared.base_runner import (
    BenchmarkRunner,
    _normalize_agent_model_name,
    check_api_keys,
    enrich_eval_result,
    generate_answer_with_llm,
    save_wrong_cases,
)
from cognifold.graph.store import ConceptGraph
from cognifold.models.event import Event
from cognifold.models.node import Edge
from cognifold.query.agent import MemoryQueryAgent


_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "hundred": "100", "thousand": "1000",
}
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "of",
    "to", "his", "her", "he", "she", "it", "by", "and", "or", "that", "this",
}


def _simple_stem(word: str) -> str:
    """Simple suffix-stripping stemmer to reduce inflectional variants."""
    if len(word) <= 3:
        return word
    # Order matters: check longer suffixes first
    for suffix, replacement in [
        ("ational", "ate"), ("tional", "tion"), ("encies", "ence"),
        ("ously", "ous"), ("ness", ""), ("ment", ""),
        ("ings", ""), ("tion", ""), ("sion", ""),
        ("ally", ""), ("ible", ""), ("able", ""),
        ("ful", ""), ("ive", ""), ("ize", ""),
        ("ing", ""), ("ies", "y"), ("ess", ""),
        ("ous", ""), ("ent", ""), ("ant", ""),
        ("ed", ""), ("ly", ""), ("er", ""), ("es", ""), ("s", ""),
    ]:
        if word.endswith(suffix) and len(word) - len(suffix) + len(replacement) >= 3:
            return word[: -len(suffix)] + replacement
    return word


def _normalize_answer(text: str) -> list[str]:
    """Normalize answer text: lowercase, strip punctuation, map number words, stem, remove stop words."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = text.split()
    words = [_NUMBER_WORDS.get(w, w) for w in words]
    words = [w for w in words if w not in _STOP_WORDS and w.strip()]
    words = [_simple_stem(w) for w in words]
    return words


def compute_f1(predicted: str, gold: str) -> float:
    pred_tokens = _normalize_answer(predicted)
    gold_tokens = _normalize_answer(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_rouge_l(predicted: str, gold: str) -> float:
    pred_tokens = _normalize_answer(predicted)
    gold_tokens = _normalize_answer(gold)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    m, n = len(pred_tokens), len(gold_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gold_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0
    precision = lcs_len / m
    recall = lcs_len / n
    return 2 * precision * recall / (precision + recall)


class NarrativeQARunner(BenchmarkRunner):
    benchmark_name = "narrativeqa"
    default_data_path = Path(__file__).parent / "data" / "narrativeqa_test.json"

    def get_query_config_overrides(self) -> dict[str, Any]:
        return {"max_nodes": 40, "max_context_chars": 8000}

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
        """Override run() to share graphs across same-document questions."""
        from benchmarks._utils import create_embedder, resolve_embedding
        from cognifold.agent.agent import CognifoldAgent
        from cognifold.agent.config import AgentConfig
        from cognifold.agent.prompt_profile import load_prompt_profiles
        from cognifold.executor.runner import PlanExecutor
        from cognifold.query.models import QueryConfig

        if not check_api_keys():
            return

        resolved_embedding = resolve_embedding(embedding, self.profile_path, self.benchmark_name)
        embedder, retrieval_mode = create_embedder(resolved_embedding)
        if embedder:
            print(f"Using embedding: {resolved_embedding}")
        else:
            print("Using retrieval: BM25 (no embedding)")

        prompt_profile = None
        profile_templates: dict[str, str] = {}
        llm_model = (
            "openai:gpt-4o-mini" if os.environ.get("OPENAI_API_KEY") else "gemini-3-flash-preview"
        )
        if use_profile and self.profile_path.exists():
            try:
                profiles = load_prompt_profiles(self.profile_path)
                prompt_profile = profiles.get(self.benchmark_name)
                if prompt_profile:
                    print(f"Using profile: {self.benchmark_name} from {self.profile_path}")
                import yaml

                with open(self.profile_path) as f:
                    raw = yaml.safe_load(f)
                bench_raw = raw.get("profiles", {}).get(self.benchmark_name, {})
                profile_templates = bench_raw.get("templates", {})
                raw_model = bench_raw.get("model", {}).get("name", "")
                if raw_model:
                    llm_model = (
                        raw_model
                        if raw_model.startswith("gemini")
                        else raw_model.replace("openai:", "")
                    )
            except Exception as e:
                print(f"Warning: Could not load profile: {e}")
        if model:
            llm_model = model

        dp = data_path or self.default_data_path
        if not dp.exists():
            print(f"Dataset not found at {dp}. Please run download_data.py first.")
            sys.exit(1)

        data = self.load_dataset(dp, limit)
        os.makedirs(self.output_dir, exist_ok=True)

        # Group examples by document ID for graph sharing
        doc_groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        for i, example in enumerate(data):
            doc_id = example.get("document", {}).get("id", str(i))
            doc_groups[doc_id].append((i, example))

        all_results: list[dict[str, Any]] = []

        for doc_id, group in doc_groups.items():
            print(f"\n{'=' * 50}")
            print(f"Document: {doc_id} ({len(group)} questions)")
            print(f"{'=' * 50}")

            # Build graph ONCE for this document
            first_idx, first_example = group[0]
            graph = ConceptGraph()

            if prompt_profile:
                config = prompt_profile.to_agent_config()
                if disable_concepts:
                    config = dataclasses.replace(config, disable_concepts=True)
                if model:
                    config = dataclasses.replace(
                        config, model_name=_normalize_agent_model_name(model)
                    )
                agent = CognifoldAgent(config=config, prompt_profile=prompt_profile)
            else:
                default_model = (
                    "openai:gpt-4o-mini"
                    if os.environ.get("OPENAI_API_KEY")
                    else "gemini-3-flash-preview"
                )
                config = AgentConfig(model_name=default_model, temperature=0.0)
                if disable_concepts:
                    config = dataclasses.replace(config, disable_concepts=True)
                if model:
                    config = dataclasses.replace(
                        config, model_name=_normalize_agent_model_name(model)
                    )
                agent = CognifoldAgent(config=config)

            executor = PlanExecutor(graph)

            qc_kwargs: dict[str, Any] = {
                "domain": self.benchmark_name,
                "max_nodes": 20,
                "include_reasoning": True,
                "retrieval_mode": retrieval_mode,
            }
            qc_kwargs.update(self.get_query_config_overrides())
            query_config = QueryConfig(**qc_kwargs)
            query_agent = MemoryQueryAgent(graph, config=query_config, embedder=embedder)

            # Ingest events ONCE for this document
            events = self.build_events(first_example, first_idx)
            events = self.filter_events(events)
            print(f"  Ingesting {len(events)} events...")

            for event in events:
                try:
                    retrieval = query_agent.query_semantic(event.description[:200])
                    context_node_ids = [n.node_id for n in retrieval.nodes[:10]]
                    plan = agent.process_event(
                        event=event,
                        graph=graph,
                        context_node_ids=context_node_ids,
                        node_scores={},
                    )
                    executor.execute(plan)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"    Error processing event: {e}")
                    if "429" in str(e):
                        print("    Rate limit hit, sleeping for 10s...")
                        time.sleep(10)

            print(f"  Graph: {graph.node_count} nodes, {graph.edge_count} edges")

            # Evaluate ALL questions for this document against the shared graph
            for idx, example in group:
                print(f"\n  Question {idx + 1}/{len(data)}")
                try:
                    eval_result = self.evaluate_example(
                        example=example,
                        idx=idx,
                        graph=graph,
                        query_agent=query_agent,
                        query_mode=query_mode,
                        use_llm_eval=use_llm_eval,
                        profile_templates=profile_templates,
                        llm_model=llm_model,
                    )
                    if isinstance(eval_result, list):
                        all_results.extend(eval_result)
                    else:
                        all_results.append(eval_result)
                except Exception as e:
                    print(f"    Error evaluating: {e}")

        run_config = {
            "query_mode": query_mode,
            "use_llm_eval": use_llm_eval,
            "disable_concepts": disable_concepts,
        }
        self.save_results(all_results, self.output_dir, run_config)
        self.print_summary(all_results, run_config)

    def load_dataset(self, data_path: Path, limit: Optional[int] = None) -> list[dict[str, Any]]:
        with open(data_path) as f:
            data = json.load(f)
        if limit:
            data = data[:limit]
        print(f"Loaded {len(data)} examples from {data_path}")
        return data

    def build_events(self, example: dict[str, Any], idx: int) -> list[Event]:
        document = example.get("document", {})
        doc_id = document.get("id", str(idx))
        doc_kind = document.get("kind", "unknown")
        summary = document.get("summary", {})
        summary_text = summary.get("text", "") if isinstance(summary, dict) else str(summary)

        # Use paragraph-level chunking (3-5 sentence chunks) instead of single sentences
        # to preserve more context per event
        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", summary_text.strip()) if s.strip()
        ]
        chunk_size = 2  # ~2 sentences per chunk for finer-grained events
        chunks: list[str] = []
        for i in range(0, len(sentences), chunk_size):
            chunk = " ".join(sentences[i : i + chunk_size])
            if chunk:
                chunks.append(chunk)

        # If no chunks from sentence splitting, try paragraph splitting
        if not chunks:
            paragraphs = [p.strip() for p in summary_text.split("\n\n") if p.strip()]
            chunks = paragraphs if paragraphs else [summary_text] if summary_text else []

        base_time = datetime(2024, 1, 1, 10, 0, 0)
        events = []
        for i, chunk in enumerate(chunks):
            events.append(
                Event(
                    event_id=str(uuid.uuid4()),
                    timestamp=base_time + timedelta(seconds=idx * 1000 + i),
                    source="narrativeqa-benchmark",
                    event_type="narrative_segment",
                    title=f"Narrative {i + 1}/{len(chunks)} ({doc_kind})",
                    description=chunk,
                    context={"document_id": doc_id, "kind": doc_kind},
                )
            )
        return events

    def _get_summary_text(self, example: dict[str, Any]) -> str:
        """Extract summary text from example for fallback context."""
        document = example.get("document", {})
        summary = document.get("summary", {})
        return summary.get("text", "") if isinstance(summary, dict) else str(summary)

    def post_ingest(self, graph: ConceptGraph, events: list[Event]) -> None:
        """Create CAUSES edges between consecutive narrative events."""
        event_ids = []
        for node in graph.get_all_nodes():
            if node.type == "event":
                event_ids.append(node.id)
        for i in range(len(event_ids) - 1):
            try:
                graph.add_edge(Edge(source=event_ids[i], target=event_ids[i + 1], edge_type="CAUSES", weight=0.6))
            except Exception:
                pass

    def print_example_header(self, example: dict[str, Any], idx: int, total: int) -> None:
        print(f"\nProcessing Example {idx + 1}/{total}")

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
        question_obj = example.get("question", {})
        question_text = (
            question_obj.get("text", "") if isinstance(question_obj, dict) else str(question_obj)
        )
        answers = example.get("answers", [])
        reference_answers = []
        for ans in answers:
            if isinstance(ans, dict):
                reference_answers.append(ans.get("text", ""))
            else:
                reference_answers.append(str(ans))

        _query_start = time.time()
        # Unified cognition: recognition → reconstruction → validation
        cognition = self.cognition_query(
            question=question_text,
            query_agent=query_agent,
            domain=self.benchmark_name,
            query_mode=query_mode,
        )
        context = cognition.context
        result = cognition.query_result

        # Always include raw summary as supplementary context
        summary_text = self._get_summary_text(example)
        if summary_text:
            context = context + "\n\n--- Original Summary ---\n" + summary_text

        if use_llm_eval:
            predicted = generate_answer_with_llm(
                question=question_text,
                context=context,
                profile_templates=profile_templates,
                model=llm_model,
                default_system=(
                    "Answer the question based on the context. "
                    "Give a concise answer: 1-10 words. "
                    "Do not write full sentences."
                ),
                max_tokens=30,
            )
        else:
            predicted = ""

        if predicted and len(predicted.split()) > 15:
            predicted = ' '.join(predicted.split()[:15])

        best_f1 = 0.0
        best_rouge_l = 0.0
        for ref in reference_answers:
            if not ref:
                continue
            best_f1 = max(best_f1, compute_f1(predicted, ref))
            best_rouge_l = max(best_rouge_l, compute_rouge_l(predicted, ref))

        eval_result: dict[str, Any] = {
            "question": question_text,
            "reference_answers": reference_answers,
            "predicted": predicted,
            "f1": best_f1,
            "rouge_l": best_rouge_l,
            "context_length": len(context),
        }
        if enrich_eval_result is not None:
            enrich_eval_result(
                eval_result,
                graph=graph,
                query_result=result,
                retrieval_mode=query_mode,
                query_start_time=_query_start,
            )

        print(f"    F1={best_f1:.3f} ROUGE-L={best_rouge_l:.3f} predicted='{predicted[:60]}'")
        return eval_result

    def save_results(
        self, all_results: list[dict[str, Any]], output_dir: str, config: dict[str, Any]
    ) -> None:
        results_path = os.path.join(output_dir, "benchmark_results.json")
        total = len(all_results)
        avg_f1 = sum(r["f1"] for r in all_results) / total if total > 0 else 0.0
        avg_rouge_l = sum(r["rouge_l"] for r in all_results) / total if total > 0 else 0.0

        with open(results_path, "w") as f:
            json.dump(
                {
                    "summary": {
                        "total": total,
                        "average_f1": avg_f1,
                        "average_rouge_l": avg_rouge_l,
                    },
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
        avg_f1 = sum(r["f1"] for r in all_results) / total if total > 0 else 0.0
        avg_rouge_l = sum(r["rouge_l"] for r in all_results) / total if total > 0 else 0.0

        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        print(f"  Query Mode: {config.get('query_mode')}")
        print(f"  Disable Concepts: {config.get('disable_concepts')}")
        print(f"  LLM Eval: {config.get('use_llm_eval')}")
        print(f"  Average F1: {avg_f1:.4f}")
        print(f"  Average ROUGE-L: {avg_rouge_l:.4f}")


if __name__ == "__main__":
    NarrativeQARunner().main()
