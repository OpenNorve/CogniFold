#!/usr/bin/env python3
"""Compare classic vs fast ingestion mode on benchmarks.

Runs both MuTual and SocialIQA benchmarks with classic (per-event agent)
and fast (batched LayeredPipeline) ingestion, comparing latency and accuracy.

Usage:
    # Set API key first
    export OPENAI_API_KEY=your-key  # or GOOGLE_API_KEY

    # Run comparison (20 samples each)
    PYTHONPATH=src python3 benchmarks/compare_fast.py --limit 20

    # Run only one benchmark
    PYTHONPATH=src python3 benchmarks/compare_fast.py --limit 20 --benchmark mutual
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

# Add project paths
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_root, "src"))
sys.path.insert(0, _project_root)

from cognifold.agent.agent import CognifoldAgent
from cognifold.agent.batch import BatchAgentProcessor
from cognifold.agent.config import AgentConfig
from cognifold.agent.prompt_profile import load_prompt_profiles
from cognifold.executor.runner import PlanExecutor
from cognifold.graph.store import ConceptGraph
from cognifold.models.event import Event
from cognifold.models.node import Node, NodeType
from cognifold.query.agent import MemoryQueryAgent
from cognifold.query.models import QueryConfig

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(_project_root) / "configs"
OUTPUT_DIR = Path(_project_root) / "benchmarks" / "compare_output"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mutual_data(limit: Optional[int] = None) -> list[dict[str, Any]]:
    """Load MuTual dataset."""
    path = Path(_project_root) / "benchmarks" / "mutual" / "data" / "mutual_dev.json"
    if not path.exists():
        print(f"MuTual data not found at {path}")
        print("Run: PYTHONPATH=src python3 benchmarks/mutual/download_data.py --split dev")
        return []
    with open(path) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    return data


def load_socialiqa_data(limit: Optional[int] = None) -> list[dict[str, Any]]:
    """Load SocialIQA dataset."""
    path = Path(_project_root) / "benchmarks" / "socialiqa" / "data" / "socialiqa_validation.json"
    if not path.exists():
        print(f"SocialIQA data not found at {path}")
        print("Run: PYTHONPATH=src python3 benchmarks/socialiqa/download_data.py")
        return []
    with open(path) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    return data


# ---------------------------------------------------------------------------
# Event conversion (same as original runners)
# ---------------------------------------------------------------------------

def mutual_to_events(example: dict[str, Any], idx: int) -> list[Event]:
    """Convert MuTual dialogue to events."""
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
                context={"example_id": idx, "turn_index": turn_idx},
            )
        )
    return events


def socialiqa_to_events(example: dict[str, Any], idx: int) -> list[Event]:
    """Convert SocialIQA example to events."""
    context_text = example.get("context", "")
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    return [
        Event(
            event_id=str(uuid.uuid4()),
            timestamp=base_time + timedelta(seconds=idx),
            source="socialiqa-benchmark",
            event_type="social_situation",
            title=f"Situation {idx + 1}",
            description=context_text,
            context={"example_id": idx},
        )
    ]


# ---------------------------------------------------------------------------
# LLM evaluation (shared)
# ---------------------------------------------------------------------------

def answer_mc_with_llm(
    question: str,
    context: str,
    options: dict[str, str],
    model: str = "gpt-4o-mini",
) -> str:
    """Use LLM to answer a multiple-choice question given context."""
    option_text = "\n".join(f"{k}: {v}" for k, v in sorted(options.items()))
    prompt = f"""Given the following context from a knowledge graph, answer the multiple-choice question.

Context:
{context}

Question: {question}

Options:
{option_text}

Answer with ONLY the letter (A, B, C, or D). Nothing else."""

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        answer = (response.choices[0].message.content or "").strip().upper()
        # Extract just the letter
        for ch in answer:
            if ch in options:
                return ch
        return answer[:1] if answer else ""
    else:
        # Try Gemini
        gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=10),
            )
            answer = (response.text or "").strip().upper()
            for ch in answer:
                if ch in options:
                    return ch
            return answer[:1] if answer else ""

    return ""


# ---------------------------------------------------------------------------
# Ingestion modes
# ---------------------------------------------------------------------------

def ingest_classic(
    events: list[Event],
    agent: CognifoldAgent,
    graph: ConceptGraph,
    query_agent: MemoryQueryAgent,
) -> float:
    """Classic per-event ingestion. Returns elapsed seconds."""
    executor = PlanExecutor(graph)
    t0 = time.time()

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
            if "429" in str(e):
                time.sleep(10)
            else:
                logger.warning("Classic ingestion error: %s", e)

    return time.time() - t0


def ingest_fast(
    events: list[Event],
    agent_config: AgentConfig,
    graph: ConceptGraph,
    prompt_profile: Any = None,
) -> float:
    """Fast batched ingestion using LayeredPipeline approach. Returns elapsed seconds."""
    t0 = time.time()

    # Layer 1: Add all events as nodes (no LLM)
    for event in events:
        node = Node(
            id=event.event_id,
            type=NodeType.EVENT,
            data={
                "title": event.title,
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "description": event.description,
                "location": event.location,
            },
            created_at=event.timestamp,
            last_accessed=event.timestamp,
        )
        if not graph.has_node(event.event_id):
            graph.add_node(node)

    # Layer 2: Batch LLM enrichment (single call for all events)
    batch_processor = BatchAgentProcessor(
        agent_config=agent_config,
        prompt_profile=prompt_profile,
    )
    plans = batch_processor.process_event_batch(
        events=events,
        graph=graph,
        context_node_ids=[],
        node_scores={},
    )

    # Execute plans
    executor = PlanExecutor(graph)
    for plan in plans:
        try:
            executor.execute(plan)
        except Exception as e:
            logger.warning("Fast plan execution error: %s", e)

    return time.time() - t0


# ---------------------------------------------------------------------------
# Run a single benchmark in both modes
# ---------------------------------------------------------------------------

def _process_one_example(
    i: int,
    example: dict[str, Any],
    mode: str,
    benchmark: str,
    agent_config: AgentConfig,
    prompt_profile: Any,
    to_events_fn: Any,
    make_question_fn: Any,
    make_options_fn: Any,
    get_correct_fn: Any,
    model: str,
) -> dict[str, Any]:
    """Process a single example (thread-safe — each gets its own graph)."""
    graph = ConceptGraph()
    events = to_events_fn(example, i)

    # Ingest
    if mode == "classic":
        agent = CognifoldAgent(config=agent_config, prompt_profile=prompt_profile)
        query_config = QueryConfig(domain=benchmark, max_nodes=20, include_reasoning=True)
        query_agent = MemoryQueryAgent(graph, config=query_config)
        ingest_time = ingest_classic(events, agent, graph, query_agent)
    else:
        ingest_time = ingest_fast(events, agent_config, graph, prompt_profile)

    # QA evaluation
    qa_start = time.time()
    query_config = QueryConfig(domain=benchmark, max_nodes=20, include_reasoning=True)
    query_agent = MemoryQueryAgent(graph, config=query_config)

    question = make_question_fn(example)
    options = make_options_fn(example)
    correct = get_correct_fn(example)

    try:
        result = query_agent.query_for_qa(
            question=question,
            domain=benchmark,
            query_mode="mergefold",
        )
        context = result.context
        predicted = answer_mc_with_llm(question, context, options, model)
        is_correct = predicted == correct
    except Exception as e:
        logger.warning("QA error for example %d: %s", i, e)
        predicted = ""
        is_correct = False
        context = ""

    qa_time = time.time() - qa_start

    status = "OK" if is_correct else "WRONG"
    print(
        f"  [{mode}] Example {i+1} ({len(events)} ev) "
        f"{status} | ingest={ingest_time:.1f}s | qa={qa_time:.1f}s | "
        f"nodes={graph.node_count} edges={graph.edge_count}",
        flush=True,
    )

    return {
        "example_idx": i,
        "is_correct": is_correct,
        "predicted": predicted,
        "correct": correct,
        "ingest_time_s": round(ingest_time, 2),
        "qa_time_s": round(qa_time, 2),
        "graph_nodes": graph.node_count,
        "graph_edges": graph.edge_count,
        "context_length": len(context),
    }


def run_comparison(
    benchmark: str,
    data: list[dict[str, Any]],
    to_events_fn: Any,
    make_question_fn: Any,
    make_options_fn: Any,
    get_correct_fn: Any,
    model: str = "gpt-4o-mini",
    modes: Optional[list[str]] = None,
    max_workers: int = 5,
) -> dict[str, Any]:
    """Run a benchmark in specified modes, return comparison results."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if modes is None:
        modes = ["classic", "fast"]
    profile_path = CONFIGS_DIR / f"{benchmark}_profile.yaml"

    # Load prompt profile
    prompt_profile = None
    agent_config = AgentConfig(model_name=f"openai:{model}", temperature=0.0)
    if profile_path.exists():
        try:
            profiles = load_prompt_profiles(profile_path)
            prompt_profile = profiles.get(benchmark)
            if prompt_profile:
                agent_config = prompt_profile.to_agent_config()
        except Exception:
            pass

    results = {"benchmark": benchmark, "samples": len(data), "classic": [], "fast": []}

    for mode in modes:
        print(f"\n{'='*60}", flush=True)
        print(f"  {benchmark.upper()} — {mode.upper()} MODE ({len(data)} samples, {max_workers} workers)", flush=True)
        print(f"{'='*60}", flush=True)

        wall_start = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _process_one_example,
                    i, example, mode, benchmark,
                    agent_config, prompt_profile,
                    to_events_fn, make_question_fn,
                    make_options_fn, get_correct_fn, model,
                ): i
                for i, example in enumerate(data)
            }

            mode_results: list[dict[str, Any]] = []
            for future in as_completed(futures):
                try:
                    mode_results.append(future.result())
                except Exception as e:
                    idx = futures[future]
                    logger.error("Example %d failed: %s", idx, e)

        # Sort by example index for deterministic output
        mode_results.sort(key=lambda r: r["example_idx"])

        wall_elapsed = time.time() - wall_start
        total_ingest_time = sum(r["ingest_time_s"] for r in mode_results)
        total_qa_time = sum(r["qa_time_s"] for r in mode_results)
        correct_count = sum(1 for r in mode_results if r["is_correct"])
        accuracy = (correct_count / len(mode_results) * 100) if mode_results else 0

        results[mode] = {
            "results": mode_results,
            "accuracy": round(accuracy, 1),
            "correct": correct_count,
            "total": len(mode_results),
            "wall_time_s": round(wall_elapsed, 2),
            "total_ingest_time_s": round(total_ingest_time, 2),
            "total_qa_time_s": round(total_qa_time, 2),
            "avg_ingest_time_s": round(total_ingest_time / len(data), 2) if data else 0,
            "avg_nodes": round(sum(r["graph_nodes"] for r in mode_results) / len(mode_results), 1) if mode_results else 0,
            "avg_edges": round(sum(r["graph_edges"] for r in mode_results) / len(mode_results), 1) if mode_results else 0,
        }

        print(f"\n  Wall time: {wall_elapsed:.1f}s | Accuracy: {accuracy:.1f}% ({correct_count}/{len(mode_results)})", flush=True)

    return results


# ---------------------------------------------------------------------------
# Benchmark-specific helpers
# ---------------------------------------------------------------------------

# MuTual
def mutual_question(ex: dict) -> str:
    article = ex.get("article", [])
    return f"Given the following dialogue, what is the best next response?\n\n" + "\n".join(article)


def mutual_options(ex: dict) -> dict[str, str]:
    opts = {}
    for j, opt in enumerate(ex.get("options", [])):
        opts[chr(ord("A") + j)] = opt
    return opts


def mutual_correct(ex: dict) -> str:
    return ex.get("answers", "A")


# SocialIQA
def socialiqa_question(ex: dict) -> str:
    return f"{ex.get('context', '')} {ex.get('question', '')}"


def socialiqa_options(ex: dict) -> dict[str, str]:
    return {
        "A": ex.get("answerA", ""),
        "B": ex.get("answerB", ""),
        "C": ex.get("answerC", ""),
    }


def socialiqa_correct(ex: dict) -> str:
    label = int(ex.get("label", 1))
    return chr(ord("A") + label - 1)


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

def print_comparison(all_results: list[dict[str, Any]]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("  FAST vs CLASSIC COMPARISON SUMMARY")
    print("=" * 80)

    header = (
        f"{'Benchmark':<12} {'Mode':<8} {'Accuracy':<10} {'Wall(s)':<10} "
        f"{'Ingest(s)':<12} {'Avg Ingest':<12} {'Avg Nodes':<10} {'Avg Edges':<10}"
    )
    print(header)
    print("-" * 80)

    for res in all_results:
        benchmark = res["benchmark"]
        for mode_key in ["classic", "fast"]:
            mode_data = res[mode_key]
            if not mode_data:
                continue
            wall = mode_data.get("wall_time_s", 0)
            print(
                f"{benchmark:<12} {mode_key:<8} "
                f"{mode_data['accuracy']:>5.1f}%    "
                f"{wall:>6.1f}s   "
                f"{mode_data['total_ingest_time_s']:>8.1f}s   "
                f"{mode_data['avg_ingest_time_s']:>8.1f}s   "
                f"{mode_data['avg_nodes']:>6.1f}    "
                f"{mode_data['avg_edges']:>6.1f}"
            )
        print()

    # Speedup summary
    print("-" * 80)
    for res in all_results:
        classic = res.get("classic", {})
        fast = res.get("fast", {})
        if classic and fast and classic.get("total_ingest_time_s", 0) > 0:
            speedup = classic["total_ingest_time_s"] / max(fast["total_ingest_time_s"], 0.01)
            acc_diff = fast["accuracy"] - classic["accuracy"]
            sign = "+" if acc_diff >= 0 else ""
            print(
                f"  {res['benchmark']}: Fast is {speedup:.1f}x faster | "
                f"Accuracy diff: {sign}{acc_diff:.1f}pp"
            )

    # Latency report
    _print_latency_report(all_results)


def _print_latency_report(all_results: list[dict[str, Any]]) -> None:
    """Print a detailed latency report at the end."""
    print("\n" + "=" * 80)
    print("  LATENCY REPORT")
    print("=" * 80)

    for res in all_results:
        benchmark = res["benchmark"]
        n = res["samples"]

        for mode_key in ["classic", "fast"]:
            mode_data = res.get(mode_key)
            if not mode_data or not mode_data.get("results"):
                continue

            items = mode_data["results"]
            ingest_times = [r["ingest_time_s"] for r in items]
            qa_times = [r["qa_time_s"] for r in items]
            total_times = [r["ingest_time_s"] + r["qa_time_s"] for r in items]

            ingest_times_sorted = sorted(ingest_times)
            p50_idx = len(ingest_times_sorted) // 2
            p90_idx = int(len(ingest_times_sorted) * 0.9)
            p50 = ingest_times_sorted[p50_idx]
            p90 = ingest_times_sorted[min(p90_idx, len(ingest_times_sorted) - 1)]

            print(f"\n  {benchmark.upper()} — {mode_key.upper()}")
            print(f"  {'─' * 50}")
            print(f"  Samples:           {n}")
            print(f"  Wall clock:        {mode_data.get('wall_time_s', 0):.1f}s")
            print(f"  Total ingest:      {sum(ingest_times):.1f}s  (sum across threads)")
            print(f"  Total QA:          {sum(qa_times):.1f}s")
            print(f"  Avg ingest/ex:     {sum(ingest_times) / n:.1f}s")
            print(f"  Avg QA/ex:         {sum(qa_times) / n:.1f}s")
            print(f"  Avg total/ex:      {sum(total_times) / n:.1f}s")
            print(f"  Median ingest:     {p50:.1f}s")
            print(f"  P90 ingest:        {p90:.1f}s")
            print(f"  Min ingest:        {min(ingest_times):.1f}s")
            print(f"  Max ingest:        {max(ingest_times):.1f}s")
            print(f"  Accuracy:          {mode_data['accuracy']:.1f}% ({mode_data['correct']}/{mode_data['total']})")
            print(f"  Avg graph nodes:   {mode_data['avg_nodes']:.1f}")
            print(f"  Avg graph edges:   {mode_data['avg_edges']:.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare classic vs fast ingestion on benchmarks")
    parser.add_argument("--limit", type=int, default=20, help="Number of examples per benchmark")
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["mutual", "socialiqa", "both"],
        default="both",
        help="Which benchmark to run",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model for evaluation")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["classic", "fast", "both"],
        default="both",
        help="Ingestion mode to run (default: both for comparison)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5)",
    )
    args = parser.parse_args()

    modes = ["classic", "fast"] if args.mode == "both" else [args.mode]

    # Check API keys
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_google = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    if not has_openai and not has_google:
        print("ERROR: No API keys found. Set OPENAI_API_KEY or GOOGLE_API_KEY.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results: list[dict[str, Any]] = []

    if args.benchmark in ("mutual", "both"):
        data = load_mutual_data(args.limit)
        if data:
            result = run_comparison(
                benchmark="mutual",
                data=data,
                to_events_fn=mutual_to_events,
                make_question_fn=mutual_question,
                make_options_fn=mutual_options,
                get_correct_fn=mutual_correct,
                model=args.model,
                modes=modes,
                max_workers=args.workers,
            )
            all_results.append(result)

    if args.benchmark in ("socialiqa", "both"):
        data = load_socialiqa_data(args.limit)
        if data:
            result = run_comparison(
                benchmark="socialiqa",
                data=data,
                to_events_fn=socialiqa_to_events,
                make_question_fn=socialiqa_question,
                make_options_fn=socialiqa_options,
                get_correct_fn=socialiqa_correct,
                model=args.model,
                modes=modes,
                max_workers=args.workers,
            )
            all_results.append(result)

    # Print comparison
    if all_results:
        print_comparison(all_results)

        # Save results
        out_path = OUTPUT_DIR / "comparison_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
