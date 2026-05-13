#!/usr/bin/env python3
"""LoCoMo benchmark runner using fast (batched) ingestion.

Runs all 10 conversations with Layer 1 (instant node creation) +
Layer 2 (batched LLM enrichment) for dramatically faster ingestion.

Usage:
    export OPENAI_API_KEY=your-key
    PYTHONPATH=src python3 -u benchmarks/locomo/run_fast.py [--limit N] [--workers N]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

# Add project paths
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_project_root, "src"))
sys.path.insert(0, _project_root)

from cognifold.agent.batch import BatchAgentProcessor
from cognifold.agent.config import AgentConfig
from cognifold.agent.prompt_profile import load_prompt_profiles
from cognifold.executor.runner import PlanExecutor
from cognifold.graph.store import ConceptGraph
from cognifold.models.event import Event
from cognifold.models.node import Node, NodeType
from cognifold.query.agent import MemoryQueryAgent
from cognifold.query.models import QueryConfig

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent / "locomo10.json"
PROFILE_PATH = Path(_project_root) / "configs" / "locomo_profile.yaml"
OUTPUT_DIR = Path(__file__).parent / "output"

# Reuse helpers from original runner
sys.path.insert(0, str(Path(__file__).parent))


def parse_locomo_timestamp(ts_str: str) -> datetime:
    """Parse LoCoMo timestamp like '1:56 pm on 8 May, 2023'."""
    if not ts_str:
        return datetime(2023, 1, 1, 12, 0, 0)
    ts_str = ts_str.strip()
    for fmt in [
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %b, %Y",
        "%H:%M on %d %B, %Y",
    ]:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return datetime(2023, 1, 1, 12, 0, 0)


_eval_model: str = "gpt-4o-mini"


def call_llm_for_eval(prompt: str, model: str = "") -> str:
    """Call LLM for evaluation (auto-detects OpenAI or Gemini)."""
    model = model or _eval_model

    if model.startswith("gemini") or (
        not os.environ.get("OPENAI_API_KEY")
        and (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    ):
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        gemini_model = model if model.startswith("gemini") else "gemini-2.0-flash"
        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=200),
        )
        return response.text or ""

    # OpenAI fallback
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return ""
    from openai import OpenAI

    client = OpenAI()
    oai_model = model if not model.startswith("gemini") else "gpt-4o-mini"
    response = client.chat.completions.create(
        model=oai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    )
    return response.choices[0].message.content or ""


def evaluate_with_llm(question: str, expected: str, context: str) -> tuple[str, str]:
    """Evaluate QA result using LLM."""
    prompt = f"""Evaluate if the context answers the question correctly.

Question: {question}
Expected Answer: {expected}
Context from knowledge graph:
{context[:3000]}

Respond with ONE of: CORRECT, PARTIAL, INCORRECT
Then explain briefly on the next line."""
    try:
        response = call_llm_for_eval(prompt)
        lines = response.strip().split("\n", 1)
        result = lines[0].strip().upper()
        explanation = lines[1].strip() if len(lines) > 1 else ""
        if "CORRECT" in result and "INCORRECT" not in result:
            return "CORRECT", explanation
        elif "PARTIAL" in result:
            return "PARTIAL", explanation
        else:
            return "INCORRECT", explanation
    except Exception as e:
        return "ERROR", str(e)


def load_data(limit: Optional[int] = None) -> list[dict[str, Any]]:
    """Load LoCoMo dataset."""
    if not DATA_PATH.exists():
        print(f"Data not found at {DATA_PATH}")
        print("Run: python3 benchmarks/locomo/download_data.py")
        sys.exit(1)
    with open(DATA_PATH) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    return data


def conversation_to_events(
    conv_data: dict[str, Any],
    limit_sessions: Optional[int] = None,
) -> list[Event]:
    """Convert a LoCoMo conversation to a list of Events."""
    session_keys = [k for k in conv_data if re.match(r"^session_\d+$", k)]
    session_keys.sort(key=lambda x: int(x.split("_")[1]))
    if limit_sessions:
        session_keys = session_keys[:limit_sessions]

    events: list[Event] = []
    for session_key in session_keys:
        turns = conv_data[session_key]
        ts_key = f"{session_key}_date_time"
        base_ts = parse_locomo_timestamp(conv_data.get(ts_key, ""))

        for turn_idx, turn in enumerate(turns):
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "")
            dia_id = turn.get("dia_id", "")

            events.append(
                Event(
                    event_id=str(uuid.uuid4()),
                    timestamp=base_ts + timedelta(seconds=turn_idx * 10),
                    source="locomo-benchmark",
                    event_type="conversation.turn",
                    title=f"{speaker}: {text[:50]}...",
                    description=text,
                    context={
                        "speaker": speaker,
                        "session_id": session_key,
                        "dialog_id": dia_id,
                    },
                )
            )
    return events


def ingest_fast(
    events: list[Event],
    agent_config: AgentConfig,
    graph: ConceptGraph,
    prompt_profile: Any = None,
    batch_size: int = 10,
) -> float:
    """Fast batched ingestion. Returns elapsed seconds."""
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

    layer1_time = time.time() - t0
    print(f"    Layer 1: {len(events)} nodes in {layer1_time:.1f}s", flush=True)

    # Layer 2: Batch LLM enrichment
    batch_processor = BatchAgentProcessor(
        agent_config=agent_config,
        prompt_profile=prompt_profile,
    )
    executor = PlanExecutor(graph)

    batches_done = 0
    plans_total = 0
    errors = 0

    for batch_start in range(0, len(events), batch_size):
        batch = events[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(events) + batch_size - 1) // batch_size

        for attempt in range(3):
            try:
                plans = batch_processor.process_event_batch(
                    events=batch,
                    graph=graph,
                    context_node_ids=[],
                    node_scores={},
                )
                for plan in plans:
                    try:
                        executor.execute(plan)
                        plans_total += 1
                    except Exception as e:
                        errors += 1
                        print(f"    Plan exec error: {e}", flush=True)
                break  # success
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait = 5 * (attempt + 1)
                    print(f"    Rate limit, retrying in {wait}s...", flush=True)
                    time.sleep(wait)
                    continue
                errors += 1
                print(f"    Batch {batch_num} failed: {type(e).__name__}: {e}", flush=True)
                break

        batches_done += 1
        if batches_done % 5 == 0 or batches_done == total_batches:
            elapsed = time.time() - t0
            print(
                f"    Layer 2: batch {batches_done}/{total_batches} | "
                f"{plans_total} plans | {elapsed:.0f}s elapsed",
                flush=True,
            )

    total_time = time.time() - t0
    print(
        f"    Done: {graph.node_count} nodes, {graph.edge_count} edges, "
        f"{errors} errors, {total_time:.1f}s total",
        flush=True,
    )
    return total_time


def evaluate_qa(
    qa_list: list[dict[str, Any]],
    graph: ConceptGraph,
    query_mode: str = "mergefold",
) -> dict[str, Any]:
    """Evaluate QA pairs against the graph. Returns metrics dict."""
    query_config = QueryConfig(
        domain="locomo",
        speaker_aware=True,
        max_nodes=20,
        include_reasoning=True,
    )
    query_agent = MemoryQueryAgent(graph, config=query_config)

    metrics = {"correct": 0, "partial": 0, "incorrect": 0, "error": 0}
    qa_results: list[dict[str, Any]] = []

    for qi, qa_item in enumerate(qa_list):
        question = qa_item.get("question")
        answer = qa_item.get("answer")
        category = qa_item.get("category", "unknown")

        if not question:
            continue

        try:
            qa_start = time.time()
            result = query_agent.query_for_qa(
                question=question,
                domain="locomo",
                query_mode=query_mode,
            )
            context = result.context
            qa_time = time.time() - qa_start

            eval_result, explanation = evaluate_with_llm(question, answer, context)
            metrics[eval_result.lower()] = metrics.get(eval_result.lower(), 0) + 1

            qa_results.append({
                "question": question,
                "expected": answer,
                "category": category,
                "result": eval_result,
                "explanation": explanation,
                "context_length": len(context),
                "query_time_ms": round(qa_time * 1000, 1),
            })

            if qi < 3:
                print(f"      Q: {question[:80]}", flush=True)
                print(f"      A: {answer} => {eval_result}", flush=True)

        except Exception as e:
            metrics["error"] += 1
            qa_results.append({
                "question": question,
                "expected": answer,
                "category": category,
                "result": "ERROR",
                "explanation": str(e),
            })

        if (qi + 1) % 50 == 0:
            print(f"      QA progress: {qi + 1}/{len(qa_list)}", flush=True)

    total_eval = metrics["correct"] + metrics["partial"] + metrics["incorrect"]
    strict = (metrics["correct"] / total_eval * 100) if total_eval > 0 else 0
    partial = (
        ((metrics["correct"] + 0.5 * metrics["partial"]) / total_eval * 100)
        if total_eval > 0
        else 0
    )

    return {
        "metrics": metrics,
        "strict_score": round(strict, 1),
        "partial_score": round(partial, 1),
        "total_eval": total_eval,
        "qa_results": qa_results,
    }


def process_one_conversation(
    i: int,
    sample: dict[str, Any],
    agent_config: AgentConfig,
    prompt_profile: Any,
    batch_size: int,
    limit_sessions: Optional[int],
    query_mode: str,
) -> dict[str, Any]:
    """Process a single conversation end-to-end (thread-safe)."""
    sample_id = sample.get("sample_id", f"conv-{i}")
    conv_data = sample.get("conversation", {})

    print(f"\n  [{sample_id}] Starting...", flush=True)

    # Convert to events
    events = conversation_to_events(conv_data, limit_sessions)
    print(f"  [{sample_id}] {len(events)} events", flush=True)

    # Ingest
    graph = ConceptGraph()
    ingest_time = ingest_fast(events, agent_config, graph, prompt_profile, batch_size)

    # Post-ingestion: fact extraction and entity indexing
    try:
        from cognifold.graph.entity_index import EntityIndex
        from cognifold.graph.fact_extraction import extract_facts

        fact_ids = extract_facts(graph)
        if fact_ids:
            print(f"  [{sample_id}] Fact extraction: {len(fact_ids)} fact nodes", flush=True)

        entity_idx = EntityIndex()
        entity_idx.build(graph)
        graph.entity_index = entity_idx
        print(f"  [{sample_id}] Entity index: {entity_idx.entity_count} entities", flush=True)
    except Exception as e:
        print(f"  [{sample_id}] Warning: post-ingest hooks failed: {e}", flush=True)

    # Evaluate QA
    qa_list = sample.get("qa", [])
    print(f"  [{sample_id}] Evaluating {len(qa_list)} QA pairs...", flush=True)
    qa_start = time.time()
    qa_eval = evaluate_qa(qa_list, graph, query_mode)
    qa_time = time.time() - qa_start

    print(
        f"  [{sample_id}] DONE | strict={qa_eval['strict_score']:.1f}% "
        f"partial={qa_eval['partial_score']:.1f}% | "
        f"ingest={ingest_time:.0f}s qa={qa_time:.0f}s | "
        f"nodes={graph.node_count} edges={graph.edge_count}",
        flush=True,
    )

    return {
        "sample_id": sample_id,
        "events": len(events),
        "ingest_time_s": round(ingest_time, 1),
        "qa_time_s": round(qa_time, 1),
        "graph_nodes": graph.node_count,
        "graph_edges": graph.edge_count,
        "strict_score": qa_eval["strict_score"],
        "partial_score": qa_eval["partial_score"],
        "metrics": qa_eval["metrics"],
        "qa_results": qa_eval["qa_results"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LoCoMo benchmark with fast ingestion")
    parser.add_argument("--limit", type=int, default=None, help="Limit conversations")
    parser.add_argument("--sessions", type=int, default=None, help="Limit sessions per conv")
    parser.add_argument("--batch-size", type=int, default=10, help="Events per LLM batch")
    parser.add_argument("--workers", type=int, default=3, help="Parallel conversations")
    parser.add_argument("--query-mode", type=str, default="mergefold", help="Query mode")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model (e.g. gpt-4o-mini)")
    args = parser.parse_args()

    # Check API keys
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY or GOOGLE_API_KEY")
        sys.exit(1)

    # Load data
    data = load_data(args.limit)
    print(f"Loaded {len(data)} conversations", flush=True)

    # Load profile
    prompt_profile = None
    agent_config = AgentConfig(model_name="openai:gpt-4o-mini", temperature=0.1)
    if PROFILE_PATH.exists():
        try:
            profiles = load_prompt_profiles(PROFILE_PATH)
            prompt_profile = profiles.get("locomo")
            if prompt_profile:
                agent_config = prompt_profile.to_agent_config()
                print("Using profile: locomo", flush=True)
        except Exception as e:
            print(f"Warning: profile load failed: {e}", flush=True)

    # Override model if requested (useful for rate limit avoidance)
    if args.model:
        global _eval_model
        raw_model = args.model
        # For agent config, need provider prefix
        if ":" in raw_model:
            model_name = raw_model
        elif raw_model.startswith("gemini"):
            model_name = raw_model  # Gemini models don't need prefix in AgentConfig
        else:
            model_name = f"openai:{raw_model}"
        agent_config = AgentConfig(
            model_name=model_name,
            temperature=agent_config.temperature,
            domain=agent_config.domain,
            max_tokens=16384,
        )
        # Also use same model family for QA evaluation
        _eval_model = raw_model
        print(f"Model override: {model_name}", flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wall_start = time.time()

    # Process conversations in parallel
    all_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_one_conversation,
                i, sample, agent_config, prompt_profile,
                args.batch_size, args.sessions, args.query_mode,
            ): i
            for i, sample in enumerate(data)
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                idx = futures[future]
                logger.error("Conversation %d failed: %s", idx, e)
                print(f"  Conversation {idx} FAILED: {e}", flush=True)

    all_results.sort(key=lambda r: r["sample_id"])
    wall_time = time.time() - wall_start

    # Print summary
    print("\n" + "=" * 80, flush=True)
    print("  LOCOMO FAST BENCHMARK RESULTS", flush=True)
    print("=" * 80, flush=True)

    hdr = "{:<10} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "Conv", "Events", "Nodes", "Edges", "Ingest", "QA", "Strict", "Partial"
    )
    print(hdr, flush=True)
    print("-" * 80, flush=True)

    total_events = 0
    total_nodes = 0
    total_edges = 0
    total_ingest = 0.0
    total_qa = 0.0
    total_strict = 0.0
    total_partial = 0.0

    for r in all_results:
        row = "{:<10} {:>6} {:>8} {:>8} {:>7.0f}s {:>7.0f}s {:>7.1f}% {:>7.1f}%".format(
            r["sample_id"], r["events"], r["graph_nodes"], r["graph_edges"],
            r["ingest_time_s"], r["qa_time_s"], r["strict_score"], r["partial_score"],
        )
        print(row, flush=True)
        total_events += r["events"]
        total_nodes += r["graph_nodes"]
        total_edges += r["graph_edges"]
        total_ingest += r["ingest_time_s"]
        total_qa += r["qa_time_s"]
        total_strict += r["strict_score"]
        total_partial += r["partial_score"]

    n = len(all_results) or 1
    print("-" * 80, flush=True)
    avg_row = "{:<10} {:>6} {:>8} {:>8} {:>7.0f}s {:>7.0f}s {:>7.1f}% {:>7.1f}%".format(
        "TOTAL", total_events, total_nodes, total_edges,
        total_ingest, total_qa, total_strict / n, total_partial / n,
    )
    print(avg_row, flush=True)
    print(f"\n  Wall clock: {wall_time:.0f}s ({wall_time/60:.1f}min)", flush=True)
    print(f"  Workers: {args.workers}", flush=True)
    print(f"  Batch size: {args.batch_size}", flush=True)

    # Save results
    out_path = OUTPUT_DIR / "fast_benchmark_results.json"
    save_data = {
        "config": {
            "mode": "fast",
            "workers": args.workers,
            "batch_size": args.batch_size,
            "query_mode": args.query_mode,
            "limit_sessions": args.sessions,
        },
        "wall_time_s": round(wall_time, 1),
        "summary": [
            {k: v for k, v in r.items() if k != "qa_results"}
            for r in all_results
        ],
        "qa_details": {
            r["sample_id"]: r["qa_results"]
            for r in all_results
        },
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {out_path}", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
