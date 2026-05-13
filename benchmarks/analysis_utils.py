"""Shared utilities for enriching benchmark evaluation results.

This module lives in benchmarks/ (NOT src/cognifold/) to avoid pyright strict mode.
It provides functions to:
  - Enrich eval_result dicts with retrieval diagnostics
  - Auto-categorize failures into standard categories
  - Save wrong cases with category breakdown

Usage in benchmark runners (with graceful fallback):
    try:
        from benchmarks.analysis_utils import enrich_eval_result, save_wrong_cases
    except ImportError:
        enrich_eval_result = None
        save_wrong_cases = None
"""

import json
import time
from pathlib import Path
from typing import Any, Optional

# Maximum context size in bytes to store in results (prevents huge JSON files)
MAX_CONTEXT_BYTES = 2048

# Standard failure categories
FAILURE_CATEGORIES = [
    "retrieval_empty",
    "retrieval_irrelevant",
    "graph_incomplete",
    "llm_extraction_error",
    "llm_verbose",
    "temporal_miss",
    "multi_hop_miss",
]


def categorize_failure(result: dict[str, Any]) -> str:
    """Auto-categorize a failed result into a standard failure category.

    Uses heuristics based on available fields. Human/Claude can override.

    Args:
        result: An eval_result dict (must have at least verdict or exact_match).

    Returns:
        One of the standard failure category strings.
    """
    ctx = result.get("retrieved_context", "")
    node_count = result.get("retrieved_node_count", -1)
    target = str(result.get("target", result.get("answer", "")))
    verdict = result.get("verdict", "")
    question = result.get("question", "")

    # 1. No context retrieved
    if node_count == 0 or (node_count == -1 and len(ctx) < 50):
        return "retrieval_empty"

    # 2. Correct verdict but failed exact match → verbose answer
    if verdict in ("CORRECT", "PARTIAL") and not result.get("exact_match", False):
        return "llm_verbose"

    # 3. Answer is in context but LLM didn't extract it
    if target and target.lower() in ctx.lower():
        return "llm_extraction_error"

    # 4. Temporal question
    temporal_keywords = ["when", "before", "after", "date", "time", "year", "month", "ago", "last", "first"]
    if any(kw in question.lower() for kw in temporal_keywords):
        return "temporal_miss"

    # 5. Multi-hop question
    if result.get("num_hops", 1) > 1:
        return "multi_hop_miss"

    # 6. Context retrieved but irrelevant
    if node_count > 0 or len(ctx) > 50:
        return "retrieval_irrelevant"

    # Default
    return "graph_incomplete"


def enrich_eval_result(
    eval_result: dict[str, Any],
    graph: Any = None,
    query_result: Any = None,
    retrieval_mode: str = "mergefold",
    query_start_time: Optional[float] = None,
) -> dict[str, Any]:
    """Enrich an eval_result dict with retrieval diagnostic fields.

    This is purely additive — it only adds new keys, never modifies existing ones.

    Args:
        eval_result: The existing evaluation result dict.
        graph: A ConceptGraph instance (optional, for node/edge counts).
        query_result: A QueryResult instance (optional, for context/nodes).
        retrieval_mode: The retrieval mode used (e.g., "mergefold", "bm25").
        query_start_time: time.time() captured before the query call.

    Returns:
        The same dict with additional diagnostic fields added.
    """
    # Graph stats
    if graph is not None:
        try:
            eval_result["graph_node_count"] = graph.node_count
            eval_result["graph_edge_count"] = graph.edge_count
        except Exception:
            pass

    # Retrieval diagnostics
    if query_result is not None:
        try:
            context = getattr(query_result, "context", "")
            # Cap context at MAX_CONTEXT_BYTES
            if isinstance(context, str) and len(context) > MAX_CONTEXT_BYTES:
                context = context[:MAX_CONTEXT_BYTES] + "...[truncated]"
            eval_result["retrieved_context"] = context

            nodes = getattr(query_result, "nodes", [])
            eval_result["retrieved_node_ids"] = [
                getattr(n, "node_id", str(n)) for n in nodes[:50]
            ]
            eval_result["retrieved_node_count"] = len(nodes)
        except Exception:
            pass

    # Retrieval mode
    eval_result["retrieval_mode"] = retrieval_mode

    # Query timing
    if query_start_time is not None:
        eval_result["query_time_ms"] = round((time.time() - query_start_time) * 1000, 1)

    # Auto-categorize if this is a wrong result
    verdict = eval_result.get("verdict", "")
    exact_match = eval_result.get("exact_match", None)
    is_wrong = verdict in ("INCORRECT", "ERROR") or (
        exact_match is False and verdict not in ("CORRECT",)
    )
    if is_wrong:
        eval_result["failure_category"] = categorize_failure(eval_result)

    return eval_result


def save_wrong_cases(
    all_results: list[dict[str, Any]],
    output_dir: str,
    filename: str = "wrong_cases.json",
) -> Optional[str]:
    """Save wrong cases with category breakdown to a JSON file.

    Args:
        all_results: List of all eval_result dicts.
        output_dir: Directory to save the wrong_cases.json file.
        filename: Output filename (default: wrong_cases.json).

    Returns:
        Path to the saved file, or None if no wrong cases.
    """
    wrong = []
    for r in all_results:
        verdict = r.get("verdict", "").upper()
        exact_match = r.get("exact_match", None)

        is_wrong = False
        if verdict in ("INCORRECT", "ERROR"):
            is_wrong = True
        elif exact_match is False and verdict not in ("CORRECT",):
            is_wrong = True

        if is_wrong:
            # Ensure failure_category is set
            if "failure_category" not in r:
                r["failure_category"] = categorize_failure(r)
            wrong.append(r)

    if not wrong:
        return None

    # Category breakdown
    categories: dict[str, int] = {}
    for w in wrong:
        cat = w.get("failure_category", "uncategorized")
        categories[cat] = categories.get(cat, 0) + 1

    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "total_results": len(all_results),
        "total_wrong": len(wrong),
        "wrong_rate": round(len(wrong) / len(all_results), 4) if all_results else 0,
        "category_breakdown": dict(sorted(categories.items(), key=lambda x: -x[1])),
        "wrong_cases": wrong,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrong cases: {len(wrong)}/{len(all_results)} saved to {output_path}")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    return str(output_path)
