#!/usr/bin/env python3
"""LoCoMo Direct LLM baseline.

Skips the graph entirely: for each QA, feed the full multi-session
conversation transcript + the question to the agent model and apply the
same Mem0-protocol J-score judge (binary, generous) used in
``run_benchmark.py``. Reports cat 1-4 J-Score so the number is
directly comparable to the CogniFold row in tab_results.

Usage:
    OPENAI_API_KEY=... python -m benchmarks.locomo.run_direct_llm \
        --model gpt-4.1-mini --judge-model gpt-4o-mini --limit 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Reuse the existing judge implementation
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from benchmarks.locomo.run_benchmark import evaluate_with_jscore  # noqa: E402

DATASET = Path(__file__).parent / "locomo10.json"


def build_full_transcript(conv_data: dict[str, Any]) -> str:
    """Render every session in chronological order as plain text."""
    session_keys = sorted(
        (k for k in conv_data if re.match(r"^session_\d+$", k)),
        key=lambda x: int(x.split("_")[1]),
    )
    parts: list[str] = []
    for sk in session_keys:
        ts_str = conv_data.get(f"{sk}_date_time", "")
        parts.append(f"\n=== {sk} ({ts_str}) ===")
        for turn in conv_data[sk]:
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "")
            parts.append(f"{speaker}: {text}")
    return "\n".join(parts).strip()


def call_openai(prompt: str, model: str, max_tokens: int = 200) -> str:
    from openai import OpenAI

    client = OpenAI()
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            print(f"    [warn] LLM call failed: {e}")
            return ""
    return ""


def answer_direct(transcript: str, question: str, model: str) -> str:
    prompt = (
        "You are answering a question about a long-running conversation between two people. "
        "Use the FULL conversation transcript provided below to answer.\n\n"
        f"Conversation transcript:\n{transcript}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely (a single phrase, date, or short sentence):"
    )
    return call_openai(prompt, model=model, max_tokens=200)


def run(model: str, judge_model: str, limit: int | None) -> dict[str, Any]:
    if not DATASET.exists():
        print(f"ERROR: {DATASET} not found. Run download_data.py first.")
        sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    with open(DATASET) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    print(f"Loaded {len(data)} conversations from {DATASET}")
    print(f"Agent model: {model}    Judge model: {judge_model}")

    j_total = 0
    j_correct = 0
    j_by_cat: dict[int, dict[str, int]] = {
        c: {"total": 0, "correct": 0} for c in (1, 2, 3, 4)
    }
    qa_results: list[dict[str, Any]] = []

    for i, sample in enumerate(data):
        sample_id = sample.get("sample_id", f"conv_{i}")
        conv = sample.get("conversation", {})
        transcript = build_full_transcript(conv)
        qa_list = sample.get("qa", [])
        print(f"\n[{i + 1}/{len(data)}] {sample_id} — {len(qa_list)} QAs, "
              f"transcript {len(transcript)} chars")

        for qi, qa in enumerate(qa_list):
            question = qa.get("question") or ""
            expected = qa.get("answer")
            category = qa.get("category", "unknown")
            cat_int = int(category) if str(category).isdigit() else 0
            if cat_int not in (1, 2, 3, 4):
                continue
            if not question:
                continue

            generated = answer_direct(transcript, question, model)
            j_ok, _reason = evaluate_with_jscore(
                question=question,
                expected=expected,
                generated=generated,
                model=judge_model,
            )

            j_total += 1
            j_by_cat[cat_int]["total"] += 1
            if j_ok:
                j_correct += 1
                j_by_cat[cat_int]["correct"] += 1

            qa_results.append({
                "sample_id": sample_id,
                "question": question,
                "expected": expected,
                "generated": generated,
                "category": category,
                "j_correct": j_ok,
            })
            mark = "✓" if j_ok else "✗"
            print(f"  [{qi + 1}/{len(qa_list)}] cat={category} {mark} "
                  f"running J={j_correct}/{j_total} = "
                  f"{(100 * j_correct / max(j_total, 1)):.1f}%")
            time.sleep(0.2)

    j_score = (j_correct / j_total * 100) if j_total > 0 else 0.0
    by_cat_pct = {
        c: (v["correct"] / v["total"] * 100) if v["total"] else 0.0
        for c, v in j_by_cat.items()
    }

    print("\n" + "=" * 60)
    print(f"LOCOMO DIRECT LLM ({model}, judge={judge_model})")
    print(f"  J-Score (cats 1-4): {j_score:.1f}% ({j_correct}/{j_total})")
    for c in (1, 2, 3, 4):
        v = j_by_cat[c]
        print(f"    cat {c}: {by_cat_pct[c]:.1f}%  ({v['correct']}/{v['total']})")
    print("=" * 60)

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"direct_llm_{ts}.json"
    summary = {
        "metadata": {
            "model": model,
            "judge_model": judge_model,
            "n_conversations": len(data),
            "timestamp": ts,
        },
        "j_score": j_score,
        "j_total": j_total,
        "j_correct": j_correct,
        "j_by_category_pct": by_cat_pct,
        "j_by_category_raw": j_by_cat,
        "qa_results": qa_results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved to {out_path}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="LoCoMo Direct LLM baseline")
    p.add_argument("--model", default="gpt-4.1-mini")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of conversations (None = all 10)")
    args = p.parse_args()
    run(model=args.model, judge_model=args.judge_model, limit=args.limit)


if __name__ == "__main__":
    main()
