#!/usr/bin/env python3
"""Re-judge LoCoMo full results with 3-vote majority protocol.

Implements the audit-recommended protocol from dial481/locomo-audit
(results-audit/RESULTS_AUDIT.md): three independent gpt-4o-mini judge
runs per question at temperature 0, majority vote (>=2/3) = CORRECT.

Uses the EverMemOS-derived judge prompt verbatim from
evaluation/config/prompts.yaml (SHA256-pinned by the audit). No graph
rebuild and no QA rerun --- only re-judges existing (question, gold,
generated) tuples stored in benchmarks/locomo/output/full10_*.json.

Usage:
    PYTHONPATH=src python scripts/rejudge_locomo_3vote.py [INPUT_JSON] [OUTPUT_JSON]
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# === Audit's exact prompts (verbatim from dial481/locomo-audit/evaluation/config/prompts.yaml) ===

SYSTEM_PROMPT = "You are an expert grader that determines if answers to questions match a gold standard answer"

USER_PROMPT_TEMPLATE = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {golden_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""


def parse_verdict(text: str) -> bool:
    """Parse judge response into binary CORRECT/WRONG. Tries JSON label first, then keyword fallback."""
    # JSON form: {"label": "CORRECT"} or {"label": "WRONG"}
    m = re.search(r'"\s*label\s*"\s*:\s*"\s*(CORRECT|WRONG)\s*"', text, re.IGNORECASE)
    if m:
        return m.group(1).upper() == "CORRECT"
    # Fallback: last keyword occurrence
    upper = text.upper()
    c = upper.rfind("CORRECT")
    w = upper.rfind("WRONG")
    return c > w


def call_judge(client, question: str, expected: str, generated: str, model: str = "gpt-4o-mini") -> tuple[bool, str]:
    """Single judge call. Returns (verdict, raw_response_text)."""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        golden_answer=expected,
        generated_answer=generated,
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=200,
    )
    raw = response.choices[0].message.content or ""
    return parse_verdict(raw), raw


def main():
    input_path = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/locomo/output/full10_iter6v2_4o-mini.json")
    output_path = Path(sys.argv[2] if len(sys.argv) > 2 else "benchmarks/locomo/output/full10_iter6v2_4o-mini_3vote.json")
    n_votes = 3
    model = "gpt-4o-mini"

    from openai import OpenAI
    client = OpenAI()

    data = json.loads(input_path.read_text())
    qa_list = data.get("qa_details", [])
    eligible = [(i, q) for i, q in enumerate(qa_list) if q.get("category") in (1, 2, 3, 4)]
    print(f"Loaded {len(qa_list)} QA entries; rejudging {len(eligible)} cat 1-4 with {n_votes}-vote majority via {model} (T=0)")
    print(f"Output: {output_path}")
    sys.stdout.flush()

    start = time.time()
    for idx, (qa_idx, qa) in enumerate(eligible):
        question = qa.get("question", "") or ""
        expected = str(qa.get("expected", "") or "")
        generated = qa.get("generated", "") or ""

        votes = []
        for vote_i in range(n_votes):
            ok = False
            for retry in range(3):
                try:
                    verdict, raw = call_judge(client, question, expected, generated, model=model)
                    votes.append({"verdict": bool(verdict), "raw": raw[:300]})
                    ok = True
                    break
                except Exception as e:
                    if retry < 2:
                        time.sleep(2 ** retry)
                    else:
                        votes.append({"verdict": False, "raw": f"ERR: {e}"})
            if not ok:
                pass  # already logged ERR

        n_correct = sum(1 for v in votes if v["verdict"])
        majority_correct = n_correct >= (n_votes // 2 + 1)
        qa["j_3vote"] = {
            "votes": votes,
            "n_correct": n_correct,
            "n_total": n_votes,
            "majority_correct": majority_correct,
        }

        if (idx + 1) % 50 == 0 or idx == len(eligible) - 1:
            elapsed = time.time() - start
            rate = (idx + 1) / max(0.01, elapsed)
            eta_min = (len(eligible) - idx - 1) / max(0.01, rate) / 60
            print(f"  {idx+1}/{len(eligible)}   {rate:.2f} q/s   ETA {eta_min:.1f}m   elapsed {elapsed/60:.1f}m")
            sys.stdout.flush()

    # ---- Aggregation: per-cat J-Score under both protocols ----
    cat_names = {1: "Multi-hop", 2: "Temporal", 3: "Open-domain", 4: "Single-hop"}

    sj = defaultdict(lambda: {"correct": 0, "total": 0})
    v3 = defaultdict(lambda: {"correct": 0, "total": 0})
    for qa in qa_list:
        cat = qa.get("category")
        if cat not in (1, 2, 3, 4):
            continue
        sj[cat]["total"] += 1
        if qa.get("j_correct"):
            sj[cat]["correct"] += 1
        if qa.get("j_3vote") and qa["j_3vote"]["majority_correct"]:
            v3[cat]["correct"] += 1
            v3[cat]["total"] += 1
        else:
            v3[cat]["total"] += 1

    sj_total_c = sum(c["correct"] for c in sj.values())
    sj_total_t = sum(c["total"] for c in sj.values())
    v3_total_c = sum(c["correct"] for c in v3.values())
    v3_total_t = sum(c["total"] for c in v3.values())

    print()
    print("=" * 64)
    print("SINGLE-JUDGE (Mem0 protocol) vs 3-VOTE MAJORITY (audit protocol)")
    print("=" * 64)
    print(f"{'Cat':<4} {'Name':<14} {'n':>6}   {'Single':>9}   {'3-Vote':>9}   {'Δ':>7}")
    for cat in (1, 2, 3, 4):
        s = sj[cat]; v = v3[cat]
        sp = 100.0 * s["correct"] / s["total"] if s["total"] else 0
        vp = 100.0 * v["correct"] / v["total"] if v["total"] else 0
        print(f"{cat:<4} {cat_names[cat]:<14} {s['total']:>6}   {sp:>8.2f}%   {vp:>8.2f}%   {vp-sp:+7.2f}")
    sp = 100.0 * sj_total_c / sj_total_t
    vp = 100.0 * v3_total_c / v3_total_t
    print(f"{'TOTAL':<4} {'':<14} {sj_total_t:>6}   {sp:>8.2f}%   {vp:>8.2f}%   {vp-sp:+7.2f}")
    print()

    data["qa_details"] = qa_list
    data["audit_3vote_summary"] = {
        "protocol": "audit (dial481/locomo-audit) 3-judge majority vote, gpt-4o-mini T=0",
        "n_votes": n_votes,
        "model": model,
        "per_cat_3vote": {str(c): {**v3[c]} for c in (1, 2, 3, 4)},
        "per_cat_single_judge": {str(c): {**sj[c]} for c in (1, 2, 3, 4)},
        "overall_3vote_pct": vp,
        "overall_single_judge_pct": sp,
    }
    output_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote {output_path} ({output_path.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
