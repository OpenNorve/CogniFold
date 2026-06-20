"""$0 static analysis of the neural-symbolic agent over ALL 133 MS questions.

No LLM, no network. For every MS question it answers:
  - Does the agent FIRE (classify_question != None)? Which family/mode?
  - What is the baseline verdict (iter33_ms_clean)? -> CORRECT or not.
This yields the agent's BLAST RADIUS:
  - fires on currently-CORRECT  = COLLATERAL SURFACE (questions it can break)
  - fires on currently-INCORRECT = WIN OPPORTUNITY  (questions it can fix)
  - does not fire               = untouched
Plus static RISK flags per firing question (the live smoke showed collateral
comes from the agent UNDER-counting; sum-mode + many-operand + include-clause
shapes are the higher-risk ones).

Usage:
  PYTHONPATH=src .venv/bin/python benchmarks/longmemeval/ns_static_analysis.py \
      --qids benchmarks/longmemeval/qid_sets/ms_only.txt \
      --base benchmarks/longmemeval/runs/iter33_ms_clean/hypothesis.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import neural_symbolic as ns

INCLUDE_CLAUSE_RE = re.compile(r"\b(including|counting|incl\.?|plus the|as well as the)\b", re.IGNORECASE)


def load_base(path: Path) -> dict[str, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                out[r["question_id"]] = r
            except Exception:
                pass
    return out


def verdict(r: dict) -> str:
    return (r.get("verdict") or (r.get("evaluation") or {}).get("result") or "").upper()


def risk_flags(question: str, fam) -> list[str]:
    flags = []
    if fam.name == ns.FAMILY_ENUMERATE_SUM and fam.mode == ns.MODE_SUM:
        flags.append("SUM")  # aggregates many operands -> under-count risk (gaming case)
    if INCLUDE_CLAUSE_RE.search(question):
        flags.append("INCLUDE-CLAUSE")  # now mitigated by the forced-include prompt rule
    if re.search(r"\b(all|every|total|entire)\b", question.lower()):
        flags.append("EXHAUSTIVE")  # demands completeness -> recall-sensitive
    return flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--show", default="fires", choices=["fires", "all"])
    args = ap.parse_args()

    qids = [q.strip() for q in Path(args.qids).read_text().split() if q.strip()]
    base = load_base(Path(args.base))

    rows = []
    for qid in qids:
        rec = base.get(qid, {})
        q = rec.get("question", "")
        fam = ns.classify_question(q) if q else None
        bv = verdict(rec)
        fires = fam is not None
        fam_label = ""
        flags = []
        if fires:
            fam_label = fam.name + (f"/{fam.mode}" if fam.mode else "")
            flags = risk_flags(q, fam)
        rows.append({
            "qid": qid, "q": q, "bv": bv, "fires": fires,
            "fam": fam_label, "flags": flags,
        })

    fired = [r for r in rows if r["fires"]]
    fire_correct = [r for r in fired if r["bv"] == "CORRECT"]      # collateral surface
    fire_wrong = [r for r in fired if r["bv"] and r["bv"] != "CORRECT"]  # win opportunity
    fire_unknown = [r for r in fired if not r["bv"]]
    no_fire = [r for r in rows if not r["fires"]]

    print(f"\n{'='*94}\nNEURAL-SYMBOLIC FULL-MS STATIC ANALYSIS  ({len(qids)} questions)\n{'='*94}")
    print(f"FIRES on              : {len(fired)}/{len(qids)}")
    print(f"  - currently CORRECT : {len(fire_correct)}   <-- COLLATERAL SURFACE (can break)")
    print(f"  - currently WRONG   : {len(fire_wrong)}   <-- WIN OPPORTUNITY (can fix)")
    if fire_unknown:
        print(f"  - baseline unknown  : {len(fire_unknown)}")
    print(f"DOES NOT FIRE         : {len(no_fire)}/{len(qids)} (untouched)")

    # family + mode breakdown over fired
    famc = Counter(r["fam"] for r in fired)
    print("\nFamily/mode breakdown (fired):")
    for fam, n in famc.most_common():
        print(f"  {fam:<28}{n}")

    # risk breakdown over the COLLATERAL SURFACE (the dangerous set)
    print(f"\nRISK on the {len(fire_correct)} collateral-surface questions (static flags):")
    sum_risk = [r for r in fire_correct if "SUM" in r["flags"]]
    incl_risk = [r for r in fire_correct if "INCLUDE-CLAUSE" in r["flags"]]
    print(f"  SUM-mode (aggregate many operands -> under-count risk): {len(sum_risk)}")
    print(f"  INCLUDE-CLAUSE (now prompt-mitigated)                : {len(incl_risk)}  {[r['qid'] for r in incl_risk]}")
    clean = [r for r in fire_correct if not r["flags"]]
    print(f"  no risk flag (simple count)                          : {len(clean)}")

    def dump(title, items):
        print(f"\n--- {title} ({len(items)}) ---")
        for r in sorted(items, key=lambda x: x["fam"]):
            fl = ("  [" + ",".join(r["flags"]) + "]") if r["flags"] else ""
            print(f"  {r['qid']:<16}{r['fam']:<26}{r['bv']:<10}{r['q'][:58]}{fl}")

    dump("WIN OPPORTUNITY (fires on currently-wrong)", fire_wrong)
    dump("COLLATERAL SURFACE (fires on currently-correct)", fire_correct)
    if args.show == "all":
        dump("DOES NOT FIRE", no_fire)

    print(f"\n{'='*94}")
    print(f"NET-EFFECT FRAMING: agent net = (wins among the {len(fire_wrong)} win-opportunities) "
          f"- (collateral among the {len(fire_correct)} collateral-surface).")
    print("Live 17-qid smoke rate: ~8 wins / ~12 fired-targets; 2 collateral / 5 fired-controls.")
    print("Naive extrapolation (PRE-fix rates, wide CI): "
          f"~{round(len(fire_wrong)*8/12)} wins vs ~{round(len(fire_correct)*2/5)} collateral.")
    print("The two fixes (lower-bound merge block + forced-include) target the collateral term.")
    print('='*94)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
