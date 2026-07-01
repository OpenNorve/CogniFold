"""Comprehensive $0 static analysis of the TAKE-MAX router over full MS-133.

Take-max changes the collateral model fundamentally vs the old hint-injection:
  - OLD (hint): every count/sum question NS fires on is at risk — the reader can
    be dragged UP or DOWN. Collateral surface = all 47 currently-correct fires.
  - NEW (take-max): NS can ONLY raise the count. So a currently-correct count/sum
    question breaks ONLY if NS OVER-counts (ratchets up wrongly). Every DOWN-ratchet
    (the observed collateral: tanks, gaming) is structurally killed. The effective
    collateral surface collapses from "all 47" to "the over-count subset" (measured
    0/10 on cached controls — the residual unknown).

This tool:
  1. Classifies all 133 MS questions, splits fires by family and take-max scope.
  2. Anchors on the $0 cached take-max replay (+7 fixes / 0 breaks).
  3. Projects full-MS under take-max for a grid of (win-rate w, over-count-rate c).
  4. Contrasts with the old hint-projection so the structural improvement is explicit.

Run:
  PYTHONPATH=src:. .venv/bin/python benchmarks/longmemeval/ns_take_max_static.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import neural_symbolic as ns

HERE = Path(__file__).parent
BASE = HERE / "runs"
QIDS = HERE / "qid_sets" / "ms_only.txt"


def load(p: Path) -> dict:
    d = {}
    if p.exists():
        with open(p) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    d[r["question_id"]] = r
                except Exception:
                    pass
    return d


def verdict(r: dict) -> bool:
    return (r.get("verdict") or (r.get("evaluation") or {}).get("result") or "").upper() == "CORRECT"


def main() -> int:
    base = load(BASE / "iter33_ms_clean" / "hypothesis.jsonl")
    qids = [q.strip() for q in QIDS.read_text().split() if q.strip()]

    # Segment every MS question.
    seg = Counter()
    cs_winopp, cs_collateral, other_fire = [], [], []
    for qid in qids:
        r = base.get(qid, {})
        fam = ns.classify_question(r.get("question", "")) if r.get("question") else None
        correct = verdict(r)
        if fam is None:
            seg["nofire_correct" if correct else "nofire_wrong"] += 1
            continue
        is_cs = fam.name == ns.FAMILY_ENUMERATE_SUM
        if is_cs:
            (cs_collateral if correct else cs_winopp).append(qid)
        else:
            other_fire.append((qid, fam.name, correct))

    nofire_correct = seg["nofire_correct"]
    nofire_wrong = seg["nofire_wrong"]
    base_correct = sum(1 for q in qids if verdict(base.get(q, {})))
    total = len(qids)

    print(f"\n{'='*90}\nTAKE-MAX FULL-MS STATIC ANALYSIS  ({total} questions, baseline {base_correct}/{total}={base_correct/total*100:.1f}%)\n{'='*90}")
    print("Segments:")
    print(f"  no-fire / correct (untouched)        : {nofire_correct}")
    print(f"  no-fire / wrong                       : {nofire_wrong}")
    print(f"  count/sum fire, baseline WRONG (win-opp, take-max can fix) : {len(cs_winopp)}")
    print(f"  count/sum fire, baseline CORRECT (collateral surface)      : {len(cs_collateral)}")
    print(f"  other-family fire (hint path, NOT take-max): {len(other_fire)}  {[o[0] for o in other_fire]}")

    print(f"\n{'-'*90}\nKEY STRUCTURAL CHANGE (take-max vs old hint):")
    print(f"  OLD hint collateral surface = ALL {len(cs_collateral)} currently-correct count/sum fires (up OR down).")
    print(f"  NEW take-max collateral     = only the OVER-COUNT subset of those {len(cs_collateral)} (down-ratchets killed).")
    print("  Cached take-max replay (ns_take_max_sim.py): +7 fixes / 0 breaks; over-count 0/10 controls.")

    # Projection. Under take-max:
    #   correct = nofire_correct
    #           + (cs_winopp * w)                  fixes among count/sum win-opp
    #           + (cs_collateral * (1 - c))        survivors (only over-count c breaks)
    #           + other_fire_correct (assume neutral hint -> keep their current verdict)
    other_correct = sum(1 for _, _, c in other_fire if c)
    floor = nofire_correct + other_correct  # untouched-correct baseline within projection
    print(f"\n{'-'*90}\nPROJECTION  correct = {nofire_correct}(nofire-ok) + {other_correct}(other-fam-ok, assumed neutral)")
    print(f"            + {len(cs_winopp)}*w (count/sum fixes) + {len(cs_collateral)}*(1-c) (collateral survivors)")
    print("\n" + f"{'w / c':<8}" + "  ".join(f"c={c:.2f}" for c in (0.00, 0.05, 0.10, 0.20)))
    for w in (0.25, 0.35, 0.50):
        row = []
        for c in (0.00, 0.05, 0.10, 0.20):
            proj = floor + len(cs_winopp) * w + len(cs_collateral) * (1 - c)
            row.append(f"{proj/total*100:5.1f}%")
        print(f"w={w:.2f} |" + "  ".join(row))
    print(f"\n  baseline = {base_correct}/{total} = {base_correct/total*100:.1f}%  (break-even line)")
    print(f"  delta(q) = {len(cs_winopp)}*w - {len(cs_collateral)}*c   ->  net-positive whenever w/c > {len(cs_collateral)}/{len(cs_winopp)} = {len(cs_collateral)/max(1,len(cs_winopp)):.2f}")

    print(f"\n{'-'*90}\nWHY THIS IS A DIFFERENT GAME than the old hint projection:")
    print("  - OLD hint: collateral c was the RAW break rate on ALL 47 fires (measured ~0.3-0.4)")
    print("    -> projection 68-76% (<= baseline). Net hinged on a big, hard-to-shrink c.")
    print("  - NEW take-max: c is only the OVER-COUNT rate (NS wrongly raising a correct answer),")
    print("    measured 0/10 so far. The down-collateral that sank the old design is GONE;")
    print("    net-positive holds for any plausibly-small over-count rate.")
    print('='*90)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    raise SystemExit(main())
