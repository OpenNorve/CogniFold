"""Compare a --neural-symbolic smoke run against the baseline, per qid.

Reports for each qid: its role (target / writer-gap / control), the baseline
verdict, the new verdict, whether the neural-symbolic agent FIRED (symbolic_pattern
starts with neural_symbolic_), and the outcome class:
  WIN        baseline wrong -> new correct
  LOSS       baseline correct -> new wrong  (COLLATERAL if NS fired on it)
  no-change  same verdict

Usage:
  PYTHONPATH=src .venv/bin/python benchmarks/longmemeval/ns_smoke_compare.py \
      --new  benchmarks/longmemeval/runs/ns_smoke17_v1/hypothesis.jsonl \
      --base benchmarks/longmemeval/runs/iter33_ms_clean/hypothesis.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROLE = {
    "60159905": "target", "gpt4_2f8be40d": "target", "gpt4_a56e767c": "target",
    "c4a1ceb8": "target", "bf659f65": "target", "d905b33f": "target",
    "gpt4_5501fe77": "target", "2ce6a0f2": "target", "d682f1a2": "target",
    "eeda8a6d": "writer-gap", "67e0d0f2": "writer-gap", "e3038f8c": "writer-gap",
    "gpt4_59c863d7": "control", "gpt4_f2262a51": "control", "46a3abf7": "control",
    "6cb6f249": "control", "28dc39ac": "control",
    # high-risk SUM controls added in the post-fix A/B
    "129d1232": "control-sum", "2b8f3739": "control-sum", "4adc0475": "control-sum",
    "3fdac837": "control-sum", "f35224e0": "control-sum",
}


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                out[r["question_id"]] = r
            except Exception:
                pass
    return out


def verdict(r):
    return (r.get("verdict") or (r.get("evaluation") or {}).get("result") or "").upper()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--base", required=True)
    args = ap.parse_args()
    new = load(Path(args.new))
    base = load(Path(args.base))

    print(f"\n{'='*92}")
    print(f"{'qid':<16}{'role':<11}{'base':<10}{'new':<10}{'NS fired':<22}outcome")
    print("=" * 92)
    wins = losses = collateral = nofire = 0
    order = sorted(new, key=lambda q: (list(ROLE).index(q) if q in ROLE else 99))
    for qid in order:
        r = new[qid]
        role = ROLE.get(qid, "?")
        bv = verdict(base.get(qid, {})) or "—"
        nv = verdict(r)
        pat = r.get("symbolic_pattern") or ""
        fired = pat if pat.startswith("neural_symbolic") else "(no)"
        if fired == "(no)":
            nofire += 1
        outcome = "no-change"
        bc, nc = bv == "CORRECT", nv == "CORRECT"
        if not bc and nc:
            outcome = "WIN ✅"
            wins += 1
        elif bc and not nc:
            outcome = "LOSS ❌"
            losses += 1
            if fired != "(no)":
                outcome = "COLLATERAL ⚠️"
                collateral += 1
        ans = (r.get("hypothesis") or "")[:24].replace("\n", " ")
        print(f"{qid:<16}{role:<11}{bv:<10}{nv:<10}{fired:<22}{outcome}   got={ans!r}")
    print("=" * 92)
    print(f"WINS={wins}  LOSSES={losses} (of which COLLATERAL={collateral})  "
          f"NS-did-not-fire={nofire}/{len(new)}")
    # control collateral specifically
    ctrl = [q for q in new if ROLE.get(q) == "control"]
    ctrl_broke = [q for q in ctrl if verdict(base.get(q, {})) == "CORRECT" and verdict(new[q]) != "CORRECT"]
    print(f"CONTROLS: {len(ctrl)} total, {len(ctrl_broke)} broken -> {ctrl_broke}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
