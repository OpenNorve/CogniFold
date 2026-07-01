"""$0 simulation of the "run both, take the higher count" complementary router.

Idea (validated by the NS-vs-baseline cross-tab): for COUNT/SUM questions the MS
error mode is under-count-dominated, and the NS-vs-baseline disagreements split
cleanly by DIRECTION — every NS win is NS ratcheting the count UP, every observed
collateral is NS ratcheting DOWN. So a router that runs the normal iter reader AND
the NS pipeline and keeps the HIGHER count should keep the up-wins and structurally
kill the down-collateral.

This simulator replays cached run results (no LLM, $0) and computes, per count/sum
question the NS agent fired on:
  - direction = sign(ns_count - baseline_count)
  - take_max verdict = NS-run verdict if NS >= baseline, else baseline verdict
and tallies vs the normal-iter baseline alone:
  FIX        baseline wrong -> take_max right   (the win we keep)
  BREAK      baseline right -> take_max wrong   (over-count collateral — the risk)
  killed     a baseline-right/NS-down case where take_max correctly keeps baseline

The BREAK count is the key unknown: take_max trades the (observed) under-count
collateral for an over-count exposure. This measures it on whatever controls the
cached runs contain (small N — a bigger control set needs a live run).

Run:
  PYTHONPATH=src:. .venv/bin/python benchmarks/longmemeval/ns_take_max_sim.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import neural_symbolic as ns

BASE = Path(__file__).parent / "runs"


def load(p: Path) -> dict:
    d = {}
    if not p.exists():
        return d
    with open(p) as fh:
        for line in fh:
            try:
                r = json.loads(line)
                d[r["question_id"]] = r
            except Exception:
                pass
    return d


def answer_count(text) -> float | None:
    """Best-effort extraction of the answer's headline count/total from prose.

    Order: a bolded **N** (the reader almost always bolds the final figure) ->
    a number right after a count/total verb -> a leading word-number -> first
    standalone digit that is not a list index ("Item 1") or inside a $-sum tail.
    """
    text = str(text)
    m = re.search(r"\*\*\s*\$?\s*([\d,]+(?:\.\d+)?)", text)
    if m:
        return ns.to_number(m.group(1))
    m = re.search(
        r"(?:attended|visited|used|have|own|completed|got|took|earned|raised|spent|total of|total)\s+"
        r"(?:about\s+|over\s+|a\s+total\s+of\s+|approximately\s+)?\$?([\d,]+(?:\.\d+)?)",
        text, re.I,
    )
    if m:
        return ns.to_number(m.group(1))
    m = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", text[:80], re.I)
    if m:
        return ns.to_number(m.group(1))
    for mm in re.finditer(r"([\d,]+(?:\.\d+)?)", text):
        pre = text[max(0, mm.start() - 6):mm.start()].lower()
        if "item" in pre or "#" in pre:
            continue
        return ns.to_number(mm.group(1))
    return None


def main() -> int:
    base = load(BASE / "iter33_ms_clean" / "hypothesis.jsonl")
    # Prefer the v1 floor run (best NS config); fill gaps from the ab22 run.
    nsrun: dict = {}
    for name in ("ns_smoke17_v1", "ns_ab22_v1"):
        for q, v in load(BASE / name / "hypothesis.jsonl").items():
            nsrun.setdefault(q, v)

    fix, brk, killed, nochange = [], [], [], []
    table = []
    for qid, nr in nsrun.items():
        fam = ns.classify_question(nr.get("question", ""))
        if not fam or fam.name != ns.FAMILY_ENUMERATE_SUM:
            continue  # take-max only defined for count/sum
        if not (nr.get("symbolic_pattern") or "").startswith("neural_symbolic"):
            continue
        b = base.get(qid, {})
        ca = answer_count(b.get("hypothesis"))
        cb = answer_count(nr.get("hypothesis"))
        bv = (b.get("verdict") or "").upper() == "CORRECT"
        nv = (nr.get("verdict") or "").upper() == "CORRECT"
        if ca is None or cb is None:
            direction = "?"
            tmv = bv or nv  # unknown direction: be charitable, flag below
        elif cb > ca:
            direction = "UP"
            tmv = nv          # take the NS pipeline's (higher) answer
        elif cb < ca:
            direction = "DOWN"
            tmv = bv          # keep the baseline's (higher) answer
        else:
            direction = "="
            tmv = nv          # equal -> same either way
        table.append((qid, fam.mode, ca, cb, direction, bv, nv, tmv))
        if not bv and tmv:
            fix.append(qid)
        elif bv and not tmv:
            brk.append(qid)
        elif bv and direction == "DOWN":
            killed.append(qid)
        else:
            nochange.append(qid)

    print(f"\n{'='*86}\nTAKE-MAX (run both, keep higher count) — $0 cached simulation\n{'='*86}")
    print(f"{'qid':<16}{'mode':<6}{'base#':<8}{'NS#':<8}{'dir':<6}{'baseOK':<8}{'NS-runOK':<10}{'take-maxOK'}")
    for qid, mode, ca, cb, d, bv, nv, tmv in sorted(table, key=lambda r: (r[4], r[0])):
        print(f"{qid:<16}{mode:<6}{ca!s:<8}{cb!s:<8}{d:<6}{bv!s:<8}{nv!s:<10}{tmv}")
    print(f"\n{'-'*86}")
    print(f"vs normal-iter baseline alone, over the {len(table)} count/sum questions NS fires on:")
    print(f"  FIX   (baseline wrong -> take-max right): {len(fix)}  {fix}")
    print(f"  BREAK (baseline right -> take-max wrong, OVER-COUNT collateral): {len(brk)}  {brk}")
    print(f"  killed-down-collateral (NS went down, take-max kept baseline)  : {len(killed)}  {killed}")
    print(f"  no-change: {len(nochange)}")
    n_ctrl = sum(1 for *_ , bv, nv, tmv in table if bv)
    print(f"\n  NET on count/sum = +{len(fix)} fixes, -{len(brk)} breaks  (= {len(fix) - len(brk):+d}).")
    print(f"  over-count BREAK rate on the {n_ctrl} currently-correct controls present: "
          f"{len(brk)}/{n_ctrl} = {(len(brk)/n_ctrl*100 if n_ctrl else 0):.0f}%  "
          f"(SMALL N — the residual unknown; needs a live run on more controls).")
    print('='*86)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    raise SystemExit(main())
