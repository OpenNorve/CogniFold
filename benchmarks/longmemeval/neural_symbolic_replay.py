"""Replay the neural-symbolic agent against CACHED retrieved context.

Why: the deterministic compute layer is proven ($0, neural_symbolic_selftest.py).
The only unproven piece is OPERAND SELECTION — whether the extraction LLM call
reliably enumerates the operands from real context. Re-ingesting all 133 MS qids
to test that is ~$1+. Instead we replay against the cached `retrieved_context` /
`full_context` already stored in a prior run's hypothesis.jsonl, so we pay ONLY
for the extraction call (~$0.003/qid), with no re-ingestion.

Two modes:
  --dry   (default, $0): for each in-family fixture, report whether the GOLD
          operands appear in the cached context (operand-presence), to estimate
          the ceiling of a retrieved-turns extraction. No LLM.
  --live  (paid): run the REAL extraction LLM call on the cached context, parse,
          compute, and compare to GT. Prints token usage at the end.

Usage:
  # $0 yield estimate
  PYTHONPATH=src .venv/bin/python benchmarks/longmemeval/neural_symbolic_replay.py \
      --cache benchmarks/longmemeval/runs/iter33_ms_clean/hypothesis.jsonl

  # paid extraction-only validation (loads .env for keys)
  PYTHONPATH=src .venv/bin/python benchmarks/longmemeval/neural_symbolic_replay.py \
      --cache benchmarks/longmemeval/runs/iter33_ms_clean/hypothesis.jsonl --live
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import neural_symbolic as ns
from neural_symbolic_selftest import FIXTURES, _num_match


def _key_token(label: str) -> str:
    """Most distinctive (longest) alpha token of a label, for presence checks."""
    toks = [t for t in re.findall(r"[a-z]+", label.lower()) if len(t) > 2]
    toks = [t for t in toks if t not in {
        "the", "and", "old", "new", "with", "pickup", "return", "bedroom",
    }]
    return max(toks, key=len) if toks else ""


def load_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            cache[r["question_id"]] = r
    return cache


def dry_report(cache: dict[str, dict]) -> None:
    print(f"\n{'='*78}\nDRY operand-presence analysis (cached context, $0)\n{'='*78}")
    print(f"{'qid':<16}{'fam':<14}{'present/total':<14}{'ceiling':<10}cached?")
    full = present_full = 0
    for fx in FIXTURES:
        qid = fx["qid"]
        if fx["fam"] is None or not fx.get("parsed"):
            continue
        rec = cache.get(qid)
        cached = "yes" if rec else "MISSING"
        ctx = ((rec or {}).get("full_context") or "") + " " + ((rec or {}).get("retrieved_context") or "")
        ctx_low = ctx.lower()
        items = fx["parsed"].get("items", []) if isinstance(fx["parsed"], dict) else []
        if items:
            tot = len(items)
            pres = 0
            for it in items:
                kt = _key_token(it.get("label", ""))
                if kt and kt in ctx_low:
                    pres += 1
            ceil = "FULL" if pres == tot else f"-{tot-pres}"
            if rec:
                full += 1
                if pres == tot:
                    present_full += 1
        else:
            # non-enumerate families: check the named operands loosely
            tot = pres = 0
            ceil = "n/a"
        disp = " (disputed)" if fx.get("disputed") else ""
        print(f"{qid:<16}{fx['fam']:<14}{f'{pres}/{tot}':<14}{ceil:<10}{cached}{disp}")
    print(f"\nOf {full} enumerate fixtures present in cache, {present_full} have ALL gold "
          f"operands in the cached context (retrieved-turns extraction ceiling).")
    print("NOTE: this cache predates the M1-M5 retrieval probes, so it UNDER-estimates "
          "what the current retrieval surfaces; treat as a lower bound.\n")


def live_report(cache: dict[str, dict], provider: str = "openrouter",
                model: str = "", effort: str = "medium") -> None:
    # Lazy imports — only needed for the paid path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from benchmarks.longmemeval.run_eval import call_llm
    from cognifold.agent.config import AgentConfig

    # Load .env for keys.
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    if provider == "commonstack":
        key = os.environ.get("COMMONSTACK_API_KEY")
        base_url = "https://api.commonstack.ai/v1"
        model_name = model or "openai:openai/gpt-5.4-mini"
    else:  # openrouter
        key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = "https://openrouter.ai/api/v1"
        model_name = model or "openai:openai/gpt-4o-mini"
    if not key:
        print(f"ERROR: missing API key for provider={provider}", file=sys.stderr)
        sys.exit(2)
    writer_config = AgentConfig(
        model_name=model_name,
        api_key=key,
        base_url=base_url,
        max_tokens=2048,
        reasoning_effort=effort,
    )

    print(f"\n{'='*78}\nLIVE extraction replay (real LLM on cached context)\n{'='*78}")
    print(f"  writer/extractor: {writer_config.model_name} @ {writer_config.base_url}")
    exact = disputed = miss = skipped = 0
    rows = []
    for fx in FIXTURES:
        qid = fx["qid"]
        if fx["fam"] is None:
            continue
        rec = cache.get(qid)
        if not rec:
            skipped += 1
            continue
        question = rec.get("question") or fx["q"]
        evidence = (rec.get("full_context") or rec.get("retrieved_context") or "")
        if not evidence:
            skipped += 1
            continue
        res = ns.resolve_neural_symbolic(
            question, nodes=None, call_llm_fn=call_llm, config=writer_config,
            evidence_text=evidence, debug=True,
        )
        ans = (res or {}).get("answer")
        ok = bool(res) and _num_match(fx["gt"], ans)
        if ok:
            exact += 1
            tag = "✓"
        elif fx.get("disputed"):
            disputed += 1
            tag = "~disp"
        else:
            miss += 1
            tag = "✗"
        dbg = (res or {}).get("_debug", {})
        rows.append((tag, qid, fx["gt"], ans, dbg.get("n_items"), (res or {}).get("pattern")))

    for tag, qid, gt, ans, n, pat in rows:
        print(f"  {tag:<6}{qid:<16} GT={gt!r:<14} got={ans!r:<16} n={n} [{pat}]")
    print(f"\nLIVE: {exact} exact, {disputed} disputed-GT, {miss} miss, {skipped} skipped (not in cache)")
    # Cost.
    try:
        from benchmarks.longmemeval.run_eval import _LLM_CALLS  # type: ignore
        print(f"  (llm calls recorded: {len(_LLM_CALLS) if _LLM_CALLS else 0})")
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="path to a hypothesis.jsonl with cached context")
    ap.add_argument("--live", action="store_true", help="run real extraction LLM (paid)")
    ap.add_argument("--provider", default="openrouter", choices=["openrouter", "commonstack"])
    ap.add_argument("--model", default="", help="override extractor model (e.g. openai:openai/gpt-5.4-mini)")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    args = ap.parse_args()
    cache = load_cache(Path(args.cache))
    print(f"Loaded {len(cache)} cached records from {args.cache}")
    if args.live:
        live_report(cache, provider=args.provider, model=args.model, effort=args.effort)
    else:
        dry_report(cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
