# iter33-MS status (2026-06-19)

## Journey
- Baseline: iter19 gpt-5.4 reader = MS 78.2% (104/133); iter19 gpt-5-mini = 82.0% (109/133).
- iter33-ms v1 (Tier1 retrieval + Tier2 reader rules + Tier3 symbolic, W1 ON, ledger OFF): clean MS-133 = **75.9% (101/133)** — BELOW baseline. Per-qid: 12 iter19-failures flipped (retrieval/abstention/compute fixes all work), but 20 regressions, dominated by COUNT-UNDERCOUNT.
- Root cause: the D-CONSOLIDATED reader rule's "merge same host/date + canonical-kind dedup" made the reader OVER-merge distinct items → ~12 undercounts (citrus 3→2, festivals 4→3, weddings 3→2, devices 4→3...). Classic precision/recall backfire (fixed 2-3 overcounts, broke ~12 undercounts).
- Surgical revert (keep "don't split one entity" for d23cf73b; remove the over-merge; add "distinct items count SEPARATELY"): targeted 32-qid check (12 flips + 20 regressions) = **15/20 regressions recovered, 9/12 flips kept** (the 3 "lost" verified as reader/judge noise on stochastic abstention/borderline cases, NOT caused by the revert).

## Current projection
- Net +12 over the 75.9% clean run → **projected full-133 MS ≈ 113/133 = 85.0%** (vs 78.2 baseline, +6.8pp). NOT yet confirmed on full 133 (banked on the targeted-subset projection).

## What works (keep)
R1 bridge-entity 2nd hop + R2 category force-include + R1-age probe (retrieval-first, the headroom); A1/A2 calibrated abstention; D-COMPUTE operand-gated; T1 role-tenure; count_among bypass=False+dedup (symbolic); W1 typed-attr.

## What backfired (removed)
D-CONSOLIDATED over-merge dedup. Lesson: do NOT reader-over-tune counting — iter19 counted better without dedup rules.

## Next: failure-driven iteration on the remaining ~17 cases (see ITER33_MS_NEXT_FAILURES.md).

---

## Neural-symbolic computation agent (2026-06-20) — EXPERIMENTAL, OFF by default, SHELVED

Built a focused-extraction + deterministic-compute agent (`neural_symbolic.py`,
`--neural-symbolic` **default OFF**, launcher env `NEURAL_SYMBOLIC_FLAG`) targeting
the count/sum/arithmetic MS failures. See the CLAUDE.md Critical rule before touching.

**Verdict: NOT net-positive at full scale → kept OFF as a tagged scaffold.**
- Mechanism works: reader DOES adopt a concrete enumerated hint; raw-turn reading
  recovers writer-gap operands (e3038f8c "25 coins"→99, 67e0d0f2 "8 edX"→20);
  percent_diff works.
- But the blast radius is structurally unfavorable: fires on ~76/133, collateral
  surface (47 currently-correct) = 1.6× win opportunity (29). At the measured
  collateral rate (~0.3–0.4), full-MS projection ≈ 68–76% (≤ 75.9% baseline).
- Live A/B: v1 (lower-bound-floor render) = 8 NS wins / 2 collateral on 17 qids;
  the post-review "two-directional render" fixes REGRESSED it (5 losses, 0 gains)
  → reverted the render to the floor. Compute (`to_number` minutes-bug, SUM
  dedup max→sum, `_norm_label` over-merge, compare vs→Yes/No) + classifier
  exclusions (`_NOT_ENUM_RE`) are correct and $0-tested; those stay.
- Rollback tag `iter33-ms-pre-symbolic`. $0 harnesses: neural_symbolic_selftest.py,
  ns_static_analysis.py, ns_smoke_compare.py, neural_symbolic_replay.py.
- Re-enable ONLY after a controlled A/B shows collateral rate < ~0.15.
