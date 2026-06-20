# Resume Point

## Last Updated
2026-06-20

## Current State (2026-06-20) — neural-symbolic MS agent: built, validated, SHELVED (OFF)
The LongMemEval MS count/sum/arithmetic **neural-symbolic agent** (`benchmarks/longmemeval/neural_symbolic.py`,
`--neural-symbolic` **default OFF**) is complete and `$0`-tested but **left OFF**: two paid A/B smokes + a
full-MS static projection show it is **NOT net-positive at full scale** (collateral surface 47 = 1.6× the
29 win opportunities; projected ~68–76% ≤ 75.9% baseline at the measured ~0.3–0.4 collateral rate). The
correct compute/classifier fixes it carries stay; the render was reverted to the v1 lower-bound floor after
the post-review "two-directional" change net-regressed the count wins. **Do NOT enable it** (see the
CLAUDE.md Critical rule). Rollback tag `iter33-ms-pre-symbolic`. Full account: `benchmarks/longmemeval/ITER33_MS_STATUS.md`,
memory `longmemeval-neural-symbolic-agent`. The higher-value MS lever remains entity-aware retrieval (M-probes).

## Current Phase
**iter30 cleanup branch** — codebase + prompt rationalization on top of the iter29 baseline. Targeting the LongMemEval-S benchmark via the `longmemeval-iter` lineage.

The cleanup is now complete on branch `iter30_cleanup`. Pending a smoke + full N=500 verification run before merging back to `longmemeval-iter`.

## Completed in This Session
- Cherry-picked CLAUDE.md, README.md, AGENT_PROTOCOL.md, COGNITION_PRINCIPLES.md, CONTRIBUTING.md, CHANGELOG.md, RESUME.md, PHASES.md from `benchmark-suli`.
- Ported `.claude/skills/{cognifold-dev, doc-guard, cognifold-create-skill}/` plus `.claude/commands/sync-skills.md`.
- Ported `.claude/hooks/pre-commit-docguard.sh` and wired it via `.claude/settings.json` (PreToolUse → Bash).
- iter30 cleanup branch landed: 9 commits, net −332 lines.
  - Removed dead code paths from iter29a: `build_calendar_block`, `build_days_ago_chart`, `build_structured_question_parse`, TR-κ 2-pass reader, plus 5 zombie CLI flags (−281).
  - Disabled 0%-acc resolver patterns (`count_among`, `order_among`) from the dispatch list.
  - Trimmed `BATCH_SYSTEM_PROMPT` from 7 rules to 4 (dropped verb-precision, SSA verbatim, serial/count).
  - Compressed `qa_answer` from 267 to 119 lines (worked examples folded, REFLECTOR MARKERS section merged into KNOWLEDGE UPDATES).
  - Removed the `(meaning DATE)` strip on MS/TR — reader now sees the absolute date anchor for all question types.
  - Added W3 START extraction pass (`_extract_start_events_pass`, `--extract-start-events`, `AgentConfig.extract_start_events`) — replaces the unreliable iter29 TR-NEW-1 writer-prompt rule.
  - Dropped reflector's STARTS section (W3 owns it now).
  - Added `scripts/run_iter30.sh` launcher with `--extract-start-events` ON.

## In Progress
Documentation rationalization — porting the docs/skills/hook harness from `benchmark-suli` onto `iter30_cleanup` so the LongMemEval iteration work follows the same quality gates as the rest of the project.

## Next Steps
1. Run a 1-qid smoke through `scripts/run_iter30.sh` once an API key with balance is available (NTU sk-proj is at `insufficient_quota`, commonstack `ak-bb5d5...` shows balance=0).
2. Run the full N=500 via the iter30 launcher.
3. Write `benchmarks/longmemeval/runs/iter30_*/CHANGES.md` per the lme_iter_folder convention.
4. Merge `iter30_cleanup` back into `longmemeval-iter`.
5. Resume LongMemEval iter improvement work — the next obvious lever is per-pattern W2 policy (currently per-type, but evidence shows the resolver path benefits and the reader path benefits unevenly across patterns).

## Branch Layout
- `longmemeval-iter` (HEAD `a088a27`) — stable baseline, iter27 stack validated at 86.80% N=500 strict.
- `iter30_cleanup` (HEAD `c38b604`) — current working branch. Cherry-picks iter29 work, then layers the iter30 cleanup commits on top.

## Recent Run Artifacts
- `benchmarks/longmemeval/runs/iter29c_tr_targeted/` — 63 / 500 records (run aborted on NTU/commonstack quota exhaustion). Apples-to-apples vs iter27 same qids: **+3.2 pp overall**, **+8.7 pp TR**, **+16.7 pp KU**. Validates that the iter29 direction is correct.
- `benchmarks/longmemeval/runs/iter29a_failed_n500/` — 73 / 500 records, archived as a regression case study (−11.6 pp vs iter27 — see CHANGELOG entry).
- `benchmarks/longmemeval/runs/iter27_gpt54mini_full_n500_W1W2/` — the 86.80% baseline used for all apples-to-apples comparisons.
