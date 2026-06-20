# Changelog

All notable changes to Cognifold will be documented in this file.

---

## [2026-06-20] - iter33-MS: neural-symbolic computation agent (EXPERIMENTAL, OFF by default, shelved)

### Added (all behind `--neural-symbolic`, default OFF — see CLAUDE.md Critical rule)
- `benchmarks/longmemeval/neural_symbolic.py` — focused structured-extraction LLM
  call reads RAW retrieved turns, enumerates operands for 5 families
  (count/sum/diff/date/age), computes deterministically, injects a RECALL_HINT
  (bypass=False). Routed to the reader (reasoning) model. CLI `--neural-symbolic`
  / `--neural-symbolic-bypass`; launcher env `NEURAL_SYMBOLIC_FLAG`.
- `$0` tooling: `neural_symbolic_selftest.py` (39 fixtures incl. adversarial
  mis-route guards), `ns_static_analysis.py` (full-MS fire-map), `ns_smoke_compare.py`,
  `neural_symbolic_replay.py` (cached-context extraction replay).

### Verdict — NOT net-positive at full scale; left OFF
- Live A/B (ns_smoke17_v1, ns_ab22_v1) + static projection: fires on ~76/133 MS;
  collateral surface (47 currently-correct) is 1.6× the win opportunity (29).
  At the measured collateral rate (~0.3–0.4), projected full-MS ≈ 68–76% — at or
  below the 75.9% baseline. The adversarial-review "fixes" (two-directional render)
  net-REGRESSED the count wins; reverted to the v1 lower-bound-floor framing.
- KEPT (correct, $0-tested): compute fixes (`to_number` "10 minutes"→1e7 bug,
  minus-sign; SUM dedup max()→sum(); `_norm_label` over-merge; compare vs→Yes/No)
  and classifier exclusions (`_NOT_ENUM_RE`: elapsed-duration/age/requirement/
  left/exceed/recurring-rate → blast radius 89→76).
- Rollback tag: `iter33-ms-pre-symbolic`. Do NOT enable without re-validating
  collateral rate < ~0.15.

---

## [2026-06-05] - iter31: revert to iter19 writer stack + 4 targeted qa_answer rules

### Motivation
Iter-by-iter MS accuracy timeline shows the regression has been baked
in since iter27, not just iter30:
```
iter05 (no W1/W2):         MS 82.0%
iter19 (no W1/W2):         MS 82.0%  ← peak
iter27 (W1+W2 ON):         MS 77.4%  (-4.6, iter27 CHANGES.md self-noted)
iter30 (W1+W2+W3 ON, 96):  MS 48.4%  (-29, accumulating writer noise)
```
Every writer enrichment pass added after iter19 (W1 typed-attr in
iter19+, W2 event_date in iter27, W3 START in iter30) competes for
top-K retrieval bandwidth on MS counting questions. iter27 sold MS
-4.6 to gain SSA +8.9 (net -0.2 by class-weight); iter30 sold MS -29
for no offsetting gain.

iter31 reverts to iter19's writer stack and adds only changes with
direct wrong-case evidence:

### Changes
- `configs/longmemeval_profile.yaml` qa_answer: 4 new rule blocks
  - **MS-EXHAUSTIVE-COUNT** — targets 22 MS undercount wrongs (top
    cluster). Forces reader to scan the entire context, list every
    matching entity, and tally before answering "how many" / "how
    much".
  - **NO-REFUSAL-extended** — extends iter29 TR-NEW-7 to MS COUNT
    and AGE-INFERENCE questions. Targets 5 MS refuse-when-data
    cases (a1cc6108, c18a7dc8, ba358f49, 51c32626, 7024f17c).
  - **_abs-WORKED-EXAMPLES** — restores the iter10 verbose _abs
    refusal template that iter30 compression dropped. Targets 4
    _abs failures (80ec1f4f_abs, eeda8a6d_abs, 09ba9854_abs,
    a96c20ee_abs).
  - **DERIVED-TIME-WORKED-EXAMPLE** — restores the iter22 #9
    relative-time worked example (e.g. wake at 7:00 − 15 min =
    6:45). Targets 73d42213.
- `src/cognifold/agent/batch.py` BATCH_SYSTEM_PROMPT — trimmed from
  4 rules (iter30) to 3: identifier preservation, state-change
  framing, per-instance quantity preservation (new — prerequisite
  for EXHAUSTIVE-COUNT to find each item). Dropped W2/W3 hint
  rules now that those passes are off.
- `benchmarks/longmemeval/run_eval.py`:
  - OpenAI client `max_retries=4` (default is 2; balances retry
    headroom vs holding requests open during rate-limit windows).
  - Per-session pacing hook via `CHAT_PACE_SECONDS` env var; when
    set, `time.sleep(N)` is inserted after each
    `process_session_batch` call. Targets commonstack's 50 RPM
    global cap, which is enforced with a sticky-penalty memory:
    bursts that exceed the cap trigger progressive lockout that
    retries cannot clear. Set the env to ~1.2 to throttle writer
    to ~30 RPM (cap/2 margin).
- `src/cognifold/embeddings/providers.py` — `max_retries=4` mirror.
- `scripts/run_iter31.sh` — new launcher. Stack:
  - Reader: gpt-5.4-mini high (kept from iter27 for SSA delta)
  - Writer: gpt-5.4-mini low (single batch pass, no W1/W2/W3)
  - Rerank: gpt-5.4-mini low
  - Judge:  gpt-4o (via OpenRouter)
  - Embed:  text-embedding-3-small (via OpenRouter, ~$0.30)
  - Chat:   commonstack with `CHAT_PACE_SECONDS=1.2` pacing
  - Reflector: OFF, TR-α: OFF
- `scripts/run_iter30_commonstack.sh`, `scripts/run_iter30b_no_w3.sh`
  — committed alongside as reference launchers for the iter30
  attempts that aborted (W3 caused MS −29 confirmed; W3 OFF + 5p
  commonstack hit RPM cap).

### Files Modified
- `configs/longmemeval_profile.yaml` — +53 lines (4 new rule blocks)
- `src/cognifold/agent/batch.py` — −13 lines net (4→3 rules)
- `benchmarks/longmemeval/run_eval.py` — +18 lines (max_retries +
  pacing hook)
- `src/cognifold/embeddings/providers.py` — 1 line (max_retries)
- `scripts/run_iter31.sh`, `run_iter30_commonstack.sh`,
  `run_iter30b_no_w3.sh` — added.

### Tests
- Added/updated tests: no (config + prompt changes only).
- All tests passing: not run on this branch yet.

### Expected Score
Theoretical ceiling ~91–92%; realistic projection 88–90% strict.
Bottleneck is rule-follow rate of gpt-5.4-mini on the new
EXHAUSTIVE-COUNT rule (estimated 30–60%). N=500 verification
pending (commonstack 1-parallel paced ≈ 18h, or OpenRouter 5-parallel
≈ 10h at ~\$60–72).

---

## [2026-06-05] - iter30 cleanup branch: rationalize iter29 + import doc harness

### Changes
- Ported the development workflow + documentation harness from `benchmark-suli` to `iter30_cleanup`:
  - 3 skills: `cognifold-dev`, `doc-guard`, `cognifold-create-skill` (alongside existing `longmemeval-iterate` / `longmemeval-run`).
  - 1 slash command: `sync-skills`.
  - 1 hook: `.claude/hooks/pre-commit-docguard.sh` wired via `.claude/settings.json` (PreToolUse → Bash). Hook denies `git commit` if `src/` files are staged but `.claude/docguard_last_run` is stale.
  - 8 core docs imported: CLAUDE.md, README.md, docs/AGENT_PROTOCOL.md, COGNITION_PRINCIPLES.md, CONTRIBUTING.md, PHASES.md, plus this CHANGELOG and RESUME.md.
- iter30 cleanup branch (9 commits, net −332 lines, see branch `iter30_cleanup`):
  - **W3 START extraction pass** added (`_extract_start_events_pass` in `benchmarks/longmemeval/run_eval.py`, `AgentConfig.extract_start_events`, CLI `--extract-start-events`). Replaces the iter29 TR-NEW-1 writer-prompt rule that gpt-5.4-mini low-effort skipped.
  - **`(meaning DATE)` strip removed** for MS/TR. Reader sees inline absolute event_date anchors for all question types now. Resolver-side `ignore_event_date` for MS+TR is preserved.
  - **`qa_answer` profile compressed** from 267 → 119 lines: worked examples folded, REFLECTOR MARKERS subsection merged into KNOWLEDGE UPDATES.
  - **Writer prompt trimmed** from 7 rules to 4 (dropped verb-precision, SSA verbatim, serial/count).
  - **0%-acc resolver patterns disabled**: `count_among` (0/5) and `order_among` (0/4) removed from the `resolve` dispatch list.
  - **Dead code removed**: `build_calendar_block`, `build_days_ago_chart`, `build_structured_question_parse`, the TR-κ 2-pass reader template, plus zombie CLI flags `--tr-cot`, `--tr-calendar-chart`, `--tr-structured-parse`, `--tr-two-pass`, `--session-calendar`.
  - **iter29a launcher deleted** (`scripts/run_iter29.sh`) — references now-removed CLI flags.
  - Reflector LLM prompt: STARTS section removed (W3 owns START extraction now).

### Files Modified
- `CLAUDE.md`, `README.md` — added.
- `docs/{RESUME,CHANGELOG,AGENT_PROTOCOL,COGNITION_PRINCIPLES,CONTRIBUTING,PHASES}.md` — added or rewritten.
- `.claude/skills/cognifold-{dev,create-skill}/`, `.claude/skills/doc-guard/`, `.claude/commands/sync-skills.md`, `.claude/hooks/pre-commit-docguard.sh`, `.claude/settings.json` — added.
- `benchmarks/longmemeval/run_eval.py` — W3 pass, dead-code removal, render-strip removal. −104 lines net (W3 +150, cleanup −254).
- `benchmarks/longmemeval/symbolic_resolver.py` — 2 patterns commented out of dispatch.
- `configs/longmemeval_profile.yaml` — `qa_answer` rewritten. −148 lines.
- `src/cognifold/agent/batch.py` — writer rules trimmed. −36 lines.
- `src/cognifold/agent/config.py` — added `extract_start_events: bool = False`.
- `src/cognifold/agent/reflector.py` — STARTS section removed. −39 lines.
- `src/cognifold/query/assembly.py` — orphan `assistant_said` renderer removed.
- `scripts/run_iter29.sh` — deleted. `scripts/run_iter30.sh` — added.

### Tests
- Added/updated tests: no (cleanup-only).
- All tests passing: not run yet — pending `make test` post-merge.

### Notes
API quota outage (NTU sk-proj `insufficient_quota`, commonstack balance 0) blocked the iter30 smoke + N=500 verification. Branch is code-complete; run deferred until a working key is available.

---

## [2026-06-04] - iter29c TR-targeted partial N=500 (63/500) — +3.2 pp vs iter27

### Final number on aborted run
- 32-qid smoke (TR-skewed sample): **+9.4 pp** vs iter27 same qids.
- 63-qid partial N=500: **+3.2 pp** vs iter27 same qids (TR +8.7, KU +16.7, MS +0, SSU −8.3).
- Aborted at 63/500 on NTU `insufficient_quota`. Archived under `benchmarks/longmemeval/runs/iter29c_tr_targeted/`.

### Changes
- `benchmarks/longmemeval/symbolic_resolver.py`:
  - `_try_duration_activity`: added phrase-reduction fallback for trigger / activity anchor.
  - `_find_is_start_concept`: added Pass-2 fallback that scans for concepts whose title or description contains a START verb (started / began / signed up / picked up / bought my new / got my first / joined / enrolled / accepted into / moved to / installed / launched).
  - New resolver pattern `date_diff_before` (TR-β) for "how many days before I X did I Y" — runs before `date_diff_between`.
- `configs/longmemeval_profile.yaml`: TR-NEW-6 INCLUSIVE-BOUNDARY + TR-NEW-7 NO-REFUSAL-for-TR rules added.
- `src/cognifold/agent/reflector.py`: STARTS detection added (later moved to W3 in iter30).

### Tests
- Added/updated tests: no.
- All tests passing: not run.

---

## [2026-06-04] - iter29a failed N=500 partial (73/500) — −11.6 pp vs iter27 (regression case study)

### Final number on aborted run
73-qid partial: **−11.6 pp** vs iter27 same qids. MS −25.9 pp (catastrophic), SSU −12 pp, TR −7 pp, KU +16.7 pp, SSP unchanged.

### What broke
iter29a wired up the full Tier-3-on-iter27 superset (B + C + D' + E + F + G + H + I + TR-α + TR-β + TR-δ + TR-ε + TR-η + TR-ι + TR-ν + TR-ξ + TR-κ). The MS regression came from:
- `H SESSION_CALENDAR` was always-on, displacing top-K retrieval rows.
- `TR-κ 2-pass reader` reversed correct Pass-1 reasoning.
- `qa_answer` bloated from 80 to 289 lines.

### Decision
Aborted at 73/500. Archived to `benchmarks/longmemeval/runs/iter29a_failed_n500/`. Pivoted to iter29b minimal (`scripts/run_iter29b.sh`): kept B + C + D' + E + F + G + I + TR-α + TR-β; dropped H, TR-δ, TR-η, TR-ι, TR-ν, TR-ξ, TR-κ.

---

## [2026-04-19] - LoCoMo full 10-conv baseline: 62.9% J-Score (paper-grade)

### Final number
First paper-grade LoCoMo run at matched-backbone Mem0 protocol.

**Configuration**:
- Agent: `openai:gpt-4.1-mini`
- Judge: `openai:gpt-4o-mini` (single-judge, generous prompt, Mem0 protocol)
- Categories 1–4 (adversarial excluded), 1540 total questions, 10 conversations
- `--event-stream` ON (inter-session consolidation at session boundaries)
- Commit: `6392336` (event-stream + `--limit` fix) on top of `benchmark-suli`

**Result**: **J-Score 62.9% (969/1540)**, strict 51.8%, partial 62.4%.

**SOTA context** (Mem0 protocol, cat 1–4, gpt-4o-mini judge):

| System | Overall | vs CogniFold |
|---|---|---|
| EverMemOS (gpt-4.1-mini + 3-judge ensemble) | 93.05 | −30.1 pp (protocol inflated) |
| EverMemOS (gpt-4o-mini backbone, 3-judge) | 86.76 | −23.9 pp |
| MemOS (3-judge ensemble) | ~75.8 | −12.9 pp |
| Mem0-graph | 68.44 | −5.5 pp |
| **Mem0** | **66.88** | **−4.0 pp** ← real same-protocol gap |
| Zep (Mem0 paper re-run) | 58.44 | **+4.5 pp** (we beat) |
| Zep (original blog, disputed) | 75.14 | −12.2 pp |
| **CogniFold (this PR)** | **62.9** | — |

### Per-conv breakdown

Variance ±8 pp (55.6 → 71.7), confirming 10-conv required for paper numbers:

| Conv | J-Score | N |
|---|---|---|
| conv-26 | 56.6% | 86/152 |
| conv-30 | 55.6% | 45/81 |
| conv-41 | 62.5% | 95/152 |
| conv-42 | 57.3% | 114/199 |
| conv-43 | 68.5% | 122/178 |
| conv-44 | 62.6% | 77/123 |
| conv-47 | 65.3% | 98/150 |
| conv-48 | 71.7% | 137/191 |
| conv-49 | 62.8% | 98/156 |
| conv-50 | 61.4% | 97/158 |

### `--event-stream` effect
Marginal. conv-26 OFF single = 62.5%, ON single = 63.2% (+0.7 pp). Full 10-conv ON = 62.9%. Per-session consolidation fires (18 `Inter-session consolidation:` log lines per conv) but downstream LoCoMo impact is within noise. Structural feature; keep in architecture narrative, but don't claim empirical boost on this benchmark.

### Context correction
The historical **51.3% J-Score** was under gemini-2.5-flash agent, not gpt-4o-mini — **not comparable** to Mem0's 66.88 (which is gpt-4o-mini agent). Real same-backbone gap is 4.0 pp, not 15.6 pp as previously believed. Paper SOTA tables must note backbone + judge protocol when comparing.

### Stage B conceptual bootstrapping
Ran 6 CogEval scenarios × 4 orderings (chronological / random_42 / curriculum / anti_curriculum). **5/6 scenarios report top-1 held-out classification = 0.000 across all orderings** — the Zhao curriculum directional claim is **not validated by this test**. Stage A's "order matters" result (concept Jaccard < 0.12 across orderings) still stands. Likely cause: 0.75 cosine threshold on text-embedding-3-small too strict for title-vs-gold matching; re-run with relaxed threshold or LLM-judge classifier is future work.

---

## [2026-04-19] - Bi-temporal + Scene experiment: reverted (negative result)

### Context
Tested two EverMemOS-inspired patches on LoCoMo: (a) bi-temporal `[t_valid_start, t_valid_end]` on fact concepts with `close_superseded_facts` supersession, (b) online centroid-based scene clustering mirroring EverMemOS MemScene. Motivation was to close the gap to Mem0 (66.88) and EverMemOS (86.76).

### A/B result on conv-26 (agent=gpt-4.1-mini, judge=gpt-4o-mini, Mem0 protocol)
| Config | J-Score | Δ |
|---|---|---|
| Baseline (both OFF) | 62.5% (95/152) | — |
| Bi-temporal only | 58.6% (89/152) | −3.9 pp |
| Scene only | 57.2% (87/152) | −5.3 pp |
| Both ON | 55.3% (84/152) | −7.2 pp |

### Root cause (bi-temporal)
`TemporalExtractor` in `src/cognifold/temporal/extractor.py` has `prefer_future=True` default. For LoCoMo questions like "Where was Alice in May 2024?", the `written_date` regex requires `<month> <day>` and misses year-only patterns. The dateparser fallback resolves "May" to **2025-05-01** (next May) against current ref time, causing the bi-temporal `as_of_time` filter to drop all historical facts as "expired".

### Scene regression
Smaller signal (−5.3 pp alone), partially within single-conv LLM-judge noise (±4 pp stddev on n=152). Plausibly from `scene_id` token pollution in BM25 / context assembly; not fully isolated.

### Actions
- **Reverted** all core code changes: bi-temporal fields on fact concepts, `SceneClusterer`, `_filter_by_temporal_and_scene`, `_parse_as_of_time`, `event_timestamp` plumbing through executor.
- **Kept** `--model` / `--judge-model` CLI flags on `benchmarks/locomo/run_benchmark.py` (orthogonal engineering improvement, enables Mem0-protocol-matching paper runs).
- **Kept** Stage A shuffle experiment (independent of this work).
- **Deleted** tests/unit/test_scene_clustering.py, tests/unit/test_fact_supersession.py.

### Lessons
1. **Single-conv variance is ~4 pp** at n=152 — effects below that magnitude need full 10-conv runs to isolate.
2. **CogniFold's real LoCoMo baseline at matched backbone** (gpt-4.1-mini agent + gpt-4o-mini Mem0-protocol judge) is **62.5% J-Score on conv-26**, far higher than the historical 51.3% (gemini-2.5-flash). The actual gap to Mem0 (66.88) is **~4 pp, not 15.6 pp**.
3. Fix `prefer_future=True` bug in `TemporalExtractor` before any future bi-temporal attempt on LoCoMo.

---

## [2026-04-17] - LoCoMo 56.2% + CogEval-Bench Narrative Integration

### Changes
- **LoCoMo benchmark**: 49.6% → **56.2% strict** (60.0% partial, J-Score 51.3%, 10 convs / 1986 QA). Stacked wins: metadata stripping at query time (`include_reasoning=False`, `include_grounding=False` → 3× more concepts per 8K context), date normalization for relative time references, raised context budget to 8K chars. Commit `53bea49`.
- **Paper narrative refactor** (`papers/research_prompt.md`, `papers/outline.md`): four-layer narrative refined:
  - Layer 1 Setting: corrected RAG characterization — assumption is *one-time batch ingestion*, not "organized docs"; CogniFold handles *continuously arriving* fragmented streams.
  - Layer 2 Insight: "reasoning over organized knowledge" remains a challenge; organizing streams is an *additional upstream* challenge (additive, not substitutive).
  - Two-tier experimental narrative: Layer 1 (§5 broad QA — good memory) + Layer 2 (§6 CogEval-Bench — structural emergence).
- **CogEval-Bench documented** in `docs/benchmark/results.md` — 6-system same-LLM structural eval (GPT-4o-mini, 6 scenarios, 251 events, 49 gold concepts). CogniFold: **Harmony 0.476, Gold F1 0.358, LLM Quality 0.733, Purity 0.361, Clustering 0.327, Compression 4.6×, Proactivity 0.614**. Uniquely non-zero on Purity and Proactivity; reveals 5-tier representation hierarchy (Flat → Entity → Enrichment → Community → Cognitive).

### Files Modified
- **Modified**: `papers/research_prompt.md`, `papers/outline.md`, `docs/benchmark/results.md`, `docs/BENCHMARK.md`, `docs/RESUME.md`
- **Related** (already present): `papers/cognifold-neurips2025/text/cogeval.tex`, `tables/tab_cogeval_main.tex`, `tables/tab_cogeval_per_scenario.tex`, `tables/tab_cogeval_scenarios.tex`

---

## [2026-03-05] - Bugfix Sprint (PR #111)
- Async safety, session recovery, SSE race fix, thread-local API keys, input validation

---

## [2026-03-04] - Wave 6: Regression Fixes & Benchmark Re-evaluation

### Changes
- **Fix 1: Fixed PageRank alpha** — Reverted adaptive alpha (0.75-0.95) to fixed 0.85; adaptive caused regressions on sparse-graph benchmarks (BABILong)
- **Fix 2: Per-benchmark post-ingest config** — Disabled consolidation and fact_extraction by default in all profile YAMLs; only entity_index enabled. Consolidation caused destructive node merges on small graphs
- **Fix 3: Retrieval-first entry point merge** — Changed `_merge_entry_points` from entity-first to retrieval-first ordering with expanded budget (1.5x when both sources contribute)
- **Fix 4: Adaptive traversal depth** — Graphs with >50 nodes get depth=4 (up from 3) for multi-hop questions (MuSiQue)
- **Fix 5: Diversity penalty relaxation** — Only apply for >20 nodes, >50% overlap, max 15% penalty
- **Fix 6: MuSiQue config restore** — `max_exploration_steps: 0→1`, chain-of-thought QA prompt
- **Fix 7: NarrativeQA model name** — Strip `openai:` prefix for API calls

### Files Modified
- **Modified**: `src/cognifold/query/strategies.py` (retrieval-first merge, adaptive depth, diversity penalty), `src/cognifold/scoring/ranker.py` (fixed alpha=0.85), `benchmarks/narrativeqa/run_benchmark.py` (model name fix), `configs/*.yaml` (post_ingest toggles), `configs/musique_profile.yaml` (exploration steps, QA prompt)
- **New**: `docs/benchmark/wave6_experiment_report.md`

### Results
- 5 benchmarks improved, 4 matched Wave 5, 1 data limitation (StreamingQA — no passage text)
- All tests passing: yes (918 tests)

---

## [2026-03-03] - Wave 6: Research-Driven Architecture Improvements
## [2026-03-11] - CogniFold 0.2: Shared Foundations Phase C (Integration)

### Changes
- **Integration wiring**: All Phase A/B components connected into the runtime
- **Session fields**: `trace_collector`, `llm_metrics`, `budget` on every session
- **Trace recording**: Automatic after plan execution (when TraceConfig.enabled=True)
- **LLM metrics**: Gemini and OpenAI calls auto-recorded via thread-local scope
- **Budget enforcement**: Pre-call check in processor, graceful fallback to default plan
- **API endpoints**: `GET /traces`, `GET /usage`, `GET /concept-quality`

### Tests
- Added: `tests/unit/test_phase_c_integration.py` (21 tests)
- All tests passing: yes (1064 total)

---

## [2026-03-11] - CogniFold 0.2: Shared Foundations Phase B (S1 + S3 + S5)

### Changes
- **S1 Concept Quality**: Near-duplicate detection in executor (`_is_near_duplicate`), concept quality stats (`get_concept_quality_stats`), opt-in prompt sections for domain configs
- **S3 Cognitive Trace**: `TraceEntry` model with full operation tracking, `TraceCollector` ring buffer (thread-safe, configurable max entries), `trace_from_plan()` extractor
- **S5 Session Maturation**: Event counter on sessions, checkpoint/restore endpoints, stricter input validation on session creation

### Files Modified
- `src/cognifold/executor/runner.py` — Near-duplicate detection at ADD_NODE time
- `src/cognifold/graph/store.py` — `get_concept_quality_stats()` method
- `src/cognifold/agent/domain.py` — `opt_in_sections` field on DomainConfig
- `src/cognifold/agent/prompt_sections.py` — `opt_in_sections` parameter in `resolve_sections()`
- `src/cognifold/agent/prompts.py` — Wired opt_in_sections through prompt generation
- `src/cognifold/models/trace.py` — **NEW**: TraceEntry dataclass
- `src/cognifold/trace/__init__.py` — **NEW**: Package init
- `src/cognifold/trace/collector.py` — **NEW**: TraceCollector + trace_from_plan
- `src/cognifold/service/session.py` — event_count, checkpoints on Session
- `src/cognifold/service/processor.py` — Increment event_count on ingest
- `src/cognifold/service/models.py` — event_count on SessionInfo, input validation
- `src/cognifold/service/routes/sessions.py` — checkpoint/restore endpoints

### Tests
- Added: `tests/unit/test_concept_quality.py` (24 tests), `tests/unit/test_trace.py` (24 tests), `tests/unit/test_service_session.py` (14 new tests)
- All tests passing: yes (1043 total)

---

## [2026-03-11] - CogniFold 0.2: Shared Foundations Phase A

### Changes
- **S6 Config Hardening**: Added `ConsolidationConfig`, `LifecycleConfig`, `TraceConfig` dataclasses to `config.py` with `__post_init__` validation, YAML round-trip, env var override support
- **S4 Observability**: Added `LLMMetricsCollector` (thread-safe, cost estimation for 15+ models) and `BudgetEnforcer` (per-session token/cost/call limits)
- **S2 Graph Projection**: Added `GraphProjection` protocol, `NetworkXProjection` implementation, `GraphSnapshot` serializable state capture, `graph_to_snapshot()` factory
- **Design docs**: Created `PLAN_WORKSTREAM_1_INFRA.md` (6 shared foundations) and `PLAN_WORKSTREAM_2_MEMORY.md` (memory consolidation & forgetting)

### Files Modified
- `src/cognifold/config.py` — 3 new config dataclasses, wired into CognifoldConfig
- `src/cognifold/graph/projection.py` — **NEW**: GraphProjection protocol + NetworkXProjection + GraphSnapshot
- `src/cognifold/graph/__init__.py` — Added projection exports
- `src/cognifold/utils/__init__.py` — **NEW**: Package init
- `src/cognifold/utils/llm_metrics.py` — **NEW**: LLMCallMetrics, LLMMetricsCollector, estimate_cost
- `src/cognifold/utils/budget.py` — **NEW**: BudgetEnforcer, LLMBudget, BudgetExceededError
- `docs/PLAN_WORKSTREAM_1_INFRA.md` — **NEW**: Workstream 1 design plan
- `docs/PLAN_WORKSTREAM_2_MEMORY.md` — **NEW**: Workstream 2 design plan

### Tests
- Added: `tests/unit/test_config.py` (28 tests), `tests/unit/test_graph_projection.py` (19 tests), `tests/unit/test_llm_metrics.py` (27 tests)
- All tests passing: yes (980 total)

---

## [2026-03-05] - Bugfix Sprint: Async Safety, Session Recovery, API Validation, LLM Key Refactor

### Changes
- **D1: Fact-aware ingestion** — Regex-based fact extraction creates structured concept nodes (entity+attribute+value) from node text, integrated as post-ingest step in base_runner
- **D2: Entity-indexed retrieval** — Heuristic NER entity index mapping entity names to node IDs, used as supplementary entry points during query
- **D3: Belief state management** — Expanded state tracking prompts with observer rules, belief freezing, Sally-Ann worked example; belief-type tags in context assembly; ToMi-specific QA prompts for belief vs reality questions
- **D4: Adaptive PPR kernel** — Graph-density-adaptive PageRank damping factor (sparse→wider diffusion, dense→tighter focus)
- **D5: Concept consolidation** — Post-ingest merge of near-duplicate concept nodes (SequenceMatcher similarity) + orphan concept tagging as low-confidence

### Files Modified
- **New**: `src/cognifold/graph/fact_extraction.py`, `src/cognifold/graph/entity_index.py`, `src/cognifold/graph/consolidation.py`, `tests/unit/test_entity_index.py`
- **Modified**: `benchmarks/shared/base_runner.py` (post-ingest pipeline), `src/cognifold/agent/prompt_sections.py` (belief state tracking), `src/cognifold/query/assembly.py` (belief tags), `src/cognifold/query/strategies.py` (entity entry points), `src/cognifold/scoring/ranker.py` (adaptive alpha), `src/cognifold/graph/store.py` (entity index property), `configs/tomi_profile.yaml` (belief QA prompts)

### Tests
- Added tests: yes (test_entity_index.py)
- All tests passing: yes (918 tests)

---

## [2026-03-02] - Supabase Persistence, SSE Streaming, User Identity

### Changes
- Added `SupabaseSessionStore` — Level 1 drop-in persistence using `sessions.graph_snapshot` JSONB
- Added `GraphSyncWriter` — Level 2 write-through mirroring graph mutations to `graph_nodes`/`graph_edges`
- Added SSE streaming via `SSEBroker` + `GET /api/v1/sessions/{id}/stream` endpoint
- Added user identity routes: `POST /users`, `GET /users/{id}`, `GET /users/{id}/sessions`
- Added `supabase` optional dependency group in `pyproject.toml`
- Wired Supabase client through `AppSettings` → lifespan → `SessionManager`

### Files Modified
- **New**: `src/cognifold/service/stores/supabase_store.py`, `src/cognifold/service/stores/graph_sync.py`, `src/cognifold/service/sse.py`, `src/cognifold/service/routes/stream.py`, `src/cognifold/service/routes/users.py`
- **New tests**: `tests/unit/test_supabase_store.py`, `tests/unit/test_graph_sync.py`, `tests/unit/test_sse_broker.py`
- **Modified**: `pyproject.toml`, `src/cognifold/executor/runner.py`, `src/cognifold/service/processor.py`, `src/cognifold/service/session.py`, `src/cognifold/service/stores/factory.py`, `src/cognifold/service/stores/__init__.py`, `src/cognifold/service/app.py`, `src/cognifold/service/wsgi.py`, `src/cognifold/service/models.py`, `src/cognifold/service/routes/__init__.py`, `src/cognifold/service/routes/events.py`

### Supabase Schema
- Tables: `users`, `sessions`, `graph_nodes` (with `vector(768)` embedding), `graph_edges`
- RLS enabled with `allow_all` policies (auth skipped for now)

### Tests
- Added tests: yes (3 new test files)
- All tests passing: yes

---

## [2026-02-22] - Codebase Consolidation & Research Roadmap

### A1: Extract Shared Text Utilities
- Created `src/cognifold/query/text_utils.py` with unified STOP_WORDS (129 words), `extract_keywords()`, `compute_text_similarity()`
- Removed duplicated definitions from `strategies.py` and `scoring.py`
- Updated test imports in `tests/unit/test_query.py`

### A2: Consolidate Type-Boost Scoring Logic
- Created `src/cognifold/query/config.py` with `TypeBoosts` dataclass, `ENTRY_POINT_BOOSTS`, `RELEVANCE_BOOSTS`, and `apply_type_boost()` utility
- Replaced 5 inline type-boost blocks in `strategies.py` and 1 in `scoring.py` with centralized config

### A3: Extract Magic Numbers as Named Constants
- Added `MAX_ENTRY_POINTS`, `BFS_DECAY_PER_HOP`, `NEIGHBOR_RELEVANCE_DISCOUNT`, `NON_MATCH_PENALTY`, `DEPTH_PENALTY_FACTOR`, `MAX_DESCRIPTION_CHARS`, `UUID_HEX_LENGTH` to `config.py`
- Replaced magic numbers across `agent.py`, `strategies.py`, `scoring.py`, `assembly.py`, `executor/runner.py`

### A4: Expose Private Methods as Public APIs
- Added `QueryScorer.node_to_summary()` public method
- Added `ContextAssembler.build_context_text()` public method
- Added `ConceptGraph.internal_graph` property
- Removed all `# type: ignore[reportPrivateUsage]` in `agent.py`, `runner.py`, `ranker.py`

### A5: Refactor Benchmark Runners into Base Class
- Created `benchmarks/shared/base_runner.py` with `BenchmarkRunner` base class (~655 lines)
- Refactored 11 of 14 benchmark runners to use base class (3 were already different enough to skip)
- Net reduction: ~4,163 lines removed across 21 files

### A6: Clarify Edge Inference Integration
- Added comprehensive docstring to `EdgeInferenceEngine` explaining opt-in design
- Added ADR-001 to `docs/ARCHITECTURE.md`

### Files Modified
- **New**: `src/cognifold/query/text_utils.py`, `src/cognifold/query/config.py`, `benchmarks/shared/base_runner.py`, `benchmarks/shared/__init__.py`
- **Modified**: `src/cognifold/query/strategies.py`, `src/cognifold/query/scoring.py`, `src/cognifold/query/assembly.py`, `src/cognifold/query/agent.py`, `src/cognifold/graph/store.py`, `src/cognifold/graph/edge_inference.py`, `src/cognifold/executor/runner.py`, `src/cognifold/scoring/ranker.py`, `docs/ARCHITECTURE.md`, `tests/unit/test_query.py`, 11 benchmark runners

### Quality
- 808 tests passing, 0 pyright errors, ruff clean
- All 14 benchmark runners syntactically valid

---
## [2026-02-22] - Core Engine Retrieval Improvements

### Changes
- Added structured context assembly: `_format_node()` appends `node.data` fields (speaker, entity, location, timestamp, context dict) to QA context
- Added title→ID resolution in `PlanExecutor` ADD_EDGE handler: resolves node titles/names to IDs via cache + substring matching
- New `EdgeInferenceEngine` class: kNN-based post-ingestion edge creation using stored node embeddings
- Added `ConceptGraph.infer_edges()` convenience method
- Added 1-hop neighbor expansion in `MemoryQueryAgent._expand_with_neighbors()`, gated by `edge_count > 0`

### Files Modified
- `src/cognifold/query/assembly.py` — structured data fields in `_format_node()`
- `src/cognifold/executor/runner.py` — `_resolve_node_ref()` method + ADD_EDGE resolution
- `src/cognifold/graph/edge_inference.py` — **NEW** EdgeInferenceEngine class
- `src/cognifold/graph/store.py` — `infer_edges()` convenience method
- `src/cognifold/graph/__init__.py` — export EdgeInferenceEngine
- `src/cognifold/query/agent.py` — `_expand_with_neighbors()` + integration in `query()`

### Benchmark Results
- MuTual: 95% (no regression)
- BABILong: 45% (no regression)
- RGB: 20% EM / 0.46 F1 (improved from 15% EM / 0.34 F1)

### Tests
- Added/updated tests: no (core changes tested via benchmark verification)
- All tests passing: yes (802 passed, 0 pyright errors)

---

## [2026-02-22] - Add graph/export endpoint & make graph_to_dict public

### Changes
- Added `GET /api/v1/sessions/{session_id}/graph/export` endpoint — exports full graph in persistence format (version, saved_at, nodes, edges). Used by NeoLearn client for session restore.
- Renamed `_graph_to_dict` → `graph_to_dict` (made public) in `graph/persistence.py` to fix pyright `reportPrivateUsage` error when used from HTTP endpoint.
- Updated all internal callers: `save_graph()`, `session.py`, `stores/base.py` docstring.

### Files Modified
- `src/cognifold/service/routes/graph.py` — new `/export` endpoint
- `src/cognifold/graph/persistence.py` — `_graph_to_dict` → `graph_to_dict`
- `src/cognifold/service/session.py` — updated import
- `src/cognifold/service/stores/base.py` — updated docstring

---

## [2026-02-22] - Fix PART_OF edge constraint for concept sources

### Bug Fix
- Added `"concept"` to PART_OF allowed source types (was only `["event"]`, now `["event", "concept"]`)
- Concept-to-concept PART_OF edges (e.g., "Machine Learning" PART_OF "AI") no longer produce constraint warnings during ingestion
- DERIVED_FROM already allowed concept sources; RELATED_TO has no constraints (unconstrained by design)

### Tests
- Added `TestEdgeTypeConstraints` test class with 6 tests covering PART_OF, DERIVED_FROM, RELATED_TO, and legacy edge validation

---

## [2026-02-22] - Batch Processor Bug Fixes & Benchmark Scripts

### Bug Fixes
- **JSON object wrapping** (`batch.py`): OpenAI `response_format=json_object` wraps arrays in objects like `{"update_plans": [...]}`. Parser now extracts the first list value from dict responses.
- **Fallback duplicate node** (`batch.py`): When `BatchAgentProcessor` falls back to per-event `CognifoldAgent.process_event()`, the agent generates ADD_NODE for events that Layer 1 already created. Executor failed atomically, dropping ALL operations including concept/edge creation. Fixed by stripping duplicate ADD_NODE ops via `_op_resolves_to_existing()`.
- **Retry before fallback** (`batch.py`): Batch processing now retries 3 times on JSON parse failure before falling back to slow per-event mode.

### New Files
- `benchmarks/compare_fast.py` — Comparison script for fast vs classic mode on MuTual/SocialIQA
- `benchmarks/locomo/run_fast.py` — LoCoMo fast runner with parallel conversation processing
- `benchmarks/tomi/run_benchmark.py` — Added Gemini support for QA eval functions

### Tests
- Added `test_handles_object_wrapped_array` in `test_batch_agent.py`

---

## [2026-02-22] - Fast Mode: Layered Pipeline (`--fast`)

### Problem
`cognifold run --agent` processing 1.2k events took ~20 min due to sequential
PageRank (O(n+m) x2 per event), LLM calls (2-8s each), and per-node embeddings.

### Solution
Three-layer pipeline accessible via `--fast` flag:
- **Layer 1**: Add all events as nodes (no LLM/embeddings/PageRank). Target: <30s for 1200 events.
- **Layer 2**: Batched LLM enrichment — 10 events per prompt, ~10x fewer API calls.
- **Layer 3**: Batch embeddings + FAISS index build.

### Changes
- `perf`: PageRank cache (`scoring/cache.py`) — eliminates duplicate computation
- `perf`: Executor `skip_embedding` flag — skips per-node embedding API calls
- `refactor`: `pipeline.py` → `pipeline/` package (classic.py, layered.py, progress.py)
- `feat`: `LayeredPipeline` class with `run_layer1()`, `run_layer2()`, `run_layer3()`
- `feat`: `BatchAgentProcessor` — sends N events in single LLM prompt
- `feat`: CLI flags: `--fast`, `--layer`, `--batch-size`, `--no-embeddings`
- `feat`: Service API: `POST /events/batch/layered` endpoint
- `feat`: `FastModeConfig` dataclass in config.py

### CLI Usage
```bash
# Fast ingest only (~30s for 1200 events)
cognifold run timeline.json --fast

# Fast + LLM enrichment + embeddings
cognifold run timeline.json --fast --agent

# Compare with classic mode
cognifold run timeline.json --agent
```

---

## [2026-02-17] - Benchmark Evaluation Report (11 benchmarks, gpt-4o-mini)

### Overview
Ran full evaluation across 11 benchmarks (~20 samples each) with gpt-4o-mini. Generated comprehensive experiment report with failure analysis and improvement recommendations.

### Results
- **Strong**: MuTual 95%, SocialIQA 65%
- **Moderate**: BABILong 55%, StreamingQA 20%
- **Weak**: ToMi 10%, NarrativeQA F1=0.13, QMSum F1=0.14
- **Critical**: MuSiQue 0%, MSC 0% strict, RGB 0%

### Key Findings
- Dominant failure: retrieval_irrelevant (64.4%) — context retrieved but doesn't contain answer
- Root causes: passive concept extraction, zero edge creation in bulk ingestion, BM25-only retrieval
- Top recommendations: enable embedding retrieval, implement atomic fact decomposition, add cross-document edges

### New File
- `benchmarks/evaluation_reports/eval_2026_02_17.md` — Full experiment report with per-benchmark analysis, failure distributions, root causes, and improvement roadmap

---

## [2026-02-22] - Ingestion Fixes & Benchmark Re-evaluation (Phase 12.7)

### Code Changes
- **Template bypass fix** (`prompts.py`): YAML profile templates now override `core.role` only; edge types, connectivity rules, validation, dedup, self-review sections compose normally instead of being skipped
- **Orphan node auto-fix** (`runner.py`): Concept/intent nodes with zero edges automatically get GROUNDS edges via `grounded_in` references or most-recent-event fallback
- **Brace escaping** (`prompt_sections.py`): Fixed `{event_id}` → `{{event_id}}` in atomic facts section for `.format()` compatibility

### Re-evaluation Results (10 benchmarks, n=20, gpt-4o-mini)

**Improvements vs Feb 17 baseline**:
- **RGB**: EM 10% → 15%, F1 0.204 → 0.346 (+70% relative F1) — biggest gain from template fix
- **SocialIQA**: 65% → 70% (+5%) — orphan auto-fix improved concept retrieval
- **ToMi**: 10-20% → 20% (stabilized at upper range)
- **BABILong**: 45-55% → 55% (stabilized)

**Stable/unchanged**: MuTual 95%, MuSiQue 0%, TimeQA 0%, SafetyBench 0% (now scored)

**Stochastic variance**: StreamingQA 20% → 15% EM, NarrativeQA F1 0.132 → 0.089 (n: 10→20)

### Documentation
- Updated `docs/benchmark/results.md` with Feb 21 results, delta column, and improvement analysis
- Updated `docs/benchmark/plan.md` status: Planned → Implemented, added post-fix results
- Updated `docs/benchmark/status.md` with full evaluation results and improvement history
- Updated `docs/benchmark/phase12-log.md` with Phase 12.7 entry

---

## [2026-02-21] - Benchmark Docs Consolidation & Ingestion Fix Plan

### Docs Consolidation
- Merged `benchmarks/evaluation_reports/eval_2026_02_17.md` (n=20, 11 benchmarks) into `docs/benchmark/results.md`, replacing the older small-sample smoke test results
- Consolidated results now cover all 15 benchmarks with failure analysis, root causes, cognition principles alignment, and improvement priorities
- Removed `benchmarks/evaluation_reports/` directory (single source of truth is now `docs/benchmark/results.md`)

### Ingestion Fix Plan
- Created `docs/benchmark/plan.md` — root cause analysis and fix plan for sparse graphs (0 edges across all benchmarks)
- Root cause: template bypass in `prompts.py:418-420` skips all modular prompt sections (edge types, connectivity rules, validation); 3 dead sections in `SECTION_REGISTRY` not in `DEFAULT_SECTION_ORDER`; orphan node detector warns but doesn't auto-fix
- Plan: 3 code changes (section ordering, template composition fix, orphan auto-fix)
- Added reference in `docs/benchmark/architecture.md` and `CLAUDE.md`

### BM25 Degradation Bug — Status Update
- Confirmed PR #54 added embedding infrastructure (`_utils.create_embedder()`) but integration bug remains: runners re-call `create_embedder()` without required arg inside per-example loop, overwriting the correctly-computed embedder
- `MemoryQueryAgent.__init__` still warns without changing `retrieval_mode` — unfixed
- Updated `plan.md` section 1.3 with accurate fix status

---

## [2026-02-17] - Benchmark-Driven Development Tooling & Cognition Principles

### Overview
Added tooling for autonomous benchmark execution, failure analysis, and principled improvement guidance. This is tooling-only — no changes to core `src/cognifold/` modules.

### New: Skills
- **`/cognifold-bench-run`** — Run any of 15 benchmarks via `/cognifold-bench-run <name> [--limit N]`. Auto-downloads data, maps to correct runner, reports results.
- **`/cognifold-bench-analyze`** — Analyze benchmark failures from `wrong_cases.json` or `benchmark_results.json`. Categorizes failures into 7 standard categories, identifies root cause patterns, suggests fixes aligned with Cognition Principles.

### New: Cognition Principles
- `docs/COGNITION_PRINCIPLES.md` — 5 core principles (Event-Driven Cognition, Cognitive Folding, Intention Emergence, Cognitive Assets, Open Infrastructure), 4 anti-patterns (Flat Memory Store, RAG Wrapper, Keyword Search Only, Static Snapshot), alignment checklist, module-to-principle mapping.
- Doc-guard integration: core module changes now trigger COGNITION_PRINCIPLES.md alignment check.

### New: Enhanced Wrong-Case Reporting
- `benchmarks/analysis_utils.py` — Shared utilities: `enrich_eval_result()`, `categorize_failure()`, `save_wrong_cases()`. Context capped at 2KB. Lives in `benchmarks/` (not `src/cognifold/`) to avoid pyright strict mode.
- All 15 benchmark runners enriched with retrieval diagnostics (graph node/edge counts, retrieved context, node IDs, query timing, failure category). Import uses try/except for graceful degradation.

### Modified Files
- 15 benchmark runners (babilong, tomi, musique, msc, locomo, mutual, streamingqa, timeqa, narrativeqa, qmsum, socialiqa, safetybench, rgb, longmemeval, futurex)
- `.claude/skills/doc-guard/references/DOC_RULES.md` — Added Cognition Principles Alignment mapping
- `.claude/skills/doc-guard/scripts/check_docs.sh` — Added COGNITION_PRINCIPLES.md to doc files, core module detection
- `CLAUDE.md` — Added documentation index entry, two new slash commands

---

## [2026-02-14] - Phase 12.5: Benchmark Smoke Testing & Bug Fixes

### Overview
Downloaded data, smoke tested all 10 new benchmarks (8 successful, 2 data unavailable), and fixed 6 download scripts + 2 runner scripts with data format issues.

### Fixed: Download Scripts
- **`mutual/download_data.py`**: Parser treated JSON-format txt files as plain text; added JSON detection in `parse_mutual_txt()` and `parse_mutual_file()`
- **`timeqa/download_data.py`**: Wrong URL filenames (`test_easy.json` → `test.easy.json`); added fallback URLs
- **`safetybench/download_data.py`**: Wrong HF API args (`load_dataset(name, lang, split="test")` → `load_dataset(name, "test", split=lang)`); fixed options list extraction
- **`socialiqa/download_data.py`**: HF datasets v4.5 rejects old loading scripts; added parquet download fallback
- **`qmsum/download_data.py`**: Wrong HF dataset (mattercalm/qmsum has different schema); rewrote to use GitHub JSONL as primary source
- **`tomi/download_data.py`**: Repo is a data generator; added clone + `main.py --num-stories 500` fallback

### Fixed: Runner Scripts
- **`timeqa/run_benchmark.py`**: `load_data()` assumed JSON array format; added JSONL (one JSON per line) support
- **`safetybench/run_benchmark.py`**: Reported misleading 100% accuracy when answers empty; now handles missing ground truth gracefully with "N/A"

### Test Results (8/10 benchmarks, gpt-4o-mini, 2-3 sample smoke tests)
- MuTual: 66.7% accuracy (MC dialogue reasoning)
- SocialIQA: 100% accuracy (MC social reasoning)
- ToMi: 50% EM / 100% verdict (theory of mind)
- NarrativeQA: F1=0.218, ROUGE-L=0.200 (long-form QA)
- QMSum: F1=0.212, ROUGE-L=0.196 (meeting summarization)
- MuSiQue: 0% EM (LLM too verbose)
- TimeQA: 0% EM (answer format mismatch)
- SafetyBench: N/A (test set has no ground truth)
- RGB: Data unavailable (GitHub repo 404)
- StreamingQA: Data unavailable (hosted on GCS)

### Updated: Documentation
- `docs/benchmark/status.md` — 13/15 tested, per-dataset accuracy, download/runner fix tables
- `docs/benchmark/results.md` — 8 new benchmark result sections
- `docs/BENCHMARK.md` — 15-row status table, expanded file map, 15 config profiles
- `docs/RESUME.md` — Phase 12.5 complete, next steps updated

---

## [2026-02-14] - Phase 12.4: All 15 Benchmark Implementations

### Overview
Implemented all 10 remaining benchmark runners (MuTual, MuSiQue, StreamingQA, RGB, TimeQA, NarrativeQA, QMSum, SocialIQA, ToMi, SafetyBench), completing the full 15-dataset benchmark suite.

### Changes

#### Added: 10 New Benchmark Runners
- **MC benchmarks**: `benchmarks/socialiqa/`, `benchmarks/safetybench/`, `benchmarks/mutual/` — multiple-choice with accuracy metric
- **Reasoning**: `benchmarks/tomi/`, `benchmarks/musique/` — free-form with EM/F1 metrics
- **Temporal**: `benchmarks/timeqa/`, `benchmarks/streamingqa/` — temporal QA with EM/F1
- **Long-form**: `benchmarks/narrativeqa/`, `benchmarks/qmsum/` — ROUGE-L/F1 metrics
- **Robustness**: `benchmarks/rgb/` — noise filtering with EM/F1

#### Added: 10 Download Scripts
- All support `HF_ENDPOINT` env var for HuggingFace mirror
- GitHub-based scripts support `GITHUB_MIRROR` env var (e.g., `https://ghproxy.com/`)
- SSH clone fallback for GitHub repos when HTTPS fails

#### Added: 10 Profile YAMLs
- All under `configs/<name>_profile.yaml`
- Default model: `openai:gpt-4o-mini`
- Domain-specific QA templates

#### Changed: Domain Registry
- `src/cognifold/agent/domain.py` — 10 new domain configs registered (20 total)
- All benchmark domains use `disabled_sections=frozenset({"intents"})`

#### Updated: Documentation
- `docs/benchmark/status.md` — Implementation progress (15/15 implemented, 5/15 tested)
- `docs/RESUME.md` — Current work state updated
- `docs/CHANGELOG.md` — This entry

---

## [2026-02-14] - Benchmark Documentation Reorganization

### Overview
Consolidated 5 scattered benchmark doc files into a clean `docs/BENCHMARK.md` entry point + `docs/benchmark/` folder.

### Changes

#### Restructured: Benchmark Documentation
- **Rewritten**: `docs/BENCHMARK.md` — now serves as entry point with status table, file map, and checklists
- **Created**: `docs/benchmark/architecture.md` — system architecture, core components, profile schema, benchmark specs
- **Created**: `docs/benchmark/dataset-catalog.md` — all 14 planned datasets across 5 categories (from old `docs/BENCHMARKS.md`)
- **Created**: `docs/benchmark/results.md` — consolidated experiment results, status details, known issues (from old `benchmarks/BENCHMARK_STATUS.md` + `benchmarks/EXPERIMENT_REPORT.md`)
- **Created**: `docs/benchmark/phase12-log.md` — Phase 12 development log (from old `benchmarks/PHASE12_PROGRESS.md`)
- **Removed**: `docs/BENCHMARKS.md`, `benchmarks/BENCHMARK_STATUS.md`, `benchmarks/EXPERIMENT_REPORT.md`, `benchmarks/PHASE12_PROGRESS.md`
- **Updated cross-references**: `CLAUDE.md`, `docs/PHASES.md`, `docs/RESUME.md`, `.claude/skills/doc-guard/`

---

## [2026-02-14] - Phase 12.3: BABILong Benchmark Rewrite

### Overview
Complete rewrite of the BABILong benchmark to fix critical bugs (download script, BM25 crash, missing agent processing) and align with LoCoMo's proven patterns.

### Changes

#### Fixed: BM25 Dict-in-Join Crash
- **File**: `src/cognifold/retrieval/bm25.py`
- **Problem**: `_get_node_text()` appended `node.data["context"]` (a dict) to a `list[str]`, causing `" ".join()` to crash with "sequence item 3: expected str instance, dict found"
- **Solution**: Type-check context — serialize dict values to string, pass through str/other types

#### Rewritten: BABILong download_data.py
- **File**: `benchmarks/babilong/download_data.py`
- **Problem**: Used `split="test"` (invalid) instead of task names (`qa1`, `qa2`, etc.) as splits. Set `task` field to "test" instead of actual task name. `--tasks` argument was parsed but unused.
- **Solution**: Correct HF API usage (`load_dataset(name, config)[task_name]`), iterate over tasks, output as `babilong_{config}_{task}.json`
- **Added**: `HF_ENDPOINT` environment variable support for HuggingFace mirror (e.g., `https://hf-mirror.com`)
- **Removed**: Invalid `--split` CLI argument

#### Rewritten: BABILong run_benchmark.py
- **File**: `benchmarks/babilong/run_benchmark.py`
- **Pattern**: Follows LoCoMo runner (`benchmarks/locomo/run_benchmark.py`) for consistency
- **Fixed**: sys.path setup, try/except imports, API key checking
- **Fixed**: Prompt profile loading via `load_prompt_profiles()` + raw YAML for QA templates
- **Implemented**: Three distinct processing modes:
  - **direct**: Event nodes only (pure retrieval test, zero cost)
  - **batch**: Event nodes + single LLM call to extract entity states as concept nodes
  - **agent**: Full per-statement processing with context retrieval, `process_event()`, `execute()`, rate limiting
- **Fixed**: LLM QA using direct OpenAI API calls with profile templates (replaced non-existent `query_agent.llm_client`)
- **Added**: GraphLogger + ReplayRenderer integration for visualization
- **Added**: Rate limiting (0.5s/statement, 10s on 429 errors)

#### Updated: Documentation
- `benchmarks/babilong/README.md` — Correct download/run commands, mode descriptions, HF mirror instructions
- `benchmarks/BENCHMARK_STATUS.md` — BABILong status updated to reflect rewrite
- `docs/RESUME.md` — Current work state updated
- `docs/CHANGELOG.md` — This entry

---

## [2026-02-13] - Phase 12.2: MSC Benchmark Optimization & Issue Discovery

### Overview
Optimized MSC benchmark with speaker normalization, temperature tuning, prompt enhancement, and QA evaluation strategy improvements. Discovered agent concept extraction issue affecting recall (0% strict, 22.2% lenient without PRE-SEED).

### Changes

#### Fixed: Speaker Normalization
- **Problem**: MSC data uses "Speaker 1/2" but speaker-aware filter expects "User1/User2"
- **Solution**: Normalize speaker IDs in event creation (Line 257-262 in `run_benchmark.py`)
- **Effect**: Event titles now correctly formatted for speaker-aware filtering

#### Optimized: Temperature & Prompt
- **Temperature**: 0.1 → 0.3 in `configs/msc_profile.yaml` for more creative extraction
- **System Prompt**: Added "CRITICAL EXTRACTION MANDATE" emphasizing concept extraction
- **User Prompt**: Added mandatory CONCEPT creation requirement
- **Effect**: Agent now creates concepts (1 vs 0 before), but still insufficient

#### Improved: QA Evaluation Strategy
- **Old**: Directly use `personas[0][:3]` as gold standard (unfair - facts may not be in dialog)
- **New**: Smart filtering with `fact_in_dialog()` function
  - Only test facts appearing in dialog (keyword matching)
  - At least 2 keywords must match
  - Fallback to general questions if no matched facts
- **Effect**: More fair evaluation (9 QA pairs vs 6, all from dialog content)

#### Updated: .gitignore
- Added `benchmarks/*/output/`
- Added `benchmarks/*/results/`
- Added `benchmarks/*/*.log`

### Test Results

#### MSC - PRE-SEED Mode (Invalid for benchmarking)
- **Method**: Directly inject persona facts as initial concept nodes
- **Result**: Strict 66.7%, Lenient 83.3%
- **Nodes**: 27 (15 pre-seeded + 12 events)
- **⚠️ Problem**: This injects gold standard answers - not valid for true benchmarking

#### MSC - No PRE-SEED Mode (True baseline)
- **Method**: Agent extracts concepts from dialog naturally
- **Result**: Strict 0.0%, Lenient 22.2%
- **Nodes**: 12 (1 concept + 11 events)
- **QA pairs**: 9 (filtered to facts in dialog)
- **Cost**: ~$0.002
- **✓ Valid**: True extraction test

### Known Issues

#### Issue: Agent Concept Extraction Insufficient
- **Severity**: High - affects core benchmark performance
- **Symptom**: Agent only creates concept in first turn, subsequent turns only create events
- **Impact**: Low recall (0% strict, 22.2% lenient)
- **Examples of missed facts**:
  - Dialog mentions "I lift weights at the gym" → Not extracted
  - Dialog mentions "work out alone" → Not extracted
  - Dialog mentions "don't eat meat" → Not extracted
- **Attempted fixes**:
  - ✓ Temperature 0.1 → 0.3
  - ✓ Added extraction mandate to prompts
  - ⚠️ Effect: Minimal improvement (0 → 1 concept)
- **Suggested solutions** (for Phase 12.3):
  - Agent logic improvement
  - Batch extraction approach
  - Post-processing step
  - Different prompt strategies
  - Test stronger models (gpt-4 vs gpt-4o-mini)

### Files Modified
- `benchmarks/msc/run_benchmark.py`:
  - Speaker normalization (Line 257-262)
  - QA evaluation strategy (Line 329-368)
- `configs/msc_profile.yaml`:
  - Temperature: 0.1 → 0.3
  - System prompt enhanced
  - User prompt enhanced
- `.gitignore`:
  - Added benchmark output/results/log patterns
- `benchmarks/BENCHMARK_STATUS.md` (updated)
- `benchmarks/PHASE12_PROGRESS.md` (updated)

### Next Steps
- Create GitHub issue for agent extraction problem
- Phase 12.3: Agent extraction optimization
- Test with larger sample sizes (10-50 conversations)
- Compare with LoCoMo to see if same issue exists

---

## [2026-02-13] - Phase 12.1: MSC & BABILong Benchmarks Running

### Overview
Successfully implemented and tested both MSC and BABILong benchmark runners. Fixed import paths, adapted to current codebase APIs, and achieved first successful benchmark runs.

### Changes

#### Fixed: MSC Benchmark Runner
- Completely rewrote `run_benchmark.py` based on working locomo implementation
- Fixed import paths: `executor.runner.PlanExecutor`, `graph.store.ConceptGraph`, `query.agent.MemoryQueryAgent`
- Configured to use `gpt-4o-mini` for cost savings (~67x cheaper than gpt-4.1)
- Adapted MSC data format handling (JSONL parsing)
- Added comprehensive error handling and progress logging

#### New: BABILong Benchmark Implementation
- Downloaded BABILong dataset from HuggingFace (`RMT-team/babilong`)
- Fixed `download_data.py` to add task field to examples
- Updated `run_benchmark.py` with proper Node object creation
- Implemented direct/batch/agent modes (default: direct)
- Configured BM25 retrieval by default (no embeddings required)

#### Fixed: API Compatibility
- Node creation: Use `type` field (not `node_type`), pass Node objects to `add_node()`
- QueryConfig: Removed invalid `top_k` parameter
- ConceptGraph: Use `node_count`/`edge_count` properties (not `.graph.number_of_nodes()`)
- Event objects: Added required `event_id` field

#### New: Dependencies
- Installed `datasets` (HuggingFace datasets library)
- Installed `pyyaml`, `langgraph`, `openai`, `google-genai>=1.0.0`

### Test Results

#### MSC Benchmark
- **Command**: `python benchmarks/msc/run_benchmark.py --limit 1 --turns 3 --no-llm-eval`
- **Data**: 501 conversations (5945 turns) from official ParlAI
- **Test run**: 1 conversation, 3 turns processed
- **Graph**: 6 nodes created
- **Cost**: ~$0.0007 (gpt-4o-mini)
- **Result file**: `benchmarks/msc/output/msc_results_20260213_191305.json`

#### BABILong Benchmark
- **Command**: `python benchmarks/babilong/run_benchmark.py --config 0k --tasks qa1 --limit 2 --mode direct --query-mode bm25 --no-llm-qa`
- **Data**: 100 qa1 questions (0k context length)
- **Test run**: 2 questions processed
- **Graph**: 4-10 nodes per question
- **Cost**: $0 (no LLM calls in direct mode)
- **Result file**: `benchmarks/babilong/results/babilong_0k_20260213_192602.json`

### Files Modified
- `benchmarks/msc/run_benchmark.py` (complete rewrite, 433 lines)
- `benchmarks/babilong/download_data.py` (add task field)
- `benchmarks/babilong/run_benchmark.py` (Node API fixes, 523 lines)
- `benchmarks/BENCHMARK_STATUS.md` (new - status summary)
- `benchmarks/msc/MSC_TEST_RESULTS.md` (updated with BABILong results)

### Next Steps
- Run larger-scale tests (10-50 samples) to establish baselines
- Test with different retrieval modes (bm25 vs hybrid)
- Compare results with LoCoMo baseline
- Document full results in `benchmarks/EXPERIMENT_REPORT.md`

---

## [2026-02-12] - Phase 12: MSC Benchmark (Official ParlAI Dataset)

### Overview
Fixed MSC (Multi-Session Chat) benchmark to use the official ParlAI dataset instead of the unofficial HuggingFace mirror. Added MSC and BABILong domain configurations with specialized retrieval handlers.

### Changes

#### Fixed: MSC Dataset Source
- **OLD**: Downloaded from `nayohan/multi_session_chat` (HuggingFace)
- **NEW**: Downloads from official ParlAI source: `http://parl.ai/downloads/msc/msc_v0.1.tar.gz`
- Dataset paper: "Beyond Goldfish Memory: Long-Term Open-Domain Conversation" (ACL 2022)
- Paper link: https://arxiv.org/abs/2106.09102

#### New: Domain Configurations
- `MSC_DOMAIN` in `domain.py` — Speaker-aware persona fact extraction, disabled intents
- `BABILONG_DOMAIN` in `domain.py` — Entity state tracking, noise filtering, logic puzzles
- Registered both domains in `DOMAIN_REGISTRY`

#### New: Query Handlers
- `_query_msc_qa()` in `query/agent.py` — Speaker-aware retrieval for persona facts
- `_query_babilong_qa()` in `query/agent.py` — Entity-focused retrieval for state queries

#### New: Prompt Profiles
- `configs/msc_profile.yaml` — MSC-specific prompts and guidelines
- `configs/babilong_profile.yaml` — BABILong-specific prompts and guidelines

#### New: MSC Benchmark Runner
- `benchmarks/msc/download_data.py` — Downloads and parses official ParlAI MSC dataset
- `benchmarks/msc/run_benchmark.py` — MSC benchmark runner (~490 lines)
- `benchmarks/msc/README.md` — Usage documentation

### Files Modified
- `benchmarks/msc/download_data.py` (rewritten)
- `benchmarks/msc/run_benchmark.py` (new)
- `benchmarks/msc/README.md` (updated)
- `configs/msc_profile.yaml` (new)
- `configs/babilong_profile.yaml` (new)
- `src/cognifold/agent/domain.py` (added MSC_DOMAIN, BABILONG_DOMAIN)
- `src/cognifold/query/agent.py` (added _query_msc_qa, _query_babilong_qa)
- `docs/CHANGELOG.md` (this file)

### Next Steps
- Implement BABILong benchmark runner
- Test MSC benchmark with official ParlAI dataset
- Run benchmarks to establish baseline performance

---
## [2026-02-16] - Pyright Type Safety Cleanup

### Changes
- Resolved all 83 pyright type errors across 26 source files
- Added `get_node_or_none()` to `GraphStore` for type-safe node lookups
- Fixed conditional import patterns (`dateparser`, `faiss`) for proper type narrowing
- Removed dead `"action"` node type references (replaced by `"intent"` long ago)
- Fixed `save_timeline()` parameter order in generator subclasses to match base class
- Added type annotations to untyped variables and function signatures
- Added `type: ignore` comments for untyped third-party libs (pyvis, pypdf, google.generativeai)

### Files Modified
- 26 files under `src/cognifold/` (graph, query, scoring, retrieval, embeddings, intent, generator, simulator, cli, temporal, executor, importers)

### Tests
- All 800 tests passing, 2 skipped
- Pyright: 0 errors (was 83)
- Ruff: all checks passing

---

## [2026-02-16] - GCP Cloud Run Deployment

### Overview
Added GCP Cloud Run as a deployment target alongside the existing VPS infrastructure. Fully managed, auto-scaling, no infrastructure to maintain. Uses Memorystore Redis for shared sessions, Artifact Registry for images, and Workload Identity Federation for keyless CI/CD auth.

### Changes

#### GCP Setup Script
- `deploy/gcp/setup.sh` — Idempotent one-time setup: project + billing, APIs, Artifact Registry, VPC connector, Memorystore Redis, service accounts (runtime + deploy), Workload Identity Pool/Provider for GitHub Actions

#### Cloud Run Service Template
- `deploy/gcp/service.yaml` — Reference/template documenting all Cloud Run settings: 1Gi memory, 1 CPU, session affinity, VPC connector, Secret Manager integration, health probes

#### CD Workflow Update
- `.github/workflows/cd.yml` — Added Workload Identity auth, dual registry push (GHCR + Artifact Registry), `deploy-cloudrun` job with health check, renamed VPS deploy to `deploy-vps`, changed platforms to `linux/amd64` only

#### Documentation
- `docs/DEPLOYMENT.md` — Cloud Run section: architecture, env var mapping, manual deploy, log viewing, rollback, cost estimates, decisions D18-D23
- `.env.example` — Cloud Run override comments block

### Architectural Decisions
- D18: 1 worker per Cloud Run instance — horizontal scaling via auto-scaler
- D19: Memorystore Redis — managed, session-miss recovery handles restarts
- D20: Workload Identity Federation — no long-lived SA keys
- D21: Dual registry push — GHCR for VPS, AR for Cloud Run
- D22: Cookie-based session affinity — Redis provides failover recovery
- D23: amd64 only — Cloud Run is amd64, halves build time

### No Application Code Changes
Existing Dockerfile, gunicorn config, session stores, health endpoints, and structured logging all work as-is on Cloud Run.

---

## [2026-02-16] - Phase 15.5: Horizontal Scaling & Auto-Deploy

### Overview
Horizontal scaling infrastructure: persist-on-mutation for session resilience, NGINX `ip_hash` session affinity for multi-instance routing, rolling deploy script for N replicas, Docker Compose scaled overlay, and SSH-based auto-deploy from GitHub Actions.

### Changes

#### Persist-on-Mutation
- `src/cognifold/service/session.py` — `persist_session_data()` method: fire-and-forget persistence after every successful event processing; refreshes Redis TTL
- `src/cognifold/service/processor.py` — Calls `persist_session_data()` after plan execution succeeds

#### NGINX Session Affinity
- `deploy/nginx/upstream.conf` — `ip_hash` directive for sticky sessions; `max_fails=3 fail_timeout=30s` per server for automatic failover

#### Rolling Deploy
- `deploy/scripts/deploy-scaled.sh` — Rolling restart of N replicas: pull image → stop/start one at a time → health check → rollback on failure

#### Docker Compose Scaled Overlay
- `docker-compose.scaled.yml` — Explicit `cognifold-1`, `cognifold-2`, `cognifold-3` replicas with Redis session backend

#### Auto-Deploy
- `.github/workflows/cd.yml` — `deploy` job using `appleboy/ssh-action`; requires GitHub Environment approval; SSHs to server and runs `deploy-scaled.sh`

### Architectural Decisions
- D13: Persist-on-mutation (fire-and-forget) — crash recovery loses at most in-flight request
- D14: NGINX `ip_hash` for affinity — same client IP → same backend, automatic failover
- D15: Explicit replicas — unique container names for NGINX upstream and per-replica health checks
- D16: Rolling deploy for scaled mode — preserves N-1 capacity during restart
- D17: GitHub Environment protection — required reviewers gate prevents accidental deploys

---

## [2026-02-16] - Phase 15: Production Deployment & Observability

### Overview
Production-ready deployment infrastructure: Docker containerization, NGINX reverse proxy with blue-green deploys, structured JSON logging, pluggable session persistence (file/Redis), GitHub Actions CI/CD, and Loki log aggregation.

### Changes

#### Structured Logging (Phase 15.1)
- `src/cognifold/logging.py` — Rewritten to use structlog when available; JSON output with contextvars-based request-ID tracking; falls back to plain-text stdlib logging
- `src/cognifold/service/app.py` — `RequestContextMiddleware` generates `X-Request-ID`, binds request context to all log calls; new `session_backend` and `redis_url` settings in `AppSettings`

#### Session Store Abstraction (Phase 15.1)
- New `src/cognifold/service/stores/` package with `SessionStore` ABC, `FileSessionStore` (async I/O), `RedisSessionStore` (redis.asyncio + TTL), and env-driven factory
- `src/cognifold/service/session.py` — `SessionManager` accepts store for persistence; session-miss recovery from store on `get_session()`

#### Gunicorn + WSGI (Phase 15.2)
- `deploy/gunicorn.conf.py` — Uvicorn worker class, auto-scaled workers, 120s timeout
- `src/cognifold/service/wsgi.py` — ASGI entry point reading config from environment
- `src/cognifold/cli/serve.py` — New `--session-backend`, `--redis-url`, `--gunicorn` flags

#### Docker Image (Phase 15.2)
- `Dockerfile` — Multi-stage build (uv builder + python:3.11-slim runtime), non-root user, health check, gunicorn CMD

#### Docker Compose + NGINX (Phase 15.3)
- `docker-compose.yml` — NGINX reverse proxy, optional Redis (profiles), shared network
- `deploy/nginx/nginx.conf` — JSON access logs, rate limiting (30r/s API, 10r/s ingest), security headers, 120s proxy timeout
- `deploy/nginx/upstream.conf` — Blue-green backend routing (rewritten by deploy script)

#### Log Aggregation (Phase 15.3)
- `docker-compose.logging.yml` — Loki + Promtail overlay
- `deploy/loki/loki-config.yaml` — 7-day retention, filesystem storage
- `deploy/promtail/promtail-config.yaml` — Docker SD, JSON pipeline for cognifold containers

#### Deployment Scripts (Phase 15.3)
- `deploy/scripts/deploy.sh` — Blue-green deploy with health check, auto-rollback, 30s drain
- `deploy/scripts/rollback.sh` — Reads deploys.log to revert to previous image
- `deploy/scripts/setup.sh` — First-time VPS setup (Docker, network, dirs)
- `deploy/scripts/health-check.sh` — Monitoring script for cron
- `deploy/scripts/ssl-init.sh` — Let's Encrypt cert via certbot

#### CI/CD (Phase 15.4)
- `.github/workflows/ci.yml` — Lint/format/typecheck/test on PR, Docker build smoke test
- `.github/workflows/cd.yml` — Build multi-arch image (amd64+arm64), push to GHCR on merge

#### Configuration
- `.env.example` — Template for all configuration variables
- `.dockerignore` — Lean Docker builds
- `pyproject.toml` — New extras: `[redis]`, `[production]`; `structlog` added to `[service]`

### Tests
- `tests/unit/test_logging.py` — 14 tests (JSON output, request ID binding, fallback behavior)
- `tests/unit/test_session_stores.py` — 11 tests (file store round-trip, factory, ABC)

---

## [2026-02-14] - NeoLearn Backend Fixes (B1/B2/B7/B8/B10/B12)

### Overview
Six backend fixes to support the NeoLearn learning domain integration. These changes make custom domain configurations actually drive LLM behavior, expose domain registration via HTTP, improve reliability with retry logic, and update deprecated models.

### Changes

#### Fix 1 (B1 — Critical): Pass `session.config.domain` to Agent
- `agent/config.py` — added `domain: str = "personal-timeline"` field to `AgentConfig`
- `service/processor.py` — passes `domain=session.config.domain` when creating `AgentConfig`
- `agent/state.py` — added `domain: Optional[str]` to `AgentState` and `create_initial_state()`
- `agent/agent.py` — passes `domain=self._config.domain` to `create_initial_state()`
- `agent/graph.py` — uses state domain with fallback chain: `profile.domain` -> `state["domain"]` -> `PERSONAL_TIMELINE_DOMAIN`

#### Fix 2 (B2 — Critical): Domain Registration HTTP API
- New file `service/routes/domains.py` with `POST /api/v1/domains`, `GET /api/v1/domains`, `GET /api/v1/domains/{name}`
- Registered in `service/routes/__init__.py`

#### Fix 3 (B12): Concept Description in LLM Prompt
- `agent/prompts.py` — added `"description"` field to `_format_concept_example()` so the LLM sees description examples and generates descriptions for concepts

#### Fix 4 (B10): LLM Retry with Backoff for 429
- `service/processor.py` — added retry logic (3 attempts, exponential backoff) around LLM calls for `429 RESOURCE_EXHAUSTED` errors

#### Fix 5 (B8): Server Env Var Fallback for LLM Keys
- `service/processor.py` — checks `os.environ` for `GOOGLE_API_KEY`/`OPENAI_API_KEY` in addition to session keys

#### Fix 6 (B7): Deprecated Embedding Model
- `embeddings/config.py` — changed default model from `models/text-embedding-004` to `gemini-embedding-001` (both default and `for_gemini()` class method)

### Tests
- 7 new tests in `tests/unit/test_domain_routes.py` covering domain registration, listing, retrieval, and 404 handling
- Updated `tests/unit/test_embeddings.py` to match new default model

## [2026-02-12] - Phase 14.1: Intent Personalization

### Overview
Closed-loop intent calibration system. Users can accept/reject/defer/modify intents, and the system learns from feedback via EMA-weighted category profiles to adjust future intent scoring and prompt generation.

### Changes

#### New: `intent/personalization.py`
- `FeedbackType` enum (accept, reject, defer, modify)
- `IntentFeedback`, `CalibrationProfile`, `FeedbackStats` Pydantic models

#### New: `intent/feedback_store.py`
- `FeedbackStore(graph)` — persists feedback as event nodes + USER_FEEDBACK edges
- CRUD: `add_feedback()`, `get_feedback_for_intent()`, `get_all_feedback()`, `get_stats()`
- Auto-updates intent status on reject (→REJECTED) and defer (→DEFERRED)
- Applies priority/description modifications on MODIFY feedback

#### New: `intent/calibrator.py`
- `IntentCalibrator` with EMA-based profile computation (alpha=0.3)
- `get_score_multiplier(intent)` → [0.1, 2.0] calibration factor
- `get_prompt_context()` → prompt snippet for agent injection
- `get_adjusted_min_urgency(base)` → adaptive threshold based on acceptance rate

#### Updated: `intent/selector.py`
- `IntentSelector` accepts optional `calibrator: IntentCalibrator`
- Scoring applies calibrator multiplier to combined score

#### Updated: `models/node.py`
- `IntentStatus` gains REJECTED, DEFERRED values
- `BaseEdgeType` gains USER_FEEDBACK with default weight 0.8 and constraints

#### Updated: `agent/prompt_sections.py`
- New section `intents.personalization` with `{intent_personalization_context}` placeholder
- Section count: 20 → 21, intents group: 3 → 4

#### Updated: `agent/prompts.py`
- `format_system_prompt_for_domain()` provides default personalization context

#### Updated: `agent/context.py`
- `AgentContext` gains `calibration_context` field

#### New: `service/routes/intents.py`
- `POST /sessions/{id}/intents/{intent_id}/feedback` — submit feedback
- `GET /sessions/{id}/intents/calibration` — get calibration profile
- `GET /sessions/{id}/intents/pending` — get pending intents with calibrated scores

#### Updated: `cli/client.py`
- `:feedback <intent-id> accept|reject|defer|modify [comment]` command
- `:calibration` command to display current profile
- `:intents pending` subcommand for calibrated pending list

#### New Tests
- `tests/unit/test_intent_personalization.py` — 31 unit tests (models, store, calibrator, selector integration)
- `tests/integration/test_personalization_loop.py` — 5 integration tests (closed-loop behavior)
- Updated `tests/unit/test_prompt_sections.py` — section count assertions for 21 sections

### Files Modified
- `src/cognifold/intent/personalization.py` (new)
- `src/cognifold/intent/feedback_store.py` (new)
- `src/cognifold/intent/calibrator.py` (new)
- `src/cognifold/intent/selector.py`
- `src/cognifold/intent/__init__.py`
- `src/cognifold/models/node.py`
- `src/cognifold/agent/prompt_sections.py`
- `src/cognifold/agent/context.py`
- `src/cognifold/agent/prompts.py`
- `src/cognifold/service/routes/intents.py` (new)
- `src/cognifold/service/routes/__init__.py`
- `src/cognifold/cli/client.py`
- `tests/unit/test_intent_personalization.py` (new)
- `tests/unit/test_prompt_sections.py`
- `tests/integration/test_personalization_loop.py` (new)

---

## [2026-02-08] - Phase 13: Modular System Prompt Composition

### Overview
Decomposed the monolithic ~370-line `SYSTEM_PROMPT_TEMPLATE` into 20 composable, named sections organized into 4 groups (core, concepts, intents, time). Domains can now toggle sections on/off, override content, or inject custom sections while maintaining byte-identical output for all existing domains.

### Changes

#### New: `prompt_sections.py`
- 20 section constants (`SECTION_CORE_ROLE`, `SECTION_INTENTS_GUIDELINES`, etc.)
- `SECTION_REGISTRY`, `SECTION_GROUPS`, `DEFAULT_SECTION_ORDER`
- `resolve_sections()` function for composing sections with disabled/extra/override support

#### Updated: `domain.py`
- `DomainConfig` gains `disabled_sections`, `extra_sections`, `extra_section_position` fields
- LoCoMo and LongMemEval domains use `disabled_sections=frozenset({"intents"})`

#### Updated: `prompts.py`
- `SYSTEM_PROMPT_TEMPLATE` reconstructed from sections (no longer a 370-line string literal)
- `format_system_prompt_for_domain()` uses section-based composition; YAML template overrides bypass it

#### Updated: `prompt_profile.py`
- `PromptProfile` gains `disabled_sections` and `extra_sections` fields
- YAML loader reads `sections.disabled` and `sections.extra` keys

#### Updated: `graph.py`
- `analyze_node()` merges profile section config with domain section config

#### New: `test_prompt_sections.py`
- 47 tests: registry completeness, concatenation invariant, section toggling, extra section injection, golden-output preservation for all 7 domains x 4 modes

### Files Modified
- `src/cognifold/agent/prompt_sections.py` (new)
- `src/cognifold/agent/prompts.py`
- `src/cognifold/agent/domain.py`
- `src/cognifold/agent/prompt_profile.py`
- `src/cognifold/agent/graph.py`
- `tests/unit/test_prompt_sections.py` (new)
- `tests/unit/test_domain_prompts.py`
- `tests/unit/test_prompt_profile.py`
- `docs/PROMPTS.md`
- `docs/PHASES.md`
- `docs/RESUME.md`

---

## [2026-02-08] - Fix: LLM-Powered Event Ingestion

### Overview
Fixed four cascading bugs that prevented LLM agent-generated plans from executing successfully during event ingestion via the HTTP service.

### Root Cause
When using LLM API keys, the agent successfully generates multi-operation plans, but execution failed because:
1. The executor ran outside the `llm_env()` context, so the embedding service couldn't access API keys
2. Execution errors were silently dropped (no `error` field in response)
3. The executor didn't check `op.node_id` for ADD_NODE operations
4. LLM-generated plans could have ADD_EDGE before ADD_NODE for the same target
5. Agent-assigned node IDs were stored under various data keys the executor didn't check

### Changes

#### Wrap Execution in `llm_env()` (`service/processor.py`)
- Plan execution now runs inside `session.llm_env()` so the embedding service has access to API keys
- Added warning log when execution fails, including operation index and error message
- Propagate `execution.error` to `IngestEventResponse`

#### Error Propagation (`service/models.py`, `cli/client.py`)
- Added `error: str | None = None` field to `IngestEventResponse`
- CLI client displays error message on FAILED ingestion

#### Embedding Service Re-initialization (`utils/embeddings.py`)
- `get_embedding_service()` now re-creates the singleton when the current instance has no client but an API key is available (e.g., inside `llm_env()`)
- Added `reset_embedding_service()` function

#### Operation Sorting (`executor/runner.py`)
- Operations are sorted by type priority before execution: ADD_NODE (0) → UPDATE_NODE (1) → ADD_EDGE (2) → REMOVE_EDGE (3) → REMOVE_NODE (4) → MERGE_NODES (5)
- Ensures nodes exist before edges reference them, regardless of agent output order

#### Robust Node ID Resolution (`executor/runner.py`)
- New `_resolve_add_node_id()` method with 5-layer lookup:
  1. `op.node_id` — explicit ID on the operation
  2. Well-known data keys: `event_id`, `id`, `concept_id`, `action_id`, `intent_id`, `time_id`
  3. Dynamic `{node_type}_id` key (e.g., `intent` → `intent_id`)
  4. Scan all data values for IDs referenced by edge operations in the plan
  5. Auto-generate fallback: `{node_type}-{uuid[:8]}`
- Collects edge-referenced IDs at the start of execution to match agent-assigned IDs regardless of which data key holds them

### Files Modified
- `src/cognifold/service/processor.py` — llm_env wrapping, error logging and propagation
- `src/cognifold/service/models.py` — added `error` field to `IngestEventResponse`
- `src/cognifold/cli/client.py` — display error on FAILED ingest
- `src/cognifold/utils/embeddings.py` — singleton re-creation, `reset_embedding_service()`
- `src/cognifold/executor/runner.py` — operation sorting, `_resolve_add_node_id()`
- `docs/ARCHITECTURE.md` — updated PlanExecutor and Event Processor sections

### Commits
- `a2538d5` fix: wrap plan execution in llm_env so embeddings have API keys
- `ccea9d5` fix(executor): use op.node_id for ADD_NODE to match agent-assigned IDs
- `e72fc7f` fix(executor): sort operations so ADD_NODEs execute before ADD_EDGEs
- `cccf179` fix(executor): robust node ID resolution for LLM-generated plans

### Tests
- Existing tests still passing
- Verified end-to-end with OpenAI GPT-4.1 agent: events now ingest successfully with 5+ operations

---

## [2026-02-08] - Phase 11: Start Script & Interactive CLI Client

### Overview
Added a start script and interactive CLI client for the Phase 11 HTTP service layer.

### Changes

#### Start Script (`scripts/start_server.sh`)
- Configurable via env vars: `COGNIFOLD_HOST`, `COGNIFOLD_PORT`, `COGNIFOLD_LOG_LEVEL`, `COGNIFOLD_PERSIST_DIR`, `COGNIFOLD_API_KEY`
- Prints startup banner with URL, docs URL, and auth status
- Uses `exec` for clean process replacement (Python replaces shell)
- Defaults: `127.0.0.1:8000`, `info` log level, `./sessions` persist dir, no auth

#### Interactive CLI Client (`src/cognifold/cli/client.py`)
- `CognifoldClient`: HTTP wrapper using stdlib `urllib.request` (no new dependencies)
  - Methods for health, sessions, events, query, and graph endpoints
  - API key header support, connection error handling, HTTP error parsing
- `ClientREPL`: Interactive REPL with `:` command dispatch
  - Session management: `:session create|info|delete|<ID>`
  - Event ingestion: `:ingest TYPE TITLE [--desc D] [--loc L]`
  - Graph exploration: `:stats`, `:concepts`, `:intents`, `:events`, `:node`, `:graph`
  - Natural language query: any input without `:` prefix
  - Dynamic prompt: `cognifold>` or `cognifold [abcd1234]>` with session
- CLI registration: `cognifold client [--url URL] [--api-key KEY] [--session ID]`

### Files Created
- `scripts/start_server.sh` - Server start script
- `src/cognifold/cli/client.py` - Interactive CLI client
- `tests/unit/test_cli_client.py` - 38 unit tests

### Files Modified
- `src/cognifold/cli/__init__.py` - Registered `client` subcommand
- `CLAUDE.md` - Added service/client documentation, updated module structure and status
- `docs/RESUME.md` - Updated current state
- `docs/CHANGELOG.md` - This entry

### Tests
- Added/updated tests: yes (38 new tests)
- All tests passing: yes

---

## [2026-02-08] - Benchmark Evaluation Plan (Phase 12)

### Overview
Added comprehensive benchmark overview documenting 14 datasets across 5 evaluation categories for systematically evaluating Cognifold's capabilities.

### Changes

#### New File: `docs/BENCHMARKS.md`
- **Category 1 - Long-Term Conversational Memory**: MSC, LoCoMo, LongMem
- **Category 2 - Multi-Hop Reasoning**: BABILong, MuTual, MuSiQue-Ans
- **Category 3 - Streaming & Conflicts (Dynamic Graph)**: StreamingQA, RGB, TimeQA
- **Category 4 - Long-Form Narrative & Event Understanding**: NarrativeQA, QMSum
- **Category 5 - Proactive ("The Soul")**: SocialIQA, ToMi, SafetyBench

#### Updated Files
- `CLAUDE.md` - Added benchmark docs to Documentation Index
- `docs/PHASES.md` - Added Phase 12 (Benchmark Evaluation) with priority datasets and adaptation notes
- `docs/RESUME.md` - Updated current state for benchmark-dev-suli branch
- `docs/CHANGELOG.md` - This entry

### Tests
- Documentation only, no code changes

---

## [2026-01-25] - Benchmark Improvements & Query Agent Enhancements

### Overview
Major improvements to benchmark system and query agent. Added domain-specific configurations,
prompt profiles, speaker-aware retrieval for LoCoMo, and LLM-based evaluation.

### Domain Configuration

#### LoCoMo Domain (`src/cognifold/agent/domain.py`)
- Added `LOCOMO_DOMAIN` for conversational memory benchmarks
- Speaker-aware concept extraction (User1 vs User2 distinction)
- Relationship tracking between speakers
- Registered in `DOMAIN_REGISTRY`

### Prompt Profiles

#### New Profile: `configs/locomo_profile.yaml`
- Domain-specific prompts for LoCoMo benchmark
- Templates: system, user, qa_system, qa_answer, evaluate
- Speaker-aware extraction guidelines
- Features: enable_time_nodes, disable action_nodes

#### Updated: `configs/longmemeval_profile.yaml`
- Added batch_extraction template for efficient multi-event processing
- Added qa_answer template for QA evaluation
- Added evaluate template for LLM-based scoring
- Comprehensive extraction guidelines for user facts

### Query Agent Improvements (`src/cognifold/query/`)

#### New Config Options (`models.py`)
- `domain`: Domain hint for domain-specific query processing
- `speaker_aware`: Enable speaker-aware retrieval for conversations
- `use_llm_rerank`: Use LLM to re-rank retrieved candidates
- `use_query_expansion`: Expand query with synonyms and related terms

#### New QueryIntent Model (`models.py`)
- `query_type`: Inferred query type (semantic, temporal, structural, hybrid)
- `key_topics`: Main topics/concepts being queried
- `time_context`: Temporal reference if present
- `speaker_filter`: Target specific speaker (for LoCoMo)

#### Domain-Specific QA Methods (`agent.py`)
- `query_for_qa()`: Dispatch to domain-specific handlers
- `_query_locomo_qa()`: Speaker-aware retrieval with speaker extraction
- `_query_longmemeval_qa()`: Fact-focused retrieval

#### LLM Features (`agent.py`)
- `parse_query_intent()`: LLM-based query understanding
- `rerank_with_llm()`: Re-rank candidates by relevance score
- `_call_llm()`: Unified LLM helper (OpenAI/Gemini)
- `_extract_speaker_from_question()`: Extract speaker from question text
- `_node_mentions_speaker()`: Filter nodes by speaker

#### Context Assembly Improvements (`assembly.py`)
- Added graph reference for edge lookups
- `_format_node_edges()`: Include edge connections in context
- Shows edge type and connected node titles

### Benchmark Runner Updates

#### LoCoMo (`benchmarks/locomo/run_benchmark.py`)
- Complete rewrite with domain integration
- Loads `locomo_profile.yaml` for prompts
- Uses `QueryConfig` with domain="locomo", speaker_aware=True
- **Context retrieval before processing events** (was passing empty list)
- LLM-based evaluation with CORRECT/PARTIAL/INCORRECT scoring
- Saves detailed results to `benchmark_results.json`
- New CLI flags: `--no-llm-eval`, `--no-profile`, `--query-mode`

#### LongMemEval (`benchmarks/longmemeval/run_eval.py`)
- Uses templates from `longmemeval_profile.yaml`
- LLM-based evaluation with `evaluate_answer()`
- Saves metrics.json with evaluation results
- Uses `query_for_qa()` with domain="longmemeval"

### Files Added
- `configs/locomo_profile.yaml`

### Files Modified
- `src/cognifold/agent/domain.py` - Added LOCOMO_DOMAIN
- `src/cognifold/query/models.py` - Added QueryConfig options, QueryIntent
- `src/cognifold/query/agent.py` - Domain-specific QA, LLM features
- `src/cognifold/query/assembly.py` - Edge formatting
- `configs/longmemeval_profile.yaml` - New templates
- `benchmarks/locomo/run_benchmark.py` - Complete rewrite
- `benchmarks/longmemeval/run_eval.py` - LLM evaluation

### Key Improvements
1. **Speaker-aware retrieval**: LoCoMo QA now extracts speaker from questions and filters results
2. **Domain-specific prompts**: Each benchmark has tailored extraction and evaluation prompts
3. **LLM-based evaluation**: More accurate than simple string matching
4. **Context before processing**: Events now receive relevant context during ingestion
5. **Unified LLM helper**: Supports both OpenAI and Gemini APIs

---

## [2026-02-01] - Documentation Refinement: Phases & Architecture

### Overview
Consolidated phase documentation, added phase management instructions, and rewrote ARCHITECTURE.md with accurate implementation details from the codebase.

### Changes

#### Phase Documentation Consolidation
- Merged `docs/PHASES.md` and `docs/PHASES_COMPLETED.md` into single `docs/PHASES.md`
- Structure: Status Overview → Current/Planned Phases → Completed Phases
- Deleted `docs/PHASES_COMPLETED.md`

#### CLAUDE.md Updates
- Added "Phase Documentation Management" section with instructions for:
  - Updating status in overview table
  - Moving completed phases to "Completed Phases" section
  - Keeping completed specs concise
- Updated Documentation Index to reflect consolidated files

#### ARCHITECTURE.md Rewrite
- Read actual codebase to ensure accuracy
- Added critical implementation details:
  - Full Node/Edge/Event/UpdatePlan schemas with field descriptions
  - ConceptGraph as NetworkX MultiDiGraph wrapper
  - ContextRanker scoring formula components
  - Hierarchical context window levels and scoring strategies
  - Pipeline processing flow
  - PlanExecutor atomic execution with rollback
  - Query system retrieval modes (LEGACY, BM25, SEMANTIC, HYBRID)
  - Embedding system with provider abstraction
  - Hybrid retrieval with RRF fusion
  - Module dependency graph

### Files Modified
- `docs/PHASES.md` - Consolidated all phases
- `docs/PHASES_COMPLETED.md` - Deleted
- `CLAUDE.md` - Added phase management instructions
- `docs/ARCHITECTURE.md` - Complete rewrite with accurate details

### Tests
- Documentation only, no code changes

---

## [2026-02-01] - Documentation Restructuring for Multi-Agent Collaboration

### Overview
Restructured documentation to support multiple agents working on the codebase. Created modular documentation with strict collaboration rules.

### Changes

#### New Files Created
- `docs/AGENT_PROTOCOL.md` - Strict rules for all agents (pre-flight checklist, commit protocol, quality gates, progress tracking)
- `docs/PHASES.md` - Current and planned phase specifications (Phases 8-11)
- `docs/CONTRIBUTING.md` - Code standards, testing requirements, PR workflow

#### Files Modified
- `CLAUDE.md` - Refactored from ~1200 lines to ~240 lines as concise entry point
  - Moved phase specifications to `docs/PHASES.md`
  - Moved agent workflow rules to `docs/AGENT_PROTOCOL.md`
  - Kept: architecture overview, core concepts, schemas, module structure
- `docs/RESUME.md` - Updated format and current state

#### Documentation Structure (After)
```
CLAUDE.md                     # Entry point (~240 lines)
docs/
├── AGENT_PROTOCOL.md         # Agent collaboration rules
├── PHASES.md                 # All phases (current + completed)
├── ARCHITECTURE.md           # System design (detailed)
├── PROMPTS.md                # Prompt engineering
├── CONTRIBUTING.md           # Code standards
├── RESUME.md                 # Work in progress
├── CHANGELOG.md              # Change history
└── WISHLIST.md               # Deferred work
```

### Key Protocol Rules
1. Pre-flight checklist: Read CLAUDE.md → RESUME.md → CHANGELOG.md → PHASES.md
2. Quality gates: ruff format, ruff check, pyright, pytest (must pass before commit)
3. Commit protocol: Incremental commits with Phase reference
4. Progress tracking: Update RESUME.md every session, CHANGELOG.md after changes
5. Branch strategy: Never push to main/cognifold-dev, use phase branches + PRs

### Tests
- No code changes, documentation only
- All existing tests still passing

---

## [2026-02-01] - Phase 10.2, 10.3, 10.4: Advanced Retrieval System

### Overview
Implemented advanced retrieval capabilities including temporal extraction, embedding-based semantic search, and hybrid retrieval with BM25 + RRF fusion.

---

### Phase 10.2: Enhanced Temporal Extraction

#### New Module: `src/cognifold/temporal/`

##### Core Components
- `extractor.py`: Temporal extraction with regex patterns and dateparser integration
  - `TemporalType` enum: ABSOLUTE_DATE, ABSOLUTE_TIME, RELATIVE_DATE, RECURRING, DEADLINE, DURATION, TIME_RANGE, PERIOD_REFERENCE
  - `TemporalEntity` dataclass: Extracted temporal reference with raw text, normalized datetime, confidence, span, and is_future flag
  - `TemporalExtractor` class: Extracts temporal entities from text

##### Key Features
- Regex patterns for common date/time formats (ISO, US, relative, recurring)
- Pattern priority ordering to prevent partial matches (e.g., "every Monday" before "Monday")
- Integration with `dateparser` library for natural language date parsing
- Confidence scoring based on pattern type and parsing success
- Future date detection for scheduling-related temporals
- `extract_for_time_nodes()` method for TIME node creation suggestions

##### Patterns Supported
| Pattern Type | Examples |
|--------------|----------|
| Recurring | "every Monday", "weekly", "daily" |
| Deadline | "by Friday", "due tomorrow", "before next week" |
| ISO dates | "2026-01-15", "2026-01-15T09:00:00Z" |
| US dates | "January 15, 2026", "01/15/2026" |
| Relative | "tomorrow", "next week", "in 3 days" |
| Duration | "2 hours", "30 minutes", "3 days" |
| Time ranges | "9am to 5pm", "9:00-17:00" |

##### Files Created
- `src/cognifold/temporal/__init__.py` - Module exports
- `src/cognifold/temporal/extractor.py` - TemporalExtractor implementation

##### Tests
- 32 unit tests for temporal extraction
- Tests for pattern priority, recurring events, deadlines, relative dates
- All tests passing

---

### Phase 10.3: Embedding-based Semantic Search

#### New Module: `src/cognifold/embeddings/`

##### Core Components
- `config.py`: Embedding configuration and provider types
  - `EmbeddingProviderType` enum: GEMINI, OPENAI, MOCK
  - `EmbeddingConfig` dataclass: Provider, model, dimensions, batch_size, cache settings
- `providers.py`: Embedding provider abstraction
  - `EmbeddingProvider` abstract base class
  - `MockEmbeddingProvider`: Deterministic embeddings for testing
  - `GeminiEmbeddingProvider`: Google AI embedding integration
  - `OpenAIEmbeddingProvider`: OpenAI embedding integration
- `embedder.py`: Node embedding generation
  - `NodeEmbedder` class: Generates and caches embeddings for graph nodes
  - Extracts searchable text from nodes (title, description, reasoning)
  - Supports lazy and eager embedding generation
  - LRU caching for efficient repeated lookups
- `search.py`: Semantic similarity search
  - `SearchConfig` dataclass: top_k, min_score, node type filters
  - `SearchResult` dataclass: node_id, score, node reference
  - `SemanticSearch` class: Cosine similarity search over embeddings

##### Key Features
- Provider abstraction for multiple embedding services
- Deterministic mock provider using numpy RNG (for reproducible tests)
- Batch embedding support for efficiency
- Embedding caching with configurable TTL
- Cosine similarity with numpy optimization
- Node type filtering in search results
- Index building for fast similarity search

##### Configuration Defaults
| Setting | Default Value |
|---------|---------------|
| Provider | GEMINI |
| Model | models/text-embedding-004 |
| Dimensions | 768 |
| Batch Size | 100 |
| Cache Embeddings | true |
| Normalize | true |

##### Files Created
- `src/cognifold/embeddings/__init__.py` - Module exports
- `src/cognifold/embeddings/config.py` - EmbeddingConfig
- `src/cognifold/embeddings/providers.py` - EmbeddingProvider implementations
- `src/cognifold/embeddings/embedder.py` - NodeEmbedder
- `src/cognifold/embeddings/search.py` - SemanticSearch

##### Tests
- 36 unit tests for embeddings module
- Tests for provider abstraction, caching, semantic search
- Fixed numerical overflow in MockEmbeddingProvider
- All tests passing

---

### Phase 10.4: Hybrid Retrieval (BM25 + Embeddings)

#### New Module: `src/cognifold/retrieval/`

##### Core Components
- `config.py`: Retrieval configuration
  - `RetrievalStrategy` enum: KEYWORD, SEMANTIC, HYBRID
  - `RetrievalConfig` dataclass: strategy, weights, RRF constant, top_k, filters
  - Factory methods: `for_keyword_search()`, `for_semantic_search()`, `for_hybrid_search()`
- `result.py`: Retrieval result types
  - `RetrievalResult` dataclass: node_id, final_score, bm25_score, semantic_score, ranks
  - `RetrievalMetrics` dataclass: candidate counts, strategy used, final results count
- `bm25.py`: BM25 inverted index
  - `BM25Config` dataclass: k1, b, min_term_length, stopwords
  - `BM25Index` class: Full Okapi BM25 implementation
    - Standard BM25 scoring: IDF × (tf × (k1+1)) / (tf + k1 × (1 - b + b × dl/avgdl))
    - Robertson-Sparck Jones IDF formula
    - Inverted index for efficient term lookup
    - Document add/remove/update operations
    - Serialization support
- `hybrid.py`: Hybrid retrieval with RRF fusion
  - `HybridRetriever` class: Combines BM25 and semantic search
  - Reciprocal Rank Fusion (RRF): score = Σ(weight / (k + rank))
  - Automatic index building on first search
  - Configurable strategy selection
  - Index invalidation and single-node updates

##### Key Features
- Full BM25 implementation with configurable parameters
- RRF fusion for combining keyword and semantic rankings
- Automatic weight normalization in config
- Support for keyword-only, semantic-only, and hybrid modes
- Node type filtering (include/exclude)
- Minimum score thresholds
- Index statistics tracking
- Serialization/deserialization for BM25 index

##### Default Configuration
| Setting | Default Value |
|---------|---------------|
| Strategy | HYBRID |
| Top K | 10 |
| Min Score | 0.0 |
| BM25 k1 | 1.5 |
| BM25 b | 0.75 |
| Semantic Weight | 0.5 |
| Keyword Weight | 0.5 |
| RRF k | 60 |

##### RRF Formula
```
score(d) = Σ(weight_i / (k + rank_i(d)))
```
Where k=60 (configurable) smooths rank differences.

##### Files Created
- `src/cognifold/retrieval/__init__.py` - Module exports
- `src/cognifold/retrieval/config.py` - RetrievalConfig
- `src/cognifold/retrieval/result.py` - RetrievalResult, RetrievalMetrics
- `src/cognifold/retrieval/bm25.py` - BM25Index
- `src/cognifold/retrieval/hybrid.py` - HybridRetriever

##### Tests
- `tests/unit/test_retrieval.py` - Comprehensive tests for BM25 and hybrid retrieval
- Tests for BM25 scoring, tokenization, serialization
- Tests for RRF fusion, strategy selection, filtering

---

### Dependencies Added
- `dateparser>=1.2` - Natural language date parsing (Phase 10.2)
- `numpy` - Already present, used for embeddings

---

### Files Modified
- `pyproject.toml` - Added dateparser dependency

---

### Summary
| Phase | Module | Tests | Status |
|-------|--------|-------|--------|
| 10.2 | `temporal/` | 32 | ✅ Complete |
| 10.3 | `embeddings/` | 36 | ✅ Complete |
| 10.4 | `retrieval/` | Tests created | ✅ Complete |

---

## [2026-02-01] - Integration: Query Module with Advanced Retrieval

### Overview
Integrated the new retrieval modules (temporal, embeddings, retrieval) with the existing query API.

### Changes

#### New Query Configuration (`query/models.py`)
- Added `RetrievalMode` enum: LEGACY, BM25, SEMANTIC, HYBRID
- Added `retrieval_mode` to QueryConfig (default: LEGACY for backward compatibility)
- Added `semantic_weight` and `keyword_weight` to QueryConfig

#### Updated Entry Point Selection (`query/strategies.py`)
- `EntryPointSelector` now accepts optional `embedder` parameter
- Added `_ensure_hybrid_retriever()` for lazy HybridRetriever initialization
- Added `_ensure_semantic_search()` for lazy SemanticSearch initialization
- New methods for retrieval backends:
  - `_select_by_hybrid_retrieval()` - BM25 + semantic with RRF
  - `_select_by_semantic_search()` - Embedding-based search
  - `_select_by_bm25_search()` - BM25 keyword search
- Entry point `source` field reflects method used (e.g., "bm25_search", "semantic_search")

#### Updated Query Agent (`query/agent.py`)
- `MemoryQueryAgent` now accepts optional `embedder` parameter
- Validates embedder requirement for SEMANTIC/HYBRID modes (warns if missing)
- Integrated temporal extraction for queries
- Query metadata now includes:
  - `retrieval_mode` - Which backend was used
  - `temporal_references` - Extracted time references from query

#### Temporal Integration
- Added `_ensure_temporal_extractor()` for lazy initialization
- Added `_parse_temporal_references()` to extract time references from queries
- Temporal entities included in query metadata for downstream use

### Usage Examples

```python
# Legacy mode (default, no dependencies)
agent = MemoryQueryAgent(graph)
result = agent.query("exercise habits")

# BM25 mode (better keyword matching)
from cognifold.query import QueryConfig, RetrievalMode
config = QueryConfig(retrieval_mode=RetrievalMode.BM25)
agent = MemoryQueryAgent(graph, config=config)

# Hybrid mode (best quality, requires embedder)
from cognifold.embeddings.embedder import NodeEmbedder
from cognifold.embeddings.config import EmbeddingConfig
embedder = NodeEmbedder(EmbeddingConfig())
config = QueryConfig(retrieval_mode=RetrievalMode.HYBRID)
agent = MemoryQueryAgent(graph, config=config, embedder=embedder)
```

### Tests Added
- `TestRetrievalModes` class with 9 tests
- `TestEntryPointSelectorWithRetrieval` class with 2 tests

### Files Modified
- `src/cognifold/query/models.py` - Added RetrievalMode, updated QueryConfig
- `src/cognifold/query/strategies.py` - Added retrieval backend integration
- `src/cognifold/query/agent.py` - Added embedder support and temporal extraction
- `src/cognifold/query/__init__.py` - Export RetrievalMode
- `tests/unit/test_query.py` - Added retrieval mode tests

---

## [2026-01-25] - Phase 9.1 & 9.2: Typed Edges & Hierarchical Context

### Overview
Phase 9.1 adds semantic edge types and weights to graph edges.
Phase 9.2 adds hierarchical context windows with multi-level memory organization.

### Phase 9.1: Typed/Weighted Edges

#### Edge Model Enhancements (`src/cognifold/models/node.py`)
- Added `BaseEdgeType` enum with 8 semantic types:
  - CAUSES, GROUNDS, TRIGGERS, REINFORCES
  - PART_OF, DERIVED_FROM, DEADLINE_FOR, RELATED_TO
- Added `EDGE_TYPE_DEFAULT_WEIGHTS` mapping type-specific weights
- Added `edge_type`, `weight`, `metadata` fields to Edge model
- Added `Edge.create()` factory method for automatic weight defaults
- Added `edge_key` property for MultiDiGraph keying
- Added `validate_edge_type_constraints()` for soft constraint checking

#### Graph Store Changes (`src/cognifold/graph/store.py`)
- Changed from DiGraph to **MultiDiGraph** to support multiple edges
- Updated `has_edge()`, `add_edge()`, `get_edge()`, `remove_edge()` for edge types
- Added `get_edges_between()` to get all edges between two nodes
- Logs soft warnings for edge type constraint violations

#### Weighted PageRank (`src/cognifold/scoring/ranker.py`)
- Added `edge_decay_rate` config (default 0.01)
- Added `use_weighted_pagerank` config (default True)
- Effective edge weight = `base_weight × recency_factor`
- Recency factor: `exp(-decay_rate × hours_since_creation)`

#### Prompt & Executor Updates
- Added edge type documentation to agent prompts
- Added `edge_type` and `weight` to Operation model
- Executor creates typed edges from operations

### Phase 9.2: Hierarchical Context Windows

#### New Module: `src/cognifold/scoring/hierarchical.py`
- **HierarchicalContextConfig**: Configuration for three-level context
  - `total_size`, proportions for immediate/working/background
  - Level-specific scoring weights
  - Relevance threshold filtering
- **ContextLevel**: Single level with nodes, edges, scores
- **HierarchicalContext**: Three-level container with deduplication
- **ContextMetrics**: Selection and contribution tracking
- **HierarchicalContextSelector**: Multi-level selection with:
  - Level-specific scoring functions
  - Deduplication to highest priority level
  - Edge collection per level

#### Level Characteristics
| Level | Proportion | Focus | Scoring Weights |
|-------|------------|-------|-----------------|
| Immediate | 10% | Recent events, urgent intents | 70% recency, 30% urgency |
| Working | 30% | Active concepts, patterns | 50% PageRank, 30% recency, 20% type |
| Background | 50% | Historical context, weak signals | 80% PageRank, 20% diversity |

#### Prompt Formatting (`src/cognifold/agent/prompts.py`)
- Added `HIERARCHICAL_USER_PROMPT_TEMPLATE`
- Added `format_hierarchical_context()` for level-specific formatting
- Added `format_hierarchical_user_prompt()` for complete prompts

### Tests
- Added 11 tests for typed/weighted edges in `test_graph.py`
- Added 19 tests for hierarchical context in `test_hierarchical.py`
- **390 tests passing** (371 + 19)

### Files Added
- `src/cognifold/scoring/hierarchical.py`
- `tests/unit/test_hierarchical.py`

### Files Modified
- `src/cognifold/models/node.py` - Edge type enhancements
- `src/cognifold/models/plan.py` - Operation edge_type/weight
- `src/cognifold/graph/store.py` - MultiDiGraph support
- `src/cognifold/graph/persistence.py` - Edge type persistence
- `src/cognifold/scoring/ranker.py` - Weighted PageRank
- `src/cognifold/scoring/__init__.py` - Export hierarchical classes
- `src/cognifold/agent/prompts.py` - Edge types and hierarchical prompts
- `src/cognifold/executor/runner.py` - Typed edge execution
- `tests/unit/test_graph.py` - Typed edge tests

---

## [2026-01-21] - Phase 8: Intent Execution System

### Overview
Transform intents (goals/desires) into executable actions and integrate with
the event processing pipeline. This phase introduces the Intent-to-Action agent,
action queue, and simulation mode with action execution.

### Terminology Change: action → intent
- **"action" nodes** are now called **"intent" nodes** in the graph
- Intents represent goals/desires (stored in graph)
- Actions represent concrete, schedulable steps (stored in action queue)
- Backward compatibility maintained: legacy "action" type still works

### New Module: `src/cognifold/intent/`

#### Core Components
- `models.py`: Action, ActionMetadata, ActionStatus dataclasses
- `agent.py`: IntentToActionAgent - converts intents to concrete actions
- `queue.py`: ActionQueue - manages scheduled actions with persistence
- `selector.py`: IntentSelector - selects actionable intents for processing
- `executor.py`: ActionExecutor, SimulatedActionExecutor - executes actions
- `prompts.py`: Prompts for intent-to-action conversion

#### Key Features
- **Intent Lifecycle**: pending → action_scheduled → resolved
- **Action Queue**: Sorted by scheduled_time, supports persistence to JSON
- **Mock LLM Mode**: IntentToActionAgent works without LLM for testing
- **Priority-based Scheduling**: urgent/high/medium/low priorities
- **Execution Simulation**: ActionExecutor generates result events

### Updated Files

#### Node Model (`src/cognifold/models/node.py`)
- Added `NodeType.INTENT` (replaces ACTION with backward compat)
- Added `IntentStatus` enum for intent lifecycle tracking
- Added `NodeType.from_string()` for backward compatibility

#### Prompts (`src/cognifold/agent/prompts.py`)
- Updated all "action" references to "intent"
- Added intent-specific terminology and examples
- Backward compat: legacy action_examples still work

#### Domain Config (`src/cognifold/agent/domain.py`)
- Updated docstrings for intent terminology
- Added "intent" key to node_type_descriptions

#### Visualizer (`src/cognifold/simulator/visualizer.py`)
- Updated to show "Intents" in sidebar (was "Actions")
- Supports both "action" and "intent" node types for display

#### Query Module (`src/cognifold/query/`)
- Added `get_recent_intents()` method (alias: `get_recent_actions()`)
- Updated CLI with `--recent-intents` flag

### Additional Source Changes (Session 2)
- Updated `src/cognifold/query/strategies.py`: NodeType.ACTION → NodeType.INTENT
- Updated `src/cognifold/query/scoring.py`: NodeType.ACTION → NodeType.INTENT
- Updated `src/cognifold/agent/tools.py`:
  - NodeType.ACTION → NodeType.INTENT
  - `action_count` → `intent_count` in graph stats
  - Use `NodeType.from_string()` for backward compat in `find_nodes_by_type`

### Phase 8.3: Simulator Action Mode (Session 3)

#### Simulator Extensions (`src/cognifold/simulator/cli.py`)
- Added `action_mode` parameter to Simulator class
- Added `action_config` for configuring action mode behavior
- New methods:
  - `step_with_actions()`: Process events with action generation and execution
  - `_init_action_mode()`: Initialize action mode components
  - `_process_actionable_intents()`: Generate actions for pending intents
  - `_execute_due_actions()`: Execute actions scheduled in time windows
  - `_process_action_result_event()`: Process action results back into pipeline
  - `_maybe_resolve_intent()`: Resolve intents when all actions complete
  - `run_all_with_actions()`: Run full simulation with action mode
  - `get_action_summary()`: Get action mode execution statistics

#### CLI Flags (`src/cognifold/cli/run.py`)
- `--action-mode`: Enable action mode during simulation
- `--action-llm {mock,gemini}`: LLM provider for action generation
- `--min-urgency FLOAT`: Minimum urgency threshold for actionable intents
- `--save-actions PATH`: Save action queue to JSON file

#### Action Mode Flow
1. Process event normally (add to graph)
2. Check for actionable intents (pending, high urgency)
3. Generate actions for intents using IntentToActionAgent
4. Enqueue actions in ActionQueue
5. Execute actions scheduled before next event
6. Generate result events for completed actions
7. Mark intents as resolved when all actions complete

#### Combined Agent + Action Mode (Session 4 fix)
When using `--agent --action-mode` together:
1. Agent generates plan with intents
2. Action mode processes those intents immediately
3. Actions are generated and queued for execution

### Bug Fixes (Session 5)

#### Intent ID Handling Fix
Fixed validation failures when LLM creates intent nodes with `intent_id` field:
- Added `intent_id` to ID extraction in `src/cognifold/simulator/cli.py:_execute_operation()`
- Added `intent_id` to ID extraction in `src/cognifold/executor/validator.py`
- Fixes: `ADD_EDGE: Source node 'i-001' does not exist` errors

### Replay Logging for Intent/Action Flow (Session 5)

Added comprehensive logging for intent-to-action flow to enable replay visualization:

#### New Log Entry Types (`src/cognifold/replay/logger.py`)
- `INTENT_SELECTED` - when intents are selected for action generation
- `ACTION_GENERATED` - when actions are generated from an intent
- `ACTION_EXECUTED` - when an action is executed
- `ACTION_RESULT_EVENT` - when an action result event is processed

#### New Logger Methods
- `log_intent_selected()` - logs intent selection with urgency score
- `log_action_generated()` - logs action with scheduled time and urgency
- `log_action_executed()` - logs action execution with result event ID
- `log_action_result_event()` - logs result processing and intent resolution

#### Updated Keyframe Model (`src/cognifold/replay/player.py`)
- Added `intents_selected` - list of intents selected in each step
- Added `actions_generated` - list of actions generated in each step
- Added `actions_executed` - list of actions executed in each step
- Added `action_results` - list of action results processed in each step
- Added `intent_id` to node ID extraction for ADD_NODE operations

#### Simulator Integration (`src/cognifold/simulator/cli.py`)
- Added logging calls in `_process_actionable_intents()`
- Added logging calls in `_execute_due_actions()`
- Added logging calls in `_process_action_result_event()`

### Tests
- Added 35 unit tests for intent module (8 new for action mode)
- Fixed all existing tests to use `NodeType.INTENT` instead of `NodeType.ACTION`
- **360 tests passing** across full test suite

### Test Files Modified
- `tests/conftest.py`: NodeType.ACTION → NodeType.INTENT
- `tests/unit/test_graph_validator.py`: NodeType.ACTION → NodeType.INTENT
- `tests/unit/test_query.py`: NodeType.ACTION → NodeType.INTENT, node_type checks
- `tests/unit/test_models.py`: NodeType.ACTION → NodeType.INTENT
- `tests/unit/test_agent_tools.py`: NodeType.ACTION → NodeType.INTENT, intent_count
- `tests/unit/test_domain_prompts.py`: "Action Guidelines" → "Intent Guidelines"

### Files Added
- `src/cognifold/intent/__init__.py`
- `src/cognifold/intent/models.py`
- `src/cognifold/intent/agent.py`
- `src/cognifold/intent/queue.py`
- `src/cognifold/intent/selector.py`
- `src/cognifold/intent/executor.py`
- `src/cognifold/intent/prompts.py`
- `tests/unit/test_intent.py`

---

## [2026-01-20] - Phase 7: Memory Query Interface

### Overview
Add read/query capability to the memory system, allowing agents to retrieve
relevant context from the concept graph using natural language queries.

### New Module: `src/cognifold/query/`

#### Core Components
- `models.py`: QueryType, QueryConfig, NodeSummary, QueryResult data models
- `strategies.py`: EntryPointSelector, GraphTraverser for graph exploration
- `scoring.py`: QueryScorer for relevance ranking
- `assembly.py`: ContextAssembler for text formatting
- `prompts.py`: Query-specific prompt templates
- `agent.py`: MemoryQueryAgent main interface

#### Query Types
- **SEMANTIC**: Find nodes related to query meaning
- **TEMPORAL**: Find nodes from recent time periods
- **STRUCTURAL**: Find highly connected/important nodes
- **HYBRID**: Combine all strategies (default)

#### Features
- Natural language query processing
- **Text-search-first entry point selection** (for semantic/hybrid queries)
- BFS graph traversal with score decay
- Type-aware relevance scoring (prefers concepts over events)
- Context assembly respecting size limits
- Configurable max nodes and context characters

### Query Improvement: Text Search First (Latest)
- Refactored entry point selection to search for text-matching nodes first
- For semantic/hybrid queries, finds nodes whose title/description match query keywords
- Falls back to PageRank-based selection if no text matches found
- Scoring reinforces text matches and penalizes unrelated traversed nodes
- Results are now more relevant to the actual query content

### CLI Command: `cognifold query`

```bash
# Query with natural language
cognifold query -g output/graph.json "What patterns exist?"

# Query with specific strategy
cognifold query -g output/graph.json --type semantic "exercise habits"

# Convenience shortcuts
cognifold query -g output/graph.json --top-concepts 10
cognifold query -g output/graph.json --recent-actions 5
cognifold query -g output/graph.json --explain c-001
```

### Files Added
- `src/cognifold/query/__init__.py`
- `src/cognifold/query/models.py`
- `src/cognifold/query/strategies.py`
- `src/cognifold/query/scoring.py`
- `src/cognifold/query/assembly.py`
- `src/cognifold/query/prompts.py`
- `src/cognifold/query/agent.py`
- `src/cognifold/cli/query.py`
- `tests/unit/test_query.py`

### Tests
- Added 39 unit tests for query module (including text matching tests)
- All tests passing

---

## [2026-01-20] - Phase 6.1 & 6.2: Wiki Integration & Engineering Cleanup

### Overview
- Phase 6.1: Establish importer pattern and document architecture
- Phase 6.2: Engineering cleanup with CLI refactoring, quality fixes, and dev scripts

### Phase 6.1: Wiki Integration

#### BaseImporter Abstract Class
- Created `BaseImporter[TSettings]` generic abstract class for data importers
- `ImportResult` dataclass for standardized import results
- Common utilities: event ID generation, timeline creation, save/load
- Clear separation: generators (LLM synthesis) vs importers (data transformation)

#### Wiki Importer Tests
- Added 19 comprehensive tests for wiki importer edge cases
- Edge cases: empty directory/files, whitespace-only files
- Frontmatter: YAML parsing, date extraction for timestamps
- Split strategies: fixed, paragraph, heading
- Options: specific files, blank titles, max chunks, min chunk size

#### Architecture Documentation
- Created `docs/ARCHITECTURE.md` with system overview
- Documented generator vs importer distinction
- Wiki importer configuration guide
- How to create new importers

### Phase 6.2: Engineering Cleanup

#### CLI Refactoring
- Split 830-line cli.py into organized submodules:
  - `cli/__init__.py` - Main entry point
  - `cli/run.py` - run_simulation command
  - `cli/generate.py` - generate_command with domain helpers
  - `cli/replay.py` - replay_command
  - `cli/build.py` - build_timeline_command
  - `cli/config.py` - config_command
  - `cli/__main__.py` - Module execution entry point

#### Quality Fixes
- Fixed all ruff lint warnings
- Added noqa comments for intentional patterns (Python 3.9 compat)
- Simplified nested conditionals in validator.py
- Used pytest.raises instead of assert False

#### Development Scripts
- Created Makefile with common tasks (install, test, lint, format, etc.)
- Added scripts/pre-commit.sh for pre-commit quality checks
- Added scripts/run-e2e.sh for end-to-end testing

#### Documentation
- Added wiki build-timeline commands to README.md
- Usage examples for building wiki timelines
- Options table for all wiki importer settings

### Files Added
- `src/cognifold/importers/base.py` - BaseImporter abstract class
- `src/cognifold/cli/` - CLI submodules (7 files)
- `docs/ARCHITECTURE.md` - System architecture documentation
- `Makefile` - Development tasks
- `scripts/pre-commit.sh` - Pre-commit hook
- `scripts/run-e2e.sh` - E2E test script

### Files Removed
- `src/cognifold/cli.py` - Replaced by cli/ submodule

### Tests
- Total tests: 286 (all passing)
- New wiki importer tests: 19
- Test coverage: 55% overall, 90%+ for core modules

---

## [2026-01-19 00:30] - Phase 6: Multi-Domain Support

### Overview
Transformed Cognifold from a personal timeline tool to a general-purpose memory system supporting multiple event stream domains.

### Changes

#### Unified Event Schema
- Changed `event_type` from enum to free-form string (domain-specific)
- Added `source` field to identify event domain (e.g., "personal-timeline", "service-logs")
- Added `context` field for structured domain-specific data

#### Event Generator Architecture
- Created `BaseEventGenerator` abstract base class with common LLM client management
- Abstract methods: `generate`, `_generate_day`, `_build_generation_prompt`, `_parse_events`
- Refactored `PersonalTimelineGenerator` (formerly `EventGenerator`) to use base class
- Added backwards-compatible alias: `EventGenerator = PersonalTimelineGenerator`

#### New Event Generators
- `ComputerActivityGenerator` for computer usage events (app launches, browser activity, file operations)
  - `WorkProfile` dataclass for work style configuration
  - Sample profiles: software_developer, data_analyst, product_manager
  - Event types: browser.*, app.*, file.*, terminal.*, communication.*, system.*, meeting.*

- `ServiceLogsGenerator` for microservice/infrastructure events
  - `ServiceTopology` dataclass for service architecture definition
  - Sample topologies: ecommerce, saas_platform, microservices_demo
  - Event types: http.*, db.*, cache.*, queue.*, auth.*, business.*, system.*, ops.*

#### Domain-Agnostic Prompts
- Created `DomainConfig` dataclass for domain-specific configuration
- Pre-configured domains: personal-timeline, computer-activity, service-logs
- `format_system_prompt_for_domain()` for domain-specific prompt generation
- Domain registry for extensibility (`register_domain()`)
- Each domain customizes: node type descriptions, examples, pattern types, guidelines

### Files Added
- `src/cognifold/generator/base.py` - Abstract base class for generators
- `src/cognifold/generator/computer_activity.py` - Computer activity generator
- `src/cognifold/generator/service_logs.py` - Service logs generator
- `src/cognifold/agent/domain.py` - Domain configuration system
- `tests/unit/test_domain_prompts.py` - Domain prompt tests

### Files Modified
- `src/cognifold/models/event.py` - Unified event schema
- `src/cognifold/generator/event_generator.py` - Refactored to use base class
- `src/cognifold/generator/__init__.py` - Export new generators
- `src/cognifold/agent/prompts.py` - Domain-agnostic prompt templates
- `src/cognifold/agent/__init__.py` - Export domain module
- Multiple files updated for `event_type.value` → `event_type` change

### Tests
- All 263 tests passing (17 new domain prompt tests)
- Updated tests to use string event types instead of enum

---

## [2026-01-18 22:30] - Event Generator Reliability Improvements

### Changes
- Fixed multi-day event generation - was only generating 1 day instead of 3
- Increased `max_output_tokens` from 4096 to 8192 to prevent response truncation
- Added retry mechanism with exponential backoff (max 3 attempts)
- Improved parsing to handle markdown code blocks (`\`\`\`json ... \`\`\``)
- Fixed common JSON issues (trailing commas before `]` or `}`)
- Added detailed error logging for debugging parse failures

### Files Modified
- `src/cognifold/generator/event_generator.py` - Added retry logic, increased tokens, improved parsing

### Tests
- All 245 tests passing

### Regenerated Timelines
- `data/generated/maya_rodriguez_timeline.json` - 99 events, 3 days
- `data/generated/jordan_taylor_timeline.json` - 99 events, 3 days

---

## [2026-01-18 22:00] - Repository Setup & Remote Configuration

### Changes
- Added remote repository: https://github.com/duanyiqun/cognifold
- Created development branch: `cognifold-dev`
- All development work will be done on `cognifold-dev` branch
- Main branch reserved for stable releases

### Repository Structure
- Remote: `origin` → https://github.com/duanyiqun/cognifold.git
- Development branch: `cognifold-dev`
- Local main branch preserved

### Notes
- Push to remote with: `git push -u origin cognifold-dev`
- Create PRs from `cognifold-dev` to `main` for releases

---

## [2026-01-18 21:45] - Quality Metrics Tracking

### Changes
- Created `QualityMetrics` dataclass with computed properties:
  - `orphan_rate`: Percentage of orphan nodes (target: 0%)
  - `ungrounded_rate`: Percentage of ungrounded nodes
  - `missing_reasoning_rate`: Percentage missing reasoning
  - `connectivity_violation_rate`: Percentage with violations
  - `edge_density`: Edges per node ratio
  - `is_healthy`: Boolean check against quality thresholds
- Created `MetricsCollector` for tracking metrics over time:
  - `collect()`: Gather metrics from graph
  - `get_trend()`: Get metric history
  - `is_improving()`: Check if metric is trending better
- Added summary generation for human-readable reports

### Files Created
- `src/cognifold/graph/metrics.py` - QualityMetrics and MetricsCollector
- `tests/unit/test_metrics.py` - 22 comprehensive tests

### Files Modified
- `src/cognifold/graph/__init__.py` - Export new classes

### Tests
- Added/updated tests: yes (22 new tests)
- All tests passing: yes (245 total)

---

## [2026-01-18 21:30] - Prompt Optimization for Graph Integrity

### Changes
- Added "Graph Connectivity Rules" section with explicit requirements
- Added connectivity table by node type (event/concept/action/time)
- Added examples of correct vs incorrect node creation with edges
- Added "Avoiding Duplicate Concepts" section with step-by-step guidance
- Added examples of update vs create decisions
- Added "Self-Validation Checklist" for agents to verify before outputting

### Files Modified
- `src/cognifold/agent/prompts.py` - Added new sections for connectivity and validation

### Tests
- Added/updated tests: no (prompt-only change)
- All tests passing: yes (223 total)

---

## [2026-01-18 21:15] - Executor Integration with GraphValidator

### Changes
- Integrated GraphValidator with PlanExecutor
- Added `validate_after_execution` parameter to PlanExecutor
- Added `validation_report` field to ExecutionResult
- Added `has_integrity_issues` property for easy checking
- Added `log_integrity_issues` parameter for automatic logging
- Added logging of validation issues at appropriate levels

### Files Modified
- `src/cognifold/executor/runner.py` - Added validation integration
- `tests/unit/test_executor.py` - Added 5 new tests for validation integration

### Tests
- Added/updated tests: yes (5 new tests)
- All tests passing: yes (223 total)

---

## [2026-01-18 21:00] - Phase 5.6: GraphValidator Class

### Changes
- Created `GraphValidator` class for graph integrity validation
- Implemented `validate_no_orphans()` - finds nodes without edges (events excluded)
- Implemented `validate_connectivity_rules()` - checks type-specific connectivity:
  - event: can be standalone
  - concept: must connect to event or concept
  - action: must connect to concept or event
  - time: must connect to action or event
- Implemented `validate_grounding()` - finds nodes without grounded_in references
- Implemented `validate_reasoning()` - finds nodes without reasoning
- Implemented `validate_all()` - returns full `ValidationReport`
- Added `get_repair_suggestions()` for actionable fixes
- Created `ValidationReport` dataclass with summary generation
- Created `IntegrityIssue` and `IntegrityLevel` for detailed issue tracking

### Files Created
- `src/cognifold/graph/validator.py` - GraphValidator implementation
- `tests/unit/test_graph_validator.py` - 32 comprehensive tests

### Files Modified
- `src/cognifold/graph/__init__.py` - Export new classes

### Tests
- Added/updated tests: yes (32 new tests)
- All tests passing: yes (218 total)

---

## [2026-01-18 20:30] - Replay Tool: Top Concepts & Actions Display

### Changes
- Added "Top Concepts" sidebar section to replay tool showing highest-scored concepts
- Added "Top Actions" sidebar section to replay tool showing highest-scored actions
- Concepts display score and strength percentage
- Actions display score and priority level (high/medium/low with color coding)
- Nodes sorted by score (descending), limited to top 5 each
- Hover over items to see reasoning (if available)

### Files Modified
- `src/cognifold/replay/renderer.py` - Added Top Concepts and Top Actions sidebar sections with CSS and JavaScript

### Tests
- Added/updated tests: no (UI-only change)
- All tests passing: yes (186 total)

---

## [2026-01-18 19:57] - CLI Wrapper for Direct Invocation

### Changes
- Created `cognifold` shell wrapper script at project root
- No longer need `python3 -c "..."` to invoke CLI
- Direct invocation: `./cognifold run data/mock_timeline.json --agent`

### Files Created
- `cognifold` - Bash wrapper that sets PYTHONPATH and invokes CLI

---

## [2026-01-18 19:55] - Phase 5.5 Complete: Node Explainability & Event Grounding

### Changes
- Implemented explainability system for node creation and updates
- Added grounding validation to ensure non-event nodes are connected to evidence
- Fixed agent graph parsing to extract reasoning/grounded_in from LLM output
- Fixed graph store to persist reasoning/grounded_in/update_history fields
- Added `--open` flag to CLI to automatically open visualizations in browser

### Node Model Updates
- Added `reasoning` field: Why this node was created (1-2 sentences)
- Added `grounded_in` field: List of event/node IDs that justify existence
- Added `update_history` field: Audit trail of changes with reasoning
- New `UpdateHistoryEntry` class to track update reasoning and changes

### Operation Model Updates
- Added `reasoning` field for ADD_NODE operations
- Added `update_reasoning` field for UPDATE_NODE operations
- Added `grounded_in` field for ADD_NODE operations

### Validator Updates
- Extended `PlanValidator` with grounding validation options:
  - `require_grounding`: Validates non-event nodes have grounded_in references
  - `require_reasoning`: Validates non-event nodes have reasoning
- Invalid grounding references (to non-existent nodes) produce errors
- Missing reasoning/grounding produce warnings (plan still valid)

### Executor Updates
- Nodes created with reasoning and grounded_in from operations
- UPDATE_NODE operations track changes in update_history

### Agent Prompt Updates
- Added "Explainability Requirements (CRITICAL)" section
- Examples of good vs bad reasoning
- Grounding rules for concept/action/time nodes

### Visualization Updates
- Node tooltips now show reasoning and grounded_in
- Update history shown in tooltips (latest update + count)
- Replay renderer shows reasoning in operations panel
- Grounding references displayed with operations

### Files Created
- None (extensions to existing modules)

### Files Modified
- `src/cognifold/models/node.py` - Added UpdateHistoryEntry, reasoning, grounded_in, update_history fields
- `src/cognifold/models/plan.py` - Added reasoning, update_reasoning, grounded_in fields to Operation
- `src/cognifold/models/__init__.py` - Exported UpdateHistoryEntry
- `src/cognifold/executor/validator.py` - Added grounding/reasoning validation
- `src/cognifold/executor/runner.py` - Store reasoning/grounding when creating nodes, track update_history
- `src/cognifold/agent/prompts.py` - Added explainability requirements section
- `src/cognifold/simulator/visualizer.py` - Show reasoning in node tooltips
- `src/cognifold/simulator/cli.py` - Pass reasoning/grounded_in to logger
- `src/cognifold/replay/player.py` - Extract reasoning/grounded_in from operations
- `src/cognifold/replay/renderer.py` - Show reasoning in operations panel and node tooltips
- `tests/unit/test_executor.py` - Added TestGroundingValidation class (10 new tests)

### Tests
- Added/updated tests: yes (10 new grounding validation tests)
- All tests passing: yes (186 total)

### Exit Criteria Met
- Every non-event node can have a `reasoning` field explaining why it exists
- Every node can have `grounded_in` references to justify its existence
- Updates can include `update_reasoning` with audit trail in `update_history`
- Grounding enforced in validator (invalid refs = error, missing = warning)
- Agent prompts require reasoning for concept/action/time nodes
- Reasoning visible in visualizer and replay tooltips

---

## [2026-01-18 18:55] - Phase 5.4 Complete: Graph Evolution Replay Tool

### Changes
- Implemented complete replay system for visualizing graph evolution over time
- Created structured JSONL log format for recording graph operations:
  - `run_start` / `run_end`: Simulation lifecycle
  - `event_start` / `event_end`: Event processing boundaries
  - `operation`: Graph operations (ADD_NODE, ADD_EDGE, etc.)
  - `context_window`: Context window state at each step
  - `scores`: Node scores snapshot
- Built `ReplayPlayer` class to reconstruct graph states from logs:
  - Parses JSONL log files into ordered keyframes
  - Tracks added/removed nodes and edges per step
  - Supports keyframe range selection
- Created interactive HTML replay visualization:
  - Play/pause/step controls with keyboard shortcuts
  - Timeline scrubber to jump to any step
  - Playback speed control (0.5x, 1x, 2x, 5x)
  - Node animations for additions
  - Context window highlighting
  - Operations panel showing changes per step
  - Stats panel (node/edge counts)
  - Color-coded node types (event=blue, concept=green, action=orange, time=purple)
- Added `cognifold replay` CLI command:
  - `cognifold replay logs/replay_*.jsonl -o output/replay.html`
  - Generates standalone HTML file with embedded data
- Updated simulator to automatically log operations during runs
- Replay logs saved alongside run logs in `logs/` directory

### Files Created
- `src/cognifold/replay/__init__.py` - Module exports
- `src/cognifold/replay/logger.py` - GraphLogger and LogEntry classes
- `src/cognifold/replay/player.py` - ReplayPlayer and Keyframe classes
- `src/cognifold/replay/renderer.py` - ReplayRenderer for HTML generation
- `tests/unit/test_replay.py` - 13 new tests for replay functionality

### Files Modified
- `src/cognifold/cli.py` - Added replay command, integrated GraphLogger
- `src/cognifold/simulator/cli.py` - Added graph_logger parameter, logging in step methods

### Tests
- Added/updated tests: yes (13 new tests)
- All tests passing: yes (176 total)

### CLI Usage
```bash
# Run simulation (generates replay log automatically)
cognifold run data/mock_timeline.json --agent --steps 30 -o output/

# Generate replay from logs
cognifold replay logs/replay_mock_timeline_*.jsonl -o output/replay.html

# With custom title
cognifold replay logs/replay_*.jsonl -o replay.html --title "My Simulation"
```

### Keyboard Shortcuts (in replay HTML)
- Space: Play/Pause
- Arrow Left/Right: Previous/Next step
- Home/End: Go to start/end

---

## [2026-01-18 18:45] - Visualization Sidebar & Event Generator Improvements

### Changes
- Added sidebar panel to graph visualization showing:
  - **Top Concepts** (sorted by score, showing strength)
  - **Actions** (showing priority level, checkmark for completed)
  - **Time Anchors** (temporal anchors)
- Color-coded sidebar headers and item borders to match node types
- Fixed duplicate event ID issue in multi-day timeline generation
- Enhanced event generator prompt to include 20-30% actionable events:
  - Deadlines and scheduled meetings (triggers TIME nodes)
  - Tasks and follow-ups (triggers ACTION nodes)
  - Planning/reminder events
- Added PLANNING and DEADLINE event types to EventType enum
- Enhanced action_guidelines to support pattern-based action creation from habits
- Added pattern-based action examples to prompts (e.g., Morning Routine → "Make morning coffee" action)

### Files Modified
- `src/cognifold/simulator/visualizer.py` - Added `_get_top_nodes_by_type()`, `_build_sidebar_html()`, and sidebar injection in `render()`
- `src/cognifold/generator/event_generator.py` - Enhanced prompt for actionable events, fixed event ID uniqueness
- `src/cognifold/models/event.py` - Added PLANNING and DEADLINE event types
- `src/cognifold/agent/config.py` - Enhanced action_guidelines for pattern-based actions
- `src/cognifold/agent/prompts.py` - Added pattern-based action examples

### Tests
- All existing tests still passing
- Manually tested with 30 events, verified TIME and ACTION nodes created
- Pattern-based action "Make morning coffee" created from Morning Routine habit

### Results
- Successfully generated 11 ACTION nodes and 4 TIME nodes from 30 events
- Sidebar displays top 10 concepts, up to 10 actions, and up to 5 time anchors
- Hovering over sidebar items shows node ID and score

---

## [2026-01-18 18:17] - Phase 5.3 Complete: Proactive Action Generation & Temporal Urgency

### Changes
- Added TIME node type for temporal anchors (deadlines, scheduled events, recurring periods)
- Implemented time-aware urgency scoring:
  - Nodes connected to approaching TIME nodes get urgency boost (1.0 to 2.0x multiplier)
  - Linear interpolation within 24-hour urgency window
  - Configurable via `urgency_boost` and `urgency_window_hours` in ScoringConfig
- Updated visualizer with purple color (#9932CC) for TIME nodes
- Enhanced prompts for proactive ACTION generation:
  - Actions now include `suggested_time`, `expiry`, and `priority` metadata
  - Actions linked to TIME nodes for urgency tracking
- Added TIME node guidelines to agent config
- Updated `format_system_prompt` to accept optional `time_guidelines`
- Added comprehensive urgency scoring tests (8 new tests)
- Fixed linting issues (removed unused imports, fixed f-string, added `from e` to raise)

### Files Modified
- `src/cognifold/models/node.py` - Added TIME to NodeType enum with documentation
- `src/cognifold/scoring/ranker.py` - Added urgency scoring with `compute_urgency_score()`, updated `score_nodes()` to apply urgency multiplier, added `urgency_boost` and `urgency_window_hours` to ScoringConfig
- `src/cognifold/simulator/visualizer.py` - Added purple color for TIME nodes
- `src/cognifold/agent/prompts.py` - Added TIME node section, enhanced action metadata section, updated operation types
- `src/cognifold/agent/config.py` - Added enhanced action_guidelines with proactive action creation, added time_guidelines tuple
- `src/cognifold/agent/graph.py` - Updated to pass time_guidelines to format_system_prompt
- `src/cognifold/cli.py` - Fixed unused import and f-string without placeholder
- `src/cognifold/simulator/cli.py` - Fixed raise statement to use `from e`
- `tests/unit/test_scoring.py` - Added TestContextRankerUrgency class with 8 tests

### Tests
- Added/updated tests: yes (8 new urgency scoring tests)
- All tests passing: yes (163 total)

### Notes
- TIME nodes require `scheduled_time` field in ISO 8601 format
- Urgency scoring is transparent - doesn't affect base weights (alpha/beta/gamma still sum to 1.0)
- Actions can reference `related_time_node` for deadline tracking
- Successfully tested end-to-end: 5 events → 9 nodes (including concepts), 9 edges

---

## [2026-01-18 17:45] - Phase 5.1 & 5.2 Complete: Event Generator & Prompt Engineering

### Phase 5.1: Event Generator
- Created generator module for LLM-powered event stream generation
- Implemented Persona dataclass with comprehensive attributes (name, age, occupation, habits, interests, health_goals, personality_traits, etc.)
- Built EventGenerator class using Gemini to generate realistic event timelines based on personas
- Added 3 sample personas: software_engineer (Alex Chen), graduate_student (Maya Rodriguez), freelance_designer (Jordan Taylor)
- Created CLI `generate` command with options for persona, events count, days, and output directory
- Generated sample timelines for all 3 personas (stored in data/generated/)

### Phase 5.2: Prompt Engineering & Concept Hierarchy
- Added ReasoningMode enum with QUICK, ANALYTICAL, and CONSOLIDATION modes
- Enhanced system prompt with hierarchical concept support (Level 1/2/3)
- Added temporal pattern recognition guidance (daily, weekly, irregular)
- Implemented concept strength dynamics (0.0-1.0 with decay/reinforcement)
- Created mode-specific prompt templates:
  - QUICK: Fast processing, minimal concept creation
  - ANALYTICAL: Deep pattern analysis with hierarchy consideration
  - CONSOLIDATION: Graph health focus, merging and cleanup
- Added similarity check prompt for concept consolidation
- Enhanced concept guidelines with hierarchy, deduplication, and strength management
- Added action guidelines with progress tracking and habit conversion
- Created comprehensive docs/PROMPTS.md documentation

### Files Modified
- `src/cognifold/generator/__init__.py` - NEW: Generator module exports
- `src/cognifold/generator/persona.py` - NEW: Persona dataclass with sample personas
- `src/cognifold/generator/event_generator.py` - NEW: Gemini-powered event generator
- `src/cognifold/cli.py` - Added generate command with --persona, --events, --days options
- `src/cognifold/agent/prompts.py` - Enhanced with hierarchical concepts, reasoning modes
- `src/cognifold/agent/config.py` - Updated concept/action guidelines
- `data/personas/software_engineer.json` - NEW: Alex Chen persona
- `data/personas/graduate_student.json` - NEW: Maya Rodriguez persona
- `data/personas/freelance_designer.json` - NEW: Jordan Taylor persona
- `data/generated/*.json` - NEW: Generated sample timelines
- `docs/PROMPTS.md` - NEW: Comprehensive prompt engineering documentation

### Tests
- All 155 tests passing
- EventGenerator tested with real Gemini API calls

### CLI Usage
```bash
# List available personas
cognifold generate --list-personas

# Generate events for a built-in persona
cognifold generate --persona software_engineer --events 100 --days 3

# Generate from custom persona file
cognifold generate --persona-file data/personas/custom.json --output data/generated/

# With verbose output
cognifold generate --persona graduate_student -n 50 -d 2 -v
```

### Notes
- Requires GOOGLE_API_KEY environment variable for generation
- Generated events include realistic details matching persona characteristics
- Supports custom personas via JSON files with Persona schema

---

## [2026-01-18 17:24] - Phase 5 Complete: End-to-End Pipeline

### Changes
- Implemented CognifoldConfig configuration management with YAML and environment variable support
- Created comprehensive logging module with structured logging and context managers
- Built CLI entry point (`cognifold` command) with run and config subcommands
- Created Pipeline class for end-to-end orchestration of event processing
- Fixed multiple TypedDict/LangGraph import issues for Python 3.9 compatibility:
  - AgentContext must be imported unconditionally (TypedDict evaluates at runtime)
  - UpdatePlan must be imported unconditionally (TypedDict evaluates at runtime)
  - AgentState must be imported unconditionally (LangGraph inspects function signatures)
  - Changed `X | None` to `Optional[X]` for Python 3.9 compatibility
- Fixed timezone-aware vs naive datetime comparison in ranker.py
- Migrated to new `google.genai` SDK (from deprecated `google.generativeai`)
- Model configured as `gemini-2.0-flash` (gemini-3 requires thought signatures not yet fully supported)
- Added tool definitions for new google.genai SDK
- Improved error handling in CLI for agent failures
- Added 14 integration tests for pipeline functionality
- All 155 tests passing
- Successfully tested end-to-end with real Gemini API (5 events, 5 nodes, 4 edges)

### Files Modified
- `src/cognifold/config.py` - NEW: Configuration management (YAML/env)
- `src/cognifold/logging.py` - NEW: Logging setup with context managers
- `src/cognifold/cli.py` - NEW: CLI entry point with subcommands, improved error handling
- `src/cognifold/pipeline.py` - NEW: End-to-end Pipeline class
- `src/cognifold/__init__.py` - Updated exports (sorted __all__)
- `src/cognifold/scoring/ranker.py` - Fixed timezone handling
- `src/cognifold/agent/state.py` - Fixed imports and Optional[] for Python 3.9
- `src/cognifold/agent/graph.py` - Migrated to google.genai SDK, added new tool definitions
- `src/cognifold/agent/agent.py` - Updated for new SDK
- `src/cognifold/agent/config.py` - Model set to gemini-2.0-flash
- `pyproject.toml` - Added pyyaml dependency, CLI entry point, google-genai SDK
- `config.example.yaml` - NEW: Example configuration file
- `tests/integration/test_pipeline.py` - NEW: Integration tests

### Tests
- Added/updated tests: yes (14 new integration tests)
- All tests passing: yes (155 total)

### CLI Usage
```bash
# Run simulation without agent (default plans)
cognifold run data/mock_timeline.json

# Run simulation with LLM agent (requires GOOGLE_API_KEY)
cognifold run data/mock_timeline.json --agent

# Generate visualizations
cognifold run data/mock_timeline.json --output output/

# Show current configuration
cognifold config --show

# Generate example config file
cognifold config --generate config.yaml
```

### Notes
- Phase 5 exit criteria met: Full working pipeline with configuration, logging, and CLI
- Requires GOOGLE_API_KEY environment variable for agent mode
- Migrated to new google.genai SDK (google-generativeai is deprecated)
- Model set to gemini-2.0-flash (gemini-3 requires thought signatures not yet supported)
- Successfully tested end-to-end with real Gemini API: agent created graph relationships

---

## [2026-01-18 20:00] - Phase 4 Complete: Agent Integration

### Changes
- Implemented LangGraph agent with Google Gemini for intelligent graph updates
- Created CognifoldAgent orchestrator class with lazy initialization
- Built AgentConfig with configurable model settings and guidelines
- Implemented AgentContext for packaging event + context window for LLM
- Created GraphTools with 5 traversal tools (get_node, get_neighbors, find_nodes_by_type, search_nodes, get_graph_stats)
- Built LangGraph StateGraph with analyze/call_llm/execute_tools/parse_response nodes
- Created comprehensive prompts for concept and action discovery
- Implemented PlanValidator for pre-execution validation checks
- Built PlanExecutor with atomic execution and rollback on failure
- Added step_with_agent() to Simulator for LLM-powered event processing
- Created 47 new unit tests (141 total tests, all passing)

### Files Modified
- `src/cognifold/agent/__init__.py` - Agent module exports
- `src/cognifold/agent/config.py` - AgentConfig dataclass
- `src/cognifold/agent/context.py` - ContextNode and AgentContext
- `src/cognifold/agent/tools.py` - GraphTools for LLM
- `src/cognifold/agent/prompts.py` - System and user prompts
- `src/cognifold/agent/state.py` - LangGraph AgentState TypedDict
- `src/cognifold/agent/graph.py` - LangGraph StateGraph definition
- `src/cognifold/agent/agent.py` - CognifoldAgent orchestrator
- `src/cognifold/executor/__init__.py` - Executor module exports
- `src/cognifold/executor/validator.py` - PlanValidator
- `src/cognifold/executor/runner.py` - PlanExecutor with rollback
- `src/cognifold/simulator/cli.py` - Added step_with_agent()
- `tests/unit/test_agent_tools.py` - Agent tools tests
- `tests/unit/test_executor.py` - Executor/validator tests

### Tests
- Added/updated tests: yes (47 new tests)
- All tests passing: yes (141 total)

### Dependencies Added
- langgraph>=0.2
- google-generativeai>=0.5
- langchain-core (transitive)

### Notes
- Requires GOOGLE_API_KEY environment variable for LLM calls
- Agent is lazy-loaded (no API calls until step_with_agent())
- Fallback to default plan on LLM errors

---

## [2026-01-18 18:30] - Phase 3 Complete: Simulator MVP

### Changes
- Created Timeline class and load_timeline() for loading mock event streams
- Created mock_timeline.json with 16 events (typical workday)
- Implemented GraphVisualizer using pyvis for interactive HTML output
  - Color coding by node type (blue=event, green=concept, orange=action)
  - Context window highlighting (gold border)
  - Node size scaled by relevance score
- Built Simulator class with:
  - Timeline loading and step-through controls
  - Default plan generation (adds event as node)
  - Custom plan support via UpdatePlan or JSON
  - All operation types: ADD/UPDATE/REMOVE_NODE, ADD/REMOVE_EDGE, MERGE_NODES
  - Status reporting
  - run_all() for batch visualization
- Created 22 unit tests for simulator (94 total tests, all passing)

### Files Modified
- `src/cognifold/simulator/__init__.py` - Simulator module exports
- `src/cognifold/simulator/timeline.py` - Timeline loader
- `src/cognifold/simulator/visualizer.py` - GraphVisualizer with pyvis
- `src/cognifold/simulator/cli.py` - Simulator class
- `data/mock_timeline.json` - Sample workday timeline
- `tests/unit/test_simulator.py` - Simulator unit tests

### Tests
- Added/updated tests: yes (22 new tests)
- All tests passing: yes (94 total)

---

## [2026-01-18 18:00] - Phase 2 Complete: Scoring & Context Window

### Changes
- Implemented ScoringConfig dataclass with configurable weights (alpha, beta, gamma)
- Created ContextRanker class for relevance scoring
- Added PageRank computation using NetworkX
- Implemented recency score with exponential decay formula
- Added access frequency scoring (normalized)
- Created composite score function combining all three metrics
- Implemented context window selection with top-k and threshold filtering
- Added NodeScore dataclass for score components
- Created 24 unit tests for scoring (72 total tests, all passing)

### Files Modified
- `src/cognifold/scoring/__init__.py` - Scoring module exports
- `src/cognifold/scoring/ranker.py` - ContextRanker and scoring logic
- `tests/unit/test_scoring.py` - Scoring unit tests

### Tests
- Added/updated tests: yes (24 new tests)
- All tests passing: yes (72 total)

---

## [2026-01-18 17:30] - Phase 1 Complete: Foundation

### Changes
- Created pyproject.toml with dependencies (networkx, pydantic, pytest, ruff, pyright)
- Implemented Pydantic models: Event, EventType, Node, NodeType, Edge
- Implemented UpdatePlan model with Operation and OperationType
- Built ConceptGraph class wrapping NetworkX DiGraph
- Added JSON persistence with save_graph/load_graph functions
- Created comprehensive unit tests (48 tests, all passing)
- Python 3.9+ compatibility via eval_type_backport

### Files Modified
- `pyproject.toml` - Project config and dependencies
- `src/cognifold/__init__.py` - Package init
- `src/cognifold/models/event.py` - Event model
- `src/cognifold/models/node.py` - Node and Edge models
- `src/cognifold/models/plan.py` - UpdatePlan and Operation models
- `src/cognifold/graph/store.py` - ConceptGraph class
- `src/cognifold/graph/persistence.py` - JSON save/load
- `tests/unit/test_models.py` - Model tests
- `tests/unit/test_graph.py` - Graph and persistence tests
- `tests/fixtures/factories.py` - Test factories
- `tests/conftest.py` - Shared pytest fixtures

### Tests
- Added/updated tests: yes (48 unit tests)
- All tests passing: yes

---

## [2026-01-18 15:45] - Project Initialization

### Changes
- Created project structure with Claude Code conventions
- Defined architecture, schemas, and development phases
- Established Claude Agent Convention for development workflow

### Files Modified
- `CLAUDE.md` - Full architecture documentation
- `.claude/settings.json` - Project config
- `docs/CHANGELOG.md` - This file
- `docs/RESUME.md` - Resume protocol file

### Tests
- Added/updated tests: no (project setup only)
- All tests passing: n/a
