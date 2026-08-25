# Cognifold Benchmark System

**This document is an extension of `CLAUDE.md`. Read `CLAUDE.md` first.**

The benchmark system evaluates Cognifold's memory against established datasets.
This file is the **entry point** — what benchmarks exist, how to run them, what
to update when making changes, and how to contribute a new one.

---

## 📄 Paper results (canonical)

Headline numbers are as reported in the technical report — [arXiv:2605.13438v3](https://arxiv.org/abs/2605.13438), *CogniFold: Always-On Proactive Memory via Cognitive Folding*. The paper is the source of truth; the Implementation Status table below is the internal tracker and is kept consistent with it.

> **Why these numbers (not the highest we can get).** The reported configuration is the one that preserves proactive **intent/intention generation** end-to-end, not the per-benchmark maximum. Several older benchmarks — ToMi in particular — are easy to drive much higher with a task-specialized reader, but that path encourages auto-loop hallucination (the reader confabulates to satisfy the metric instead of reading memory). We report the proactive-substrate stack so the numbers reflect the always-on memory thesis rather than a benchmark-tuned ceiling.

### LongMemEval (Table 5) — J-Score, N=500

Stack: build `gpt-4o-mini`, answer `gpt-5.4-mini`, judge `gpt-4o`.

| System | SSU | SSA | SSP | MS | KU | TR | **Overall** |
|---|---|---|---|---|---|---|---|
| Chronos (High) | 98.6 | 100.0 | 100.0 | 88.7 | 100.0 | 95.5 | 95.6 |
| Mastra | 95.7 | 94.6 | 100.0 | 87.2 | 96.2 | 95.5 | 94.9 |
| **CogniFold** | **97.1** | **100.0** | **93.3** | **91.0** | **94.9** | **88.7** | **93.0** |
| ENGRAM | 97.1 | 87.5 | 93.3 | 60.2 | 74.4 | 55.6 | 71.4 |
| Zep | 92.9 | 80.4 | 56.7 | 57.9 | 83.3 | 62.4 | 71.2 |

### LoCoMo (Table 4) — J-Score, Mem0 protocol

Stack: `gpt-4o-mini` read/write, judge `gpt-4o-mini`, `--event-stream`.

| Category | CogniFold | ENGRAM | MemOS | Zep |
|---|---|---|---|---|
| Single-Hop | 90.49 | 79.90 | 81.09 | 79.79 |
| Multi-Hop | 67.38 | 79.79 | 67.49 | 74.11 |
| Temporal | 78.50 | 70.79 | 75.18 | 67.71 |
| Open Domain | 50.00 | 72.92 | 55.90 | 66.04 |
| **Overall** | **81.23** | **77.55** | **75.80** | **75.14** |
| Overall F1 | 35.71 | 21.08 | 45.27 | 41.23 |

### CogEval-Bench (Table 3) & downstream (Fig. 4)

Stack: `gpt-4o-mini` extraction/reader, `text-embedding-3-small`.

- **CogEval-Bench** — Harmony **0.476** (GraphRAG 0.323), Gold-F1 **0.358**, LLM-Quality **0.733**, Purity **0.361** (all others 0.000), Proactivity **0.614** (all others 0.000), Compression **4.6×** (GraphRAG 1.2×, Mem0 1.0×). Only CogniFold is non-zero on Purity **and** Proactivity.
- **MuSiQue** — F1 **58.7** vs HippoRAG 2 49.3.
- **BABILong** — **85.0** vs ARMT (fine-tuned) 83.8.
- **ToMi** — **83.5** vs AutoToM 80.2.

---

## ⚠️ Consolidation and `--event-stream` — how it actually works

Inter-session consolidation (`merge_similar_concepts` + `prune_orphan_concepts`)
is central to the always-on memory thesis, but the mechanism differs by runner:

- **LoCoMo**: consolidation is gated behind the `--event-stream` flag, which
  exists **only** in `benchmarks/locomo/run_benchmark.py`. Paper-grade LoCoMo
  runs MUST pass it (`scripts/reproduce.sh locomo` does so automatically).
  Sanity-check the log for `Inter-session consolidation:` lines.
- **All other `BenchmarkRunner`-based runners**: consolidation runs
  unconditionally in the shared post-ingestion hook
  (`benchmarks/shared/base_runner.py`, step after ingestion). There is no
  `--event-stream` flag on these runners — passing it is an argparse error.

Canonical LoCoMo (full 10-conv, Mem0 protocol):

```bash
PYTHONPATH=src python -u -m benchmarks.locomo.run_benchmark \
    --event-stream --model openai:gpt-4o-mini
```

Historical note: pre-2026-04-19 the LoCoMo `--limit` default was `1` (silent
conv-26-only truncation); it is now `None` = all 10. If a full run logs
`Loaded 1 conversations`, that regression is back.

---

## Implementation Status

Nine benchmark directories exist in-tree today. Rows for retired benchmarks
(MSC, FutureX, TimeQA, QMSum, SocialIQA, SafetyBench, RGB) keep their last
recorded numbers for the paper's record but have **no code in this repo**.

| Benchmark | Location | Status | Accuracy (latest) | Note |
|-----------|----------|--------|-------------------|------|
| **LoCoMo** | `benchmarks/locomo/` | In-tree, tested | **81.23% J-Score overall** (paper Table 4) | vs ENGRAM 77.55 · MemOS 75.80 · Zep 75.14 |
| **LongMemEval** | `benchmarks/longmemeval/` | In-tree, tested | **93.0% J-Score overall** (paper Table 5, N=500) | headline benchmark; entry point is `run_eval.py`, not `run_benchmark.py` — see below |
| **CogEval-Bench** | `benchmarks/cogeval/` | In-tree, tested | **Harmony 0.476, Purity 0.361, Proactivity 0.614, 4.6×** | dataset is generated (`generate_dataset.py`), not downloaded |
| **BABILong** | `benchmarks/babilong/` | In-tree, tested | **85.0** (paper Fig. 4) | exceeds ARMT — fine-tuned (83.8) |
| **MuTual** | `benchmarks/mutual/` | In-tree, tested | **93.2% acc** (N=500) | cleanest runner — use as the template |
| **MuSiQue-Ans** | `benchmarks/musique/` | In-tree, tested | **F1 58.7** (paper Fig. 4, N=500) | exceeds HippoRAG 2 (49.3) |
| **NarrativeQA** | `benchmarks/narrativeqa/` | In-tree, tested | **F1 0.720 / ROUGE-L 0.712** (N=500) | |
| **ToMi** | `benchmarks/tomi/` | In-tree, tested | **83.5 EM** (paper Fig. 4) | exceeds AutoToM (80.2) |
| **StreamingQA** | `benchmarks/streamingqa/` | In-tree, tested | **78.4% EM / F1 0.573** (N=500) | |
| MSC / FutureX / TimeQA / QMSum / SocialIQA / SafetyBench / RGB | *(removed)* | Retired | historical | code no longer in tree |

---

## File Map

### Benchmark code (`benchmarks/`)

```
benchmarks/
├── shared/                      # THE shared infrastructure — read this first
│   ├── base_runner.py           #   BenchmarkRunner ABC + run() pipeline + LLM helpers
│   ├── baseline_runner.py       #   Direct-LLM / RAG baselines
│   ├── graph_evolution_tracker.py
│   └── stats_utils.py           #   wilson_ci, bootstrap_ci_mean
├── _utils.py                    # embedding resolution (the canonical embedder factory)
├── analysis_utils.py            # wrong-case enrichment + save_wrong_cases
├── compare_fast.py              # classic-vs-fast ingestion A/B
├── babilong/    cogeval/    locomo/    longmemeval/
├── musique/     mutual/     narrativeqa/
├── streamingqa/ tomi/
└── scripts/plot_graph_evolution.py
```

Each benchmark dir: `run_benchmark.py` (runner), `download_data.py` (dataset
fetch), `README.md`. Exceptions: **longmemeval** uses `run_eval.py` +
`symbolic_resolver.py` + `runs/<label>/` snapshots + `HISTORY.md`;
**cogeval** generates its dataset via `generate_dataset.py` and ships extra
baseline/experiment runners.

### Entry points

| What | Command |
|---|---|
| Any base-runner benchmark | `bash scripts/reproduce.sh {cogeval,locomo,musique,narrativeqa,tomi,babilong,mutual,streamingqa}` |
| All of the above | `bash scripts/reproduce.sh all` |
| LongMemEval | `bash scripts/parallel_longmemeval.sh [N_PARALLEL] [STRATIFIED] [TOTAL_LIMIT] [LABEL]` — or the Claude Code skills `.claude/skills/longmemeval-run` (one-shot) / `longmemeval-iterate` (iteration campaign) |
| Runner sanity check | `PYTHONPATH=src python test_benchmarks.py` (8 base runners; longmemeval not covered) |

**Datasets are NOT auto-downloaded.** Run `benchmarks/<name>/download_data.py`
before the first run (a missing dataset exits with an error). LongMemEval is
the exception — `run_eval.py` self-downloads.

**Reproducing paper numbers**: `reproduce.sh` defaults to `MODEL=openai:gpt-5`;
the paper stack is `gpt-4o-mini` (build) — set `MODEL=openai:gpt-4o-mini` to
match published numbers.

### Config profiles (`configs/`)

One `configs/<name>_profile.yaml` per benchmark: `babilong`, `locomo`,
`longmemeval`, `musique`, `mutual`, `narrativeqa`, `streamingqa`, `tomi`
(cogeval currently has none — known gap). Prompt changes go in the YAML, not in
Python. `configs/prompt_profiles.yaml` is the general scenario-profile registry.

### Core modules shared by all benchmarks (`src/cognifold/`)

| File | Role |
|------|------|
| `agent/domain.py` | `DomainConfig` per benchmark + `register_domain()` |
| `query/agent.py` | `MemoryQueryAgent` — main read interface, `query_for_qa()` |
| `query/models.py` | `QueryConfig`, `QueryResult`, `NodeSummary` |
| `query/assembly.py` | Formats nodes into context text |
| `query/strategies.py` | Query modes (mergefold, rag, base, episodic) |
| `retrieval/bm25.py` | BM25 retrieval engine |
| `symbolic/` | Symbolic state/belief trackers used during ingestion + ToMi |

Changes here affect **every** benchmark — smoke-test after touching them
(see Checklists).

### Environment variables

`.env` keys (see `.env.example`): `OPENAI_API_KEY` / `GOOGLE_API_KEY` required
for LLM modes. Benchmark-specific:

| Variable | Used by | Meaning |
|---|---|---|
| `MODEL` | `reproduce.sh` | reader/build model (`provider:model` syntax) |
| `PYTHON` | `reproduce.sh` | python binary override |
| `READER_MODEL` `WRITER_MODEL` `JUDGE_MODEL` `RERANK_MODEL` `EMBED_MODEL` | `parallel_longmemeval.sh` | per-role model overrides |
| `EMBEDDING_API_KEY` `EMBEDDING_BASE_URL` | embeddings provider | separate key/endpoint for embeddings (defaults to api.openai.com when key set) |
| `JUDGE_API_KEY` `JUDGE_BASE_URL` | longmemeval | separate judge endpoint |
| `OPENROUTER_API_KEY` | shell scripts | shim: exported as `OPENAI_API_KEY` + OpenRouter base URL |
| `HF_ENDPOINT` `GITHUB_MIRROR` | `download_data.py` scripts | mirrors (e.g. hf-mirror.com) |
| `COGNIFOLD_ABLATE_KNN=1` `COGNIFOLD_ABLATE_MERGE=1` | `base_runner.py` | ablation switches (skip kNN edge inference / concept merge) |
| `EXTRACT_TYPED_ATTRIBUTES` `RESOLVE_EVENT_DATES` | longmemeval | W1 (default on) / W2 (default off) passes |
| `QID_LIST_FILE` | `parallel_longmemeval.sh` | run a fixed qid subset |

Model string syntax everywhere: `provider:model` — `openai:gpt-4o-mini`,
`gemini:gemini-2.5-flash` (bare `gemini-*` also accepted). OpenAI models MUST
carry the `openai:` prefix so agent dispatch routes correctly
(`benchmarks/shared/base_runner.py::_normalize_agent_model_name`).

---

## Contributing a new benchmark

`benchmarks/mutual/run_benchmark.py` (182 lines) is the reference
implementation — copy its shape. `locomo` and `narrativeqa` predate the shared
base and duplicate its orchestration; **do not copy them**.

1. **Create the directory**:

   ```
   benchmarks/<name>/
   ├── __init__.py
   ├── run_benchmark.py        # subclass benchmarks.shared.base_runner.BenchmarkRunner
   ├── download_data.py        # dataset fetch (honor HF_ENDPOINT/GITHUB_MIRROR mirrors)
   └── README.md               # quick setup + one working command
   ```

   Subclass `BenchmarkRunner` and implement only the three abstract methods —
   `load_dataset()`, `build_events()`, `evaluate_example()` — plus optional
   hooks (`add_extra_args`, `post_ingest`, `get_query_config_overrides`, …).
   **Never copy the `run()` loop**; if `run()` doesn't fit, add a hook to the
   base instead.

2. **Create the profile**: `configs/<name>_profile.yaml` (copy
   `configs/mutual_profile.yaml` as a starting point; set
   `profiles.<name>.model.name` and `embedding.model`).

3. **Register the domain** with `register_domain()` in your runner (or, if it
   must ship in-tree, add `<NAME>_DOMAIN` in `src/cognifold/agent/domain.py`
   — but the long-term direction is YAML-registered domains, see
   `docs/CODE_HEALTH.md` #3).

4. **Unified CLI** — the base `main()` already provides `--limit`,
   `--query-mode {base,rag,episodic,mergefold}`, `--disable-concepts`,
   `--no-llm-eval`, `--no-profile`, `--visualize`, `--data`, `--embedding`,
   `--model`. Add benchmark-specific flags via `add_extra_args()` only.
   (`--judge-model` and `--event-stream` are currently locomo/longmemeval
   specials, not base flags.)

5. **Wire the hardcoded lists** (all four, or your benchmark is invisible):
   - `scripts/reproduce.sh` — both the `case` in `run_one` **and** the
     `all` loop
   - `test_benchmarks.py` — the `RUNNERS` list
   - the **Implementation Status** table and **File Map** in this file

6. **Outputs**: `run()` writes `benchmarks/<name>/output/benchmark_results.json`
   and `wrong_cases.json` (both allowlisted in `.gitignore`; everything else
   under `output/` is ignored). Run from the **repo root** — the output dir is
   a relative path.

7. **Verify cheaply before spending**: `--limit 1` first; then
   `PYTHONPATH=src python test_benchmarks.py` must pass.

### When modifying an existing benchmark

1. Read `configs/<name>_profile.yaml` first — prompt changes go there, not in code.
2. Test with `--limit 1` before full runs (API cost).
3. Record results (date, model, config, command) in the run output and update
   the status table here if numbers change significantly.

### When modifying the query system

`src/cognifold/query/` is shared by all benchmarks. After changes:

```bash
# Quick smoke test
PYTHONPATH=src python -m benchmarks.locomo.run_benchmark --limit 1 --sessions 1

# Retrieval quality without LLM cost
PYTHONPATH=src python -m benchmarks.babilong.run_benchmark --config 0k --tasks qa1 --limit 5 --no-llm-eval
```

---

## Quick Start

```bash
# Environment
export OPENAI_API_KEY="sk-..."          # or put it in .env
export HF_ENDPOINT=https://hf-mirror.com  # optional mirror

uv sync                                  # install

# Fetch a dataset, then run a minimal-cost test
PYTHONPATH=src python benchmarks/babilong/download_data.py
PYTHONPATH=src python -m benchmarks.babilong.run_benchmark \
    --config 0k --tasks qa1 --limit 3 --no-llm-eval

# Full paper run for one benchmark
bash scripts/reproduce.sh musique

# LongMemEval full N=500 (hours; ~$80–150 on the recommended stack)
bash .claude/skills/longmemeval-run/scripts/run.sh
```

## Known gaps (tracked)

- `cogeval` has no README, no `download_data.py`, no config profile.
- `longmemeval` is absent from `reproduce.sh` and `test_benchmarks.py`; its
  flags drift from the unified set (`--llm-eval` vs `--no-llm-eval`).
- `locomo`, `narrativeqa`, `longmemeval` re-implement the shared `run()`
  orchestration; `baseline_runner.py` duplicates LLM/metric helpers.
- Result JSON schemas differ per benchmark; there is no cross-benchmark
  aggregate file.

Details, evidence, and fix directions: `docs/CODE_HEALTH.md`.
