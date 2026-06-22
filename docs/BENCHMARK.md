# Cognifold Benchmark System

**This document is an extension of `CLAUDE.md`. Read `CLAUDE.md` first.**

The benchmark system evaluates Cognifold's memory against established datasets. This file is the **entry point** — it tells you what benchmarks exist, where to find everything, and what to update when making changes.

Detailed documentation lives in `docs/benchmark/`.

---

## 📄 Paper results (canonical)

Headline numbers are as reported in the technical report — [arXiv:2605.13438v3](https://arxiv.org/abs/2605.13438), *CogniFold: Always-On Proactive Memory via Cognitive Folding*. The paper is the source of truth; the Implementation Status table below is the internal tracker and is kept consistent with it.

> **Why these numbers (not the highest we can get).** The reported configuration is the one that preserves proactive **intent/intention generation** end-to-end, not the per-benchmark maximum. Several older benchmarks — ToMi in particular — are easy to drive much higher with a task-specialized reader, but that path encourages auto-loop hallucination (the reader confabulates to satisfy the metric instead of reading memory). We report the proactive-substrate stack so the numbers reflect the always-on memory thesis rather than a benchmark-tuned ceiling. See PR discussion for detail.

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

## ⚠️ ALWAYS pass `--event-stream` (every benchmark, not just LoCoMo)

All benchmark runners have `event_stream` default OFF, but **paper-grade runs MUST enable it** to activate per-session inter-session consolidation (`merge_similar_concepts` + `prune_orphan_concepts`). Canonical LoCoMo (full 10-conv, Mem0 protocol):
```bash
PYTHONPATH=src python -u -m benchmarks.locomo.run_benchmark \
    --event-stream --model openai:gpt-4.1-mini
```
Sanity check log for `Inter-session consolidation:` lines. Pre-2026-04-19 `--limit` default was `1` (silent conv-26-only truncation); fixed to `None` = all 10. If log shows `Loaded 1 conversations` on a full run → regression.

---

## Implementation Status

| Benchmark | Location | Status | Accuracy (latest) | Primary Blocker / Note |
|-----------|----------|--------|-------------------|------------------------|
| **LoCoMo** | `benchmarks/locomo/` | Tested | **81.23% J-Score overall** (paper Table 4, Mem0 protocol, gpt-4o-mini read/write + gpt-4o-mini judge, `--event-stream`; Single-Hop 90.49 / Multi-Hop 67.38 / Temporal 78.50 / Open 50.00; F1 35.71) | vs ENGRAM 77.55 · MemOS 75.80 · Zep 75.14 |
| **LongMemEval** | `benchmarks/longmemeval/` | Tested | **93.0% J-Score overall** (paper Table 5, N=500, build gpt-4o-mini / answer gpt-5.4-mini / judge gpt-4o; SSA 100.0 / SSU 97.1 / KU 94.9 / SSP 93.3 / MS 91.0 / TR 88.7) | vs Mastra 94.9 · ENGRAM 71.4 · Zep 71.2; Chronos (High) 95.6. MS lever: see PR #26/#27 |
| **MSC** | `benchmarks/msc/` | Tested | N/A (excluded from Feb 21 full eval) | Agent concept extraction too passive |
| **BABILong** | `benchmarks/babilong/` | Tested | **85.0** (paper Fig. 4; proactive-substrate stack, not benchmark-tuned ceiling) | exceeds ARMT — fine-tuned (83.8) |
| **FutureX** | `benchmarks/futurex/` | Tested | N/A (no GT) | Pipeline verified, needs real MiroFlow |
| **MuTual** | `benchmarks/mutual/` | Tested | **93.2% acc** (N=500) | Near-SOTA (~97% GPT-4o zero-shot) |
| **MuSiQue-Ans** | `benchmarks/musique/` | Tested | **F1 58.7** (paper Fig. 4, N=500) | exceeds HippoRAG 2 (49.3) |
| **TimeQA** | `benchmarks/timeqa/` | Tested | 0.0% EM (Feb 21, n=20) | Temporal reasoning absent |
| **NarrativeQA** | `benchmarks/narrativeqa/` | Tested | **F1 0.720 / ROUGE-L 0.712** (Apr 8, N=500, GPT-4o) | Scoring normalization + summary detruncation |
| **QMSum** | `benchmarks/qmsum/` | Tested | F1=0.143, ROUGE-L=0.139 (N=281) | Gemini thinking-token truncation unresolved |
| **SocialIQA** | `benchmarks/socialiqa/` | Tested | **78.4% acc** (N=500) | LLM internal commonsense sufficient |
| **ToMi** | `benchmarks/tomi/` | Tested | **83.5 EM** (paper Fig. 4; proactive-substrate stack — a task-specialized reader scores far higher but invites auto-loop confabulation, so not used) | exceeds AutoToM (80.2) |
| **SafetyBench** | `benchmarks/safetybench/` | Tested | **94.3% acc** (N=35) | Exceeds GPT-4 zero-shot (88.9%); direct mode |
| **StreamingQA** | `benchmarks/streamingqa/` | Tested | **78.4% EM / F1 0.573** (N=500) | Answer-seeded fact events + containment EM |
| **RGB** | `benchmarks/rgb/` | Tested | 80.0% EM / F1 0.860 (N=20, pilot) | Wave 7 fix |
| **CogEval-Bench** (structural) | `papers/cognifold-neurips2025/` | Validated | **Harmony 0.476, Purity 0.361, Proactivity 0.614, Compression 4.6×** (6 systems × 6 scenarios, GPT-4o-mini) | Only CogniFold non-zero on Purity + Proactivity; 5-tier hierarchy revealed |

See [docs/benchmark/results.md](benchmark/results.md) for detailed experiment results and known issues.

---

## File Map

### Documentation (docs/)

| File | Content |
|------|---------|
| **`docs/BENCHMARK.md`** | **This file** — entry point, status overview, checklists |
| `docs/benchmark/status.md` | **Implementation progress tracker** (15/15 done, accuracy data, CLI reference) |
| `docs/benchmark/architecture.md` | System architecture, core components, profile schema, conventions |
| `docs/benchmark/dataset-catalog.md` | All 15 planned datasets across 6 categories |
| `docs/benchmark/results.md` | Experiment results, current status details, known issues |
| `docs/benchmark/phase12-log.md` | Phase 12 detailed work log (historical record) |
| **`test_benchmarks.py`** | **Test suite** — verifies all 15 runners, unified CLI args, data files |

### Benchmark Code (benchmarks/)

```
benchmarks/
├── locomo/
│   ├── run_benchmark.py        # Main runner (agent mode)
│   ├── download_data.py        # Fetches locomo10.json from GitHub
│   ├── locomo10.json           # Dataset (10 conversations, ~66K lines)
│   └── README.md
├── longmemeval/
│   ├── __init__.py
│   └── run_eval.py             # Runner (batch + turn modes)
├── msc/
│   ├── run_benchmark.py        # Main runner (speaker-aware)
│   ├── download_data.py        # Downloads ParlAI dataset
│   ├── data/                   # Downloaded data (501 conversations)
│   └── README.md
├── babilong/
│   ├── run_benchmark.py        # Runner (direct/batch/agent modes)
│   ├── download_data.py        # Downloads from HuggingFace
│   ├── data/                   # Downloaded data
│   └── README.md
├── futurex/
│   ├── run_benchmark.py        # Async runner (simulated tools)
│   ├── run_miroflow_benchmark.py  # Real MiroFlow entry point
│   ├── miroflow_adapter.py     # Cognifold <> MiroFlow adapter
│   ├── cognifold_orchestrator.py  # Custom MiroFlow orchestrator
│   ├── download_data.py
│   └── futurex_data.jsonl      # 96 prediction tasks
├── mutual/
│   ├── run_benchmark.py        # MC dialogue reasoning (4 choices)
│   ├── download_data.py        # GitHub zip download, JSON parser
│   ├── data/                   # 886 dev examples
│   └── README.md
├── musique/
│   ├── run_benchmark.py        # Multi-hop QA (EM/F1)
│   ├── download_data.py        # HuggingFace download
│   ├── data/                   # 4,834 examples
│   └── README.md
├── timeqa/
│   ├── run_benchmark.py        # Time-sensitive QA (EM/F1)
│   ├── download_data.py        # GitHub JSONL download
│   ├── data/                   # 2,997 easy + hard examples
│   └── README.md
├── narrativeqa/
│   ├── run_benchmark.py        # Long-form QA (ROUGE-L/F1)
│   ├── download_data.py        # HuggingFace download
│   ├── data/                   # 10,557 examples
│   └── README.md
├── qmsum/
│   ├── run_benchmark.py        # Meeting summarization (ROUGE-L)
│   ├── download_data.py        # GitHub JSONL + HF fallback
│   ├── data/                   # 35 meetings, 281 queries
│   └── README.md
├── socialiqa/
│   ├── run_benchmark.py        # MC social reasoning (3 choices)
│   ├── download_data.py        # HF + parquet fallback
│   ├── data/                   # 1,954 validation examples
│   └── README.md
├── tomi/
│   ├── run_benchmark.py        # Theory of Mind (EM)
│   ├── download_data.py        # GitHub + clone/generate fallback
│   ├── data/                   # 2,988 generated examples
│   └── README.md
├── safetybench/
│   ├── run_benchmark.py        # Safety MC (4 choices, no GT)
│   ├── download_data.py        # HuggingFace download
│   ├── data/                   # 11,435 test examples
│   └── README.md
├── streamingqa/
│   ├── run_benchmark.py        # Temporal knowledge QA
│   ├── download_data.py        # GCS download (requires manual)
│   └── README.md
└── rgb/
    ├── run_benchmark.py        # Noise robustness (EM/F1)
    ├── download_data.py        # GitHub (currently 404)
    └── README.md
```

### Config Profiles (configs/)

| File | Benchmark |
|------|-----------|
| `configs/locomo_profile.yaml` | LoCoMo |
| `configs/longmemeval_profile.yaml` | LongMemEval |
| `configs/msc_profile.yaml` | MSC |
| `configs/babilong_profile.yaml` | BABILong |
| `configs/futurex_profile.yaml` | FutureX |
| `configs/mutual_profile.yaml` | MuTual |
| `configs/musique_profile.yaml` | MuSiQue-Ans |
| `configs/timeqa_profile.yaml` | TimeQA |
| `configs/narrativeqa_profile.yaml` | NarrativeQA |
| `configs/qmsum_profile.yaml` | QMSum |
| `configs/socialiqa_profile.yaml` | SocialIQA |
| `configs/tomi_profile.yaml` | ToMi |
| `configs/safetybench_profile.yaml` | SafetyBench |
| `configs/streamingqa_profile.yaml` | StreamingQA |
| `configs/rgb_profile.yaml` | RGB |

### Core Modules (src/cognifold/)

These are **shared** by all benchmarks. Changes here affect everything.

| File | Role |
|------|------|
| `agent/domain.py` | `DomainConfig` per benchmark (LOCOMO_DOMAIN, MSC_DOMAIN, BABILONG_DOMAIN, etc.) |
| `query/agent.py` | `MemoryQueryAgent` — main read interface, `query_for_qa()` |
| `query/models.py` | `QueryConfig`, `QueryResult`, `NodeSummary` |
| `query/assembly.py` | Formats nodes into context text |
| `query/strategies.py` | Query mode implementations (mergefold, rag, base, episodic) |
| `retrieval/bm25.py` | BM25 retrieval engine |
| `utils/embeddings.py` | OpenAI embedding service (text-embedding-3-small) |

---

## Checklists

### When Adding a New Benchmark

1. **Create benchmark directory**:
   ```
   benchmarks/<name>/
   ├── __init__.py
   ├── run_benchmark.py        # argparse CLI, standard phases
   ├── download_data.py        # Data download (optional)
   └── README.md               # Quick setup
   ```

2. **Create config profile**: `configs/<name>_profile.yaml`
   - See [docs/benchmark/architecture.md](benchmark/architecture.md) for the profile schema

3. **Register domain**: Add `<NAME>_DOMAIN` in `src/cognifold/agent/domain.py`

4. **Required CLI flags** (unified across all benchmarks):
   - `--limit N` — number of samples to process
   - `--query-mode MODE` — `base` | `rag` | `episodic` | `mergefold` (default: `mergefold`)
   - `--disable-concepts` — episodic mode (events only, no concepts)
   - `--no-llm-eval` — disable LLM-based evaluation
   - `--no-profile` — skip YAML prompt profile
   - `--visualize` — generate replay HTML

5. **Required phases in runner**:
   1. Data loading (with `--limit`)
   2. Graph initialization (fresh `ConceptGraph` per sample or global)
   3. Ingestion (event -> agent -> executor -> graph)
   4. QA evaluation (query_for_qa -> answer generation -> LLM judge)
   5. Metrics output (accuracy, node count, context length)

6. **Update documentation**:
   - Update `docs/benchmark/status.md` — mark dataset as Done, fill in runner/profile/accuracy
   - Add row to the **Implementation Status** table in this file
   - Add entry to `docs/benchmark/results.md` (even if no results yet)
   - Add to the **File Map** sections in this file

### When Modifying an Existing Benchmark

1. **Read the profile YAML first** (`configs/<domain>_profile.yaml`)
2. **Prompt changes go in the YAML**, not in Python code
3. **Test with `--limit 1`** before running full datasets (API cost)
4. **Update `docs/benchmark/results.md`** after any run that produces new results

### After Running Experiments

1. **Record results** in `docs/benchmark/results.md` with:
   - Date, model, config used
   - Accuracy metrics (strict, partial/lenient)
   - Node count, context length
   - Command used to reproduce
2. **Update status table** in this file if accuracy changed significantly

### When Modifying the Query System

The query system (`src/cognifold/query/`) is shared by all benchmarks. After changes:

```bash
# Quick smoke test
PYTHONPATH=src python benchmarks/locomo/run_benchmark.py --limit 1 --sessions 1

# Check retrieval quality
PYTHONPATH=src python benchmarks/babilong/run_benchmark.py --config 0k --tasks qa1 --limit 5 --no-llm-eval
```

---

## Quick Start

### Environment

```bash
# Required for LLM-based modes
export OPENAI_API_KEY="sk-..."

# Optional (HuggingFace mirror, e.g. in China)
export HF_ENDPOINT=https://hf-mirror.com

# Install
uv pip install -e .
```

### Verify All Runners

```bash
# Run the test suite — checks all 15 runners, unified CLI args, and data files
PYTHONPATH=src python test_benchmarks.py
```

### Minimal Cost Tests

```bash
# BABILong - 3 examples, no LLM eval
PYTHONPATH=src python benchmarks/babilong/run_benchmark.py \
    --config 0k --tasks qa1 --limit 3 --no-llm-eval

# MSC - 1 conversation
PYTHONPATH=src python benchmarks/msc/run_benchmark.py --limit 1 --query-mode rag

# LoCoMo - 1 conversation, 1 session
PYTHONPATH=src python benchmarks/locomo/run_benchmark.py --limit 1 --sessions 1
```

### Full Benchmarks (costs tokens)

```bash
# BABILong - full agent mode
PYTHONPATH=src python benchmarks/babilong/run_benchmark.py \
    --config 0k --tasks qa1,qa2 --limit 20 --query-mode mergefold

# MSC - multiple conversations
PYTHONPATH=src python benchmarks/msc/run_benchmark.py --limit 10 --query-mode mergefold

# LoCoMo - 2 conversations, all sessions, with visualization
PYTHONPATH=src python benchmarks/locomo/run_benchmark.py --limit 2 --sessions 5 --visualize
```

---

## Detail Documentation

- **[Implementation Status](benchmark/status.md)** — progress tracker for all 15 datasets (TODO / Done)
- **[Architecture & Components](benchmark/architecture.md)** — system design, core components, profile schema, LLM config
- **[Dataset Catalog](benchmark/dataset-catalog.md)** — all 15 planned datasets across 6 categories
- **[Experiment Results](benchmark/results.md)** — results, status details, known issues
- **[Phase 12 Work Log](benchmark/phase12-log.md)** — historical record of Phase 12 development
