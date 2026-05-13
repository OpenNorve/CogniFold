# Cognifold Benchmark System

**This document is an extension of `CLAUDE.md`. Read `CLAUDE.md` first.**

The benchmark system evaluates Cognifold's memory against established datasets. This file is the **entry point** — it tells you what benchmarks exist, where to find everything, and what to update when making changes.

Detailed documentation lives in `docs/benchmark/`.

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
| **LoCoMo** | `benchmarks/locomo/` | Tested | **82.8% J-Score (iter3)** (May 5, 1275/1540 cat 1–4 QA, strict 65.9% / partial 82.0%, gpt-4.1-mini + gpt-4o-mini judge, `--event-stream`, Mem0 protocol; entity-anchored EVENT boost + raw-turn preservation + question decomposition) · prior 62.9% (Apr 19 baseline) · 56.2% strict (Apr 17, gemini-2.5-flash, 1986 QA) | +15.9 pp over Mem0 (66.88); +7.0 pp over Memobase v0.0.37 (75.78); -8.4 pp to MemMachine v0.2 (91.23, has Cohere reranker) |
| **LongMemEval** | `benchmarks/longmemeval/` | Tested | 0% | Retrieval returns empty context |
| **MSC** | `benchmarks/msc/` | Tested | N/A (excluded from Feb 21 full eval) | Agent concept extraction too passive |
| **BABILong** | `benchmarks/babilong/` | Tested | **96.0% EM** (N=100) | Intent routing; solved |
| **FutureX** | `benchmarks/futurex/` | Tested | N/A (no GT) | Pipeline verified, needs real MiroFlow |
| **MuTual** | `benchmarks/mutual/` | Tested | **93.2% acc** (N=500) | Near-SOTA (~97% GPT-4o zero-shot) |
| **MuSiQue-Ans** | `benchmarks/musique/` | Tested | **41.2% EM / F1 0.587** (Apr 8, N=500, GPT-4o) | **Exceeds HippoRAG 2 F1=0.486** |
| **TimeQA** | `benchmarks/timeqa/` | Tested | 0.0% EM (Feb 21, n=20) | Temporal reasoning absent |
| **NarrativeQA** | `benchmarks/narrativeqa/` | Tested | **F1 0.720 / ROUGE-L 0.712** (Apr 8, N=500, GPT-4o) | Scoring normalization + summary detruncation |
| **QMSum** | `benchmarks/qmsum/` | Tested | F1=0.143, ROUGE-L=0.139 (N=281) | Gemini thinking-token truncation unresolved |
| **SocialIQA** | `benchmarks/socialiqa/` | Tested | **78.4% acc** (N=500) | LLM internal commonsense sufficient |
| **ToMi** | `benchmarks/tomi/` | Tested | **91.6% EM** (N=500) | Symbolic belief tracker |
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
