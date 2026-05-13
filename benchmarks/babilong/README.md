# BABILong Benchmark

Evaluates Cognifold's ability to perform multi-hop logical reasoning while filtering noise from massive contexts.

## Dataset

**BABILong** ([Kuratov et al., 2024](https://arxiv.org/abs/2406.10149)) extends the classic bAbI logic tasks with long noisy contexts (0k-1M tokens). It tests whether the system can maintain precise entity state tracking and logical connections despite massive distractors.

- **Source**: [RMT-team/babilong](https://huggingface.co/datasets/RMT-team/babilong) on HuggingFace
- **Paper**: "BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack"
- **Tasks**: 20 reasoning tasks (qa1-qa20)
- **Context lengths**: 0k, 1k, 2k, 4k, 8k, 16k, 32k, 128k, 256k, 512k, 1M tokens
- **Question format**: Logic puzzles with precise answers (usually 1-2 words)

## Download Data

```bash
cd benchmarks/babilong

# Default: qa1 at 0k context
python download_data.py

# Specific tasks and context length
python download_data.py --config 0k --tasks qa1,qa2,qa3

# Longer context
python download_data.py --config 4k --tasks qa1

# Using HuggingFace mirror (recommended for China)
export HF_ENDPOINT=https://hf-mirror.com
python download_data.py --config 0k --tasks qa1
```

**Options:**
```
--config CONFIG      Context length (0k, 1k, 2k, 4k, 8k, 16k, 32k, 128k, ...)
--tasks TASKS        Comma-separated task names (default: qa1). Available: qa1-qa10
--samples SIZE       Sample size: 100 (default), 1k, or 5k
--output-dir DIR     Output directory (default: benchmarks/babilong/data/)
```

Data files are saved as `data/babilong_{config}_{task}.json` (e.g., `data/babilong_0k_qa1.json`).

## Run Benchmark

### Quick Test (no LLM, zero cost)

```bash
python run_benchmark.py --config 0k --tasks qa1 --limit 5 --mode direct --no-llm-qa
```

### With LLM-based QA

```bash
export OPENAI_API_KEY='...'
python run_benchmark.py --config 0k --tasks qa1 --limit 10
```

### Processing Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **direct** | Add event nodes only, test pure retrieval | Verifying retrieval pipeline, zero cost |
| **batch** | Add events + one LLM call to extract entity states as concepts | Longer contexts (4k+), cost-efficient |
| **agent** | Full per-statement agent processing with context retrieval | Short contexts (0k-2k), highest quality |

```bash
# Direct mode (default) - retrieval only
python run_benchmark.py --config 0k --tasks qa1 --limit 5 --mode direct --no-llm-qa

# Batch mode - single LLM extraction
python run_benchmark.py --config 0k --tasks qa1 --limit 5 --mode batch

# Agent mode - full Cognifold pipeline
python run_benchmark.py --config 0k --tasks qa1 --limit 5 --mode agent
```

### All Options

```
--data PATH          Path to dataset JSON (auto-detected from --config/--tasks)
--config CONFIG      Context length (0k, 1k, 2k, 4k, 8k, etc.)
--tasks TASKS        Comma-separated task names (default: qa1)
--limit N            Process first N questions
--mode MODE          Processing mode: direct, batch, or agent
--query-mode MODE    Retrieval mode: legacy, bm25 (default), hybrid, agentic
--no-llm-qa          Use simple extraction instead of LLM for QA
--visualize          Generate graph replay HTML visualizations
--profile PATH       Path to profile YAML (default: configs/babilong_profile.yaml)
--output DIR         Output directory for results
```

## Metrics

- **Exact match**: Answer matches target exactly (case-insensitive)
- **Contains match**: Target is substring of answer
- **Correct (verdict)**: CORRECT verdict from LLM evaluation
- **Per-task accuracy**: Breakdown by task type (qa1, qa2, etc.)

## What It Tests

1. **Entity state tracking**: Track precise locations and possessions
2. **Noise filtering**: Ignore irrelevant filler text
3. **Multi-hop reasoning**: Connect facts across multiple statements
4. **Concept updates**: UPDATE existing entity states (not create duplicates)
5. **Scalability**: Performance degradation with increasing context length

## Tasks Overview

| Task | Description | Hops |
|------|-------------|------|
| qa1  | Single supporting fact | 1 |
| qa2  | Two supporting facts | 2 |
| qa3  | Three supporting facts | 3 |
| qa4  | Two argument relations | 2 |
| qa5  | Three argument relations | 3 |
| qa6  | Yes/no questions | 1 |
| qa7  | Counting | 1 |
| qa8  | Lists/sets | 1 |
| qa9  | Simple negation | 1 |
| qa10 | Indefinite knowledge | 1 |

## Notes

- Uses `babilong_profile.yaml` with BABILONG_DOMAIN config
- Each question gets a fresh graph (questions are independent)
- Noise detection: filters sentences without tracked entities/actions
- Agent mode includes rate limiting (0.5s/statement + 10s on 429 errors)
- For contexts >4k, use `--mode batch` to avoid per-statement LLM calls
- Visualization requires `--visualize` flag (generates HTML replay files)
