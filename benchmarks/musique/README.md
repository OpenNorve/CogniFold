# MuSiQue Benchmark

Multi-hop QA requiring chained reasoning across paragraphs (TACL 2022).

## Dataset
- **Source**: [dgslibisey/MuSiQue](https://huggingface.co/datasets/dgslibisey/MuSiQue) or [bdsaglam/musique](https://huggingface.co/datasets/bdsaglam/musique)
- **Size**: ~25K multi-hop questions (2-4 hops)
- **Task**: Answer questions that require connecting facts across multiple paragraphs

## Quick Start

```bash
python download_data.py
python run_benchmark.py --limit 10 --mode direct --query-mode bm25
```

## Evaluation
- **Metrics**: Exact Match (EM), Token-level F1
