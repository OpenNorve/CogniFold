# StreamingQA Benchmark

Temporal knowledge QA where facts change over time (ICML 2022).

## Dataset
- **Source**: [google-deepmind/streamingqa](https://github.com/google-deepmind/streamingqa)
- **Size**: 14-year news corpus with temporal QA
- **Task**: Answer questions about facts that may have changed over time

## Quick Start

```bash
python download_data.py
python run_benchmark.py --limit 10 --mode direct --query-mode bm25
```

## Evaluation
- **Metrics**: Exact Match (EM), Token-level F1
