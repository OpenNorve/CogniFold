# NarrativeQA Benchmark

Question answering on full books and movie scripts (TACL 2018).

## Dataset
- **Source**: [deepmind/narrativeqa](https://huggingface.co/datasets/deepmind/narrativeqa)
- **Size**: ~46,765 QA pairs across 1,567 documents
- **Task**: Answer questions about entire books or movie scripts using summaries

## Quick Start

```bash
python download_data.py
python run_benchmark.py --limit 10 --mode direct --query-mode bm25
```

## Evaluation
- **Metrics**: ROUGE-L, Token-level F1
