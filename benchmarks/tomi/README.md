# ToMi Benchmark

Theory of Mind evaluation - tracking agent beliefs about object locations (EMNLP 2019).

## Dataset
- **Source**: [facebookresearch/ToMi](https://github.com/facebookresearch/ToMi)
- **Size**: Generated stories with Sally-Anne style belief tracking
- **Task**: Answer questions about what agents believe (may differ from reality)

## Quick Start

```bash
python download_data.py
python run_benchmark.py --limit 10 --mode direct --query-mode bm25
```

## Evaluation
- **Metric**: Accuracy (exact match on location/entity answers)
