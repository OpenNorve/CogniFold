# MuTual Benchmark

Multi-turn dialogue reasoning with multiple-choice responses (ACL 2020).

## Dataset
- **Source**: [Nealcly/MuTual](https://github.com/Nealcly/MuTual)
- **Size**: ~8,860 dialogues with 4-choice responses
- **Task**: Select the most appropriate next response in a dialogue

## Quick Start

```bash
python download_data.py
python run_benchmark.py --limit 10 --mode direct --query-mode bm25
```

## Evaluation
- **Metric**: Accuracy (% correct response selected)
