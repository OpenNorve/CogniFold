# LoCoMo Benchmark for Cognifold

This directory contains the benchmark integration for the [LoCoMo dataset](https://github.com/snap-research/locomo), evaluating very long-term conversational memory.

## Setup

1.  Download the dataset:
    ```bash
    python download_data.py
    ```
    This will download `locomo10.json` to this directory.

## Usage

Run the benchmark script:
```bash
python run_benchmark.py
```

This will:
1.  Load the LoCoMo dataset.
2.  Initialize a Cognifold agent.
3.  Ingest the conversation history as events.
4.  Run the QA evaluation and report results.

## Requirements

The benchmark uses the core `CognifoldAgent` which requires an LLM backend. Ensure you have one of the following environment variables set:
- `OPENAI_API_KEY`: For OpenAI models.
- `GOOGLE_API_KEY`: For Gemini models.

## Options

You can limit the scope for testing:
```bash
python run_benchmark.py --limit 1 --sessions 2
```
This processes only the first conversation and its first 2 sessions.
