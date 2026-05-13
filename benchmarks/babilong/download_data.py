#!/usr/bin/env python3
"""Download BABILong dataset from HuggingFace.

BABILong extends bAbI logic tasks with long noisy contexts (0k-1M tokens).
Tests reasoning ability in massive haystack scenarios.

Dataset: RMT-team/babilong
Paper: https://arxiv.org/abs/2406.10149

HuggingFace dataset structure:
  - Config (subset) = context length tier: 0k, 1k, 2k, 4k, 8k, 16k, 32k, 128k, ...
  - Split = task name: qa1, qa2, ..., qa10
  - Example: load_dataset("RMT-team/babilong", "0k")["qa1"]

To use a HuggingFace mirror (e.g. in China):
  export HF_ENDPOINT=https://hf-mirror.com

Usage:
    python download_data.py
    python download_data.py --config 0k --tasks qa1,qa2,qa3
    python download_data.py --config 4k --tasks qa1 --samples 1k
"""

import argparse
import json
import os
from pathlib import Path


def download_babilong(
    config: str = "0k",
    tasks: list[str] | None = None,
    samples: str = "100",
    output_dir: str | None = None,
) -> list[Path]:
    """Download BABILong dataset from HuggingFace.

    The HF dataset uses config for context length (0k, 1k, ...) and
    splits for task names (qa1, qa2, ..., qa10).

    Args:
        config: Context length configuration (0k, 1k, 2k, 4k, 8k, 16k, 32k, 128k, ...).
        tasks: List of task names (qa1-qa10). Defaults to ["qa1"].
        samples: Sample size ("100" for 100 samples, "1k" for 1000, "5k" for 5000).
        output_dir: Output directory. Defaults to benchmarks/babilong/data/.

    Returns:
        List of paths to saved JSON files.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required. Install with: pip install datasets")
        raise SystemExit(1) from None

    if tasks is None:
        tasks = ["qa1"]

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Determine dataset name based on sample size
    if samples == "1k":
        dataset_name = "RMT-team/babilong-1k-samples"
    elif samples == "5k":
        dataset_name = "RMT-team/babilong-train-5k-samples"
    else:
        dataset_name = "RMT-team/babilong"

    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    print("Downloading BABILong dataset...")
    print(f"  Dataset: {dataset_name}")
    print(f"  Config (context length): {config}")
    print(f"  Tasks: {', '.join(tasks)}")
    if hf_endpoint:
        print(f"  HF Mirror: {hf_endpoint}")

    # Load the full config (all splits/tasks available)
    ds = load_dataset(dataset_name, config)

    available_splits = list(ds.keys())
    print(f"  Available tasks: {', '.join(available_splits)}")

    output_paths: list[Path] = []

    for task in tasks:
        if task not in ds:
            print(f"  WARNING: task '{task}' not found. Available: {available_splits}")
            continue

        task_data = ds[task]
        data = []
        for example in task_data:
            example_dict = dict(example)
            example_dict["task"] = task
            data.append(example_dict)

        file_path = out_path / f"babilong_{config}_{task}.json"
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        avg_len = sum(len(x["input"]) for x in data) // len(data) if data else 0
        print(f"  Saved {len(data)} examples to {file_path}")
        print(f"    Task: {task}, Avg context: {avg_len} chars")
        output_paths.append(file_path)

    return output_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BABILong dataset from HuggingFace")
    parser.add_argument(
        "--config",
        type=str,
        default="0k",
        help="Context length config (0k, 1k, 2k, 4k, 8k, 16k, 32k, 128k, ...)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="qa1",
        help="Comma-separated task names (default: qa1). Available: qa1-qa20",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="100",
        choices=["100", "1k", "5k"],
        help="Sample size (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: benchmarks/babilong/data/)",
    )

    args = parser.parse_args()

    task_list = [t.strip() for t in args.tasks.split(",")]

    try:
        paths = download_babilong(
            config=args.config,
            tasks=task_list,
            samples=args.samples,
            output_dir=args.output_dir,
        )
        print(f"\nDownloaded {len(paths)} task file(s).")
    except Exception as e:
        print(f"\nError: {e}")
        raise SystemExit(1) from e
