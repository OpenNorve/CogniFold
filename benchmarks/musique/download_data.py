#!/usr/bin/env python3
"""Download MuSiQue dataset from HuggingFace.

MuSiQue (Multi-hop Questions via Single-hop Question Composition) is a
multi-hop question answering dataset that requires reasoning across multiple
paragraphs. Each question has a decomposition showing the reasoning steps.

Dataset: bdsaglam/musique (or dgslibisey/MuSiQue)
Paper: https://arxiv.org/abs/2108.00573

To use a HuggingFace mirror (e.g. in China):
  export HF_ENDPOINT=https://hf-mirror.com

Usage:
    python download_data.py
    python download_data.py --split validation
    python download_data.py --output data/
"""

import argparse
import json
import os
from pathlib import Path

# Try multiple known HF dataset names for MuSiQue
DATASET_CANDIDATES = [
    "bdsaglam/musique",
    "dgslibisey/MuSiQue",
]


def download_musique(
    split: str = "validation",
    output_dir: str | None = None,
) -> Path:
    """Download MuSiQue dataset from HuggingFace.

    Tries multiple known dataset names since the dataset may be hosted
    under different accounts.

    Args:
        split: Dataset split (train, validation). Default: validation.
        output_dir: Output directory. Defaults to benchmarks/musique/data/.

    Returns:
        Path to the saved JSON file.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required. Install with: pip install datasets")
        raise SystemExit(1) from None

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    hf_endpoint = os.environ.get("HF_ENDPOINT", "")
    print("Downloading MuSiQue dataset...")
    print(f"  Split: {split}")
    if hf_endpoint:
        print(f"  HF Mirror: {hf_endpoint}")

    # Try each dataset name until one works
    ds = None
    dataset_name = None
    for candidate in DATASET_CANDIDATES:
        try:
            print(f"  Trying dataset: {candidate}...")
            ds = load_dataset(candidate, split=split)
            dataset_name = candidate
            print(f"  Successfully loaded: {candidate}")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if ds is None:
        print("ERROR: Could not load MuSiQue from any known source.")
        print("Known sources tried:")
        for c in DATASET_CANDIDATES:
            print(f"  - {c}")
        print("\nAlternative: download manually from https://github.com/StonyBrookNLP/musique")
        raise SystemExit(1)

    # Extract fields
    data = []
    columns = ds.column_names
    print(f"  Available columns: {columns}")

    for example in ds:
        item = {
            "id": example.get("id", ""),
            "question": example.get("question", ""),
            "answer": example.get("answer", example.get("answers", "")),
        }

        # Paragraphs may be stored as a list of dicts or nested structure
        if "paragraphs" in example:
            item["paragraphs"] = example["paragraphs"]
        elif "context" in example:
            item["paragraphs"] = example["context"]

        # Question decomposition (multi-hop steps)
        if "question_decomposition" in example:
            item["question_decomposition"] = example["question_decomposition"]
        elif "decomposition" in example:
            item["question_decomposition"] = example["decomposition"]

        # Answerable flag if present
        if "answerable" in example:
            item["answerable"] = example["answerable"]

        data.append(item)

    file_path = out_path / f"musique_{split}.json"
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"  Saved {len(data)} examples to {file_path}")
    print(f"  Dataset: {dataset_name}")
    print("  Fields: id, question, answer, paragraphs, question_decomposition")

    return file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MuSiQue dataset from HuggingFace")
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split (default: validation)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: benchmarks/musique/data/)",
    )

    args = parser.parse_args()

    try:
        path = download_musique(split=args.split, output_dir=args.output)
        print(f"\nDownload complete: {path}")
    except Exception as e:
        print(f"\nError: {e}")
        raise SystemExit(1) from e
