#!/usr/bin/env python3
"""Download NarrativeQA dataset from HuggingFace.

NarrativeQA tests reading comprehension over full-length books and movie
scripts. Questions require understanding narrative elements like plot,
characters, and themes.

Dataset: deepmind/narrativeqa
Paper: https://arxiv.org/abs/1712.07040

To use a HuggingFace mirror (e.g. in China):
  export HF_ENDPOINT=https://hf-mirror.com

Usage:
    python download_data.py
    python download_data.py --split test
    python download_data.py --output data/
"""

import argparse
import json
import os
from pathlib import Path


def download_narrativeqa(
    split: str = "test",
    output_dir: str | None = None,
) -> Path:
    """Download NarrativeQA dataset from HuggingFace.

    Args:
        split: Dataset split (train, validation, test). Default: test.
        output_dir: Output directory. Defaults to benchmarks/narrativeqa/data/.

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
    print("Downloading NarrativeQA dataset...")
    print("  Dataset: deepmind/narrativeqa")
    print(f"  Split: {split}")
    if hf_endpoint:
        print(f"  HF Mirror: {hf_endpoint}")

    ds = load_dataset("deepmind/narrativeqa", split=split)

    print(f"  Available columns: {ds.column_names}")

    data = []
    for example in ds:
        item = {}

        # Document metadata
        doc = example.get("document", {})
        if isinstance(doc, dict):
            item["document"] = {
                "id": doc.get("id", ""),
                "kind": doc.get("kind", ""),
                "summary": doc.get("summary", {}).get("text", "")
                if isinstance(doc.get("summary"), dict)
                else doc.get("summary", ""),
            }
        else:
            item["document"] = {"id": str(doc), "kind": "", "summary": ""}

        # Question
        question = example.get("question", {})
        if isinstance(question, dict):
            item["question"] = question.get("text", str(question))
        else:
            item["question"] = str(question)

        # Answers (typically two reference answers)
        answers = example.get("answers", [])
        if isinstance(answers, list):
            item["answers"] = []
            for ans in answers:
                if isinstance(ans, dict):
                    item["answers"].append(ans.get("text", str(ans)))
                else:
                    item["answers"].append(str(ans))
        elif isinstance(answers, dict):
            # Some versions store answers differently
            item["answers"] = [answers.get("text", str(answers))]
        else:
            item["answers"] = [str(answers)]

        data.append(item)

    file_path = out_path / f"narrativeqa_{split}.json"
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    # Print stats
    doc_kinds = {}
    for item in data:
        kind = item.get("document", {}).get("kind", "unknown")
        doc_kinds[kind] = doc_kinds.get(kind, 0) + 1

    print(f"  Saved {len(data)} examples to {file_path}")
    print("  Fields: document (id, kind, summary), question, answers")
    print(f"  Document types: {dict(sorted(doc_kinds.items()))}")

    return file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NarrativeQA dataset from HuggingFace")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: benchmarks/narrativeqa/data/)",
    )

    args = parser.parse_args()

    try:
        path = download_narrativeqa(split=args.split, output_dir=args.output)
        print(f"\nDownload complete: {path}")
    except Exception as e:
        print(f"\nError: {e}")
        raise SystemExit(1) from e
