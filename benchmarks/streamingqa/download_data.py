#!/usr/bin/env python3
"""Download StreamingQA dataset.

StreamingQA evaluates the ability to answer questions about information
encountered in a continuous stream of news articles. It tests temporal
knowledge and the ability to handle evolving information.

Source: https://github.com/google-deepmind/streamingqa
Paper: https://arxiv.org/abs/2205.11388

The dataset is hosted on Google Cloud Storage as gzipped JSONL files.

Usage:
    python download_data.py
    python download_data.py --split eval
    python download_data.py --output data/
"""

import argparse
import gzip
import json
import os
import urllib.request
from pathlib import Path

# Google Cloud Storage URLs (primary source)
GCS_BASE = "https://storage.googleapis.com/dm-streamingqa"
GCS_SPLITS = {
    "eval": f"{GCS_BASE}/streaminqa_eval.jsonl.gz",
    "train": f"{GCS_BASE}/streaminqa_train.jsonl.gz",
    "valid": f"{GCS_BASE}/streaminqa_valid.jsonl.gz",
}


def download_streamingqa(
    split: str = "eval",
    output_dir: str | None = None,
) -> Path:
    """Download StreamingQA dataset from Google Cloud Storage.

    Args:
        split: Dataset split (eval, train, valid). Default: eval.
        output_dir: Output directory. Defaults to benchmarks/streamingqa/data/.

    Returns:
        Path to the saved JSON file.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    file_path = out_path / f"streamingqa_{split}.json"
    gcs_url = GCS_SPLITS.get(split)

    if not gcs_url:
        print(f"ERROR: Unknown split '{split}'. Available: {list(GCS_SPLITS.keys())}")
        raise SystemExit(1)

    print("Downloading StreamingQA dataset...")
    print(f"  Split: {split}")
    print(f"  URL: {gcs_url}")

    try:
        req = urllib.request.Request(
            gcs_url,
            headers={"User-Agent": "CogniFold-Benchmark/1.0"},
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            gz_data = response.read()

        print(f"  Downloaded {len(gz_data) / (1024 * 1024):.1f} MB (compressed)")

        # Decompress and parse JSONL
        data = []
        content = gzip.decompress(gz_data).decode("utf-8")
        for line in content.strip().split("\n"):
            line = line.strip()
            if line:
                data.append(json.loads(line))

        print(f"  Parsed {len(data)} examples")

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        # Print stats
        recent = sum(1 for d in data if d.get("recent_or_past") == "recent")
        past = len(data) - recent
        written = sum(1 for d in data if d.get("written_or_generated") == "written")
        generated = len(data) - written

        print(f"  Saved to {file_path}")
        print(f"  Recent: {recent}, Past: {past}")
        print(f"  Written: {written}, Generated: {generated}")
        print("  Fields: qa_id, question, answers, question_ts, evidence_ts, ...")

        return file_path

    except Exception as e:
        print(f"  Download failed: {e}")
        print(f"\n  Please manually download from: {gcs_url}")
        print(f"  Then decompress and place at: {file_path}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download StreamingQA dataset")
    parser.add_argument(
        "--split",
        type=str,
        default="eval",
        choices=["eval", "train", "valid"],
        help="Dataset split (default: eval)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: benchmarks/streamingqa/data/)",
    )

    args = parser.parse_args()

    try:
        path = download_streamingqa(split=args.split, output_dir=args.output)
        print(f"\nDownload complete: {path}")
    except Exception as e:
        print(f"\nError: {e}")
        raise SystemExit(1) from e
