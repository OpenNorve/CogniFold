#!/usr/bin/env python3
"""Download MuTual dataset from GitHub.

MuTual is a retrieval-based dialogue reasoning dataset built from Chinese
student English listening comprehension exams. Each example has a multi-turn
dialogue context and 4 response candidates.

Source: https://github.com/Nealcly/MuTual
Paper: https://arxiv.org/abs/2004.04494

The dataset is downloaded as a zip from GitHub and the dev split txt files
are parsed into a structured JSON format.

Usage:
    python download_data.py
    python download_data.py --split dev
    python download_data.py --output data/
"""

import argparse
import io
import json
import os
import re
import urllib.request
import zipfile
from pathlib import Path

GITHUB_ZIP_URL = "https://github.com/Nealcly/MuTual/archive/refs/heads/master.zip"

# Git clone fallback
GIT_SSH_URL = "git@github.com:Nealcly/MuTual.git"
GIT_HTTPS_URL = "https://github.com/Nealcly/MuTual.git"


def _mirror_url(url: str) -> str:
    """Apply GITHUB_MIRROR env var prefix to a github URL if set.

    Usage: export GITHUB_MIRROR=https://ghproxy.com/
    """
    mirror = os.environ.get("GITHUB_MIRROR", "").rstrip("/")
    if mirror and ("github.com" in url or "raw.githubusercontent.com" in url):
        return f"{mirror}/{url}"
    return url


def parse_mutual_file(content: str) -> dict | None:
    """Parse a single MuTual txt file into structured format.

    MuTual txt files contain a single JSON object with fields:
        answers: str (e.g. "B")
        options: list[str] (4 response candidates)
        article: str (multi-turn dialogue, speakers separated by whitespace)
        id: str (e.g. "dev_1")

    Args:
        content: Raw text content of a MuTual file.

    Returns:
        Parsed example dict, or None if parsing fails.
    """
    content = content.strip()
    if not content:
        return None

    # MuTual txt files are JSON objects
    try:
        obj = json.loads(content)
        article_text = obj.get("article", "")
        # Split dialogue into turns (each turn starts with "m :" or "f :")
        turns = re.split(r"(?<=\.)(?=\s+[mf] :)", article_text)
        article_lines = [t.strip() for t in turns if t.strip()]

        return {
            "id": obj.get("id", ""),
            "article": article_lines,
            "options": obj.get("options", []),
            "answers": obj.get("answers"),
        }
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: plain text format
    lines = content.split("\n")
    article_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(\d+)\s+(.+)$", line)
        if match:
            article_lines.append(match.group(2))

    if not article_lines:
        return None

    return {
        "article": article_lines,
        "options": [],
        "answers": None,
    }


def parse_mutual_txt(content: str, filename: str) -> dict | None:
    """Parse a MuTual dev txt file.

    Each file is a JSON object with: answers, options, article, id.

    Args:
        content: Raw text of the file.
        filename: Name of the file for ID extraction.

    Returns:
        Parsed example dict or None.
    """
    content = content.strip()
    if not content:
        return None

    # MuTual txt files are JSON objects
    try:
        obj = json.loads(content)
        article_text = obj.get("article", "")
        # Split dialogue into turns (each turn starts with "m :" or "f :")
        turns = re.split(r"(?<=\.)(?=\s+[mf] :)", article_text)
        article_lines = [t.strip() for t in turns if t.strip()]

        return {
            "id": obj.get("id", Path(filename).stem),
            "article": article_lines,
            "options": obj.get("options", []),
            "answers": obj.get("answers"),
        }
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: plain text format
    lines = content.split("\n")
    dialogue_lines = []
    options = []
    answer = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        opt_match = re.match(r"^([ABCD])\s+(.+)$", line)
        if opt_match:
            options.append(opt_match.group(2))
            continue

        ans_match = re.match(r"^([ABCD])$", line)
        if ans_match:
            answer = ans_match.group(1)
            continue

        dial_match = re.match(r"^(\d+\s+)?(.+)$", line)
        if dial_match:
            dialogue_lines.append(dial_match.group(2))

    file_id = Path(filename).stem

    return {
        "id": file_id,
        "article": dialogue_lines,
        "options": options if options else [],
        "answers": answer,
    }


def _git_clone_mutual(split: str, out_path: Path) -> Path:
    """Git clone fallback for downloading MuTual data."""
    import subprocess
    import tempfile

    for clone_url in [GIT_SSH_URL, GIT_HTTPS_URL]:
        try:
            print(f"  Cloning: {clone_url}")
            with tempfile.TemporaryDirectory() as tmpdir:
                subprocess.run(
                    ["git", "clone", "--depth", "1", clone_url, tmpdir],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
                data = []
                target_dir = Path(tmpdir) / "data" / "mutual" / split
                if not target_dir.exists():
                    # Try alternative paths
                    for alt in [f"mutual/{split}", split]:
                        alt_dir = Path(tmpdir) / alt
                        if alt_dir.exists():
                            target_dir = alt_dir
                            break

                if target_dir.exists():
                    for txt_file in sorted(target_dir.glob("*.txt")):
                        content = txt_file.read_text(encoding="utf-8")
                        parsed = parse_mutual_txt(content, txt_file.name)
                        if parsed:
                            data.append(parsed)

                if data:
                    file_path = out_path / f"mutual_{split}.json"
                    with open(file_path, "w") as f:
                        json.dump(data, f, indent=2)
                    print(f"  Saved {len(data)} examples to {file_path}")
                    return file_path
                print("  WARNING: Cloned but data not found at expected path")
        except Exception as e:
            print(f"  Clone failed ({clone_url}): {e}")
            continue

    raise RuntimeError(
        f"Could not download MuTual data. Set GITHUB_MIRROR env var or "
        f"manually clone: git clone {GIT_SSH_URL}"
    )


def download_mutual(
    split: str = "dev",
    output_dir: str | None = None,
) -> Path:
    """Download MuTual dataset from GitHub and parse into JSON.

    Downloads the repository zip, extracts the specified split's txt files,
    and parses them into a structured JSON format.

    Args:
        split: Dataset split (dev, test, train). Default: dev.
        output_dir: Output directory. Defaults to benchmarks/mutual/data/.

    Returns:
        Path to the saved JSON file.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Downloading MuTual dataset from GitHub...")
    mirror = os.environ.get("GITHUB_MIRROR", "")
    if mirror:
        print(f"  GitHub mirror: {mirror}")
    zip_url = _mirror_url(GITHUB_ZIP_URL) if mirror else GITHUB_ZIP_URL
    print(f"  URL: {zip_url}")
    print(f"  Split: {split}")

    # Download zip
    print("  Downloading zip archive...")
    try:
        req = urllib.request.Request(
            zip_url,
            headers={"User-Agent": "CogniFold-Benchmark/1.0"},
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            zip_data = response.read()
    except Exception as e:
        print(f"  Zip download failed: {e}")
        print("  Trying git clone fallback...")
        return _git_clone_mutual(split, out_path)

    print(f"  Downloaded {len(zip_data) / (1024 * 1024):.1f} MB")

    # Extract relevant files from zip
    data = []
    target_prefix = f"MuTual-master/data/mutual/{split}/"

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        matching_files = [
            name
            for name in zf.namelist()
            if name.startswith(target_prefix) and name.endswith(".txt")
        ]

        print(f"  Found {len(matching_files)} files in {target_prefix}")

        for fname in sorted(matching_files):
            try:
                content = zf.read(fname).decode("utf-8")
                basename = os.path.basename(fname)
                parsed = parse_mutual_txt(content, basename)
                if parsed:
                    data.append(parsed)
            except Exception as e:
                print(f"  WARNING: Failed to parse {fname}: {e}")
                continue

    if not data:
        # Fallback: try the "mutual" directory without the nested "data" path
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            all_names = zf.namelist()
            # Print some entries for debugging
            sample = [n for n in all_names if "mutual" in n.lower()][:10]
            print(f"  Zip contents (sample): {sample}")
            print("  WARNING: No files found matching expected path structure.")
            print("  The repository structure may have changed.")

    file_path = out_path / f"mutual_{split}.json"
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  Saved {len(data)} examples to {file_path}")
    print("  Fields: id, article (dialogue lines), options (4 choices), answers (A/B/C/D)")

    return file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MuTual dataset from GitHub")
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "test", "train"],
        help="Dataset split (default: dev)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: benchmarks/mutual/data/)",
    )

    args = parser.parse_args()

    try:
        path = download_mutual(split=args.split, output_dir=args.output)
        print(f"\nDownload complete: {path}")
    except Exception as e:
        print(f"\nError: {e}")
        raise SystemExit(1) from e
