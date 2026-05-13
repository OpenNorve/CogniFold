#!/usr/bin/env python3
"""Download ToMi (Theory of Mind Inventory) dataset from GitHub.

ToMi tests theory of mind reasoning: given a story about characters moving
objects and entering/exiting rooms, answer questions about beliefs.

Source: https://github.com/facebookresearch/ToMi
Paper: https://arxiv.org/abs/1904.09728

The dataset is downloaded as raw text files from GitHub and parsed into
structured JSON format.

Usage:
    python download_data.py
    python download_data.py --output data/
"""

import argparse
import io
import json
import os
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/facebookresearch/ToMi/master/data"
FILES = {
    "test": f"{BASE_URL}/test.txt",
    "test_trace": f"{BASE_URL}/test.trace",
}
ARCHIVE_URL = "https://raw.githubusercontent.com/facebookresearch/ToMi/master/tomi_balanced_story_types.zip"

# SSH clone fallback
GIT_SSH_URL = "git@github.com:facebookresearch/ToMi.git"
GIT_HTTPS_URL = "https://github.com/facebookresearch/ToMi.git"


def _mirror_url(url: str) -> str:
    """Apply GITHUB_MIRROR env var prefix to a github URL if set.

    Usage: export GITHUB_MIRROR=https://ghproxy.com/
    """
    mirror = os.environ.get("GITHUB_MIRROR", "").rstrip("/")
    if mirror and ("github.com" in url or "raw.githubusercontent.com" in url):
        return f"{mirror}/{url}"
    return url


def download_file(url: str) -> str:
    """Download a text file from a URL (tries mirror first).

    Args:
        url: URL to download.

    Returns:
        Text content of the file.
    """
    for attempt_url in [_mirror_url(url), url] if _mirror_url(url) != url else [url]:
        try:
            req = urllib.request.Request(
                attempt_url,
                headers={"User-Agent": "CogniFold-Benchmark/1.0"},
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read().decode("utf-8")
        except Exception:
            if attempt_url != url:
                print(f"  Mirror failed, trying direct: {url}")
                continue
            raise
    # unreachable but makes type checker happy
    raise RuntimeError(f"Failed to download {url}")


def parse_tomi_data(text_content: str) -> list[dict]:
    """Parse ToMi text format into structured examples.

    ToMi format: numbered lines forming stories, with questions indicated
    by a tab-separated answer at the end. Stories are separated by line
    number resets (going back to 1).

    Example:
        1 Ethan entered the bathroom.
        2 Aiden entered the bathroom.
        3 The asparagus is in the blue_box.
        4 Aiden exited the bathroom.
        5 Ethan moved the asparagus to the green_box.
        6 Where does Aiden think that the asparagus is?	blue_box

    Args:
        text_content: Raw text of the ToMi data file.

    Returns:
        List of parsed example dicts.
    """
    examples = []
    current_story = []
    current_id = 0

    for line in text_content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Parse line number
        match = re.match(r"^(\d+)\s+(.+)$", line)
        if not match:
            continue

        line_num = int(match.group(1))
        content = match.group(2)

        # Detect story reset (line number goes back to 1 or lower)
        if line_num <= 1 and current_story:
            current_story = []

        # Check if this line contains a question (has tab-separated answer)
        if "\t" in content:
            parts = content.split("\t")
            question = parts[0].strip()
            answer = parts[1].strip() if len(parts) > 1 else ""

            # Determine question type from the question text
            question_type = "unknown"
            if "think" in question.lower() or "believe" in question.lower():
                question_type = "belief"  # Theory of mind question
            elif "where" in question.lower() and "really" in question.lower():
                question_type = "reality"  # Reality question
            elif "where" in question.lower():
                question_type = "memory"  # Memory question
            elif "search" in question.lower():
                question_type = "search"

            current_id += 1
            examples.append(
                {
                    "id": current_id,
                    "story": list(current_story),
                    "question": question,
                    "answer": answer,
                    "question_type": question_type,
                }
            )
        else:
            current_story.append(content)

    return examples


def _archive_fallback() -> tuple[str, str | None] | None:
    """Download the published ToMi archive and extract test files.

    The current upstream repo no longer exposes ``data/test.txt`` directly,
    but it still ships a zip archive with pre-generated train/val/test files.
    """
    try:
        print(f"  Downloading archive fallback: {ARCHIVE_URL}")
        req = urllib.request.Request(
            _mirror_url(ARCHIVE_URL),
            headers={"User-Agent": "CogniFold-Benchmark/1.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            archive_bytes = response.read()
    except Exception as e:
        print(f"  Archive fallback failed: {e}")
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
            test_text = zf.read("tomi_balanced_story_types/fb_all_test.txt").decode("utf-8")
            trace_text = zf.read("tomi_balanced_story_types/fb_all_test.trace").decode("utf-8")
            return test_text, trace_text
    except Exception as e:
        print(f"  Archive extraction failed: {e}")
        return None


def _git_clone_fallback(out_path: Path) -> str | None:
    """Try git clone and generate data via the repo's main.py.

    The ToMi repo is a data *generator* — it does not ship pre-built data files.
    We clone the repo, run `python main.py --num-stories 500`, and read the
    generated test.txt.
    """
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
                # Check if pre-built data exists (unlikely but check)
                test_file = Path(tmpdir) / "data" / "test.txt"
                if test_file.exists():
                    return test_file.read_text(encoding="utf-8")

                # Generate data using the repo's main.py
                main_py = Path(tmpdir) / "main.py"
                if main_py.exists():
                    print("  Generating data via main.py (500 stories)...")
                    subprocess.run(
                        [
                            sys.executable,
                            "main.py",
                            "--num-stories",
                            "500",
                        ],
                        cwd=tmpdir,
                        check=True,
                        capture_output=True,
                        timeout=300,
                    )
                    test_file = Path(tmpdir) / "data" / "test.txt"
                    if test_file.exists():
                        print(f"  Generated {test_file.stat().st_size} bytes")
                        return test_file.read_text(encoding="utf-8")
                    print("  WARNING: main.py ran but test.txt not generated")
                else:
                    print("  WARNING: No main.py found in cloned repo")
        except Exception as e:
            print(f"  Clone/generate failed ({clone_url}): {e}")
            continue
    return None


def download_tomi(output_dir: str | None = None) -> Path:
    """Download and parse ToMi test dataset from GitHub.

    Args:
        output_dir: Output directory. Defaults to benchmarks/tomi/data/.

    Returns:
        Path to the saved JSON file.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Downloading ToMi dataset from GitHub...")
    mirror = os.environ.get("GITHUB_MIRROR", "")
    if mirror:
        print(f"  GitHub mirror: {mirror}")
    print(f"  URL: {FILES['test']}")

    # Download test.txt (try direct/mirror first, then git clone fallback)
    text_content = None
    try:
        print("  Downloading test.txt...")
        text_content = download_file(FILES["test"])
        print(f"  Downloaded {len(text_content)} bytes")
    except Exception as e:
        print(f"  Direct download failed: {e}")
        archive_result = _archive_fallback()
        if archive_result is not None:
            text_content, trace_content = archive_result
            print(f"  Loaded {len(text_content)} bytes from archive fallback")
        else:
            print("  Trying git clone fallback...")
            text_content = _git_clone_fallback(out_path)

    if text_content is None:
        raise RuntimeError(
            "Could not download ToMi data. Set GITHUB_MIRROR env var or "
            f"manually clone: git clone {GIT_SSH_URL}"
        )

    # Try to download trace file (optional metadata)
    if trace_content is None:
        try:
            print("  Downloading test.trace...")
            trace_content = download_file(FILES["test_trace"])
            print(f"  Downloaded {len(trace_content)} bytes")
        except Exception as e:
            print(f"  WARNING: Could not download test.trace: {e}")
            print("  Continuing without trace metadata.")

    # Parse the test data
    data = parse_tomi_data(text_content)

    # If trace data is available, enrich examples with story type info
    if trace_content:
        trace_lines = trace_content.strip().split("\n")
        # Trace format varies; try to parse and attach metadata
        for i, trace_line in enumerate(trace_lines):
            if i < len(data):
                data[i]["story_type"] = trace_line.strip()

    file_path = out_path / "tomi_test.json"
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    # Print stats
    question_types = {}
    for item in data:
        qt = item.get("question_type", "unknown")
        question_types[qt] = question_types.get(qt, 0) + 1

    print(f"  Saved {len(data)} examples to {file_path}")
    print("  Fields: id, story, question, answer, question_type")
    print(f"  Question types: {dict(sorted(question_types.items()))}")

    return file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ToMi dataset from GitHub")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: benchmarks/tomi/data/)",
    )

    args = parser.parse_args()

    try:
        path = download_tomi(output_dir=args.output)
        print(f"\nDownload complete: {path}")
    except Exception as e:
        print(f"\nError: {e}")
        raise SystemExit(1) from e
