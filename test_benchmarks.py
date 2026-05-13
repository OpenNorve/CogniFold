#!/usr/bin/env python3
"""Quick test - verify all 8 benchmark runners import and accept unified CLI args."""
import json
import subprocess
import sys
from pathlib import Path

RUNNERS = [
    "benchmarks/locomo/run_benchmark.py",
    "benchmarks/babilong/run_benchmark.py",
    "benchmarks/mutual/run_benchmark.py",
    "benchmarks/musique/run_benchmark.py",
    "benchmarks/streamingqa/run_benchmark.py",
    "benchmarks/narrativeqa/run_benchmark.py",
    "benchmarks/tomi/run_benchmark.py",
    "benchmarks/cogeval/run_benchmark.py",
]

UNIFIED_ARGS = ["--query-mode", "--disable-concepts", "--limit", "--no-llm-eval", "--embedding"]

PYTHON = str(Path(".venv/bin/python")) if Path(".venv/bin/python").exists() else "python"

print("\n" + "=" * 60)
print("Cognifold Benchmark Test Suite")
print("=" * 60 + "\n")

# Test 1: Core imports
print("[1/4] Testing core imports...")
try:
    sys.path.insert(0, "src")
    from cognifold.agent.agent import CognifoldAgent
    from cognifold.agent.config import AgentConfig
    from cognifold.executor.runner import PlanExecutor
    from cognifold.graph.store import ConceptGraph
    from cognifold.models.event import Event
    from cognifold.query.agent import MemoryQueryAgent
    from cognifold.query.models import QueryConfig

    print("  OK - all core modules imported\n")
except Exception as e:
    print(f"  FAIL - import error: {e}\n")
    sys.exit(1)

# Test 2: All runners --help
print(f"[2/4] Testing {len(RUNNERS)} runners (--help)...")
passed = 0
failed = []
for script in RUNNERS:
    name = Path(script).parent.name
    if not Path(script).exists():
        failed.append((name, "file not found"))
        continue
    result = subprocess.run(
        [PYTHON, script, "--help"],
        capture_output=True,
        timeout=10,
        env={**__import__("os").environ, "PYTHONPATH": "src:."},
    )
    if result.returncode == 0:
        passed += 1
    else:
        err = result.stderr.decode().strip().split("\n")[-1] if result.stderr else "unknown"
        failed.append((name, err))

print(f"  {passed}/{len(RUNNERS)} passed")
for name, err in failed:
    print(f"  FAIL - {name}: {err}")
print()

# Test 3: Unified CLI args present in all runners
print("[3/4] Checking unified CLI args...")
missing_args = []
for script in RUNNERS:
    name = Path(script).parent.name
    if not Path(script).exists():
        continue
    result = subprocess.run(
        [PYTHON, script, "--help"],
        capture_output=True,
        timeout=10,
        env={**__import__("os").environ, "PYTHONPATH": "src:."},
    )
    help_text = result.stdout.decode()
    for arg in UNIFIED_ARGS:
        if arg not in help_text:
            missing_args.append((name, arg))

if not missing_args:
    print(f"  OK - all {len(RUNNERS)} runners have {UNIFIED_ARGS}\n")
else:
    print(f"  {len(missing_args)} missing args:")
    for name, arg in missing_args:
        print(f"    {name}: missing {arg}")
    print()

# Test 4: Data files exist
print("[4/4] Checking data files...")
data_checks = [
    ("locomo", "benchmarks/locomo/locomo10.json"),
    ("mutual", "benchmarks/mutual/data/mutual_dev.json"),
    ("musique", "benchmarks/musique/data/musique_validation.json"),
    ("streamingqa", "benchmarks/streamingqa/data/streamingqa_eval.json"),
    ("narrativeqa", "benchmarks/narrativeqa/data/narrativeqa_test.json"),
    ("tomi", "benchmarks/tomi/data/tomi_test.json"),
    ("babilong", None),  # downloaded on-the-fly from HuggingFace
    ("cogeval", None),  # generated synthetically
]
found = sum(1 for _, p in data_checks if p and Path(p).exists())
total_local = sum(1 for _, p in data_checks if p)
print(f"  {found}/{total_local} local data files present ({len(data_checks) - total_local} use on-the-fly download)")
for name, p in data_checks:
    if p and not Path(p).exists():
        print(f"    MISSING - {name}: {p}")
print()

# Summary
print("=" * 60)
total_ok = passed == len(RUNNERS) and not missing_args
if total_ok:
    print(f"ALL TESTS PASSED - {len(RUNNERS)} runners verified")
else:
    print(f"SOME TESTS FAILED - {passed}/{len(RUNNERS)} runners, {len(missing_args)} missing args")
print("=" * 60)
print(f"\nTo run a smoke test:")
print(f"  PYTHONPATH=src {PYTHON} benchmarks/mutual/run_benchmark.py --limit 3 --query-mode mergefold")
