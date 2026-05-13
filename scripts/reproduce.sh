#!/usr/bin/env bash
# Reproduce a CogniFold paper benchmark end-to-end.
#
# Usage:
#   bash scripts/reproduce.sh             # canonical run: LoCoMo (full 10-conv, ~1 h)
#   bash scripts/reproduce.sh BENCHMARK   # one of (paper order):
#                                         #   cogeval locomo musique narrativeqa
#                                         #   tomi babilong mutual streamingqa
#   bash scripts/reproduce.sh all         # all 8 paper benchmarks back-to-back
#
# Each runner downloads its dataset on first call (no manual setup needed) and
# writes results to benchmarks/<name>/output/benchmark_results.json.

set -euo pipefail
cd "$(dirname "$0")/.."

# Load .env if present (OPENAI_API_KEY, optional GOOGLE_API_KEY)
[ -f .env ] && set -a && source .env && set +a

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set. Edit .env (cp .env.example .env) or export it." >&2
    exit 1
fi

if [ -n "${PYTHON:-}" ]; then
    PYBIN="$PYTHON"
elif [ -x ".venv/bin/python" ]; then
    PYBIN=".venv/bin/python"
else
    PYBIN="$(command -v python3 || command -v python || true)"
fi
[ -x "$PYBIN" ] || { echo "ERROR: no python interpreter found (set PYTHON=... to override)" >&2; exit 1; }

MODEL="${MODEL:-openai:gpt-4o-mini}"

run_one() {
    local bench="$1"
    local module entry extra
    case "$bench" in
        locomo)       module="benchmarks.locomo.run_benchmark";    extra=(--event-stream) ;;
        cogeval)      module="benchmarks.cogeval.run_benchmark";   extra=() ;;
        musique|narrativeqa|tomi|babilong|mutual|streamingqa)
                      module="benchmarks.${bench}.run_benchmark";  extra=() ;;
        *)
            echo "ERROR: unknown benchmark '$bench'" >&2
            exit 2 ;;
    esac
    echo "==> $bench  ($module)"
    PYTHONPATH=src "$PYBIN" -u -m "$module" --model "$MODEL" "${extra[@]}"
    echo "    results: benchmarks/$bench/output/benchmark_results.json"
    echo
}

TARGET="${1:-locomo}"
if [ "$TARGET" = "all" ]; then
    for b in cogeval locomo musique narrativeqa tomi babilong mutual streamingqa; do
        run_one "$b"
    done
else
    run_one "$TARGET"
fi
