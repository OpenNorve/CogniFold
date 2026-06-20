#!/usr/bin/env bash
# unified-longmemeval — MS validation run (Stage 2b).
#
# Same retrieval/ledger/resolver stack as iter31 (the verified TR 88.7 config),
# pointed at the 133 MS qids, with ONE deliberate change from run_iter31.sh:
#
#   WRITER = gpt-4o-mini (routed to OpenRouter), NOT gpt-5.4-mini.
#
# Why: iter27 CHANGES.md verdict — "gpt-5.4-mini ≈ gpt-5-mini, no measurable
# advantage; NEUTRAL, use whichever is cheaper." gpt-4o-mini is the canonical
# iter19 writer AND the MS-branch "faithful iter19" writer where MS 82.0 was
# measured, so it is the most faithful writer for an MS test. It is also far
# cheaper, and the writer is the dominant ingestion cost (~50 sessions/qid).
# run_iter31.sh used gpt-5.4-mini writer only to keep all chat on one
# commonstack provider — a convenience, not a quality requirement.
#
# Stack:
#   Reader:  gpt-5.4-mini high  (commonstack)   — composite default
#   Writer:  gpt-4o-mini        (OpenRouter)    — cheap + faithful for MS
#   Rerank:  gpt-5.4-mini low   (commonstack)
#   Judge:   gpt-4o             (OpenRouter)
#   Embed:   text-embedding-3-small (OpenRouter)
#   W1/W2/W3: OFF   AGG: 15000   ledger: 42 emitters (8 TR + 34 MS)
#
# NEURAL_SYMBOLIC_FLAG (env, default empty/OFF): if set to "--neural-symbolic" it
# enables the EXPERIMENTAL neural-symbolic count/sum agent. It is NOT net-positive
# at full scale (collateral surface 1.6x the win opportunity; projected ~68-76% <=
# 75.9% baseline). DO NOT set it for benchmark/paper runs — see the CLAUDE.md
# "Neural-symbolic computation agent" Critical rule. Leave unset for all real runs.
#
# Usage:  bash scripts/run_unified_ms.sh <QID_LIST_FILE> <LABEL> [N_PARALLEL]

set -uo pipefail
cd "$(dirname "$0")/.."

# CRITICAL: force this worktree's src/cognifold ahead of any editable install
# in .venv (which may point at a DIFFERENT checkout). The unified branch carries
# iter31 src changes (writer-rule-4 activity_start/start_date in agent/batch.py)
# that the main checkout lacks — without this, the run would silently use the
# wrong writer code (a stack swap).
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

[ -f .env ] && set -a && source .env && set +a

QID_LIST_FILE="${1:?need QID_LIST_FILE path}"
LABEL="${2:?need LABEL}"
N_PARALLEL="${3:-10}"

if [ -z "${COMMONSTACK_API_KEY:-}" ]; then
    echo "ERROR: COMMONSTACK_API_KEY missing in .env" >&2; exit 1
fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY missing in .env" >&2; exit 1
fi

# Reader / rerank → commonstack (gpt-5.4-mini)
export OPENAI_API_KEY="$COMMONSTACK_API_KEY"
export OPENAI_BASE_URL="https://api.commonstack.ai/v1"
unset OPENAI_ORGANIZATION
unset READER_API_KEY READER_BASE_URL

# Writer → OpenRouter (gpt-4o-mini); commonstack does not serve gpt-4o-mini
export WRITER_API_KEY="$OPENROUTER_API_KEY"
export WRITER_BASE_URL="https://openrouter.ai/api/v1"

# Judge → OpenRouter (commonstack lacks gpt-4o)
export JUDGE_API_KEY="$OPENROUTER_API_KEY"
export JUDGE_BASE_URL="https://openrouter.ai/api/v1"

# Embed → OpenRouter (commonstack has no /embeddings)
export EMBEDDING_API_KEY="$OPENROUTER_API_KEY"
export EMBEDDING_BASE_URL="https://openrouter.ai/api/v1"

# gpt-4o-mini is non-reasoning → writer-reasoning-effort is ignored; leave unset.
unset WRITER_REASONING_EFFORT

BASE="benchmarks/longmemeval"
FINAL_DIR="$BASE/runs/$LABEL"
mkdir -p "$FINAL_DIR" logs

echo "unified MS run: $LABEL"
echo "  qid file:  $QID_LIST_FILE"
echo "  parallel:  $N_PARALLEL"
echo "  reader:    commonstack  → openai/gpt-5.4-mini (high)"
echo "  writer:    openrouter   → openai/gpt-4o-mini   <<< changed from gpt-5.4-mini"
echo "  rerank:    commonstack  → openai/gpt-5.4-mini (low, pool=100)"
echo "  judge:     openrouter   → openai/gpt-4o"
echo "  embed:     openrouter   → openai/text-embedding-3-small"
echo "  W1/W2/W3:  OFF    AGG: 15000    ledger: 42 emitters"
echo "  output:    $FINAL_DIR"

TODO_FILE=$(mktemp)
trap "rm -f $TODO_FILE" EXIT
.venv/bin/python <<PY > "$TODO_FILE"
import json, sys
from pathlib import Path
text = Path("$QID_LIST_FILE").read_text().strip()
raw = [t.strip() for chunk in text.split("\n") for t in chunk.split(",")]
target_qids = [q for q in raw if q]
done = set()
final_hyp = Path("$FINAL_DIR/hypothesis.jsonl")
if final_hyp.exists():
    for line in open(final_hyp):
        try:
            done.add(json.loads(line)["question_id"])
        except Exception:
            pass
todo = [q for q in target_qids if q not in done]
sys.stderr.write(f"target={len(target_qids)} done={len(done)} todo={len(todo)}\n")
for q in todo:
    print(q)
PY

N_TODO=$(wc -l < "$TODO_FILE")
if [ "$N_TODO" -eq 0 ]; then
    echo "Nothing to do — all qids already in $FINAL_DIR/hypothesis.jsonl"
    exit 0
fi
if [ "$N_TODO" -lt "$N_PARALLEL" ]; then
    N_PARALLEL="$N_TODO"
fi
echo "Launching $N_PARALLEL parallel batches over $N_TODO todo qids"

CHUNK=$(( N_TODO / N_PARALLEL ))
REM=$(( N_TODO - CHUNK * N_PARALLEL ))

PIDS=()
BATCH_DIRS=()
START=1
for ((i=0; i<N_PARALLEL; i++)); do
    LIMIT=$CHUNK
    if [ "$i" -lt "$REM" ]; then LIMIT=$((CHUNK + 1)); fi
    END=$(( START + LIMIT - 1 ))
    IDS_CSV=$(sed -n "${START},${END}p" "$TODO_FILE" | paste -sd, -)
    OUTDIR="$BASE/output_ums_b$((i+1))"
    LOG="logs/unified_ms_b$((i+1)).log"
    BATCH_DIRS+=("$OUTDIR")
    echo "  batch $((i+1)): $LIMIT qid → $OUTDIR (log: $LOG)"
    rm -rf "$OUTDIR" && mkdir -p "$OUTDIR"
    nohup .venv/bin/python -u -m benchmarks.longmemeval.run_eval \
        --model openai:openai/gpt-5.4-mini \
        --writer-model openai:openai/gpt-4o-mini \
        --judge-model openai:openai/gpt-4o \
        --embedding openai:openai/text-embedding-3-small \
        --symbolic-resolver --symbolic-temporal --symbolic-bypass \
        --tr-topic-timeline \
        --llm-rerank --rerank-model openai:openai/gpt-5.4-mini \
        --rerank-reasoning-effort low --rerank-pool 100 \
        --agg-max-context-chars 15000 \
        --extract-typed-attributes \
        ${NEURAL_SYMBOLIC_FLAG:-} \
        --question-ids "$IDS_CSV" \
        --output-dir "$OUTDIR" \
        --batch-mode --llm-eval \
        > "$LOG" 2>&1 &
    PIDS+=($!)
    START=$(( END + 1 ))
done

echo "PIDs: ${PIDS[*]}"
echo "Waiting..."
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "  batch pid=$pid failed (see logs/unified_ms_b*.log)"
        FAIL=$((FAIL+1))
    fi
done

.venv/bin/python <<PY
import json
from collections import Counter
from pathlib import Path

final_hyp = Path("$FINAL_DIR/hypothesis.jsonl")
records = {}
if final_hyp.exists():
    for line in open(final_hyp):
        try:
            r = json.loads(line); records[r["question_id"]] = r
        except Exception:
            pass
for batch in sorted(Path("$BASE").glob("output_ums_b*")):
    hp = batch / "hypothesis.jsonl"
    if not hp.exists():
        continue
    for line in open(hp):
        try:
            r = json.loads(line); records[r["question_id"]] = r
        except Exception:
            pass

with open(final_hyp, "w") as f:
    for r in records.values():
        f.write(json.dumps(r) + "\n")

c = Counter(r.get("verdict") for r in records.values())
total = len(records)
strict = c["CORRECT"]/total*100 if total else 0
partial = (c["CORRECT"] + 0.5*c["PARTIAL"])/total*100 if total else 0
with open("$FINAL_DIR/metrics.json", "w") as f:
    json.dump({
        "correct": c["CORRECT"], "partial": c["PARTIAL"],
        "incorrect": c["INCORRECT"], "error": c.get("ERROR", 0),
        "total": total,
        "score_strict": strict, "score_partial": partial,
    }, f, indent=2)
print(f"merged: {total} results — {c['CORRECT']}/{total} = {strict:.2f}% strict, {partial:.2f}% partial")
PY

if [ "$FAIL" -eq 0 ]; then
    for d in "${BATCH_DIRS[@]}"; do rm -rf "$d"; done
    echo "Done. Result at $FINAL_DIR/{hypothesis,metrics}.json"
else
    echo "WARNING: $FAIL batches failed; per-batch dirs preserved." >&2
    exit 1
fi
