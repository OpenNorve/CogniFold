# Cognifold

A dynamic concept graph system that processes real-time event streams and maintains an evolving knowledge representation.

---

## ⚠️ CRITICAL — Must-Remember Operational Rules

**Universal rule: streaming consolidation must be ON for every benchmark.** CogniFold's always-on memory story REQUIRES that `merge_similar_concepts` + `prune_orphan_concepts` (and related consolidation) actually run during/after ingestion. Without it, the "sleep-like" per-session/per-batch consolidation that's central to the design isn't being tested.

**Per-benchmark status**:

| Benchmark | Consolidation default | Action required |
|---|---|---|
| **LoCoMo** | `event_stream` defaults to **False** in `benchmarks/locomo/run_benchmark.py` | **MUST pass `--event-stream`** every run. |
| **All others** (MuSiQue, NarrativeQA, ToMi, BABILong, MuTual, StreamingQA, MSC, LongMemEval) | `base_runner.py` runs consolidation automatically post-ingestion (no flag) | Do not disable. |

**Canonical LoCoMo command** (gpt-4.1-mini agent + gpt-4o-mini judge, Mem0 protocol, **all 10 conv**):
```bash
PYTHONPATH=src python -u -m benchmarks.locomo.run_benchmark \
    --event-stream \
    --model openai:gpt-4.1-mini \
    --judge-model gpt-4o-mini
```

(Pre-2026-04-19 the runner had `--limit` default = **1** — a smoke-test default that silently truncated full runs to conv-26 only. Fixed to `None` (= all 10). If you see `Loaded 1 conversations` in log when expecting full run → check git for regression.)

**Verification (LoCoMo)**: log must contain `Inter-session consolidation: N merges, M orphans` lines. If absent → `--event-stream` was forgotten → **rerun**.

**Verification (other benchmarks)**: log should contain `Consolidation: N merges, M orphans tagged` (from `base_runner` post-ingest). If absent, check that `merge_similar_concepts` import succeeded.

**History**: 2026-04-19 confirmed all 4 prior LoCoMo smoke runs (conv-26 baseline 62.5% J-Score, bi-temporal A/B, scene A/B) ran **WITHOUT** `--event-stream` — those numbers are OFF-state baselines. ON-state numbers required for paper.

---

**Neural-symbolic computation agent (LongMemEval MS): EXPERIMENTAL, OFF by default — do NOT enable.** The count/sum/arithmetic agent (`benchmarks/longmemeval/neural_symbolic.py`, CLI `--neural-symbolic` **default OFF**; launcher env `NEURAL_SYMBOLIC_FLAG`) is built but **shelved**. Two paid A/B smokes + a full-MS static projection (2026-06-20) show it is **NOT net-positive at full scale**: it fires on ~76/133 MS questions, but the collateral surface (47 currently-CORRECT) is 1.6× the win opportunity (29 currently-wrong), and at the measured collateral rate (~0.3–0.4) the projected full-MS score is ~68–76% — **at or below** the 75.9% baseline. **NEVER pass `--neural-symbolic` / set `NEURAL_SYMBOLIC_FLAG` in any benchmark or paper run, and NEVER flip the default to ON, without first re-validating (a controlled A/B) that the collateral rate is below ~0.15.** Its compute + classification fixes are correct and $0-tested (`neural_symbolic_selftest.py`); the agent is kept only as a tagged, reversible scaffold (`git tag iter33-ms-pre-symbolic`) for future operand-selection work. Details: memory `longmemeval-neural-symbolic-agent`, `benchmarks/longmemeval/ITER33_MS_STATUS.md`.

## Quick Start for Agents

**Read these files in order before writing any code:**

| Order | File | Purpose |
|-------|------|---------|
| 1 | `CLAUDE.md` | This file - project overview |
| 2 | `docs/AGENT_PROTOCOL.md` | **REQUIRED** - Strict rules for all agents |
| 3 | `docs/RESUME.md` | Current work-in-progress |
| 4 | `docs/CHANGELOG.md` (last 50 lines) | Recent changes |
| 5 | `docs/PHASES.md` | Current phase specifications |

---

## Documentation Index

| File | Purpose |
|------|---------|
| `docs/AGENT_PROTOCOL.md` | Workflow rules, commit protocol, quality gates |
| `docs/PHASES.md` | All phase specifications (current, planned, completed) |
| `docs/ARCHITECTURE.md` | System design, data flow, critical implementation details |
| `docs/SERVICE_API.md` | HTTP service API reference (request/response schemas) |
| `docs/PROMPTS.md` | Prompt engineering guide |
| `docs/CONTRIBUTING.md` | Code standards, testing, PR workflow |
| `docs/RESUME.md` | Current work state (update every session) |
| `docs/CHANGELOG.md` | Change history (append after changes) |
| `docs/WISHLIST.md` | Deferred work and ideas |
| `docs/WISHLIST_QUERY_MEMORY.md` | Query/retrieval improvements roadmap (EverMemOS gap analysis) |
| `docs/benchmark/` | Benchmark details (architecture, dataset catalog, results, phase12 log, ingestion fix plan) |
| `docs/BENCHMARK.md` | Benchmark datasets and evaluation plan (14 datasets, 5 categories) |
| `docs/DEPLOYMENT.md` | Production deployment design (Docker, NGINX, Redis, CI/CD, logging) |
| `docs/COGNITION_PRINCIPLES.md` | Core cognition principles, anti-patterns, alignment checklist |
| `docs/RESEARCH_INTENTION_EMERGENCE.md` | Research roadmap: memory → intention emergence (EN) |
| `docs/RESEARCH_INTENTION_EMERGENCE_ZH.md` | Research roadmap: memory → intention emergence (ZH) |
| `docs/RESEARCH_INTENTION_EMERGENCE.md` | Research roadmap: memory→intention emergence, cognitive science foundations |
| `docs/PLAN_WORKSTREAM_1_INFRA.md` | CogniFold 0.2 Workstream 1: Shared foundations + infra hardening plan |
| `docs/PLAN_WORKSTREAM_2_MEMORY.md` | CogniFold 0.2 Workstream 2: Memory consolidation & forgetting plan |

### Phase Documentation Management

When completing a phase:
1. Update status in "Phase Status Overview" table (📋 → 🔄 → ✅)
2. Move the phase spec from "Current & Planned Phases" to "Completed Phases" section
3. Keep completed phase specs concise (remove task checklists, keep key concepts)
4. Update `docs/RESUME.md` with completion status

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              COGNIFOLD SYSTEM                               │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │ Event Stream │  (mock JSON / real-time source)
    └──────┬───────┘
           │
           ▼
┌─────────────────────┐
│   Event Ingestion   │  Validates & queues incoming events
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐      ┌─────────────────────────────────────┐
│  Context Selector   │◄────►│          Concept Graph              │
│                     │      │  ┌─────────────────────────────┐    │
│  • Compute scores   │      │  │  Nodes                      │    │
│  • PageRank         │      │  │  ├── event (blue)           │    │
│  • Recency decay    │      │  │  ├── concept (green)        │    │
│  • Access frequency │      │  │  ├── intent (orange)        │    │
│                     │      │  │  └── time (purple)          │    │
│  Output: Context    │      │  ├─────────────────────────────┤    │
│  Window (top-k)     │      │  │  Edges (typed, weighted)    │    │
└──────────┬──────────┘      │  └─────────────────────────────┘    │
           │                 │                                     │
           ▼                 │  Storage: NetworkX + JSON           │
┌─────────────────────┐      └──────────────────▲──────────────────┘
│   LangGraph Agent   │                         │
│   (Gemini / OpenAI) │                         │
│                     │                         │
│  Input:             │                         │
│  • New event        │                         │
│  • Context window   │                         │
│    (hierarchical)   │                         │
│                     │                         │
│  Output:            │                         │
│  • UpdatePlan       │                         │
└──────────┬──────────┘                         │
           │                                    │
           ▼                                    │
┌─────────────────────┐                         │
│     Executor        │─────────────────────────┘
│                     │
│  • Validates plan   │
│  • Applies ops      │
│  • Atomic updates   │
└─────────────────────┘
```

---

## Core Concepts

### Node Types

| Type | Description |
|------|-------------|
| `event` | Direct representation of input events |
| `concept` | Higher-level patterns that emerge from events |
| `intent` | Goals/desires that emerge from patterns (pending → action_scheduled → resolved → rejected/deferred) |
| `time` | Temporal anchors (deadlines, scheduled times, recurring periods) |

### Edge Types (Semantic)

| Type | Weight | Usage |
|------|--------|-------|
| GROUNDS | 0.9 | Event grounds Concept/Intent |
| CAUSES | 0.9 | Event causes Event |
| TRIGGERS | 0.8 | Concept triggers Intent |
| REINFORCES | 0.7 | Event reinforces Concept |
| PART_OF | 0.7 | Event/Concept → Concept (subtopic, structural) |
| DERIVED_FROM | 0.6 | Concept derived from another |
| DEADLINE_FOR | 0.6 | Time is deadline for Intent |
| RELATED_TO | 0.5 | Generic relationship (default) |
| USER_FEEDBACK | 0.8 | Feedback event → Intent |

### Context Window (Hierarchical)

| Level | % | Content |
|-------|---|---------|
| Immediate | 10% | Recent events, urgent intents |
| Working | 30% | Active concepts, patterns |
| Background | 50% | Historical context |
---

## Event Schema

```json
{
  "event_id": "uuid",
  "timestamp": "ISO 8601 datetime",
  "source": "domain identifier (e.g., 'personal-timeline', 'computer-activity', 'service-logs')",
  "event_type": "string (free-form, domain-specific: meal, browser.page_visit, http.request, etc.)",
  "title": "short description",
  "description": "detailed description (optional)",
  "location": "string (optional)",
  "duration_minutes": "integer (optional)",
  "context": {"domain-specific structured data"},
  "metadata": {}
}
```

---

## Update Plan Schema

The agent produces an **UpdatePlan** containing operations:

```json
{
  "plan_id": "uuid",
  "trigger_event_id": "the event that triggered this update",
  "reasoning": "agent's explanation",
  "operations": [
    {"op": "ADD_NODE", "node_type": "event|concept|intent|time", "data": {...}},
    {"op": "ADD_EDGE", "source_id": "...", "target_id": "...", "edge_type": "GROUNDS", "weight": 0.9},
    {"op": "UPDATE_NODE", "node_id": "...", "data": {...}},
    {"op": "REMOVE_NODE", "node_id": "..."},
    {"op": "REMOVE_EDGE", "source_id": "...", "target_id": "..."},
    {"op": "MERGE_NODES", "node_ids": [...], "merged_data": {...}}
  ]
}
```

---

## Module Structure

```
src/cognifold/
├── models/           # Pydantic schemas (Event, Node, Edge, UpdatePlan)
├── graph/            # NetworkX wrapper, persistence, validation, edge inference, projection protocol
├── scoring/          # PageRank + hierarchical context scoring
├── agent/            # LangGraph agent, prompts, domain configs
├── executor/         # Plan execution with validation
├── generator/        # LLM-based event generators
├── importers/        # Data importers (wiki, etc.)
├── query/            # Query system (strategies, assembly, agent, LLM utilities, prompts)
├── temporal/         # Temporal extraction
├── embeddings/       # Embedding providers, semantic search, optional FAISS ANN index
├── retrieval/        # BM25, hybrid, and agentic multi-round retrieval
├── feedback/         # Self-organizing feedback (reinforcement, plasticity, decay, activation)
├── intent/           # Intent-to-action system
├── replay/           # Graph evolution replay tool
├── service/          # HTTP service layer (FastAPI)
├── simulator/        # Interactive visualization
├── cli/              # CLI subcommands (run, query, generate, replay, serve, client, etc.)
└── utils/            # Shared utilities (embedding service, LLM metrics/budget)
```

---

## CLI Commands

**Run the CLI with:**
```bash
# If installed via pip/uv
cognifold [command] [options]

# Or run directly from source
PYTHONPATH=src python -m cognifold [command] [options]
```

### 1. Generate Sample Events

```bash
# List available personas/profiles
cognifold generate --domain personal-timeline --list
cognifold generate --domain computer-activity --list
cognifold generate --domain service-logs --list

# Generate personal timeline (requires GOOGLE_API_KEY)
GOOGLE_API_KEY=<key> cognifold generate \
  --domain personal-timeline \
  --persona software_engineer \
  --events 50 \
  --days 2 \
  -o data/generated/
```

**Expected output:** Timeline JSON file at `data/generated/<persona>_timeline.json`

### 2. Run Simulation with Agent

```bash
# Run simulation with agent (requires GOOGLE_API_KEY)
GOOGLE_API_KEY=<key> cognifold run data/generated/alex_chen_timeline.json \
  --agent \
  --output output/ \
  --save-graph output/my_graph.json

# Fast mode: 3-layer pipeline (Layer 1 <30s, then progressive enrichment)
cognifold run data/generated/alex_chen_timeline.json --fast \
  --save-graph output/fast_graph.json

# Fast mode with LLM enrichment (all 3 layers)
GOOGLE_API_KEY=<key> cognifold run data/generated/alex_chen_timeline.json \
  --fast --agent --batch-size 10 \
  --save-graph output/fast_agent_graph.json

# Compare classic vs fast
cognifold run timeline.json --agent              # Classic: sequential, ~20min
cognifold run timeline.json --fast --agent       # Fast: layered, ~5-8min
```

**Expected output:**
- Visualization: `output/graph_<timestamp>.html`
- Graph: `output/my_graph.json`
- Replay log: `logs/replay_<timeline>_<timestamp>.jsonl`

### 3. Generate Replay Visualization

```bash
# Generate replay from log file
cognifold replay logs/replay_alex_chen_timeline_*.jsonl \
  -o output/replay.html \
  --open
```

**Expected output:** Interactive HTML replay at `output/replay.html`

### 4. Query the Graph

```bash
# Basic query (uses legacy keyword matching)
cognifold query --graph output/my_graph.json "morning routine"

# Query with BM25 retrieval (better keyword matching)
cognifold query --graph output/my_graph.json \
  --retrieval bm25 \
  "exercise habits"

# Query with hybrid retrieval (default, requires GOOGLE_API_KEY for embeddings)
GOOGLE_API_KEY=<key> cognifold query --graph output/my_graph.json \
  --retrieval hybrid \
  "what are my fitness patterns"

# Agentic retrieval (multi-round with LLM sufficiency check, requires LLM API key)
GOOGLE_API_KEY=<key> cognifold query --graph output/my_graph.json \
  --retrieval agentic \
  "what connections exist between my diet and exercise"

# Interactive query mode
cognifold query --graph output/my_graph.json \
  --retrieval bm25 \
  --interactive

# Get top concepts
cognifold query --graph output/my_graph.json --top-concepts 10

# Get recent intents
cognifold query --graph output/my_graph.json --recent-intents 5
```

**Expected output:** Formatted context with concepts, intents, and events

### 5. Start the HTTP Service

```bash
# Start with the convenience script
./scripts/start_server.sh

# Or manually
cognifold serve --host 127.0.0.1 --port 8000

# Configure via environment variables
COGNIFOLD_PORT=9000 COGNIFOLD_API_KEY=secret ./scripts/start_server.sh
```

**Expected output:** Server running at `http://127.0.0.1:8000`, docs at `/docs`

### Cloud Run (Production)

```bash
# Get the Cloud Run service URL
gcloud run services describe cognifold --region=us-central1 --project=cognifold-production --format="value(status.url)"
```

### 6. Interactive Client

```bash
# Connect to a running server
cognifold client

# With custom URL and API key
cognifold client --url http://localhost:9000 --api-key secret

# Auto-connect to an existing session
cognifold client --session <session-id>
```

**Interactive commands:**
- `:session create` / `:session delete` / `:session <ID>` — session management
- `:ingest TYPE TITLE [--desc D] [--loc L]` — ingest events
- `:stats` / `:concepts` / `:intents` / `:events` — graph exploration
- `:node <ID>` / `:graph [N]` — inspect nodes
- `:expand <ID> [N] [--direction D] [--max M]` — BFS expand from node by N layers
- `:load <FILE>` — load a graph JSON file
- Type any text without `:` to query the graph in natural language

### 7. Build Timeline from Data

```bash
# Build timeline from wiki/markdown files
cognifold build-timeline --source wiki --input data/wiki/ -o data/timeline.json
```

**Expected output:** Timeline JSON file at the specified output path

### 8. Configuration

```bash
# Show current configuration
cognifold config --show

# Generate example config file
cognifold config --generate config.yaml
```

### Quick Test Workflow

```bash
# Set API key once
export GOOGLE_API_KEY='your-api-key'

# 1. Generate events (or use existing data/generated/alex_chen_timeline.json)
cognifold generate --domain personal-timeline --persona software_engineer --events 30

# 2. Run simulation with agent
cognifold run data/generated/alex_chen_timeline.json --agent -o output/ --save-graph output/test_graph.json

# 3. Generate replay
cognifold replay logs/replay_alex_chen_timeline_*.jsonl -o output/test_replay.html

# 4. Query the graph
cognifold query --graph output/test_graph.json --retrieval bm25 --interactive
```

---

## Slash Commands

Claude Code slash commands are available for common CLI operations:

| Command | Purpose |
|---------|---------|
| `/cognifold-generate` | Generate sample event timelines |
| `/cognifold-run` | Run simulation with agent |
| `/cognifold-replay` | Generate replay visualization |
| `/cognifold-query` | Query the concept graph |
| `/cognifold-test` | Full end-to-end test workflow |
| `/doc-guard` | Check & fix documentation completeness before PRs |
| `/cognifold-bench-run` | Run a benchmark evaluation (downloads data, runs runner, reports results) |
| `/cognifold-bench-analyze` | Analyze benchmark failures (categorize, root cause, suggest fixes) |
| `/longmemeval-run` | Run the LongMemEval-S benchmark with the recommended stack (gpt-5.4-mini reader/writer + gpt-4o judge + reflector + W3 START extraction) |
| `/longmemeval-iterate` | Run an iter-improvement loop on LongMemEval: pick failure cluster → propose patch → write CHANGES.md → verify on hard100 → escalate to N=500 |

Each command shows usage, options, and examples. Use `/cognifold-test` to verify the full workflow.

---

## Tech Stack

- Python 3.9+ (target 3.11+)
- LangGraph for agent orchestration
- Google Gemini (`google-genai`) / OpenAI for LLM
- NetworkX for graph operations
- Pydantic for schema validation
- FastAPI + uvicorn for HTTP service
- FAISS (optional) for fast ANN vector search
- pyvis for visualization
- numpy + scipy for embeddings and similarity
- dateparser for temporal extraction
- ruff for linting/formatting
- pyright for type checking
- pytest for testing

---

## Current Status

**Active Phases**: Phase 12 - Benchmark Evaluation (ingestion fixes & re-evaluation), Phase 15 - Production Deployment & Observability (in progress)

Phase 14.1 (Intent Personalization) is complete — added closed-loop intent calibration with user feedback (accept/reject/defer/modify), EMA-based category profiles, adaptive scoring, and 3 new service API endpoints. Phase 11.1 (Query Memory Improvements) added FAISS ANN index, agentic multi-round retrieval, and unified embedding system. Default retrieval is now HYBRID (auto-degrades to BM25 without embeddings). See `docs/PHASES.md` for detailed specifications and `docs/RESUME.md` for current work state.

---

## Agent Protocol Summary

> **Full details in `docs/AGENT_PROTOCOL.md`**

1. **Check branches** - Before starting, check for unmerged remote branches ahead of `cognifold-dev`. Warn user if found.
2. **Read** required files before coding
3. **Plan** multi-step tasks with TodoWrite
4. **Code** with types, tests, following standards
5. **Verify** with `make quality` (ruff lint + format) and `make typecheck` (pyright) and `make test` (pytest)
6. **Commit** immediately after each small task
7. **Log** changes to CHANGELOG.md
8. **Update** RESUME.md with current state

### Branch Check Command
```bash
git fetch --all && git branch -r --no-merged origin/cognifold-dev
```
If branches are listed, ask user before proceeding.
