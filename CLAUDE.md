# CLAUDE.md

Guidance for Claude Code (and human contributors) working in this repository.
CogniFold is a brain-inspired always-on agent memory: it folds an event stream
into a typed concept graph (`event` → `concept` → `intent`) and surfaces
proactive context without being asked. Deeper reading: `README.md` (product
overview), `docs/ARCHITECTURE.md` (technical authority), `docs/BENCHMARK.md`
(benchmark system), `docs/PHILOSOPHY.md` (design intent).

## Commands

```bash
uv sync                          # fastest install (uses uv.lock); or: pip install -e ".[dev,agent,service]"
make test                        # python -m pytest tests/ -v
make lint                        # ruff check src/ tests/
make format                      # ruff format src/ tests/
make typecheck                   # pyright src/  (pyright STRICT, not mypy)
make check                       # lint + format --check   (CI runs check + typecheck + test)
make serve                       # ./scripts/start_server.sh (HTTP service on :8000)
bash scripts/reproduce.sh <bench>  # paper benchmarks; see docs/BENCHMARK.md
bash scripts/pre-commit.sh       # local quality gate (cp to .git/hooks/pre-commit)
```

- **Tool versions are pinned** in the `dev` extra: `ruff==0.15.17`, `pyright==1.1.408`.
  Do not upgrade casually — `ruff format --check` will churn across the tree.
- Lint/format/typecheck scope is **`src/` and `tests/` only**. `benchmarks/`,
  `scripts/`, and root-level `.py` files are not checked by CI.
- Python: CI, Docker, and pyright all use **3.11**. (`requires-python` says
  ">=3.10" and ruff targets py39 — treat 3.11 as the real floor.)

## Architecture map

Write path: `Generator/Importer → Event stream → CognifoldAgent (LangGraph) →
UpdatePlan → PlanExecutor (validate → atomic execute → rollback) → ConceptGraph`.
Read path: `query → MemoryQueryAgent (legacy|bm25|semantic|hybrid|agentic) →
GraphTraverser → scorer → ContextAssembler → QueryResult`. Proactive read:
`HierarchicalContextSelector` bands (immediate/working/background) — no query needed.

Packages under `src/cognifold/` (~40K lines, 150 files):

| Package | Role |
|---|---|
| `models/` | Pydantic schemas: Node, Edge, Event, UpdatePlan (bottom layer, no internal deps) |
| `graph/` | NetworkX MultiDiGraph wrapper, persistence, validation, metrics |
| `scoring/` | PageRank + recency ranking, context window selection |
| `executor/` | Plan validation + atomic execution with rollback |
| `agent/` | LangGraph write-path agent, prompts, domain configs |
| `query/` | MemoryQueryAgent, strategies, assembly, LLM calls |
| `retrieval/` | BM25, hybrid RRF fusion, agentic multi-round |
| `embeddings/` | Provider ABC: Gemini/OpenAI/Mock, optional FAISS |
| `pipeline/` | Orchestration — `classic.Pipeline` (public API, MCP) and `LayeredPipeline` (CLI run, service) are BOTH alive by design |
| `intent/` | Intent-to-action queue, executor, calibrator |
| `symbolic/` | Symbolic belief/state trackers (used by benchmarks + query) |
| `temporal/` | Date parsing, temporal entity extraction |
| `service/` | FastAPI HTTP service: routes, session stores (file/redis/supabase), SSE |
| `mcp/` | MCP server (`cognifold-mcp`): remember/query/graph_stats/list_intents |
| `cli/` | `cognifold` CLI: run, query, generate, replay, serve, client |
| `generator/` `importers/` `simulator/` `replay/` | Event generation, data import, timeline sim, HTML replay |
| `brain/` | `memory_coverage.json` — **single source of truth** for the coverage figure (pages.yml copies it to docs-site) |

**Dependency layering rule** (`docs/ARCHITECTURE.md` §Module Dependencies):
`models → graph → scoring/executor → agent → query`; `pipeline` may depend on
everything; `service`/`cli`/`simulator` sit on top. **Nothing below `service/`
should import from `service/`** — the existing `service.llm_keys` imports in
`config.py`, `agent/`, `query/`, `utils/`, `embeddings/` are a known violation
slated for cleanup (`docs/CODE_HEALTH.md` #4); do not add new ones.

**ADR-001**: `EdgeInferenceEngine` is library code only — never auto-invoke it
during ingestion; orphan reconnection is handled by
`PlanExecutor._detect_orphan_nodes`.

## Conventions

- Every new file starts with `from __future__ import annotations` (128/150
  files already do). PEP 604 unions (`str | None`), builtin generics
  (`dict[str, Any]`), `if TYPE_CHECKING:` for cycle-avoiding imports.
- Pyright runs **strict** on `src/` — annotate everything, including `-> None`.
- Google-style docstrings (`Args:` / `Returns:` / `Raises:`) on modules,
  classes, and public functions.
- Logging: `logger = logging.getLogger(__name__)` at module level; call
  `cognifold.logging.setup_logging()` once at process entry. Never use
  structlog directly — the stdlib integration handles JSON output.
- **Pydantic only in `models/`** (wire/graph data). Configuration and internal
  value objects use stdlib `@dataclass` (48 files follow this).
- Optional heavy deps (structlog, redis, supabase, faiss, langgraph) are
  imported lazily with graceful fallback (e.g. hybrid retrieval degrades to
  BM25 without an embedder). Keep that property.
- Node IDs use typed prefixes: `e-` event, `c-` concept, `i-` intent, `t-`
  time; canonical form is `<prefix>-<n>` (e.g. `e-001`).
- Prompt changes for benchmarks go in `configs/<name>_profile.yaml`, not in
  Python code.

## Testing reality

`pytest tests/` is a **smoke suite only** (3 assertions). A green pytest does
not verify behavior. Real verification paths:

- `PYTHONPATH=src python test_benchmarks.py` — all runners import + unified CLI
- `python scripts/test_retrieval_api.py` — all retrieval modes
- `python scripts/e2e_supabase_test.py` — service end-to-end
- Benchmark runs with `--limit 1` (cheap) before full runs (API cost)
- CI's docker-build job boots the service and polls `/health`

When you touch a deterministic core (`graph/`, `executor/`, `scoring/`,
`retrieval/bm25`, `query/text_utils`, `config.py`), add real unit tests under
`tests/` — growing that net is an explicit goal (`docs/CODE_HEALTH.md` #1).

## Benchmarks quick reference

Full guide: `docs/BENCHMARK.md`. Entry points:

| What | How |
|---|---|
| Any paper benchmark | `bash scripts/reproduce.sh {cogeval,locomo,musique,narrativeqa,tomi,babilong,mutual,streamingqa,all}` |
| LongMemEval (headline) | `bash scripts/parallel_longmemeval.sh` or the Claude Code skills `.claude/skills/longmemeval-run` / `longmemeval-iterate` |
| Datasets | run `benchmarks/<name>/download_data.py` first — runners do NOT auto-download |

Hard rules (from the skills — keep them):
- **Judge lock**: LongMemEval/LoCoMo judge is `openai:gpt-4o` — never substitute
  (breaks comparability with published baselines).
- **Branch lock**: the LongMemEval iteration campaign commits only to
  `longmemeval-iter`.
- Reproducing paper numbers requires the paper stack (build `gpt-4o-mini`);
  `reproduce.sh` defaults to `MODEL=openai:gpt-5` — override `MODEL` to match.

## Branches, PRs, deployment

- `main` and `cognifold-dev` run CI (ruff + pyright + pytest + docker health).
- Pushes to `cognifold-dev` auto-open a promote PR to `cognifold-stable`;
  `cognifold-stable` deploys to Cloud Run via `cd.yml`.
- `pages.yml` deploys `docs-site/` on pushes to `main`; the brain-coverage
  figure comes only from `src/cognifold/brain/memory_coverage.json`.
- PRs: single purpose, green CI, English titles/descriptions/review comments.
- PR review runs automatically (`claude-code-review.yml`); `@claude` in a
  comment summons interactive help (`claude.yml`).

## Gotchas

- Code comments referencing `my_prompt.md §1.2` etc. now resolve to
  `.claude/skills/longmemeval-iterate/references/model-config.md`.
- Default model names disagree across `config.py` / `config.example.yaml` /
  `configs/cognifold.yaml` — pass models explicitly rather than trusting a
  default (`docs/CODE_HEALTH.md` #13).
- `configs/` is **excluded from the Docker image** (`.dockerignore`) — the
  service must not require it at runtime.
- `benchmarks/*/output/` is gitignored **except** `benchmark_results.json` and
  `wrong_cases.json` (allowlisted).
- Benchmark runners must run from the repo root (relative `output_dir`).
- Undocumented ablation switches: `COGNIFOLD_ABLATE_KNN=1`,
  `COGNIFOLD_ABLATE_MERGE=1` (read in `benchmarks/shared/base_runner.py`).
- `cognifold.utils.embeddings` is deprecated (import-time warning) — new code
  uses `cognifold.embeddings.create_provider()`.
