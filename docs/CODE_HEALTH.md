# Code Health — Structural Findings & Fix Directions

Evidence-based inventory of structural problems in `src/cognifold/` (~40,131
lines, 150 files) and `benchmarks/` (~19,800 lines), ranked by impact ×
confidence. Each finding names files/lines, the concrete cost, and a fix
direction. Companion to `CLAUDE.md` and `docs/BENCHMARK.md`.

Status legend: 🔴 fix before building on top · 🟠 pay down opportunistically ·
🟡 nice-to-have.

---

## Tier 1 — Highest impact

### 1. 🔴 The test suite is 28 lines for 40K lines of source

`tests/` contains only `test_smoke.py` — 3 assertions (public exports exist,
`add_node`/`get_node`, `get_node_or_none` miss). **Every package has zero
tests**: `query/` (4,458 lines), `agent/` (5,730), `service/` (3,640),
`graph/`, `executor/`, `retrieval/`, `scoring/`, `intent/`, all others.
`pyproject.toml` ships pytest + pytest-asyncio and CI runs them, so the
infrastructure exists — the net does not.

**Cost**: every refactoring in this document is unsafe; regressions in a
40K-line codebase with LLM-nondeterministic behavior go undetected until a
$100 benchmark run.

**Fix**: land unit tests for the LLM-free, deterministic cores first — they
are trivially testable: `graph/store.py`, `executor/runner.py`
(validation/rollback), `scoring/ranker.py`, `retrieval/bm25.py`,
`query/text_utils.py`, `config.py`. Target: each Tier-1/Tier-2 refactor below
lands *with* tests for the code it touches.

### 2. 🔴 `query/agent.py` (1,799 lines) hardcodes 14 benchmark names into the core library

`query/agent.py:789-808` is a literal dispatch dict —
`{"locomo": self._query_locomo_qa, "longmemeval": ..., "msc": ..., ...}` —
mapping to 14 private `_query_<bench>_qa` methods (lines 811–1261). Several
differ only by a magic `max_nodes` (40 / 15 / 5 / 30). The `query()` method
alone is 263 lines; `MemoryQueryAgent` has 44 methods. Half the entries
(`msc`, `qmsum`, `rgb`, `socialiqa`, `safetybench`, `futurex`) refer to
benchmarks whose directories no longer exist in the repo.

**Cost**: the shipped library carries leaderboard tuning; adding benchmark #9
means editing the core query engine; library users get 14 dead branches.

**Fix**: introduce a `QueryProfile` dataclass (max_nodes, boosts, prompt
template) passed into `query()`; move per-benchmark values into
`benchmarks/*/` or `configs/*_profile.yaml`; delete the dispatch dict and the
14 methods.

### 3. 🔴 `agent/domain.py` (1,500 lines): 15 of 21 domains are benchmark configs

Module-level `DomainConfig` literals for LOCOMO, FUTUREX, LONGMEMEVAL, MSC,
BABILONG, MUTUAL, MUSIQUE, STREAMINGQA, RGB, TIMEQA, NARRATIVEQA, QMSUM,
SOCIALIQA, TOMI, SAFETYBENCH (lines 419–1310). The extension point
`register_domain()` (line 1494) already exists and is unused for these. The
two hardcoded lists have already drifted: `TIMEQA_DOMAIN` exists here with no
`_query_timeqa_qa` counterpart in `query/agent.py`.

**Fix**: keep the ~6 real product domains in-tree; move benchmark domains to
YAML under `configs/` (which already holds the per-benchmark profiles) and
load via `register_domain()` from each runner.

### 4. 🔴 Layering inversion: five core packages import from `service/`

`cognifold.service.llm_keys` is imported by `agent/` (7 sites), `query/` (2,
incl. `query/llm.py:16`), `utils/` (2), `embeddings/` (2), and
`config.py:161` — the **root config module depends on the HTTP service
layer**. Symptom at scale: 287 function-local `from cognifold...` imports
across `src/` exist to dodge the resulting cycles (worst: `query/agent.py` 16,
`pipeline/classic.py` 16, `pipeline/layered.py` 15, `executor/runner.py` 15).

**Fix**: `llm_keys.py` has no FastAPI dependency — it is misfiled. Move it to
`cognifold/llm/keys.py` (or `utils/`), keep a deprecation re-export in
`service/`, and promote the deferred imports back to top level. One move,
13+ call sites cleaned.

---

## Tier 2 — Duplication and dead weight

### 5. 🟠 `replay/renderer.py::_build_html` is a 1,328-line function

Lines 115–1442: one f-string containing a complete HTML+CSS+JS application.
Every literal `{` in the CSS/JS must be doubled (`:root {{`); no JS/CSS
tooling can lint it. `simulator/visualizer.py` contains a second, independent
HTML renderer (`_build_sidebar_html` 184 lines, `render` 143 lines).

**Fix**: extract to `replay/templates/replay.{html,css,js}` as package data;
inject only the JSON payload. Do the same for the simulator, or merge the two
renderers.

### 6. 🟠 The four LLM generators are copy-paste clones (~400 duplicated lines)

`generator/{event_generator,service_logs,claude_code,computer_activity}.py`
(2,174 lines total): `_parse_events` is byte-identical in all four except the
ID prefix (`e-`/`svc-`/`cc-`/`ca-`); the 40-line retry/generate block is
identical in `claude_code.py:288-327` and `computer_activity.py:324-363`, with
a fifth copy in `generator/base.py:122-154` — the base class they all inherit
from and bypass.

**Fix**: hoist `_parse_events(text, date, id_prefix)` and the retry-wrapped
`generate_day()` into `generator/base.py`; subclasses supply only the prompt
builder and ID prefix.

### 7. 🟠 ~1,080 lines of provably dead code

Zero references across `src/`, `benchmarks/`, `scripts/`, `tests/`:

| File | Lines | Note |
|---|---|---|
| `scoring/hierarchical.py` | 450 | `HierarchicalContextSelector` never instantiated in-repo — **but README's Python quick-start showcases it**. Decide: wire it into the pipeline/service, or delete it *and* fix the README example. |
| `graph/projection.py` | 347 | re-exported by `graph/__init__.py` but never consumed; also defines `GraphSnapshot`, colliding with a different `GraphSnapshot` in `executor/runner.py:78` |
| `retrieval/cross_encoder_rerank.py` | 103 | drags a `BAAI/bge-reranker-v2-m3` dependency |
| `query/probe.py` | 98 | `SymbolicProbeAgent` unused |
| `graph/metrics_extended.py` | 87 | not even exported |
| `agent/prompts.py:631,691` | ~80 | two formatters that only render the dead `HierarchicalContext` |

### 8. 🟠 Two parallel embedding stacks; the deprecated one is on a hot path with a key-leak hazard

`cognifold/embeddings/` is the real stack (provider ABC + factory).
`utils/embeddings.py` (122 lines) raises a `DeprecationWarning` at import yet
is still called from the core executor (`executor/runner.py:301,310,575,580`).
It is hardcoded to `text-embedding-3-small` behind a **process-global
singleton** that caches a client built from whichever API key was live first —
in the multi-session service this is a cross-session key/model leak, and it
ignores `EmbeddingConfig`.

**Fix**: rewrite the 4 call sites in `executor/runner.py` against
`cognifold.embeddings.create_provider()`; delete `utils/embeddings.py`.

### 9. 🟡 Four stopword sets, four cosine implementations

Stopwords: `graph/entity_index.py:24`, `retrieval/bm25.py:32`,
`query/text_utils.py:15`, `query/agent.py:50`. Cosine:
`graph/store.py:561`, `query/agent.py:166`, `embeddings/search.py:428`,
`graph/metrics_extended.py:62`. They will drift.

**Fix**: one `utils/text.py` (tokenize + stopwords) and one
`utils/vector.py::cosine`.

---

## Tier 3 — Service layer & configuration

### 10. 🔴 Session LLM API keys are persisted in plaintext

`service/session.py:346` (`_persist_session`) and `:365` include
`"llm_api_keys": session.llm_api_keys` in the persisted blob — written as
plain JSON to `sessions/<id>/session.json`, Redis, and Supabase. Clients post
real keys in the session-create body (`cli/client.py:756-767`).

**Fix**: strip `llm_api_keys` from the persisted payload (keep in-memory
only), or encrypt at rest and re-supply per request.

### 11. 🔴 No per-user authorization — any valid API key reads any session

`service/auth.py` (37 lines) is the whole model: a flat key set returning the
key string. `Session.user_id` exists (`session.py:38`) but no route checks it;
all 20+ `get_session(session_id)` call sites across `routes/` fetch by ID
alone. Auth defaults to **off** (`AppSettings.api_keys=None`, and
`cli/serve.py:92` warns then proceeds).

**Fix**: auth dependency returns a principal; add a shared
`get_owned_session(session_id, principal)` dependency used by every route.

### 12. 🟠 `--workers` is a silent no-op; session TTL is dead config

`cli/serve.py:138-144` passes the app **object** to `uvicorn.run`, so
`workers=` is ignored — `cognifold serve --workers 8` runs one process, no
error. `SessionManager.cleanup_expired()` (`session.py:313`) has zero callers,
so `COGNIFOLD_SESSION_TTL_HOURS` does nothing. The `get_session` recovery path
(`session.py:228`) mutates `self._sessions` without the lock and bypasses the
`max_sessions` eviction check. `--session-backend` offers `file|redis` while
the factory also supports `supabase` (env-only).

**Fix**: pass `"cognifold.service.wsgi:app"` as an import string; start a
lifespan task that periodically calls `cleanup_expired()`; take the lock in
the recovery branch; align the CLI choices with the factory.

### 13. 🟠 Config sprawl: 6 config modules, 2 conflicting YAML defaults, 4 default model names

Six config modules totaling ~960 lines (`config.py` 477, plus
`query/retrieval/agent/embeddings/cli` configs). `config.example.yaml` says
`gemini-3-flash-preview` / `max_nodes: 50`; `configs/cognifold.yaml` says
`openai:gpt-5.2` / `max_nodes: 20`; neither is loaded by default. Default
model names in code: `config.py:17` `gemini-2.5-flash`, `agent/config.py:29`
`gemini-3-flash-preview`, `query/llm.py:101` `gpt-4o`, `query/models.py:92`
`openai:gpt-5`. The CLI default retrieval mode is `legacy`
(`cli/query.py:100`) while the library default is `HYBRID`
(`query/models.py:102`).

**Fix**: one `pydantic-settings` root (`env_prefix="COGNIFOLD_"`) replaces the
hand-rolled `from_env` walrus chains (`config.py:200-260`); keep exactly one
example YAML; define ONE default-model constant.

### 14. 🟡 Silent cross-provider fallback in `query/llm.py`

An OpenAI failure is logged at `logger.debug` and silently retried on Gemini
with a **different model** (`query/llm.py:113,143`) — a bad OpenAI key
produces plausible answers from the wrong model with no visible signal.

**Fix**: log at `warning` with the substituted model, and surface the
fallback in the result metadata.

---

## Benchmarks-side duplication (companion to docs/BENCHMARK.md "Known gaps")

- `locomo/run_benchmark.py` (1,102 lines) bypasses `BenchmarkRunner` and
  re-defines `check_api_keys` / `call_llm_for_eval` / `evaluate_with_llm` +
  an 80-line argparse copy. `narrativeqa` subclasses but copy-pastes the
  entire `run()` loop (lines 123–297). `longmemeval/run_eval.py` (2,390
  lines) reuses nothing.
- `shared/baseline_runner.py` (1,229 lines) duplicates provider detection,
  LLM calls, F1/EM, and `wilson_ci_95` (vs `stats_utils.wilson_ci`); its
  registry still lists removed benchmarks (socialiqa, rgb, qmsum, safetybench).
- Two embedder factories: `benchmarks/_utils.py` (canonical, used by
  base_runner) vs `benchmarks/embedding_utils.py` (legacy; its docstring still
  claims "All benchmark runners should use this"). Delete the legacy one.
- `compute_f1`/`normalize_answer` re-implemented in `musique`, `streamingqa`,
  `narrativeqa`, and `baseline_runner` — move to `shared/stats_utils.py`.
- Root leftovers to remove: `my_prompt.md` (self-declared moved to skills),
  `history.md` (superseded by `benchmarks/longmemeval/HISTORY.md`),
  `generate_demo.py` (unreferenced, unlinted).

---

## Checked and NOT a problem (do not re-litigate)

- **`symbolic/` is not dead** — used by `benchmarks/shared/base_runner.py`,
  `benchmarks/tomi/`, and `query/agent.py:497`. It is benchmark-coupled
  (finding #2) but live.
- **`pipeline/` classic vs layered** — both alive by design: `classic.Pipeline`
  is the public export + MCP path; `LayeredPipeline` serves CLI run + the
  service. Two execution models is a cost, not accidental duplication.
- **Exception discipline** — 78 `except Exception`, 0 bare `except`, only 9
  silent swallows (mostly cost telemetry). Acceptable; #14 is the one real
  case.
- **TODO/FIXME density** — trivially low (max 2 per file).
- **`service/llm_keys` thread-local design** — correct as written; the global
  client caches in `query/llm.py:24-25` are keyed by key value, deliberately.

## Suggested sequencing

1. #1 tests for deterministic cores (unblocks everything)
2. #4 move `llm_keys` (small, mechanical, removes the inversion)
3. #7 delete dead code + README fix (pure win, shrinks the surface)
4. #8 retire the deprecated embedding stack (closes the key-leak hazard)
5. #2 + #3 de-benchmark the core library (the big one; do behind tests)
6. #10/#11 before any real multi-user deployment
7. #5, #6, #9, #12, #13, #14 opportunistically
