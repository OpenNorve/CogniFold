# CogniFold Research Methodology — North-Star Operating System

> This document answers only **"how we research"**, not "what we research."
> It defines the rules for turning neuroscience / cognitive-science principles into **validated, useful** memory capabilities.
> Audience: CogniFold authors + build-in-public readers. Specific topics (temporal reasoning, forgetting, dual-track…) are out of scope here — they fall out of the process.

---

## 0. Get the North Star right

- **North Star = a capability**: an always-on memory that continuously accumulates / consolidates / forgets / anticipates, and **measurably helps the agent**.
- **Brain-inspiration = a source of hypotheses**: the brain is the existence proof that this kind of memory works, and an inexhaustible library of mechanisms. It is a **compass + idea pool, not the goal.**
- **Corollary (this is what decides whether we dare to cut things)**: a brain mechanism with no payoff gets dropped — we carry no "biological-fidelity" debt. Planes don't flap their wings.

---

## 1. What drives the agenda: problems pull, neuroscience supplies

| Role | Who | Function |
|---|---|---|
| **Generator** | neuroscience / cogsci | supplies candidate mechanisms to the backlog |
| **Gate** | the agent's **measured deficits** | decides **which to pull, and when** |

- Don't march to "cover the whole brain" (= museum risk, building to resemble); pull by "fix what actually hurts" (= building to be useful).
- This gives direction to "consolidate whatever pays off": go to neuroscience **with a known deficit in hand**, instead of casting a wide net and hoping.

---

## 2. The constitution (5 rules)

1. **Program-level North Star = transfer width.** A single experiment winning doesn't count; a mechanism must **reproduce its gain on ≥2 unrelated tasks** to be "consolidated." Otherwise it's just overfitting one benchmark.
2. **Borrow at Marr's computational / algorithmic level, never the implementation level.** Borrow "what problem it solves, by what algorithm," not spikes / neurotransmitters / specific brain regions.
3. **Validation gate = the four-piece kit + a pre-registered kill threshold.** Claim / operationalization / isolated ablation / kill criterion — no merge without all four; the threshold is **frozen before the run**.
4. **Dual-evidence measurement = CogEval structure + external-task payoff.** Both must move to keep it; CogEval-only = self-serving, and fatal to credibility.
5. **Publish negative results = the kill log.** Daring to post "tried this brain mechanism, no gain, removed it" is what makes build-in-public credible; an all-success repo convinces no one.

---

## 3. The selection mechanism: a gated funnel ★ (core of this doc)

```
 gap ─▶ ① literature ─▶ ② survey/ ─▶ ③ feasibility ─▶ ④ pre-deploy ─▶ ⑤ consolidate
        retrieval        abstraction    ideation         validation       (ledger)
  │         │               │               │                │
  └─────────┴───────────────┴───────────────┴────────────────┘
                     any gate fails → Kill log (public)
```
**Design principle: cheap deaths on the left (paper-only); the expensive part (engineering + compute) is spent only on candidates that pass the ③ feasibility gate.** Each stage has a defined **output artifact** and a **go / no-go gate**.

---

### Stage 0 — Gap intake
- **Purpose**: candidates may only originate from a real, measurable deficit — not from "feels brain-like."
- **Do**: localize the symptom to a metric (a benchmark weakness / a cluster of failure cases / a capability the always-on vision needs but we lack).
- **Output**: `gap ticket` — symptom + the metric that quantifies it + current baseline.
- **Gate**: is this a **quantifiable, real** gap? (not a vibe)

### Stage 1 — Literature retrieval
- **Purpose**: for this gap, find mechanisms in neuro / cogsci that solve the **analogous computational problem**.
- **Do**:
  - Search by **cognitive-function / problem** terms, not by our code's terms (search `memory consolidation` / `active forgetting` / `event segmentation`, not `graph dedup`).
  - **Map the landscape with reviews first**, then expand from seed papers via forward / backward citations (Semantic Scholar / Connected Papers / Google Scholar / arXiv).
  - Prefer **canonical, repeatedly-validated** sources; distrust single novel results.
  - LLM-assisted survey is allowed, but **verify each claim against the primary source** (guard against second-hand distortion).
- **Output**: an **annotated mini-bibliography** (3–8 papers) + a one-line candidate-mechanism description.
- **Gate**: is there a **mechanism-level** (algorithmic) description, not just a phenomenon?

### Stage 2 — Survey & abstraction
- **Purpose**: strip the mechanism down to a level we can implement.
- **Do**: use Marr's levels — what computational problem does it solve, by what algorithm? Strip the implementation level (spikes / transmitters / regions).
- **Output**: a `mechanism card` —
  (a) cognitive claim + citations; (b) the computational problem it solves; (c) the algorithm at an implementable level; (d) **what it predicts will improve** (linked back to the Stage-0 metric).
- **Gate**: can it be stated at the algorithmic level **and** does that level map onto our primitives (graph ops / scoring / agent)?

### Stage 3 — Feasibility & operationalization ★ the cheap elimination point
- **Purpose**: judge "worth doing? cleanly measurable?" **without writing real code**.
- **Do**:
  - Map the mechanism onto CogniFold's real primitives (`node/edge ops`, `scoring`, `consolidation`, `retrieval`).
  - Design the **minimal change** + the **minimal ablation** (with / without).
  - Estimate **cost** (LLM calls, latency, complexity) and **conflict risk** (does it fight existing ops?).
- **Output**: a one-page `design doc` — the concrete operation / the A-B design / the metric(s) it should move / expected effect size / cost.
- **Gate**: **feasibility × clean measurability (the ablation isolates the mechanism) × expected payoff**. **Most candidates should die here** — decidable on paper, no engineering spent.

### Stage 4 — Pre-deployment validation ★ guards against "looks real, measures hollow"
- **Purpose**: before integrating into the main system, validate the mechanism's usefulness in isolation.
- **Do** (order matters):
  1. **Pre-register**: before running, freeze the hypothesis / metric / keep-threshold (= kill criterion) / ablation design.
  2. **Minimal prototype + trigger self-check**: implement in a branch / sandbox, smoke on a small slice, and **first confirm the mechanism actually fired** — verify the artifacts it claims to produce really appear. (Lesson: we once had an "Intents" column filled with 2–22 on paper while the real graphs had 0.)
  3. **Isolated ablation**: same seeds, everything else held fixed, A/B on the **CogEval structural probe + an external-task slice**.
  4. **Dual evidence + significance**: both structure and task clear the noise (multiple seeds / confidence intervals).
  5. **Cost / regression check**: no latency / cost blow-up, no regression on already-consolidated metrics.
- **Output**: a `validation report` (pre-registration + results + verdict).
- **Gate**: meets the pre-registered threshold + dual evidence + no unacceptable regression → promote to Stage 5; otherwise → **Kill**.

### Stage 5 — Consolidate (deploy)
- **Purpose**: turn a surviving mechanism into a permanent capability that **can't be silently broken** by future changes.
- **Do**: merge to main; record it in the `validated-mechanism ledger`; **add its probe to the permanent regression suite**.
- **Output**: merged code + ledger entry + standing probe.

### Kill & negative results (throughout)
- Any gate failure → a `kill log` entry: **what / why it died / evidence**. Public.

### Periodic re-audit
- Consolidated mechanisms can fail later (mechanism interactions / data drift) → periodically re-run the standing probes; demote or remove what no longer holds.

---

## 4. Artifacts (what the research produces)

| Stage | Artifact | Nature |
|---|---|---|
| 0 | Gap ticket | deficit + quantifying metric |
| 1 | Annotated mini-bibliography | candidate sources |
| 2 | Mechanism card | algorithm-level mechanism + prediction |
| 3 | Design doc | operation + ablation + cost |
| 4 | Validation report | pre-registration + results + verdict |
| 5 | Ledger entry + standing probe | a consolidated capability |
| any | Kill log | negative result (public) |

---

## 5. Cadence

- **Daily**: push candidates through the funnel and post the day's progress build-in-public; whenever something is in flight, **produce at least one verdict (promote or kill)**; update the ledger + kill log as part of the daily push.
- **Most candidates stop at Stage 1–3** (cheap); only a few reach Stage 4 (compute-heavy), so a daily cadence is sustainable — the bottleneck is paper-level triage, not compute.
- **Periodically**: re-audit consolidated mechanisms.

---

## Differentiation vs SOTA

The North-Star process says a mechanism only earns its place if it pays off where incumbents leave a gap. Here is where CogniFold's surviving mechanisms diverge from the current memory-system landscape:

- **Proactive intent (unique).** CogniFold's INTENT (`i-`) layer crystallizes goals from graph *topology* — it remembers to act, not just to recall. No mainstream memory system models prospective memory as an emergent structural property; they are all reactive retrieval stores. This is the differentiator the compass points at, not a bolt-on.
- **CLS tri-layer substrate.** Episodic EVENTs fold into semantic CONCEPTs into prefrontal INTENTs, mapped to hippocampal → neocortical → prefrontal regions (see `src/cognifold/brain/memory_coverage.json` for the honest ~60% taxonomy coverage). Most systems collapse memory into a single embedding store; the tri-layer is what makes consolidation and proactivity expressible at all.
- **Explainable graph rewrites.** Accumulation / compression / decay / completion are transparent, auditable graph operations — test-time learning with no gradient updates and no surface-text rewriting. The rewrite log *is* the explanation.

Positioned against named incumbents:

| System | Core strength | What CogniFold adds |
|---|---|---|
| **Mem0** | Portable, framework-agnostic memory layer | Proactive intent + tri-layer structure rather than flat extracted facts |
| **Zep / Graphiti** | Temporal knowledge graph, bi-temporal edges | Prospective-memory layer and CLS consolidation on top of the temporal graph |
| **EverOS** | Local-first, human-readable markdown memory | Emergent typed/weighted graph topology where goals surface on their own, not hand-authored notes |

Honesty clause (per the North-Star kill discipline): non-declarative systems — procedural, priming, conditioning — and affective/sensory memory are **planned, not modeled**. The published coverage number recomputes from the per-system status so it cannot drift upward without the evidence.

---

## In one line

**Let problems drive and the brain supply; put every mechanism through the cheap "feasibility" gate and the hard "pre-deployment validation" gate; quantify and publish both success (transfer width) and failure (kill).** Lock this process down, and "what to research next" falls out of the gaps on its own.
