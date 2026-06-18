# Cognifold Prompt Engineering Guide

This document describes the prompting system used by the Cognifold agent to maintain and update the concept graph.

## Overview

The Cognifold agent uses a structured prompt system with:
1. **System Prompt**: Defines the agent's role, graph structure, and guidelines
2. **User Prompt**: Provides context for each event processing step
3. **Reasoning Modes**: Different processing strategies for different situations

## Modular Section Architecture (Phase 13)

The system prompt is composed from 20 named sections organized into 4 groups. Domains can toggle sections on/off, override content, or inject custom sections.

### Section Groups

| Group | Sections | Purpose |
|-------|----------|---------|
| **core** | `core.role`, `core.graph_structure`, `core.edge_types`, `core.tools`, `core.output_format`, `core.explainability`, `core.connectivity`, `core.dedup`, `core.self_review`, `core.validation`, `core.operations`, `core.rules` | Always included (12 sections) |
| **concepts** | `concepts.hierarchy`, `concepts.temporal_patterns`, `concepts.strength`, `concepts.guidelines` | Concept discovery and management (4 sections) |
| **intents** | `intents.guidelines`, `intents.metadata`, `intents.patterns` | Intent/action creation (3 sections) |
| **time** | `time.nodes` | TIME node creation (1 section) |

### Module Layout

| File | Purpose |
|------|---------|
| `src/cognifold/agent/prompt_sections.py` | Section constants, registry, groups, `resolve_sections()` |
| `src/cognifold/agent/prompts.py` | `SYSTEM_PROMPT_TEMPLATE`, `format_system_prompt_for_domain()` |
| `src/cognifold/agent/domain.py` | `DomainConfig` with `disabled_sections`, `extra_sections` |
| `src/cognifold/agent/prompt_profile.py` | YAML profile loading with section config |

### How It Works

1. `SYSTEM_PROMPT_TEMPLATE` is reconstructed from sections: `"".join(SECTION_REGISTRY[k] for k in DEFAULT_SECTION_ORDER)`
2. `format_system_prompt_for_domain()` uses `resolve_sections()` to build the prompt, respecting `disabled_sections` and `extra_sections`
3. YAML profile `template` overrides bypass section composition entirely

### Domain Section Control

`DomainConfig` has three section-related fields:

```python
DomainConfig(
    # ...existing fields...
    disabled_sections=frozenset({"intents"}),  # Disable entire group
    extra_sections={"custom.speaker": "## Speaker\n..."},  # Inject custom sections
    extra_section_position="before_rules",  # Where to inject extras
)
```

**Positions for extra sections:**
- `"before_rules"` (default) — before the "Important Rules" section
- `"after_tools"` — after the "Available Tools" section
- `"after_rules"` — at the very end

### Adding a New Domain

1. Create a `DomainConfig` in `domain.py` with domain-specific examples and guidelines
2. Set `disabled_sections` to remove irrelevant sections (e.g., `frozenset({"intents"})` for memory benchmarks)
3. Add `extra_sections` for domain-specific prompt content
4. Register in `DOMAIN_REGISTRY`

Example:

```python
MY_DOMAIN = DomainConfig(
    name="my-domain",
    description="Description of my domain",
    event_description="events in my domain",
    disabled_sections=frozenset({"intents", "time"}),
    extra_sections={
        "custom.scoring": "\n\n## Custom Scoring\nScore events by importance..."
    },
    # ...other fields...
)
```

### YAML Profile Section Config

Profiles can override section settings via the `sections` key:

```yaml
profiles:
  my-profile:
    domain: my-domain
    sections:
      disabled:
        - intents
        - time
      extra:
        custom.speaker: "## Speaker Attribution\nHandle speakers..."
```

## Scenario Profiles (Gallery)

Prompt profiles are named, ready-to-use bundles of domain + reasoning mode +
model + guidelines (+ optional custom template and section toggles). They live
in `configs/prompt_profiles.yaml` and are loaded by
`cognifold.agent.prompt_profile.load_prompt_profiles(path)`, which returns a
`dict[str, PromptProfile]` keyed by profile name.

The shipped profiles:

| Profile          | Target scenario                                  | Mode       | When to use                                                                 | Provider / model                  |
|------------------|--------------------------------------------------|------------|-----------------------------------------------------------------------------|-----------------------------------|
| `personal-v1`    | Personal timeline (daily-life event streams)     | quick      | Fast ingest of personal activity logs; minimal concept creation             | default (from config)             |
| `wiki-v1`        | Wiki / notes / long-form documents               | analytical | Deep analysis of document chunks; synthesis-focused actions; no TIME nodes   | default (temperature 0.4)         |
| `wiki-v3-openai` | Wiki / novels via OpenAI with a strict template  | analytical | OpenAI run that enforces connectivity + mandatory actions per concept       | `openai:gpt-5.2-2025-12-11`       |
| `wiki-v3-gemini` | Wiki / novels via Gemini with a strict template  | analytical | Same strict graph-building rules on a Gemini model                          | `gemini-2.5-flash`                |
| `wiki-v4-openai` | Wiki / novels via OpenAI (refined v3 template)   | analytical | Latest OpenAI wiki template; prioritizes high-value synthesis actions       | `openai:gpt-5.2-2025-12-11`       |

Notes:
- "Mode" is the `ReasoningMode` enum — `quick`, `analytical`, or
  `consolidation`. The wiki profiles use the strict custom `templates.system`
  override (which bypasses section composition); the personal/wiki-v1 profiles
  use the default composed prompt for their domain.
- A profile's model (`model.name`) overrides the config model only when set;
  otherwise the model comes from `CognifoldConfig` / `AgentConfig`.

### Listing and using profiles from the CLI

List every profile (name + domain + mode + model) and exit:

```bash
cognifold --list-profiles
# point at a different profiles file:
cognifold --list-profiles --prompt-profiles path/to/profiles.yaml
```

Run graph-building with a profile (agent mode required — the profile configures
the graph-update LLM agent):

```bash
# --profile is an alias for --prompt-profile
cognifold run examples/wiki/notes_timeline.json --agent --prompt-profile wiki-v1
cognifold run data/timeline.json --agent --profile wiki-v3-openai
# custom profiles file:
cognifold run data/timeline.json --agent --profile my-profile \
  --prompt-profiles configs/my_profiles.yaml
```

The same `--profile` works in fast (layered) mode:

```bash
cognifold run data/timeline.json --fast --agent --profile wiki-v1
```

If an unknown profile name is given, the command prints the available profiles
and exits non-zero:

```text
Error: Prompt profile not found: nope
Available profiles: personal-v1, wiki-v1, wiki-v4-openai, wiki-v3-openai, wiki-v3-gemini
```

For `query`:

```bash
cognifold query --graph output/graph.json --profile wiki-v1 -v "key themes?"
```

`query` accepts `--profile` / `--prompt-profile` for parity and validates the
name (so typos fail fast and the resolved profile is shown in `--verbose`
output). **It does not change retrieval results** — `query` is read-only
retrieval over a pre-built graph, whereas profiles shape graph *building* via
`run`. Use profiles when building the graph; query the result however you like.

### Authoring a custom profile

A profile is one entry under the top-level `profiles:` key. All fields are
optional except an implicit identity (the YAML key becomes `profile_id`). The
full shape understood by `load_prompt_profiles`:

```yaml
profiles:
  my-profile:                      # -> PromptProfile.profile_id
    domain: wiki                   # DomainConfig name (DOMAIN_REGISTRY key)
    mode: analytical               # ReasoningMode: quick | analytical | consolidation
    model:
      name: openai:gpt-5.2-2025-12-11  # optional; overrides config model
      temperature: 0.3
      max_tokens: 4096
      max_exploration_steps: 3
    guidelines:                    # injected into {concept_guidelines}/{action_guidelines}
      concept:
        - Prefer updating existing concepts over duplicates
      action:
        - Create synthesis actions for strong recurring concepts
      time:
        - Create TIME nodes only for explicit dates
    templates:                     # optional; OVERRIDES section composition entirely
      system: |
        You are a cognitive graph agent...
        {concept_guidelines}
        {action_guidelines}
      user: |
        Process this event: {event}
    sections:                      # optional; only applies when NOT using templates.system
      disabled:                    # section or group names to exclude
        - intents
        - time
      extra:                       # custom sections to inject
        custom.speaker: "## Speaker Attribution\n..."
    features:                      # free-form flags consumed downstream
      enable_time_nodes: false
```

Then load and verify:

```bash
cognifold --list-profiles --prompt-profiles configs/my_profiles.yaml
cognifold run data/timeline.json --agent --profile my-profile \
  --prompt-profiles configs/my_profiles.yaml
```

### Toggling sections via DomainConfig

Profiles inherit their domain's section composition from
`DomainConfig` (`src/cognifold/agent/domain.py`). The relevant fields:

- `disabled_sections: frozenset[str]` — exclude individual sections
  (e.g., `core.tools`) or whole groups (`"intents"`, `"time"`, `"concepts"`,
  `"core"`, `"symbolic"`). Group names are expanded to their member sections.
  Example: `LOCOMO_DOMAIN` uses `disabled_sections=frozenset({"intents"})` to
  drop all intent sections for the benchmark.
- `extra_sections: dict[str, str]` — inject custom prompt text keyed by a
  custom section name (e.g., `CLAUDE_CODE_DOMAIN`'s `claude_code.tool_context`).
- `extra_section_position: str` — where extras are injected:
  `"before_rules"` (default), `"after_tools"`, or `"after_rules"`.

A profile can override these per-run via its `sections:` block (`disabled` /
`extra` above), which feeds `PromptProfile.disabled_sections` /
`PromptProfile.extra_sections`. Section names and groups are defined in
`SECTION_REGISTRY` / `SECTION_GROUPS` and resolved by `resolve_sections()` in
`src/cognifold/agent/prompt_sections.py`.

> Note: when a profile sets `templates.system`, section composition is bypassed
> entirely — the raw template (with `{concept_guidelines}` / `{action_guidelines}`
> placeholders) is used as-is. Use `sections:` only with the default composed prompt.

## System Prompt Structure

The system prompt includes these sections in order:

### Core Sections (always present)

1. **core.role** — Agent role definition and capabilities
2. **core.graph_structure** — Four node types (EVENT, CONCEPT, INTENT, TIME)
3. **core.edge_types** — Semantic edge types with weights
4. **core.tools** — Available graph exploration tools
5. **core.output_format** — JSON response structure
6. **core.explainability** — Reasoning and grounding requirements
7. **core.connectivity** — Orphan node prevention rules
8. **core.dedup** — Duplicate concept avoidance
9. **core.self_review** — Plan self-review for better connectivity
10. **core.validation** — Self-validation checklist
11. **core.operations** — Operation type reference
12. **core.rules** — Important rules

### Concept Sections (toggleable)

13. **concepts.hierarchy** — Hierarchical concept creation
14. **concepts.temporal_patterns** — Temporal pattern detection
15. **concepts.strength** — Concept strength dynamics
16. **concepts.guidelines** — Domain-specific concept guidelines

### Intent Sections (toggleable)

17. **intents.guidelines** — Proactive intent creation
18. **intents.metadata** — Intent temporal metadata
19. **intents.patterns** — Pattern-based intent examples

### Time Sections (toggleable)

20. **time.nodes** — TIME node creation and usage

## Reasoning Modes

### Quick Mode
Fast processing with minimal concept creation:
- Add event with basic connections
- Only create concepts with overwhelming evidence
- Target: 2-4 operations maximum

### Analytical Mode
Deep pattern analysis:
- Examine temporal patterns across context window
- Look for hidden connections and emerging themes
- Consider creating hierarchical concepts
- Use tools to explore beyond context window

### Consolidation Mode
Focus on graph health:
- Identify similar or duplicate concepts to merge
- Find weak concepts (strength < 0.3) to remove
- Create parent concepts for clusters
- Reconnect orphan nodes

## User Prompt Templates

### Standard Template
Used for Quick mode and default processing:
- New event details
- Context window nodes
- Basic analysis task

### Analytical Template
Used for deep analysis:
- Event details, context window
- Pattern analysis checklist
- Hierarchical analysis prompts

### Consolidation Template
Used for graph cleanup:
- Full context window, graph statistics
- Consolidation task list

### Hierarchical Context Template
Used with Phase 9.2 hierarchical context windows:
- Immediate, working, and background context levels
- Priority-aware analysis instructions

## Configuration

Prompts can be configured via `AgentConfig`:

```python
from cognifold.agent.config import AgentConfig

config = AgentConfig(
    model_name="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=4096,
    max_exploration_steps=3,
    concept_guidelines=(...),
    action_guidelines=(...),
)
```

## File Locations

| File | Purpose |
|------|---------|
| `src/cognifold/agent/prompt_sections.py` | Section constants, registry, `resolve_sections()` |
| `src/cognifold/agent/prompts.py` | Prompt templates and formatters |
| `src/cognifold/agent/domain.py` | Domain configs with section control |
| `src/cognifold/agent/prompt_profile.py` | YAML profile loading |
| `src/cognifold/agent/config.py` | Agent configuration with guidelines |
| `src/cognifold/agent/graph.py` | LangGraph agent definition |
