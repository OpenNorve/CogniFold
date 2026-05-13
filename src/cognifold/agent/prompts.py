"""Prompts for the Cognifold agent.

This module provides domain-agnostic prompts that can be customized for different
event stream domains (personal timeline, computer activity, service logs, etc.).
"""

from __future__ import annotations

import json
from enum import Enum
from typing import TYPE_CHECKING, Any

from cognifold.agent.prompt_sections import (
    DEFAULT_SECTION_ORDER,
    SECTION_REGISTRY,
    resolve_sections,
)

if TYPE_CHECKING:
    from cognifold.agent.domain import DomainConfig


class ReasoningMode(Enum):
    """Reasoning modes for the agent."""

    QUICK = "quick"  # Fast processing, minimal concept creation
    ANALYTICAL = "analytical"  # Deep pattern analysis
    CONSOLIDATION = "consolidation"  # Focus on merging and cleanup


# Domain-agnostic base prompt template — reconstructed from composable sections.
# See ``prompt_sections.py`` for the individual section constants.
SYSTEM_PROMPT_TEMPLATE = "".join(SECTION_REGISTRY[k] for k in DEFAULT_SECTION_ORDER)

# Mode-specific prompt additions
MODE_PROMPTS = {
    ReasoningMode.QUICK: """
## Reasoning Mode: QUICK

In quick mode, focus on efficiency:
- Add the event with basic connections to obvious concepts
- Only create new concepts if overwhelming evidence exists
- Skip deep pattern analysis
- Prefer existing concepts over creating new ones
- Target: 2-4 operations maximum
""",
    ReasoningMode.ANALYTICAL: """
## Reasoning Mode: ANALYTICAL

In analytical mode, perform deep analysis:
- Examine temporal patterns across the context window
- Look for hidden connections and emerging themes
- Consider creating hierarchical concepts
- Analyze concept strength changes carefully
- Use tools to explore beyond the context window
- Be thorough in your reasoning
""",
    ReasoningMode.CONSOLIDATION: """
## Reasoning Mode: CONSOLIDATION

In consolidation mode, focus on graph health:
- Identify similar or duplicate concepts that should be merged
- Look for weak concepts (strength < 0.3) that could be removed
- Find orphan nodes that need connections
- Create parent concepts for clusters of related concepts
- Prefer MERGE_NODES and REMOVE_NODE operations
- Strengthen the conceptual hierarchy
""",
}

# Backward compatibility aliases
SYSTEM_PROMPT_BASE = SYSTEM_PROMPT_TEMPLATE
SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE

USER_PROMPT_TEMPLATE = """## New Event

{event_details}

## Context Window

These are the most relevant nodes currently in the graph:

{context_window}

## Task

Analyze this event and determine the appropriate graph updates. Consider:
1. How does this event relate to existing nodes in the context?
2. Are there emerging patterns that warrant a new CONCEPT?
3. Should any existing concepts be strengthened?
4. Are there any implicit goals or desires that warrant an INTENT?

**IMPORTANT - Edge Types**: For every ADD_EDGE operation, you MUST include `edge_type`:
- Event → Concept: use `"edge_type": "grounds"` (event provides evidence)
- Event → Intent: use `"edge_type": "triggers"` (event activates goal)
- Concept → Intent: use `"edge_type": "triggers"` (pattern suggests action)
- Concept → Concept: use `"edge_type": "derived_from"` or `"edge_type": "related_to"`

Example ADD_EDGE with edge_type:
```json
{{"op": "ADD_EDGE", "source_id": "e-001", "target_id": "c-morning-routine", "edge_type": "grounds"}}
```

Explore the graph with tools if needed, then provide your update plan as JSON.
"""

# Enhanced user prompt for analytical mode
ANALYTICAL_USER_PROMPT_TEMPLATE = """## New Event

{event_details}

## Context Window

These are the most relevant nodes currently in the graph:

{context_window}

## Pattern Analysis

Look for these pattern types:
- **Temporal patterns**: Does this event occur at similar times to others?
- **Activity clusters**: Are there related activities that form a theme?
- **Social patterns**: Are the same people/locations involved repeatedly?
- **Behavioral sequences**: Does this event typically follow/precede others?

## Hierarchical Analysis

Consider the concept hierarchy:
- What specific concepts (Level 1) relate to this event?
- Do multiple Level 1 concepts share a parent theme (Level 2)?
- Is there an even more abstract pattern emerging (Level 3)?

## Task

Perform a thorough analysis and provide your update plan:
1. Add the event as a node
2. Connect to existing concepts and strengthen where appropriate
3. Create new concepts only with clear evidence
4. Consider hierarchical relationships
5. Look for opportunities to consolidate similar concepts

Provide your detailed update plan as JSON.
"""

# Consolidation-specific prompt
CONSOLIDATION_USER_PROMPT_TEMPLATE = """## Graph Health Check

{context_window}

## Statistics

{graph_stats}

## Consolidation Tasks

Focus on improving the graph structure:

1. **Merge Candidates**: Find concepts with similar titles or overlapping evidence
2. **Weak Concepts**: Identify concepts with strength < 0.3 for potential removal
3. **Orphan Nodes**: Find nodes with no edges that need connections
4. **Hierarchy Gaps**: Look for related concepts that need a parent category
5. **Redundant Edges**: Find edges that don't add semantic value

## Task

Analyze the graph and provide a consolidation plan. You may:
- Skip adding the event (if doing a pure consolidation pass)
- Focus on MERGE_NODES operations for similar concepts
- Use REMOVE_NODE for weak or redundant nodes
- Create parent concepts to organize clusters
- Reconnect orphan nodes

Provide your consolidation plan as JSON.
"""


def _format_hierarchy_examples(examples: list[dict[str, str]]) -> str:
    """Format hierarchy examples for the prompt."""
    if not examples:
        return "**Level 1 (Specific)**: Direct observations\n**Level 2 (Category)**: Related patterns grouped together\n**Level 3 (Abstract)**: High-level themes"

    lines = []
    for ex in examples[:2]:  # Show max 2 examples
        lines.append(f'**Level 1 (Specific)**: "{ex.get("level1", "")}"')
        lines.append(f'**Level 2 (Category)**: "{ex.get("level2", "")}"')
        lines.append(f'**Level 3 (Abstract)**: "{ex.get("level3", "")}"')
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_pattern_types(patterns: list[str]) -> str:
    """Format pattern types for the prompt."""
    if not patterns:
        return "- **Daily**: Same time each day\n- **Weekly**: Specific days of the week\n- **Triggered**: Based on context or conditions"
    return "\n".join(f"- {p}" for p in patterns)


def _format_concept_example(examples: list[dict[str, Any]]) -> str:
    """Format a concept example for JSON display."""
    if not examples:
        return json.dumps(
            {
                "op": "ADD_NODE",
                "node_type": "concept",
                "data": {
                    "concept_id": "c-example",
                    "title": "Example Concept",
                    "level": 1,
                    "parent_concept": "c-parent",
                    "strength": 0.6,
                    "evidence_count": 3,
                },
            },
            indent=2,
        )

    ex = examples[0]
    return json.dumps(
        {
            "op": "ADD_NODE",
            "node_type": "concept",
            "data": {
                "concept_id": ex.get("concept_id", "c-example"),
                "title": ex.get("title", "Example Concept"),
                "description": ex.get("description", "Brief description of this concept."),
                "level": 1,
                "parent_concept": "c-parent",
                "strength": ex.get("strength", 0.6),
                "evidence_count": ex.get("evidence_count", 3),
            },
        },
        indent=2,
    )


def _format_time_example(examples: list[dict[str, Any]]) -> str:
    """Format a time node example."""
    if not examples:
        return json.dumps(
            {
                "op": "ADD_NODE",
                "node_type": "time",
                "data": {
                    "id": "t-example",
                    "title": "Scheduled Event",
                    "scheduled_time": "2026-01-18T14:00:00Z",
                    "recurrence": "weekly",
                    "urgency_window_hours": 24,
                },
            },
            indent=2,
        )

    ex = examples[0]
    return json.dumps(
        {
            "op": "ADD_NODE",
            "node_type": "time",
            "data": {
                "id": ex.get("id", "t-example"),
                "title": ex.get("title", "Scheduled Event"),
                "scheduled_time": ex.get("scheduled_time", "2026-01-18T14:00:00Z"),
                "recurrence": ex.get("recurrence"),
                "urgency_window_hours": ex.get("urgency_window_hours", 24),
            },
        },
        indent=2,
    )


def _format_intent_example(examples: list[dict[str, Any]]) -> str:
    """Format an intent node example."""
    if not examples:
        return json.dumps(
            {
                "op": "ADD_NODE",
                "node_type": "intent",
                "data": {
                    "intent_id": "i-example",
                    "title": "Recommended Intent",
                    "description": "Description of the goal or desire",
                    "status": "pending",
                    "priority": "high",
                    "suggested_time": "2026-01-18T10:00:00Z",
                    "expiry": "2026-01-18T14:00:00Z",
                },
            },
            indent=2,
        )

    ex = examples[0]
    # Support both legacy "action_id" and new "intent_id" fields
    intent_id = ex.get("intent_id") or ex.get("action_id", "i-example")
    return json.dumps(
        {
            "op": "ADD_NODE",
            "node_type": "intent",
            "data": {
                "intent_id": intent_id,
                "title": ex.get("title", "Recommended Intent"),
                "description": ex.get("description", "Description of the goal or desire"),
                "status": "pending",
                "priority": ex.get("priority", "medium"),
                "suggested_time": ex.get("suggested_time", "2026-01-18T10:00:00Z"),
                "pattern_source": ex.get("pattern_source"),
            },
        },
        indent=2,
    )


# Backward compatibility alias
_format_action_example = _format_intent_example


def _format_pattern_intent_examples(intent_examples: list[dict[str, Any]]) -> str:
    """Format pattern-based intent examples."""
    if not intent_examples:
        return """When a recurring pattern is detected, create intents based on that pattern:

```json
{
  "op": "ADD_NODE",
  "node_type": "intent",
  "data": {
    "intent_id": "i-pattern-based",
    "title": "Pattern-Based Intent",
    "description": "Intent derived from established pattern",
    "status": "pending",
    "priority": "medium",
    "pattern_source": "c-detected-pattern"
  }
}
```"""

    examples_text = []
    for ex in intent_examples[:2]:
        # Support both legacy "action_id" and new "intent_id" fields
        intent_id = ex.get("intent_id") or ex.get("action_id")
        ex_json = json.dumps(
            {
                "op": "ADD_NODE",
                "node_type": "intent",
                "data": {
                    "intent_id": intent_id,
                    "title": ex.get("title"),
                    "description": ex.get("description"),
                    "status": "pending",
                    "priority": ex.get("priority", "medium"),
                    "pattern_source": ex.get("pattern_source"),
                },
            },
            indent=2,
        )
        examples_text.append(f"```json\n{ex_json}\n```")

    return "\n\n".join(examples_text)


# Backward compatibility alias
_format_pattern_action_examples = _format_pattern_intent_examples


def format_system_prompt_for_domain(
    domain: DomainConfig,
    mode: ReasoningMode | None = None,
    template: str | None = None,
) -> str:
    """Format the system prompt for a specific domain.

    Args:
        domain: Domain configuration with examples and guidelines.
        mode: Optional reasoning mode to include mode-specific guidance.

    Returns:
        Formatted system prompt string.
    """
    # Get node type descriptions with defaults
    node_descs = domain.node_type_descriptions
    event_desc = node_descs.get("event", "Direct representations of incoming events")
    concept_desc = node_descs.get("concept", "Higher-level patterns that emerge from events")
    # Support both "intent" and legacy "action" keys
    intent_desc = node_descs.get(
        "intent",
        node_descs.get("action", "Goals, desires, or intentions that may become concrete actions"),
    )
    time_desc = node_descs.get(
        "time", "Temporal anchors representing deadlines or scheduled events"
    )

    # Format guidelines - support both "intent" and legacy "action" guidelines
    concept_text = "\n".join(f"- {g}" for g in domain.concept_guidelines)
    intent_guidelines = getattr(domain, "intent_guidelines", None) or domain.action_guidelines
    intent_text = "\n".join(f"- {g}" for g in intent_guidelines)

    # Get intent examples - support both "intent_examples" and legacy "action_examples"
    intent_examples = getattr(domain, "intent_examples", None) or domain.action_examples

    format_kwargs = {
        "domain_description": domain.description,
        "event_type_desc": event_desc,
        "concept_type_desc": concept_desc,
        "intent_type_desc": intent_desc,
        "time_type_desc": time_desc,
        "hierarchy_examples": _format_hierarchy_examples(domain.hierarchy_examples),
        "concept_hierarchy_example": _format_concept_example(domain.concept_examples),
        "temporal_pattern_examples": _format_pattern_types(domain.pattern_types),
        "time_node_example": _format_time_example(domain.time_examples),
        "intent_node_example": _format_intent_example(intent_examples),
        "pattern_intent_examples": _format_pattern_intent_examples(intent_examples),
        "concept_guidelines": concept_text
        or "- Look for recurring patterns\n- Identify emerging themes",
        "intent_guidelines": intent_text
        or "- Identify goals and desires\n- Create proactive recommendations",
        "intent_personalization_context": "(No user feedback yet — generate intents based on patterns.)",
    }

    # Section-based composition — always compose all sections
    disabled = getattr(domain, "disabled_sections", frozenset()) or frozenset()
    extras = getattr(domain, "extra_sections", {}) or {}
    extras_pos = getattr(domain, "extra_section_position", "before_rules") or "before_rules"
    opt_ins = getattr(domain, "opt_in_sections", frozenset()) or frozenset()

    if template:
        # YAML profile template overrides core.role only;
        # other sections (edge types, connectivity, validation) still compose.
        sections = resolve_sections(
            disabled_sections=disabled,
            section_overrides={"core.role": template.format(**format_kwargs)},
            extra_sections=extras,
            extra_section_position=extras_pos,
            opt_in_sections=opt_ins,
        )
        parts = []
        for _key, content in sections:
            try:
                parts.append(content.format(**format_kwargs))
            except (KeyError, IndexError):
                parts.append(content)
        prompt = "".join(parts)
    else:
        sections = resolve_sections(
            disabled_sections=disabled,
            extra_sections=extras,
            extra_section_position=extras_pos,
            opt_in_sections=opt_ins,
        )
        parts = [content.format(**format_kwargs) for _key, content in sections]
        prompt = "".join(parts)

    # Add time guidelines if provided
    if domain.time_guidelines:
        time_text = "\n".join(f"- {g}" for g in domain.time_guidelines)
        prompt += f"\n\n## Time Node Guidelines\n\n{time_text}"

    # Add mode-specific guidance if specified
    if mode and mode in MODE_PROMPTS:
        prompt += "\n" + MODE_PROMPTS[mode]

    return prompt


def format_system_prompt(
    concept_guidelines: tuple[str, ...],
    intent_guidelines: tuple[str, ...] | None = None,
    time_guidelines: tuple[str, ...] | None = None,
    mode: ReasoningMode | None = None,
    *,
    action_guidelines: tuple[str, ...] | None = None,  # Legacy alias
) -> str:
    """Format the system prompt with guidelines and optional mode.

    This is the legacy function for backwards compatibility.
    For domain-specific prompts, use format_system_prompt_for_domain() instead.

    Args:
        concept_guidelines: Rules for concept discovery.
        intent_guidelines: Rules for intent discovery (formerly action_guidelines).
        time_guidelines: Optional rules for TIME node creation.
        mode: Optional reasoning mode to include mode-specific guidance.
        action_guidelines: Legacy alias for intent_guidelines.

    Returns:
        Formatted system prompt string.
    """
    # Import here to avoid circular import
    from cognifold.agent.domain import PERSONAL_TIMELINE_DOMAIN, DomainConfig

    # Support legacy action_guidelines parameter
    effective_intent_guidelines = intent_guidelines or action_guidelines or ()

    # Create a modified domain config with the provided guidelines
    base_domain = PERSONAL_TIMELINE_DOMAIN
    domain = DomainConfig(
        name=base_domain.name,
        description=base_domain.description,
        event_description=base_domain.event_description,
        node_type_descriptions=base_domain.node_type_descriptions,
        concept_examples=list(base_domain.concept_examples),
        action_examples=list(base_domain.action_examples),
        time_examples=list(base_domain.time_examples),
        pattern_types=list(base_domain.pattern_types),
        hierarchy_examples=list(base_domain.hierarchy_examples),
        concept_guidelines=concept_guidelines,
        action_guidelines=effective_intent_guidelines,
        time_guidelines=time_guidelines or (),
    )

    return format_system_prompt_for_domain(domain, mode)


def format_user_prompt(
    event_details: str,
    context_window: str,
    mode: ReasoningMode | None = None,
    graph_stats: str | None = None,
) -> str:
    """Format the user prompt with event and context.

    Args:
        event_details: Formatted event information.
        context_window: Formatted context window nodes.
        mode: Optional reasoning mode for specialized prompts.
        graph_stats: Optional graph statistics for consolidation mode.

    Returns:
        Formatted user prompt string.
    """
    if mode == ReasoningMode.ANALYTICAL:
        return ANALYTICAL_USER_PROMPT_TEMPLATE.format(
            event_details=event_details,
            context_window=context_window,
        )
    elif mode == ReasoningMode.CONSOLIDATION:
        return CONSOLIDATION_USER_PROMPT_TEMPLATE.format(
            context_window=context_window,
            graph_stats=graph_stats or "No statistics available",
        )
    else:
        # Quick mode or no mode specified - use standard template
        return USER_PROMPT_TEMPLATE.format(
            event_details=event_details,
            context_window=context_window,
        )


# Concept consolidation helper prompts
SIMILARITY_CHECK_PROMPT = """Analyze these two concepts and determine if they should be merged:

Concept 1: {concept1}
Concept 2: {concept2}

Consider:
- Are they semantically similar or overlapping?
- Would merging them create a clearer, stronger concept?
- Is there enough distinction to keep them separate?

Respond with JSON:
```json
{{
  "should_merge": true/false,
  "reason": "explanation",
  "merged_title": "suggested title if merging",
  "merged_strength": 0.X
}}
```
"""


def format_similarity_check_prompt(concept1: dict[str, Any], concept2: dict[str, Any]) -> str:
    """Format prompt for checking concept similarity.

    Args:
        concept1: First concept data.
        concept2: Second concept data.

    Returns:
        Formatted prompt string.
    """
    import json

    return SIMILARITY_CHECK_PROMPT.format(
        concept1=json.dumps(concept1, indent=2),
        concept2=json.dumps(concept2, indent=2),
    )


# Hierarchical context prompt template (Phase 9.2)
HIERARCHICAL_USER_PROMPT_TEMPLATE = """## New Event

{event_details}

## Immediate Context (high priority - focus here)
These are the most relevant nodes for the current event.

### Nodes
{immediate_nodes}

### Relationships
{immediate_edges}

## Working Context (medium priority - consider these)
Related concepts and patterns.

### Nodes
{working_nodes}

### Relationships
{working_edges}

## Background Context (low priority - reference only)
Broader historical context.

### Nodes
{background_nodes}

### Relationships
{background_edges}

## Task

Analyze this event and determine the appropriate graph updates. Consider:
1. **Immediate context** contains the most relevant nodes - prioritize connections here
2. **Working context** has related patterns - look for reinforcement opportunities
3. **Background context** is for reference - use for long-term pattern recognition

Explore the graph with tools if needed, then provide your update plan as JSON.
"""


def format_hierarchical_context(
    context: Any,  # HierarchicalContext
    node_formatter: Any | None = None,  # Callable[[Node], str]
    edge_formatter: Any | None = None,  # Callable[[Edge], str]
) -> dict[str, str]:
    """Format a hierarchical context for the prompt.

    Args:
        context: HierarchicalContext object.
        node_formatter: Optional function to format a node to string.
        edge_formatter: Optional function to format an edge to string.

    Returns:
        Dictionary with formatted strings for each level's nodes and edges.
    """
    from cognifold.models.node import NodeType

    def default_format_node(node: Any) -> str:
        """Default node formatter."""
        node_type = node.type.value if isinstance(node.type, NodeType) else str(node.type)
        title = node.data.get("title", node.id)
        score = (
            context.immediate.node_scores.get(node.id)
            or context.working.node_scores.get(node.id)
            or context.background.node_scores.get(node.id)
            or 0.0
        )
        return f"- **{title}** ({node_type}, id={node.id}, score={score:.3f})"

    def default_format_edge(edge: Any) -> str:
        """Default edge formatter with type and weight."""
        edge_type = edge.edge_type or "related_to"
        weight = edge.weight
        return f"- {edge.source} --[{edge_type} ({weight:.2f})]--> {edge.target}"

    format_node = node_formatter or default_format_node
    format_edge = edge_formatter or default_format_edge

    def format_level_nodes(level: Any) -> str:
        """Format nodes for a level."""
        if not level.nodes:
            return "No nodes at this priority level."
        return "\n".join(format_node(n) for n in level.nodes)

    def format_level_edges(level: Any) -> str:
        """Format edges for a level."""
        if not level.edges:
            return "No relationships."
        return "\n".join(format_edge(e) for e in level.edges)

    return {
        "immediate_nodes": format_level_nodes(context.immediate),
        "immediate_edges": format_level_edges(context.immediate),
        "working_nodes": format_level_nodes(context.working),
        "working_edges": format_level_edges(context.working),
        "background_nodes": format_level_nodes(context.background),
        "background_edges": format_level_edges(context.background),
    }


def format_hierarchical_user_prompt(
    event_details: str,
    context: Any,  # HierarchicalContext
    node_formatter: Any | None = None,
    edge_formatter: Any | None = None,
) -> str:
    """Format the user prompt with hierarchical context.

    Args:
        event_details: Formatted event information.
        context: HierarchicalContext object.
        node_formatter: Optional function to format a node to string.
        edge_formatter: Optional function to format an edge to string.

    Returns:
        Formatted user prompt string.
    """
    formatted = format_hierarchical_context(context, node_formatter, edge_formatter)
    return HIERARCHICAL_USER_PROMPT_TEMPLATE.format(
        event_details=event_details,
        **formatted,
    )
