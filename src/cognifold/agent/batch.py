"""Batch agent processor for Layer 2 of the fast pipeline.

Sends N events in a single LLM prompt and parses a JSON array of
UpdatePlans from the response. Falls back to per-event processing
on parse failure.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cognifold.agent.config import AgentConfig
    from cognifold.agent.prompt_profile import PromptProfile
    from cognifold.graph.store import ConceptGraph
    from cognifold.models.event import Event
    from cognifold.models.plan import UpdatePlan

logger = logging.getLogger(__name__)

BATCH_SYSTEM_PROMPT = """\
You are Cognifold, a cognitive memory system that maintains an evolving knowledge graph.

## Node Types
- **event**: Direct representation of input events
- **concept**: Higher-level patterns that emerge from events (e.g., "morning routine", "exercise habit")
- **intent**: Goals or desires that emerge from patterns (e.g., "improve sleep", "eat healthier")
- **time**: Temporal anchors (deadlines, scheduled times)

## Edge Types
- **grounds** (0.9): event → concept/intent — event provides evidence
- **causes** (0.9): event → event — causal relationship
- **triggers** (0.8): concept/event → intent — triggers a goal
- **reinforces** (0.7): event → concept — strengthens a pattern
- **part_of** (0.7): concept → concept — sub-concept
- **derived_from** (0.6): concept → concept — derived relationship
- **related_to** (0.5): any → any — generic relationship

## Your Task
You are given a BATCH of events to process together. For each event:
1. The event node already exists in the graph (added in Layer 1) — do NOT add event nodes
2. Create concepts and intents that emerge from the events
3. Create edges linking events to concepts/intents using proper edge types
4. Look for cross-event patterns within this batch

## Output Format
Return a JSON array of UpdatePlan objects, one per event that needs graph updates.
Events that don't warrant new concepts/edges can be skipped.

```json
[
  {
    "plan_id": "batch-<event_id>",
    "trigger_event_id": "<event_id>",
    "reasoning": "Why these updates are needed",
    "operations": [
      {"op": "ADD_NODE", "node_type": "concept", "node_id": "c-<descriptive-id>",
       "data": {"title": "...", "description": "...", "strength": 0.5},
       "reasoning": "Why this concept exists",
       "grounded_in": ["<event_id>"]},
      {"op": "ADD_EDGE", "source_id": "<event_id>", "target_id": "c-<id>",
       "edge_type": "grounds", "weight": 0.9}
    ]
  }
]
```

IMPORTANT:
- Do NOT include ADD_NODE operations for events — they already exist
- Every concept/intent MUST have at least one edge connecting it
- Use descriptive node IDs (e.g., "c-morning-routine", not "c-001")
- Return an empty array [] if no updates are needed
"""


def _build_system_prompt(domain_name: str | None) -> str:
    """Build a domain-aware system prompt for the batch processor.

    Appends domain-specific sections (description, node types, concept
    guidelines, intent guidelines, intent examples, pattern types) to the
    base ``BATCH_SYSTEM_PROMPT``.  Falls back to the base prompt when the
    domain is unknown or *None*.
    """
    if not domain_name:
        return BATCH_SYSTEM_PROMPT

    from cognifold.agent.domain import DOMAIN_REGISTRY

    domain = DOMAIN_REGISTRY.get(domain_name)
    if domain is None:
        return BATCH_SYSTEM_PROMPT

    sections: list[str] = [BATCH_SYSTEM_PROMPT]

    # Domain description
    sections.append(f"\n## Domain: {domain.name}\n{domain.description}")

    # Node type descriptions
    if domain.node_type_descriptions:
        lines = ["## Domain Node Types"]
        for ntype, desc in domain.node_type_descriptions.items():
            lines.append(f"- **{ntype}**: {desc}")
        sections.append("\n".join(lines))

    # Concept guidelines
    if domain.concept_guidelines:
        lines = ["## Concept Guidelines"]
        for g in domain.concept_guidelines:
            lines.append(f"- {g}")
        sections.append("\n".join(lines))

    # Intent / action guidelines
    if domain.action_guidelines:
        lines = ["## Intent Guidelines"]
        for g in domain.action_guidelines:
            lines.append(f"- {g}")
        sections.append("\n".join(lines))

    # Intent examples
    if domain.action_examples:
        import json as _json

        lines = ["## Intent Examples"]
        for ex in domain.action_examples:
            lines.append(f"```json\n{_json.dumps(ex, indent=2)}\n```")
        sections.append("\n".join(lines))

    # Pattern types
    if domain.pattern_types:
        lines = ["## Pattern Types to Detect"]
        for p in domain.pattern_types:
            lines.append(f"- {p}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _format_events_for_batch(
    events: list[Event],
    context_summary: str,
) -> str:
    """Format a batch of events into a single prompt."""
    parts = [f"## Context Window\n\n{context_summary}\n"]
    parts.append(f"## Event Batch ({len(events)} events)\n")

    for i, event in enumerate(events, 1):
        lines = [
            f"### Event {i}",
            f"- ID: {event.event_id}",
            f"- Timestamp: {event.timestamp.isoformat()}",
            f"- Type: {event.event_type}",
            f"- Title: {event.title}",
        ]
        if event.description:
            lines.append(f"- Description: {event.description}")
        if event.location:
            lines.append(f"- Location: {event.location}")
        parts.append("\n".join(lines))

    parts.append("\nAnalyze these events and return your update plans as a JSON array.")
    return "\n\n".join(parts)


def _format_context_summary(
    graph: ConceptGraph,
    context_ids: list[str],
    node_scores: dict[str, float],
) -> str:
    """Format a brief context summary for the batch prompt."""
    if not context_ids:
        return "(Empty graph — these are the first events)"

    lines = []
    for nid in context_ids[:30]:  # limit context size
        node = graph.get_node_or_none(nid)
        if node is None:
            continue
        score = node_scores.get(nid, 0.0)
        title = node.data.get("title", nid)
        lines.append(f"- [{node.type.value.upper()}] {nid}: {title} [score={score:.3f}]")

    return "\n".join(lines) if lines else "(No context nodes)"


def _parse_plans_response(raw: str, events: list[Event]) -> list[UpdatePlan]:
    """Parse the LLM response into a list of UpdatePlan objects."""
    from cognifold.models.plan import Operation, OperationType, UpdatePlan

    # Extract JSON from response (handle markdown code blocks)
    text = raw.strip()
    if text.startswith("```"):
        # Remove code fence
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try parsing as JSON array
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON array in the text
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                logger.warning("Failed to parse batch response as JSON")
                logger.warning("Raw response (first 500 chars): %s", text[:500])
                return []
        else:
            logger.warning("Failed to parse batch response as JSON")
            logger.warning("Raw response (first 500 chars): %s", text[:500])
            return []

    if not isinstance(data, list):
        # OpenAI json_object mode wraps arrays in an object like {"update_plans": [...]}
        # Extract the first list value from the object
        if isinstance(data, dict):
            for val in data.values():
                if isinstance(val, list):
                    data = val
                    break
            else:
                data = [data]
        else:
            data = [data]

    plans: list[UpdatePlan] = []

    for plan_data in data:
        if not isinstance(plan_data, dict):
            continue

        ops: list[Operation] = []
        for op_data in plan_data.get("operations", []):
            try:
                op_type = OperationType(op_data.get("op", ""))
            except ValueError:
                continue

            ops.append(
                Operation(
                    op=op_type,
                    node_type=op_data.get("node_type"),
                    node_id=op_data.get("node_id"),
                    data=op_data.get("data"),
                    source_id=op_data.get("source_id"),
                    target_id=op_data.get("target_id"),
                    edge_type=op_data.get("edge_type"),
                    weight=op_data.get("weight"),
                    node_ids=op_data.get("node_ids"),
                    merged_data=op_data.get("merged_data"),
                    reasoning=op_data.get("reasoning"),
                    grounded_in=op_data.get("grounded_in"),
                )
            )

        if ops:
            trigger = plan_data.get("trigger_event_id", "")
            plans.append(
                UpdatePlan(
                    plan_id=plan_data.get("plan_id", f"batch-{trigger}"),
                    trigger_event_id=trigger,
                    reasoning=plan_data.get("reasoning", "Batch enrichment"),
                    operations=ops,
                )
            )

    return plans


def _op_resolves_to_existing(op: Any, existing_ids: set[str]) -> bool:
    """Check if an ADD_NODE operation would create a node that already exists."""
    # Check explicit node_id
    if op.node_id and op.node_id in existing_ids:
        return True
    # Check well-known data keys (mirrors PlanExecutor._resolve_add_node_id)
    data = op.data or {}
    for key in ("event_id", "id", "concept_id", "action_id", "intent_id", "time_id"):
        val = data.get(key)
        if val and isinstance(val, str) and val in existing_ids:
            return True
    return False


class BatchAgentProcessor:
    """Processes batches of events through a single LLM call.

    Falls back to per-event processing via CognifoldAgent on failure.
    """

    def __init__(
        self,
        agent_config: AgentConfig | None = None,
        prompt_profile: PromptProfile | None = None,
    ) -> None:
        from cognifold.agent.config import AgentConfig

        self._config = agent_config or AgentConfig()
        self._prompt_profile = prompt_profile

    def process_event_batch(
        self,
        events: list[Event],
        graph: ConceptGraph,
        context_node_ids: list[str],
        node_scores: dict[str, float],
    ) -> list[UpdatePlan]:
        """Process a batch of events in a single LLM call.

        Args:
            events: Batch of events to process.
            graph: The concept graph.
            context_node_ids: Context window node IDs.
            node_scores: Node scores dict.

        Returns:
            List of UpdatePlans (one per event that needs updates).
        """
        context_summary = _format_context_summary(graph, context_node_ids, node_scores)
        user_prompt = _format_events_for_batch(events, context_summary)
        system_prompt = _build_system_prompt(self._config.domain)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                raw_response = self._call_llm(system_prompt, user_prompt)
                plans = _parse_plans_response(raw_response, events)

                if plans:
                    logger.info("Batch of %d events produced %d plans", len(events), len(plans))
                    return plans

                if attempt < max_retries - 1:
                    logger.warning(
                        "Batch produced no plans, retrying (%d/%d)", attempt + 1, max_retries
                    )
                    continue
                logger.warning(
                    "Batch produced no plans after %d attempts, falling back to per-event",
                    max_retries,
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "Batch LLM call failed (%s), retrying (%d/%d)", e, attempt + 1, max_retries
                    )
                    continue
                logger.warning(
                    "Batch LLM call failed after %d attempts (%s), falling back to per-event",
                    max_retries,
                    e,
                )

        # Fallback: process individually only after all retries exhausted
        return self._fallback_per_event(events, graph, context_node_ids, node_scores)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM with system + user prompt and return raw text."""
        model_name = self._config.model_name

        if model_name.startswith("openai:"):
            return self._call_openai(system_prompt, user_prompt, model_name)
        else:
            return self._call_gemini(system_prompt, user_prompt, model_name)

    def _call_gemini(self, system_prompt: str, user_prompt: str, model_name: str) -> str:
        from google import genai
        from google.genai import types

        from cognifold.service.llm_keys import get_api_key

        api_key = get_api_key("GOOGLE_API_KEY") or get_api_key("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)

        gen_config = types.GenerateContentConfig(
            temperature=self._config.temperature,
            max_output_tokens=self._config.max_tokens,
            system_instruction=system_prompt,
            response_mime_type="application/json",
        )

        response: Any = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=gen_config,
        )

        if (
            getattr(response, "candidates", None)
            and response.candidates[0]
            and getattr(response.candidates[0], "content", None)
            and getattr(response.candidates[0].content, "parts", None)
        ):
            return "".join(
                p.text
                for p in response.candidates[0].content.parts
                if hasattr(p, "text") and p.text
            )

        raise RuntimeError("No response from Gemini")

    def _call_openai(self, system_prompt: str, user_prompt: str, model_name: str) -> str:
        from openai import OpenAI

        from cognifold.service.llm_keys import get_api_key

        client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))
        actual_model = model_name.replace("openai:", "")

        response = client.chat.completions.create(
            model=actual_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if content:
            return content

        raise RuntimeError("No response from OpenAI")

    def _fallback_per_event(
        self,
        events: list[Event],
        graph: ConceptGraph,
        context_node_ids: list[str],
        node_scores: dict[str, float],
    ) -> list[UpdatePlan]:
        """Fall back to processing events one at a time via CognifoldAgent."""
        from cognifold.agent.agent import CognifoldAgent
        from cognifold.models.plan import OperationType

        agent = CognifoldAgent(config=self._config, prompt_profile=self._prompt_profile)
        plans: list[UpdatePlan] = []

        # Collect event IDs already in the graph (from Layer 1)
        existing_ids = {e.event_id for e in events if graph.has_node(e.event_id)}

        for event in events:
            try:
                plan = agent.process_event(
                    event=event,
                    graph=graph,
                    context_node_ids=context_node_ids,
                    node_scores=node_scores,
                )
                # Strip ADD_NODE ops for events that already exist (Layer 1 added them).
                # The node_id may be on op.node_id or inside op.data["event_id"].
                if existing_ids:
                    filtered_ops = [
                        op
                        for op in plan.operations
                        if not (
                            op.op == OperationType.ADD_NODE
                            and _op_resolves_to_existing(op, existing_ids)
                        )
                    ]
                    if len(filtered_ops) != len(plan.operations):
                        plan = plan.model_copy(update={"operations": filtered_ops})
                plans.append(plan)
            except Exception as e:
                logger.warning("Per-event fallback failed for %s: %s", event.event_id, e)

        return plans
