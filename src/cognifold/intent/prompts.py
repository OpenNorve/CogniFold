"""Prompts for the Intent-to-Action agent.

This module provides the prompts used by the IntentToActionAgent
to convert high-level intents into concrete, schedulable actions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

# System prompt for the Intent-to-Action agent
INTENT_TO_ACTION_SYSTEM_PROMPT = """You are an action planner for Cognifold. Your role is to convert
high-level intents (goals/desires) into concrete, executable actions.

## Your Task

Given an intent and its context, generate 1-3 specific, actionable steps that would
help achieve the intent. Each action should have:

1. **What**: A clear, specific description of what to do
2. **When**: A scheduled execution time based on urgency and context
3. **How long**: Estimated duration in minutes

## Input Format

You will receive:
- The intent with its title, description, and reasoning
- Related context from the concept graph (events, concepts)
- The current time

## Output Format

Return valid JSON with this structure:

```json
{
  "actions": [
    {
      "title": "Short action title (max 50 chars)",
      "description": "Detailed description of what to do",
      "scheduled_time": "ISO 8601 datetime when to execute",
      "estimated_duration_minutes": 30,
      "urgency": "low|medium|high|urgent"
    }
  ],
  "reasoning": "Brief explanation of why these actions were chosen"
}
```

## Guidelines

**Timing Decisions:**
- Urgent intents: Schedule within hours (today)
- High priority: Schedule within 24 hours (next day morning)
- Medium priority: Schedule within 48 hours
- Low priority: Schedule within a week

**Action Granularity:**
- Each action should be completable in one sitting (15-120 minutes)
- Break down large tasks into multiple actions if needed
- Actions should be independent (can be executed alone)

**Action Types:**
- Research/Information gathering
- Preparation/Planning
- Communication/Outreach
- Implementation/Execution
- Review/Verification

## Important Rules

1. Generate 1-3 actions maximum per intent
2. Each action must have a specific scheduled_time (not vague)
3. Actions should be concrete and verifiable (you know when it's done)
4. Consider dependencies but keep actions as independent as possible
5. Return ONLY valid JSON, no additional text
"""


def format_intent_to_action_user_prompt(
    intent_title: str,
    intent_description: str,
    intent_reasoning: str,
    context_nodes: list[dict[str, Any]],
    current_time: datetime,
    intent_priority: str = "medium",
) -> str:
    """Format the user prompt for intent-to-action conversion.

    Args:
        intent_title: Title of the intent.
        intent_description: Description of the intent.
        intent_reasoning: Why this intent was created.
        context_nodes: Related nodes from the graph.
        current_time: Current timestamp for scheduling.
        intent_priority: Priority level (low, medium, high, urgent).

    Returns:
        Formatted user prompt string.
    """
    # Format context nodes
    context_text = ""
    if context_nodes:
        context_items = []
        for node in context_nodes[:10]:  # Limit to 10 nodes
            node_type = node.get("type", "unknown")
            title = node.get("title", node.get("id", "Unknown"))
            desc = node.get("description", "")[:100]
            context_items.append(f"- [{node_type}] {title}: {desc}")
        context_text = "\n".join(context_items)
    else:
        context_text = "No additional context available."

    return f"""## Intent to Convert

**Title:** {intent_title}
**Description:** {intent_description}
**Reasoning:** {intent_reasoning}
**Priority:** {intent_priority}

## Related Context

{context_text}

## Current Time

{current_time.isoformat()}

## Task

Generate 1-3 concrete, schedulable actions for this intent.
Consider the priority level and context when deciding timing.

Return your response as JSON.
"""


def format_action_result_event_prompt(
    action_title: str,
    action_description: str,
    intent_id: str,
    execution_time: datetime,
) -> str:
    """Format a prompt describing an action result event.

    This is used when processing action_result events back through
    the event processing pipeline.

    Args:
        action_title: Title of the completed action.
        action_description: Description of the action.
        intent_id: ID of the originating intent.
        execution_time: When the action was executed.

    Returns:
        Description for the result event.
    """
    return (
        f"Completed action: {action_title}. "
        f"{action_description} "
        f"This action was executed to fulfill intent {intent_id}. "
        f"Executed at {execution_time.isoformat()}."
    )
