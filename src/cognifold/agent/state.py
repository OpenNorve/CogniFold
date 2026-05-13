"""LangGraph state definition for the Cognifold agent."""

# ruff: noqa: UP006, UP035, UP045
# Note: This file intentionally uses Optional[] and List[] instead of X | None and list[]
# for Python 3.9 compatibility with TypedDict. TypedDict evaluates type annotations
# at runtime when the class is defined, and LangGraph's StateGraph uses get_type_hints().
# Using the union operator | causes "unsupported operand type(s)" errors in Python 3.9.

from __future__ import annotations

from typing import Any, List, Optional, TypedDict

from cognifold.agent.config import AgentConfig
from cognifold.agent.context import AgentContext
from cognifold.agent.prompt_profile import PromptProfile
from cognifold.models.plan import UpdatePlan


class _MessageRequired(TypedDict):
    """Required fields for a conversation message."""

    role: str  # "system", "user", "assistant", or "tool"
    content: str
    tool_calls: Optional[List[dict]]
    tool_call_id: Optional[str]


class Message(_MessageRequired, total=False):
    """A message in the conversation history.

    Optional underscore-prefixed fields carry Gemini SDK objects that
    must survive round-trips so thought_signature is preserved (gemini-3).
    """

    _gemini_parts: Optional[List[Any]]  # Raw Gemini parts with thought_signature
    _tool_name: Optional[str]  # Original tool name for function_response


class AgentState(TypedDict):
    """State for the LangGraph agent.

    This state is passed through the graph and updated by each node.
    """

    # Input context
    context: AgentContext
    config: AgentConfig
    prompt_profile: Optional[PromptProfile]
    domain: Optional[str]

    # Conversation history
    messages: List[Message]

    # Tool execution tracking
    exploration_steps: int
    max_exploration_steps: int

    # Output
    update_plan: Optional[UpdatePlan]
    error: Optional[str]

    # Parsing state
    raw_response: Optional[str]
    parse_attempts: int


def create_initial_state(
    context: AgentContext,
    config: Optional[AgentConfig] = None,
    prompt_profile: Optional[PromptProfile] = None,
    domain: Optional[str] = None,
    max_exploration_steps: int = 3,
) -> AgentState:
    """Create the initial agent state.

    Args:
        context: The agent context with event and graph.
        max_exploration_steps: Maximum tool call iterations.

    Returns:
        Initial AgentState ready for processing.
    """
    if config is None:
        config = AgentConfig()

    if config.disable_concepts:
        import dataclasses

        config = dataclasses.replace(
            config,
            concept_guidelines=(
                "DO NOT create any CONCEPT nodes.",
                "Only create EVENT nodes.",
                "If you see a pattern, ignore it.",
            ),
        )

    return AgentState(
        context=context,
        config=config,
        prompt_profile=prompt_profile,
        domain=domain,
        messages=[],
        exploration_steps=0,
        max_exploration_steps=max_exploration_steps,
        update_plan=None,
        error=None,
        raw_response=None,
        parse_attempts=0,
    )
