"""Intent-to-Action Agent for Cognifold.

This module provides the agent that converts high-level intents (goals/desires)
into concrete, schedulable actions with specific execution times.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from cognifold.intent.models import Action, ActionMetadata
from cognifold.intent.prompts import (
    INTENT_TO_ACTION_SYSTEM_PROMPT,
    format_intent_to_action_user_prompt,
)

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph
    from cognifold.models.node import Node

logger = logging.getLogger(__name__)


class IntentToActionAgent:
    """Agent that converts intents into concrete, schedulable actions.

    The IntentToActionAgent takes an intent node and context from the graph,
    then uses an LLM to generate specific, actionable steps with scheduled
    execution times.

    Example:
        >>> from cognifold.intent import IntentToActionAgent
        >>> agent = IntentToActionAgent(llm_provider="gemini")
        >>> actions = agent.generate_actions(
        ...     intent=intent_node,
        ...     context=context_nodes,
        ...     current_time=datetime.now(),
        ... )
        >>> for action in actions:
        ...     print(f"{action.title} @ {action.scheduled_time}")

    Attributes:
        llm_provider: The LLM provider to use ("gemini" or "mock").
        model_name: The model name for the LLM.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        llm_provider: str = "gemini",
        model_name: str = "gemini-3-flash-preview",
        temperature: float = 0.5,
    ) -> None:
        """Initialize the agent.

        Args:
            llm_provider: LLM provider ("gemini" or "mock" for testing).
            model_name: Model name to use.
            temperature: Sampling temperature (0.0-1.0).
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.temperature = temperature

        if llm_provider == "gemini":
            self._init_gemini()
        else:
            self._model = None

    def _init_gemini(self) -> None:
        """Initialize the Gemini model."""
        try:
            import google.generativeai as genai  # type: ignore[import-untyped]

            self._model = genai.GenerativeModel(  # type: ignore[reportPrivateImportUsage]
                model_name=self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "response_mime_type": "application/json",
                },
            )
        except ImportError:
            logger.warning("Gemini not available, using mock mode")
            self._model = None
            self.llm_provider = "mock"
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini: {e}")
            self._model = None
            self.llm_provider = "mock"

    def generate_actions(
        self,
        intent: Node,
        context: list[Node],
        current_time: datetime,
        max_actions: int = 3,
    ) -> list[Action]:
        """Generate concrete actions from an intent.

        Args:
            intent: The intent node to convert.
            context: Related context nodes from the graph.
            current_time: Current time for scheduling.
            max_actions: Maximum number of actions to generate.

        Returns:
            List of Action objects with scheduled times.
        """
        # Extract intent data
        intent_data = intent.data
        intent_title = intent_data.get("title", intent.id)
        intent_description = intent_data.get("description", "")
        intent_reasoning = intent.reasoning or intent_data.get("reasoning", "")
        intent_priority = intent_data.get("priority", "medium")

        # Format context for the prompt
        context_nodes = [
            {
                "type": node.type.value,
                "id": node.id,
                "title": node.data.get("title", node.id),
                "description": node.data.get("description", ""),
            }
            for node in context[:10]  # Limit context
        ]

        # Generate actions using LLM or mock
        if self.llm_provider == "mock" or self._model is None:
            return self._generate_mock_actions(
                intent_id=intent.id,
                intent_title=intent_title,
                intent_priority=intent_priority,
                current_time=current_time,
                max_actions=max_actions,
            )

        return self._generate_llm_actions(
            intent_id=intent.id,
            intent_title=intent_title,
            intent_description=intent_description,
            intent_reasoning=intent_reasoning,
            intent_priority=intent_priority,
            context_nodes=context_nodes,
            current_time=current_time,
            max_actions=max_actions,
        )

    def _generate_llm_actions(
        self,
        intent_id: str,
        intent_title: str,
        intent_description: str,
        intent_reasoning: str,
        intent_priority: str,
        context_nodes: list[dict[str, Any]],
        current_time: datetime,
        max_actions: int,
    ) -> list[Action]:
        """Generate actions using the LLM.

        Args:
            intent_id: ID of the intent.
            intent_title: Title of the intent.
            intent_description: Description of the intent.
            intent_reasoning: Reasoning for the intent.
            intent_priority: Priority level.
            context_nodes: Related context nodes.
            current_time: Current time.
            max_actions: Maximum actions to generate.

        Returns:
            List of Action objects.
        """
        # Format the user prompt
        user_prompt = format_intent_to_action_user_prompt(
            intent_title=intent_title,
            intent_description=intent_description,
            intent_reasoning=intent_reasoning,
            context_nodes=context_nodes,
            current_time=current_time,
            intent_priority=intent_priority,
        )

        try:
            # Call the LLM
            if self._model is None:
                raise RuntimeError("LLM model not initialized")
            chat = self._model.start_chat(history=[])
            response = chat.send_message(
                [  # type: ignore[arg-type]
                    {"role": "user", "parts": [INTENT_TO_ACTION_SYSTEM_PROMPT]},
                    {
                        "role": "model",
                        "parts": ["I understand. I will convert intents into concrete actions."],
                    },
                    {"role": "user", "parts": [user_prompt]},
                ]
            )

            # Parse the response
            response_text = response.text.strip()
            result = json.loads(response_text)

            # Convert to Action objects
            actions = []
            for i, action_data in enumerate(result.get("actions", [])[:max_actions]):
                action_id = f"act-{intent_id}-{i:02d}-{uuid.uuid4().hex[:8]}"

                # Parse scheduled time
                scheduled_str = action_data.get("scheduled_time", "")
                try:
                    scheduled_time = datetime.fromisoformat(scheduled_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    # Default to priority-based scheduling
                    scheduled_time = self._default_schedule_time(current_time, intent_priority, i)

                action = Action(
                    action_id=action_id,
                    intent_id=intent_id,
                    title=action_data.get("title", f"Action for {intent_title}")[:50],
                    description=action_data.get("description", ""),
                    scheduled_time=scheduled_time,
                    created_at=current_time,
                    metadata=ActionMetadata(
                        urgency=action_data.get("urgency", intent_priority),
                        estimated_duration_minutes=action_data.get(
                            "estimated_duration_minutes", 30
                        ),
                    ),
                )
                actions.append(action)

            logger.info(f"Generated {len(actions)} actions for intent {intent_id}")
            return actions

        except Exception as e:
            logger.error(f"Failed to generate actions via LLM: {e}")
            # Fallback to mock
            return self._generate_mock_actions(
                intent_id=intent_id,
                intent_title=intent_title,
                intent_priority=intent_priority,
                current_time=current_time,
                max_actions=max_actions,
            )

    def _generate_mock_actions(
        self,
        intent_id: str,
        intent_title: str,
        intent_priority: str,
        current_time: datetime,
        max_actions: int,
    ) -> list[Action]:
        """Generate mock actions for testing.

        Args:
            intent_id: ID of the intent.
            intent_title: Title of the intent.
            intent_priority: Priority level.
            current_time: Current time.
            max_actions: Maximum actions.

        Returns:
            List of mock Action objects.
        """
        actions = []

        # Generate 1-2 mock actions
        num_actions = min(2, max_actions)

        for i in range(num_actions):
            action_id = f"act-{intent_id}-{i:02d}-{uuid.uuid4().hex[:8]}"
            scheduled_time = self._default_schedule_time(current_time, intent_priority, i)

            if i == 0:
                title = f"Research: {intent_title}"[:50]
                desc = f"Research and gather information about {intent_title}"
            else:
                title = f"Execute: {intent_title}"[:50]
                desc = f"Take action on {intent_title} based on research"

            action = Action(
                action_id=action_id,
                intent_id=intent_id,
                title=title,
                description=desc,
                scheduled_time=scheduled_time,
                created_at=current_time,
                metadata=ActionMetadata(
                    urgency=intent_priority,
                    estimated_duration_minutes=30 if i == 0 else 45,
                ),
            )
            actions.append(action)

        return actions

    def _default_schedule_time(
        self,
        current_time: datetime,
        priority: str,
        action_index: int,
    ) -> datetime:
        """Calculate default schedule time based on priority.

        Args:
            current_time: Current time.
            priority: Priority level.
            action_index: Index of the action (for spacing).

        Returns:
            Scheduled datetime.
        """
        # Base delay by priority
        if priority == "urgent":
            base_delay = timedelta(hours=1)
        elif priority == "high":
            base_delay = timedelta(hours=12)
        elif priority == "medium":
            base_delay = timedelta(hours=24)
        else:  # low
            base_delay = timedelta(hours=48)

        # Add spacing between actions
        action_spacing = timedelta(hours=2 * action_index)

        return current_time + base_delay + action_spacing

    def generate_actions_for_intents(
        self,
        intents: list[Node],
        graph: ConceptGraph,
        current_time: datetime,
        max_actions_per_intent: int = 3,
    ) -> dict[str, list[Action]]:
        """Generate actions for multiple intents.

        Args:
            intents: List of intent nodes to process.
            graph: The concept graph for context.
            current_time: Current time.
            max_actions_per_intent: Max actions per intent.

        Returns:
            Dictionary mapping intent IDs to lists of actions.
        """
        result: dict[str, list[Action]] = {}

        for intent in intents:
            # Get context for this intent (neighbors in the graph)
            context_ids = graph.get_neighbors(intent.id) + graph.get_predecessors(intent.id)
            context = [graph.get_node(nid) for nid in context_ids if graph.has_node(nid)]

            # Generate actions
            actions = self.generate_actions(
                intent=intent,
                context=context,
                current_time=current_time,
                max_actions=max_actions_per_intent,
            )

            if actions:
                result[intent.id] = actions

        return result
