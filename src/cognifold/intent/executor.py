"""Action executor for simulated action execution.

This module provides the ActionExecutor class which simulates action
execution by generating result events that feed back into the event
processing pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from cognifold.intent.models import Action
from cognifold.models.event import Event

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ActionExecutor:
    """Simulates action execution by generating result events.

    In simulation mode, we assume all actions succeed and generate
    corresponding result events that are processed by the event pipeline.

    In a real implementation, this could integrate with external systems
    to actually execute actions.

    Example:
        >>> from cognifold.intent import ActionExecutor, Action
        >>> executor = ActionExecutor()
        >>> action = Action(...)
        >>> result_event = executor.execute(action, datetime.now())
        >>> print(result_event.title)
        "Completed: Research sleep techniques"
    """

    def __init__(
        self,
        success_rate: float = 1.0,
        include_details: bool = True,
    ) -> None:
        """Initialize the executor.

        Args:
            success_rate: Probability of action success (0.0-1.0).
                         Use 1.0 for deterministic simulation.
            include_details: Include detailed action info in result events.
        """
        self.success_rate = success_rate
        self.include_details = include_details

    def execute(
        self,
        action: Action,
        execution_time: datetime | None = None,
    ) -> tuple[Event, Action]:
        """Execute an action and return the result event.

        For simulation, this generates a synthetic event representing
        the action being completed.

        Args:
            action: The action to execute.
            execution_time: When the action is being executed.
                           Defaults to action.scheduled_time.

        Returns:
            Tuple of (result_event, updated_action).
        """
        import random

        execution_time = execution_time or action.scheduled_time

        # Determine if action succeeds (for probabilistic simulation)
        success = random.random() < self.success_rate

        if success:
            result_event = self._create_success_event(action, execution_time)
            updated_action = action.mark_completed(
                {
                    "status": "success",
                    "execution_time": execution_time.isoformat(),
                }
            )
        else:
            result_event = self._create_failure_event(action, execution_time)
            updated_action = action.mark_failed("Simulated failure")

        logger.info(f"Executed action {action.action_id}: {'success' if success else 'failure'}")

        return result_event, updated_action

    def _create_success_event(
        self,
        action: Action,
        execution_time: datetime,
    ) -> Event:
        """Create a success result event for an action.

        Args:
            action: The completed action.
            execution_time: When the action was executed.

        Returns:
            Event representing the successful action.
        """
        event_id = f"e-{action.action_id}-result"

        description = f"Successfully completed: {action.description}"
        if self.include_details:
            description += (
                f"\n\nAction Details:"
                f"\n- Action ID: {action.action_id}"
                f"\n- Intent ID: {action.intent_id}"
                f"\n- Duration: {action.metadata.estimated_duration_minutes} minutes"
            )

        metadata: dict[str, Any] = {
            "action_id": action.action_id,
            "intent_id": action.intent_id,
            "outcome": "success",
            "execution_duration_minutes": action.metadata.estimated_duration_minutes,
        }

        return Event(
            event_id=event_id,
            timestamp=execution_time,
            event_type="action_result",
            title=f"Completed: {action.title}",
            description=description,
            metadata=metadata,
        )

    def _create_failure_event(
        self,
        action: Action,
        execution_time: datetime,
    ) -> Event:
        """Create a failure result event for an action.

        Args:
            action: The failed action.
            execution_time: When the action was attempted.

        Returns:
            Event representing the failed action.
        """
        event_id = f"e-{action.action_id}-result"

        description = f"Failed to complete: {action.description}"
        if self.include_details:
            description += (
                f"\n\nAction Details:"
                f"\n- Action ID: {action.action_id}"
                f"\n- Intent ID: {action.intent_id}"
            )

        metadata: dict[str, Any] = {
            "action_id": action.action_id,
            "intent_id": action.intent_id,
            "outcome": "failure",
            "error": "Simulated failure",
        }

        return Event(
            event_id=event_id,
            timestamp=execution_time,
            event_type="action_result",
            title=f"Failed: {action.title}",
            description=description,
            metadata=metadata,
        )

    def execute_batch(
        self,
        actions: list[Action],
        execution_times: list[datetime] | None = None,
    ) -> list[tuple[Event, Action]]:
        """Execute multiple actions in order.

        Args:
            actions: List of actions to execute.
            execution_times: Optional list of execution times.
                           Defaults to each action's scheduled_time.

        Returns:
            List of (result_event, updated_action) tuples.
        """
        if execution_times is None:
            execution_times = [a.scheduled_time for a in actions]

        if len(execution_times) != len(actions):
            raise ValueError("execution_times must match actions length")

        results = []
        for action, exec_time in zip(actions, execution_times):
            result = self.execute(action, exec_time)
            results.append(result)

        return results

    def preview_execution(
        self,
        action: Action,
        execution_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Preview what an action execution would produce.

        Does not actually execute the action or generate events.

        Args:
            action: The action to preview.
            execution_time: Planned execution time.

        Returns:
            Dictionary with preview information.
        """
        execution_time = execution_time or action.scheduled_time

        return {
            "action_id": action.action_id,
            "intent_id": action.intent_id,
            "title": action.title,
            "execution_time": execution_time.isoformat(),
            "result_event_id": f"e-{action.action_id}-result",
            "result_event_type": "action_result",
            "estimated_duration_minutes": action.metadata.estimated_duration_minutes,
        }


class SimulatedActionExecutor(ActionExecutor):
    """Action executor specifically for simulation mode.

    Extends ActionExecutor with simulation-specific features like
    time compression and batch processing.
    """

    def __init__(
        self,
        time_compression: float = 1.0,
        include_details: bool = True,
    ) -> None:
        """Initialize the simulated executor.

        Args:
            time_compression: Time compression factor for simulation.
                             1.0 = real-time, 10.0 = 10x faster.
            include_details: Include detailed info in result events.
        """
        super().__init__(success_rate=1.0, include_details=include_details)
        self.time_compression = time_compression

    def get_execution_duration(self, action: Action) -> float:
        """Get the simulated execution duration in seconds.

        Args:
            action: The action to get duration for.

        Returns:
            Duration in seconds (compressed if time_compression > 1).
        """
        base_minutes = action.metadata.estimated_duration_minutes
        base_seconds = base_minutes * 60
        return base_seconds / self.time_compression

    def execute_with_callback(
        self,
        action: Action,
        execution_time: datetime,
        on_complete: Callable[..., Any] | None = None,
    ) -> tuple[Event, Action]:
        """Execute an action with optional completion callback.

        Args:
            action: The action to execute.
            execution_time: Execution time.
            on_complete: Optional callback function.

        Returns:
            Tuple of (result_event, updated_action).
        """
        result = self.execute(action, execution_time)

        if on_complete:
            try:
                on_complete(result)
            except Exception as e:
                logger.warning(f"Callback failed: {e}")

        return result
