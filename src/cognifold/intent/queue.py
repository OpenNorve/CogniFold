"""Action queue for managing scheduled actions.

This module provides the ActionQueue class which manages scheduled
actions waiting for execution. Actions are sorted by scheduled_time
for efficient processing.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from cognifold.intent.models import Action, ActionStatus

logger = logging.getLogger(__name__)


@dataclass
class ActionQueue:
    """Manages scheduled actions waiting for execution.

    The ActionQueue maintains a sorted list of actions by scheduled_time.
    It provides methods for enqueueing, dequeuing, and querying actions.

    Actions are stored separately from the concept graph as they are
    transient/operational rather than semantic knowledge.

    Attributes:
        actions: List of scheduled actions.

    Example:
        >>> from datetime import datetime, timedelta
        >>> queue = ActionQueue()
        >>> action = Action(
        ...     action_id="act-001",
        ...     intent_id="i-001",
        ...     title="Research sleep techniques",
        ...     description="...",
        ...     scheduled_time=datetime.now() + timedelta(hours=2),
        ... )
        >>> queue.enqueue(action)
        >>> due_actions = queue.get_actions_before(datetime.now() + timedelta(hours=3))
    """

    actions: list[Action] = field(default_factory=list)

    def enqueue(self, action: Action) -> None:
        """Add an action and maintain sorted order by scheduled_time.

        Args:
            action: Action to add to the queue.
        """
        # Check for duplicate action_id
        if any(a.action_id == action.action_id for a in self.actions):
            logger.warning(f"Action {action.action_id} already exists, updating")
            self.actions = [a for a in self.actions if a.action_id != action.action_id]

        self.actions.append(action)
        # Sort by scheduled_time
        self.actions.sort(key=lambda a: a.scheduled_time)
        logger.debug(f"Enqueued action {action.action_id} @ {action.scheduled_time}")

    def enqueue_many(self, actions: list[Action]) -> None:
        """Add multiple actions at once.

        Args:
            actions: List of actions to add.
        """
        for action in actions:
            self.enqueue(action)

    def dequeue(self) -> Action | None:
        """Remove and return the first action in the queue.

        Returns:
            The first action, or None if queue is empty.
        """
        if not self.actions:
            return None
        return self.actions.pop(0)

    def peek(self) -> Action | None:
        """Return the first action without removing it.

        Returns:
            The first action, or None if queue is empty.
        """
        return self.actions[0] if self.actions else None

    def get_actions_before(self, time: datetime) -> list[Action]:
        """Get all actions scheduled before the given time.

        Args:
            time: Cutoff time.

        Returns:
            List of actions scheduled before the time (still in queue).
        """
        return [
            a for a in self.actions if a.scheduled_time <= time and a.status == ActionStatus.QUEUED
        ]

    def get_actions_between(
        self,
        start: datetime,
        end: datetime,
    ) -> list[Action]:
        """Get actions scheduled between two times.

        Args:
            start: Start of time window (inclusive).
            end: End of time window (exclusive).

        Returns:
            List of actions in the time window.
        """
        return [
            a
            for a in self.actions
            if start <= a.scheduled_time < end and a.status == ActionStatus.QUEUED
        ]

    def get_actions_for_intent(self, intent_id: str) -> list[Action]:
        """Get all actions associated with an intent.

        Args:
            intent_id: ID of the intent.

        Returns:
            List of actions for the intent.
        """
        return [a for a in self.actions if a.intent_id == intent_id]

    def get_action(self, action_id: str) -> Action | None:
        """Get a specific action by ID.

        Args:
            action_id: ID of the action.

        Returns:
            The action, or None if not found.
        """
        for action in self.actions:
            if action.action_id == action_id:
                return action
        return None

    def update_action(self, action: Action) -> bool:
        """Update an existing action.

        Args:
            action: Updated action (matched by action_id).

        Returns:
            True if action was found and updated.
        """
        for i, existing in enumerate(self.actions):
            if existing.action_id == action.action_id:
                self.actions[i] = action
                # Re-sort in case scheduled_time changed
                self.actions.sort(key=lambda a: a.scheduled_time)
                return True
        return False

    def mark_executing(self, action_id: str) -> bool:
        """Mark an action as currently executing.

        Args:
            action_id: ID of the action to mark.

        Returns:
            True if action was found and updated.
        """
        action = self.get_action(action_id)
        if action:
            return self.update_action(action.mark_executing())
        return False

    def mark_completed(
        self,
        action_id: str,
        result: dict[str, Any] | None = None,
    ) -> bool:
        """Mark an action as completed.

        Args:
            action_id: ID of the action to mark.
            result: Optional result data.

        Returns:
            True if action was found and updated.
        """
        action = self.get_action(action_id)
        if action:
            return self.update_action(action.mark_completed(result))
        return False

    def remove_action(self, action_id: str) -> bool:
        """Remove an action from the queue.

        Args:
            action_id: ID of the action to remove.

        Returns:
            True if action was found and removed.
        """
        for i, action in enumerate(self.actions):
            if action.action_id == action_id:
                self.actions.pop(i)
                return True
        return False

    def remove_completed(self) -> int:
        """Remove all completed or cancelled actions.

        Returns:
            Number of actions removed.
        """
        before_count = len(self.actions)
        self.actions = [
            a
            for a in self.actions
            if a.status not in (ActionStatus.COMPLETED, ActionStatus.CANCELLED)
        ]
        removed = before_count - len(self.actions)
        if removed > 0:
            logger.debug(f"Removed {removed} completed/cancelled actions")
        return removed

    def clear(self) -> None:
        """Remove all actions from the queue."""
        self.actions.clear()

    @property
    def size(self) -> int:
        """Number of actions in the queue."""
        return len(self.actions)

    @property
    def queued_count(self) -> int:
        """Number of actions with QUEUED status."""
        return sum(1 for a in self.actions if a.status == ActionStatus.QUEUED)

    @property
    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return len(self.actions) == 0

    def __len__(self) -> int:
        """Return the number of actions."""
        return len(self.actions)

    def __iter__(self) -> Iterator[Action]:
        """Iterate over actions in scheduled order."""
        return iter(self.actions)

    def to_dict(self) -> dict[str, Any]:
        """Serialize queue for persistence.

        Returns:
            Dictionary representation of the queue.
        """
        return {
            "version": "1.0",
            "actions": [a.to_dict() for a in self.actions],
            "serialized_at": datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionQueue:
        """Deserialize queue from dictionary.

        Args:
            data: Dictionary from to_dict().

        Returns:
            ActionQueue instance.
        """
        queue = cls()
        for action_data in data.get("actions", []):
            try:
                action = Action.from_dict(action_data)
                queue.actions.append(action)
            except Exception as e:
                logger.warning(f"Failed to deserialize action: {e}")
        # Ensure sorted
        queue.actions.sort(key=lambda a: a.scheduled_time)
        return queue

    def save(self, path: str | Path) -> None:
        """Save the queue to a JSON file.

        Args:
            path: Path to the output file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved action queue to {path} ({len(self.actions)} actions)")

    @classmethod
    def load(cls, path: str | Path) -> ActionQueue:
        """Load a queue from a JSON file.

        Args:
            path: Path to the input file.

        Returns:
            ActionQueue instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        queue = cls.from_dict(data)
        logger.info(f"Loaded action queue from {path} ({len(queue.actions)} actions)")
        return queue

    def summary(self) -> str:
        """Generate a summary of the queue.

        Returns:
            Human-readable summary string.
        """
        status_counts = {}
        for action in self.actions:
            status = action.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        lines = [f"ActionQueue: {len(self.actions)} total actions"]
        for status, count in sorted(status_counts.items()):
            lines.append(f"  - {status}: {count}")

        if self.actions:
            next_action = self.peek()
            if next_action:
                lines.append(f"  Next: {next_action.title} @ {next_action.scheduled_time}")

        return "\n".join(lines)
