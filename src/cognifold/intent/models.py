"""Models for the intent execution system.

This module defines the data structures for actions, which are
concrete, schedulable steps derived from intents (goals/desires).

Actions differ from intents:
- Intents are stored in the graph (node type: "intent")
- Actions are stored in the ActionQueue (not in the graph)
- Intents represent goals; actions represent concrete steps to achieve them
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ActionStatus(str, Enum):
    """Status of an action in its lifecycle.

    Lifecycle:
    - QUEUED: Action is scheduled and waiting for execution
    - EXECUTING: Action is currently being executed
    - COMPLETED: Action has been executed successfully
    - FAILED: Action execution failed
    - CANCELLED: Action was cancelled before execution
    """

    QUEUED = "queued"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ActionMetadata:
    """Metadata associated with an action.

    Provides additional context about the action's urgency,
    duration, and other execution-relevant information.
    """

    urgency: str = "medium"  # low, medium, high, urgent
    estimated_duration_minutes: int = 15
    tags: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "urgency": self.urgency,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "tags": self.tags,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionMetadata:
        """Create from dictionary."""
        return cls(
            urgency=data.get("urgency", "medium"),
            estimated_duration_minutes=data.get("estimated_duration_minutes", 15),
            tags=data.get("tags", []),
            context=data.get("context", {}),
        )


@dataclass
class Action:
    """A concrete, schedulable action derived from an intent.

    Actions represent executable steps with specific timing. They are
    stored in the ActionQueue, not in the concept graph.

    Attributes:
        action_id: Unique identifier for this action.
        intent_id: ID of the intent this action was derived from.
        title: Short description of the action.
        description: Detailed description of what to do.
        scheduled_time: When to execute this action.
        created_at: When this action was created.
        status: Current status in the lifecycle.
        metadata: Additional execution metadata.
        result: Result of execution (populated after completion).

    Example:
        >>> action = Action(
        ...     action_id="act-001",
        ...     intent_id="i-001",
        ...     title="Search sleep improvement techniques",
        ...     description="Research sleep techniques and prepare summary",
        ...     scheduled_time=datetime(2026, 1, 22, 9, 0),
        ...     metadata=ActionMetadata(urgency="medium"),
        ... )
    """

    action_id: str
    intent_id: str
    title: str
    description: str
    scheduled_time: datetime
    created_at: datetime = field(default_factory=datetime.now)
    status: ActionStatus = ActionStatus.QUEUED
    metadata: ActionMetadata = field(default_factory=ActionMetadata)
    result: dict[str, Any] | None = None

    def is_due(self, current_time: datetime) -> bool:
        """Check if this action is due for execution.

        Args:
            current_time: The current time to check against.

        Returns:
            True if the action should be executed now.
        """
        return self.status == ActionStatus.QUEUED and current_time >= self.scheduled_time

    def mark_executing(self) -> Action:
        """Create a new Action with EXECUTING status."""
        return Action(
            action_id=self.action_id,
            intent_id=self.intent_id,
            title=self.title,
            description=self.description,
            scheduled_time=self.scheduled_time,
            created_at=self.created_at,
            status=ActionStatus.EXECUTING,
            metadata=self.metadata,
            result=self.result,
        )

    def mark_completed(self, result: dict[str, Any] | None = None) -> Action:
        """Create a new Action with COMPLETED status.

        Args:
            result: Optional result data from execution.

        Returns:
            New Action with COMPLETED status.
        """
        return Action(
            action_id=self.action_id,
            intent_id=self.intent_id,
            title=self.title,
            description=self.description,
            scheduled_time=self.scheduled_time,
            created_at=self.created_at,
            status=ActionStatus.COMPLETED,
            metadata=self.metadata,
            result=result or {"status": "success"},
        )

    def mark_failed(self, error: str) -> Action:
        """Create a new Action with FAILED status.

        Args:
            error: Error message describing the failure.

        Returns:
            New Action with FAILED status.
        """
        return Action(
            action_id=self.action_id,
            intent_id=self.intent_id,
            title=self.title,
            description=self.description,
            scheduled_time=self.scheduled_time,
            created_at=self.created_at,
            status=ActionStatus.FAILED,
            metadata=self.metadata,
            result={"status": "failed", "error": error},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the action.
        """
        return {
            "action_id": self.action_id,
            "intent_id": self.intent_id,
            "title": self.title,
            "description": self.description,
            "scheduled_time": self.scheduled_time.isoformat(),
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "metadata": self.metadata.to_dict(),
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Action:
        """Create an Action from a dictionary.

        Args:
            data: Dictionary with action data.

        Returns:
            Action instance.
        """
        return cls(
            action_id=data["action_id"],
            intent_id=data["intent_id"],
            title=data["title"],
            description=data["description"],
            scheduled_time=datetime.fromisoformat(data["scheduled_time"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            status=ActionStatus(data.get("status", "queued")),
            metadata=ActionMetadata.from_dict(data.get("metadata", {})),
            result=data.get("result"),
        )

    def __str__(self) -> str:
        """String representation."""
        return (
            f"Action({self.action_id}: {self.title} "
            f"[{self.status.value}] @ {self.scheduled_time.isoformat()})"
        )

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"Action(action_id={self.action_id!r}, intent_id={self.intent_id!r}, "
            f"title={self.title!r}, status={self.status.value!r})"
        )
