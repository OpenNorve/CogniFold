"""Async task tracking for the service layer."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TaskRecord:
    """An async task record."""

    task_id: str
    session_id: str
    status: str = "pending"
    result: Any = None
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None


class TaskTracker:
    """In-memory tracker for async tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}

    def create_task(self, session_id: str) -> TaskRecord:
        """Create a new pending task."""
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        record = TaskRecord(task_id=task_id, session_id=session_id)
        self._tasks[task_id] = record
        return record

    def get_task(self, task_id: str) -> TaskRecord | None:
        """Look up a task by ID."""
        return self._tasks.get(task_id)

    def complete_task(self, task_id: str, result: Any) -> None:
        """Mark a task as completed."""
        record = self._tasks.get(task_id)
        if record is not None:
            record.status = "completed"
            record.result = result
            record.completed_at = datetime.now()

    def fail_task(self, task_id: str, error: str) -> None:
        """Mark a task as failed."""
        record = self._tasks.get(task_id)
        if record is not None:
            record.status = "failed"
            record.error = error
            record.completed_at = datetime.now()

    def set_running(self, task_id: str) -> None:
        """Mark a task as running."""
        record = self._tasks.get(task_id)
        if record is not None:
            record.status = "running"

    def cleanup_old_tasks(self, max_age_hours: float = 24.0) -> int:
        """Remove completed/failed tasks older than max_age_hours."""
        now = datetime.now()
        to_remove: list[str] = []
        for tid, record in self._tasks.items():
            if record.status in ("completed", "failed"):
                age = (now - record.created_at).total_seconds() / 3600.0
                if age > max_age_hours:
                    to_remove.append(tid)
        for tid in to_remove:
            del self._tasks[tid]
        return len(to_remove)
