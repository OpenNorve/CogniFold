"""Structured logging for graph evolution replay.

This module defines the log format and provides a logger class for recording
graph operations during simulation runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class LogEntryType(str, Enum):
    """Types of log entries for replay."""

    # Simulation lifecycle
    RUN_START = "run_start"
    RUN_END = "run_end"

    # Event processing
    EVENT_START = "event_start"
    EVENT_END = "event_end"

    # Graph operations
    OPERATION = "operation"

    # Context window
    CONTEXT_WINDOW = "context_window"

    # Scoring snapshot
    SCORES = "scores"

    # Intent/Action flow (Phase 8)
    INTENT_SELECTED = "intent_selected"
    ACTION_GENERATED = "action_generated"
    ACTION_EXECUTED = "action_executed"
    ACTION_RESULT_EVENT = "action_result_event"


@dataclass
class LogEntry:
    """A single log entry for replay.

    Attributes:
        timestamp: When the entry was recorded.
        entry_type: Type of log entry.
        step: Event step number (1-indexed).
        data: Entry-specific data.
    """

    timestamp: str
    entry_type: LogEntryType
    step: int
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "entry_type": self.entry_type.value,
            "step": self.step,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LogEntry:
        """Create from dictionary."""
        return cls(
            timestamp=d["timestamp"],
            entry_type=LogEntryType(d["entry_type"]),
            step=d["step"],
            data=d["data"],
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> LogEntry:
        """Create from JSON string."""
        return cls.from_dict(json.loads(s))


@dataclass
class GraphLogger:
    """Logger for recording graph evolution during simulation.

    The logger writes JSONL (JSON Lines) format, with one JSON object per line.
    This format is efficient for appending and streaming.

    Example output format:
        {"timestamp": "2026-01-18T12:00:00", "entry_type": "run_start", "step": 0, "data": {...}}
        {"timestamp": "2026-01-18T12:00:01", "entry_type": "event_start", "step": 1, "data": {...}}
        {"timestamp": "2026-01-18T12:00:01", "entry_type": "operation", "step": 1, "data": {...}}
        ...

    Attributes:
        log_path: Path to the log file.
        entries: In-memory list of log entries (for testing/inspection).
    """

    log_path: Path | None = None
    entries: list[LogEntry] = field(default_factory=list)
    _file_handle: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Open the log file if path is provided."""
        if self.log_path:
            self.log_path = Path(self.log_path)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(self.log_path, "w")  # noqa: SIM115

    def _now(self) -> str:
        """Get current timestamp."""
        return datetime.now().isoformat()

    def _write(self, entry: LogEntry) -> None:
        """Write entry to file and memory."""
        self.entries.append(entry)
        if self._file_handle:
            self._file_handle.write(entry.to_json() + "\n")
            self._file_handle.flush()

    def log_run_start(
        self,
        timeline_path: str,
        total_events: int,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Log the start of a simulation run.

        Args:
            timeline_path: Path to the timeline file.
            total_events: Total number of events to process.
            config: Optional configuration snapshot.
        """
        entry = LogEntry(
            timestamp=self._now(),
            entry_type=LogEntryType.RUN_START,
            step=0,
            data={
                "timeline_path": timeline_path,
                "total_events": total_events,
                "config": config or {},
            },
        )
        self._write(entry)

    def log_run_end(
        self,
        total_steps: int,
        node_count: int,
        edge_count: int,
    ) -> None:
        """Log the end of a simulation run.

        Args:
            total_steps: Number of events processed.
            node_count: Final node count.
            edge_count: Final edge count.
        """
        entry = LogEntry(
            timestamp=self._now(),
            entry_type=LogEntryType.RUN_END,
            step=total_steps,
            data={
                "total_steps": total_steps,
                "node_count": node_count,
                "edge_count": edge_count,
            },
        )
        self._write(entry)

    def log_event_start(
        self,
        step: int,
        event_id: str,
        event_type: str,
        title: str,
        timestamp: str,
        event_data: dict[str, Any] | None = None,
    ) -> None:
        """Log the start of event processing.

        Args:
            step: Current step number.
            event_id: Event ID.
            event_type: Type of event.
            title: Event title.
            timestamp: Event timestamp.
            event_data: Full event data.
        """
        entry = LogEntry(
            timestamp=self._now(),
            entry_type=LogEntryType.EVENT_START,
            step=step,
            data={
                "event_id": event_id,
                "event_type": event_type,
                "title": title,
                "event_timestamp": timestamp,
                "event_data": event_data or {},
            },
        )
        self._write(entry)

    def log_event_end(
        self,
        step: int,
        event_id: str,
        operations_count: int,
        reasoning: str | None = None,
    ) -> None:
        """Log the end of event processing.

        Args:
            step: Current step number.
            event_id: Event ID.
            operations_count: Number of operations applied.
            reasoning: Agent's reasoning (if available).
        """
        entry = LogEntry(
            timestamp=self._now(),
            entry_type=LogEntryType.EVENT_END,
            step=step,
            data={
                "event_id": event_id,
                "operations_count": operations_count,
                "reasoning": reasoning,
            },
        )
        self._write(entry)

    def log_operation(
        self,
        step: int,
        op_type: str,
        op_data: dict[str, Any],
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Log a graph operation.

        Args:
            step: Current step number.
            op_type: Operation type (ADD_NODE, ADD_EDGE, etc.).
            op_data: Operation data.
            success: Whether the operation succeeded.
            error: Error message if failed.
        """
        entry = LogEntry(
            timestamp=self._now(),
            entry_type=LogEntryType.OPERATION,
            step=step,
            data={
                "op_type": op_type,
                "op_data": op_data,
                "success": success,
                "error": error,
            },
        )
        self._write(entry)

    def log_context_window(
        self,
        step: int,
        context_node_ids: list[str],
    ) -> None:
        """Log the context window state.

        Args:
            step: Current step number.
            context_node_ids: Node IDs in the context window.
        """
        entry = LogEntry(
            timestamp=self._now(),
            entry_type=LogEntryType.CONTEXT_WINDOW,
            step=step,
            data={
                "context_node_ids": context_node_ids,
            },
        )
        self._write(entry)

    def log_scores(
        self,
        step: int,
        scores: dict[str, float],
    ) -> None:
        """Log node scores snapshot.

        Args:
            step: Current step number.
            scores: Mapping of node ID to score.
        """
        entry = LogEntry(
            timestamp=self._now(),
            entry_type=LogEntryType.SCORES,
            step=step,
            data={
                "scores": scores,
            },
        )
        self._write(entry)

    def log_intent_selected(
        self,
        step: int,
        intent_id: str,
        intent_title: str,
        urgency_score: float,
        status: str,
    ) -> None:
        """Log when an intent is selected for action generation.

        Args:
            step: Current step number.
            intent_id: ID of the selected intent.
            intent_title: Title of the intent.
            urgency_score: Urgency score that triggered selection.
            status: Intent status before selection.
        """
        entry = LogEntry(
            timestamp=self._now(),
            entry_type=LogEntryType.INTENT_SELECTED,
            step=step,
            data={
                "intent_id": intent_id,
                "intent_title": intent_title,
                "urgency_score": urgency_score,
                "status": status,
            },
        )
        self._write(entry)

    def log_action_generated(
        self,
        step: int,
        action_id: str,
        intent_id: str,
        action_title: str,
        scheduled_time: str,
        urgency: str,
    ) -> None:
        """Log when an action is generated from an intent.

        Args:
            step: Current step number.
            action_id: ID of the generated action.
            intent_id: ID of the source intent.
            action_title: Title of the action.
            scheduled_time: When the action is scheduled.
            urgency: Action urgency level.
        """
        entry = LogEntry(
            timestamp=self._now(),
            entry_type=LogEntryType.ACTION_GENERATED,
            step=step,
            data={
                "action_id": action_id,
                "intent_id": intent_id,
                "action_title": action_title,
                "scheduled_time": scheduled_time,
                "urgency": urgency,
            },
        )
        self._write(entry)

    def log_action_executed(
        self,
        step: int,
        action_id: str,
        intent_id: str,
        action_title: str,
        execution_time: str,
        result_event_id: str,
    ) -> None:
        """Log when an action is executed.

        Args:
            step: Current step number.
            action_id: ID of the executed action.
            intent_id: ID of the source intent.
            action_title: Title of the action.
            execution_time: When the action was executed.
            result_event_id: ID of the result event generated.
        """
        entry = LogEntry(
            timestamp=self._now(),
            entry_type=LogEntryType.ACTION_EXECUTED,
            step=step,
            data={
                "action_id": action_id,
                "intent_id": intent_id,
                "action_title": action_title,
                "execution_time": execution_time,
                "result_event_id": result_event_id,
            },
        )
        self._write(entry)

    def log_action_result_event(
        self,
        step: int,
        result_event_id: str,
        action_id: str,
        intent_id: str,
        outcome: str,
        intent_resolved: bool,
    ) -> None:
        """Log when an action result event is processed.

        Args:
            step: Current step number.
            result_event_id: ID of the result event.
            action_id: ID of the completed action.
            intent_id: ID of the related intent.
            outcome: Outcome of the action (success/failure).
            intent_resolved: Whether the intent was resolved.
        """
        entry = LogEntry(
            timestamp=self._now(),
            entry_type=LogEntryType.ACTION_RESULT_EVENT,
            step=step,
            data={
                "result_event_id": result_event_id,
                "action_id": action_id,
                "intent_id": intent_id,
                "outcome": outcome,
                "intent_resolved": intent_resolved,
            },
        )
        self._write(entry)

    def close(self) -> None:
        """Close the log file."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def __enter__(self) -> GraphLogger:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()


def load_log(path: str | Path) -> list[LogEntry]:
    """Load log entries from a JSONL file.

    Args:
        path: Path to the log file.

    Returns:
        List of log entries.
    """
    path = Path(path)
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(LogEntry.from_json(line))
    return entries
