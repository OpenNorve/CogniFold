"""Cognitive trace collector with ring buffer and plan extraction."""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime

from cognifold.config import TraceConfig
from cognifold.models.plan import UpdatePlan
from cognifold.models.trace import TraceEntry


class TraceCollector:
    """Collects cognitive trace entries from event processing.

    Uses a ring buffer (deque with maxlen) to cap memory usage.
    Thread-safe for concurrent event processing (deque is thread-safe
    for append/pop on CPython due to the GIL).
    """

    def __init__(self, config: TraceConfig) -> None:
        self._config = config
        self._entries: deque[TraceEntry] = deque(maxlen=config.max_entries)

    @property
    def enabled(self) -> bool:
        """Whether trace collection is enabled."""
        return self._config.enabled

    def record(self, entry: TraceEntry) -> None:
        """Record a trace entry if tracing is enabled."""
        if not self._config.enabled:
            return
        self._entries.append(entry)

    def get_entries(self, limit: int = 0, event_id: str | None = None) -> list[TraceEntry]:
        """Get trace entries, optionally filtered by event_id.

        Args:
            limit: Maximum number of entries to return (0 = all).
                   Returns the *most recent* entries when limited.
            event_id: If provided, only return entries for this event.

        Returns:
            List of matching TraceEntry objects.
        """
        entries = list(self._entries)
        if event_id:
            entries = [e for e in entries if e.event_id == event_id]
        if limit > 0:
            entries = entries[-limit:]
        return entries

    def get_entries_since(self, since: datetime) -> list[TraceEntry]:
        """Get entries after a given timestamp.

        Args:
            since: Only return entries with timestamp >= since.

        Returns:
            List of matching TraceEntry objects.
        """
        return [e for e in self._entries if e.timestamp >= since]

    def clear(self) -> None:
        """Remove all collected entries."""
        self._entries.clear()

    @property
    def count(self) -> int:
        """Number of entries currently stored."""
        return len(self._entries)

    def to_jsonl(self) -> str:
        """Serialize all entries as JSONL string.

        Returns:
            Newline-delimited JSON, one entry per line.
        """
        lines: list[str] = []
        for entry in self._entries:
            lines.append(json.dumps(entry.to_dict()))
        return "\n".join(lines)


def trace_from_plan(plan: UpdatePlan, event_id: str) -> TraceEntry:
    """Extract a TraceEntry from an executed UpdatePlan.

    Walks the plan's operations and populates the trace entry fields
    based on operation types.

    Args:
        plan: The executed UpdatePlan.
        event_id: The event that triggered this plan.

    Returns:
        A populated TraceEntry.
    """
    entry = TraceEntry(event_id=event_id, plan_reasoning=plan.reasoning or "")
    for op in plan.operations:
        entry.operation_count += 1
        op_type = op.op.value if hasattr(op.op, "value") else str(op.op)

        if op_type == "ADD_NODE" and op.node_type in ("concept", "intent"):
            node_id = ""
            if op.data:
                node_id = (
                    op.data.get("id", "")
                    or op.data.get("concept_id", "")
                    or op.data.get("intent_id", "")
                )
            entry.activated_concepts.append(node_id)

        elif op_type == "ADD_EDGE":
            entry.new_edges.append((op.source_id or "", op.target_id or "", op.edge_type or ""))

        elif op_type == "REMOVE_NODE":
            entry.removed_nodes.append(op.node_id or "")

        elif op_type == "REMOVE_EDGE":
            entry.removed_edges.append((op.source_id or "", op.target_id or ""))

        elif op_type == "MERGE_NODES":
            merged_id = op.merged_data.get("id", "") if op.merged_data else ""
            entry.merged_nodes.append((op.node_ids or [], merged_id))

    return entry
