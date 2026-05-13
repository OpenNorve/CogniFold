"""Cognitive trace model for recording event processing outcomes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TraceEntry:
    """A single cognitive trace entry recording what happened during event processing."""

    event_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    activated_concepts: list[str] = field(default_factory=list)  # concept/intent node IDs
    new_edges: list[tuple[str, str, str]] = field(default_factory=list)  # (src, tgt, edge_type)
    updated_weights: list[tuple[str, str, float]] = field(
        default_factory=list
    )  # (src, tgt, weight)
    removed_nodes: list[str] = field(default_factory=list)
    removed_edges: list[tuple[str, str]] = field(default_factory=list)
    merged_nodes: list[tuple[list[str], str]] = field(default_factory=list)  # (orig_ids, merged_id)
    plan_reasoning: str = ""
    operation_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON persistence."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "activated_concepts": self.activated_concepts,
            "new_edges": [list(e) for e in self.new_edges],
            "updated_weights": [list(w) for w in self.updated_weights],
            "removed_nodes": self.removed_nodes,
            "removed_edges": [list(e) for e in self.removed_edges],
            "merged_nodes": [[list(ids), mid] for ids, mid in self.merged_nodes],
            "plan_reasoning": self.plan_reasoning,
            "operation_count": self.operation_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceEntry:
        """Deserialize from dict."""
        ts_raw = data.get("timestamp")
        if isinstance(ts_raw, str):
            ts = datetime.fromisoformat(ts_raw)
        elif isinstance(ts_raw, datetime):
            ts = ts_raw
        else:
            ts = datetime.now(timezone.utc)

        return cls(
            event_id=data["event_id"],
            timestamp=ts,
            activated_concepts=data.get("activated_concepts", []),
            new_edges=[tuple(e) for e in data.get("new_edges", [])],  # type: ignore[misc]
            updated_weights=[tuple(w) for w in data.get("updated_weights", [])],  # type: ignore[misc]
            removed_nodes=data.get("removed_nodes", []),
            removed_edges=[tuple(e) for e in data.get("removed_edges", [])],  # type: ignore[misc]
            merged_nodes=[(list(ids), mid) for ids, mid in data.get("merged_nodes", [])],
            plan_reasoning=data.get("plan_reasoning", ""),
            operation_count=data.get("operation_count", 0),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
