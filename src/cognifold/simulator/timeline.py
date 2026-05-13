"""Mock timeline loader for simulation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cognifold.models.event import Event


@dataclass
class Timeline:
    """A timeline of events for simulation.

    Attributes:
        timeline_id: Unique identifier for the timeline.
        description: Human-readable description.
        events: List of events in chronological order.
    """

    timeline_id: str
    description: str
    events: list[Event]

    def __len__(self) -> int:
        """Return the number of events in the timeline."""
        return len(self.events)

    def __iter__(self):
        """Iterate over events in the timeline."""
        return iter(self.events)

    def __getitem__(self, index: int) -> Event:
        """Get event by index."""
        return self.events[index]


def load_timeline(path: str | Path) -> Timeline:
    """Load a timeline from a JSON file.

    Args:
        path: Path to the timeline JSON file.

    Returns:
        A Timeline object with parsed events.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file contains invalid data.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Timeline file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return _parse_timeline(data)


def _parse_timeline(data: dict[str, Any]) -> Timeline:
    """Parse timeline data from a dictionary."""
    if "events" not in data:
        raise ValueError("Timeline data missing 'events' key")

    events: list[Event] = []
    for event_data in data["events"]:
        event = Event(
            event_id=event_data["event_id"],
            timestamp=event_data["timestamp"],
            event_type=event_data["event_type"],
            title=event_data["title"],
            description=event_data.get("description"),
            location=event_data.get("location"),
            duration_minutes=event_data.get("duration_minutes"),
            metadata=event_data.get("metadata", {}),
        )
        events.append(event)

    # Sort events by timestamp
    events.sort(key=lambda e: e.timestamp)

    return Timeline(
        timeline_id=data.get("timeline_id", "unknown"),
        description=data.get("description", ""),
        events=events,
    )
