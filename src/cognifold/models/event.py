"""Event model for incoming events in the stream."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Legacy event types for personal timelines.

    Note: This enum is kept for backwards compatibility.
    For new domains, use free-form strings for event_type.
    """

    MEAL = "meal"
    WORK = "work"
    STUDY = "study"
    EXERCISE = "exercise"
    SOCIAL = "social"
    REST = "rest"
    TRANSIT = "transit"
    ENTERTAINMENT = "entertainment"
    PLANNING = "planning"  # Calendar reviews, scheduling, reminders
    DEADLINE = "deadline"  # Project due dates, submission deadlines


class Event(BaseModel):
    """An event from the input stream.

    Events represent discrete occurrences in a timeline that get processed
    into the concept graph. The schema is domain-agnostic - the same Event
    model works for personal timelines, computer activity, service logs, etc.

    Example (personal timeline):
        Event(
            event_id="e-001",
            timestamp=datetime.now(),
            source="personal-timeline",
            event_type="meal",
            title="Breakfast at home",
            context={"food": ["oatmeal", "coffee"], "location": "kitchen"}
        )

    Example (computer activity):
        Event(
            event_id="ca-001",
            timestamp=datetime.now(),
            source="computer-activity",
            event_type="browser.page_visit",
            title="Visited GitHub",
            context={"url": "https://github.com", "browser": "Chrome", "tab_count": 5}
        )

    Example (service logs):
        Event(
            event_id="svc-001",
            timestamp=datetime.now(),
            source="service-logs",
            event_type="http.request",
            title="POST /api/users",
            context={"method": "POST", "endpoint": "/api/users", "status": 201, "latency_ms": 45}
        )
    """

    event_id: str = Field(..., description="Unique identifier for the event")
    timestamp: datetime = Field(..., description="When the event occurred")
    source: str = Field(
        default="personal-timeline",
        description="Event source/domain identifier (e.g., 'personal-timeline', 'computer-activity', 'service-logs')",
    )
    event_type: str = Field(
        ...,
        description="Free-form event type string (domain-specific, e.g., 'meal', 'browser.page_visit', 'http.request')",
    )
    title: str = Field(..., description="Short description of the event")
    description: str | None = Field(default=None, description="Detailed description")
    location: str | None = Field(
        default=None, description="Where the event occurred (for physical events)"
    )
    duration_minutes: int | None = Field(default=None, ge=0, description="Duration in minutes")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured domain-specific data (opaque to core system, interpreted by LLM)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional unstructured data",
    )

    model_config = {"frozen": True}
