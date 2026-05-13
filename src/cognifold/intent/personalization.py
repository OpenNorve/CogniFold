"""Intent personalization models for user feedback and calibration.

Phase 14.1: Provides data models for the intent feedback loop —
users can accept, reject, defer, or modify intents, and the system
learns from this feedback to calibrate future intent generation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class FeedbackType(str, Enum):
    """Types of user feedback on an intent."""

    ACCEPT = "accept"  # User agrees with the intent
    REJECT = "reject"  # Intent is irrelevant or unwanted
    DEFER = "defer"  # Timing is wrong, maybe later
    MODIFY = "modify"  # Direction is right, needs adjustment


class IntentFeedback(BaseModel):
    """A single piece of user feedback on an intent node."""

    feedback_id: str = Field(..., description="Unique feedback identifier")
    intent_id: str = Field(..., description="ID of the intent being evaluated")
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    user_comment: str | None = Field(default=None, description="Optional user comment")
    modified_priority: str | None = Field(
        default=None, description="New priority if feedback_type is MODIFY"
    )
    modified_description: str | None = Field(
        default=None, description="Adjusted description if feedback_type is MODIFY"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="When feedback was given")
    category_tags: list[str] = Field(
        default_factory=list, description="Auto-extracted category tags from the intent"
    )


class CalibrationProfile(BaseModel):
    """User intent preference profile built from aggregated feedback."""

    category_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Per-category preference weight (EMA). >1 = preferred, <1 = disliked",
    )
    priority_bias: float = Field(
        default=0.0,
        description="System over/under-estimation of priority (-1 to 1)",
    )
    acceptance_rate: float = Field(
        default=0.5,
        description="Overall acceptance rate (0.0-1.0)",
    )
    rejection_patterns: list[str] = Field(
        default_factory=list,
        description="Keywords frequently present in rejected intents",
    )
    preferred_patterns: list[str] = Field(
        default_factory=list,
        description="Keywords frequently present in accepted intents",
    )
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="When this profile was last recomputed",
    )


class FeedbackStats(BaseModel):
    """Aggregate statistics for intent feedback."""

    total_count: int = 0
    accept_count: int = 0
    reject_count: int = 0
    defer_count: int = 0
    modify_count: int = 0
    acceptance_rate: float = 0.0
    by_category: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Feedback counts grouped by category tag",
    )
