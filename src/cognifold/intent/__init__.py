"""Intent execution system for Cognifold.

This module handles the conversion of intents (goals/desires) into
concrete, schedulable actions and manages their execution lifecycle.

Key concepts:
- Intent: A goal or desire stored in the graph (node type: "intent")
- Action: A concrete, schedulable step with execution time (not in graph)
- ActionQueue: Manages scheduled actions waiting for execution
- IntentToActionAgent: Converts intents to actions using LLM
- IntentSelector: Selects which intents should have actions generated

Phase 8.1: Intent-to-Action Agent
Phase 8.2: Pipeline Integration & Action Queue
Phase 8.3: Simulation with Action Execution
Phase 14.1: Intent Personalization (feedback, calibration)
"""

from cognifold.intent.agent import IntentToActionAgent
from cognifold.intent.calibrator import IntentCalibrator
from cognifold.intent.executor import ActionExecutor, SimulatedActionExecutor
from cognifold.intent.feedback_store import FeedbackStore
from cognifold.intent.models import (
    Action,
    ActionMetadata,
    ActionStatus,
)
from cognifold.intent.personalization import (
    CalibrationProfile,
    FeedbackStats,
    FeedbackType,
    IntentFeedback,
)
from cognifold.intent.queue import ActionQueue
from cognifold.intent.selector import IntentScore, IntentSelector

__all__ = [
    "Action",
    "ActionExecutor",
    "ActionMetadata",
    "ActionQueue",
    "ActionStatus",
    "CalibrationProfile",
    "FeedbackStats",
    "FeedbackStore",
    "FeedbackType",
    "IntentCalibrator",
    "IntentFeedback",
    "IntentScore",
    "IntentSelector",
    "IntentToActionAgent",
    "SimulatedActionExecutor",
]
