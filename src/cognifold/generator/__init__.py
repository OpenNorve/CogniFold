"""Event stream generation module."""

from cognifold.generator.base import BaseEventGenerator
from cognifold.generator.claude_code import (
    ClaudeCodeGenerator,
    SessionProfile,
    get_session_profile,
)
from cognifold.generator.computer_activity import (
    ComputerActivityGenerator,
    WorkProfile,
    get_work_profile,
)
from cognifold.generator.event_generator import EventGenerator, PersonalTimelineGenerator
from cognifold.generator.persona import Persona
from cognifold.generator.service_logs import (
    ServiceLogsGenerator,
    ServiceTopology,
    get_service_topology,
)

__all__ = [
    "BaseEventGenerator",
    "ClaudeCodeGenerator",
    "ComputerActivityGenerator",
    "EventGenerator",
    "Persona",
    "PersonalTimelineGenerator",
    "ServiceLogsGenerator",
    "ServiceTopology",
    "SessionProfile",
    "WorkProfile",
    "get_service_topology",
    "get_session_profile",
    "get_work_profile",
]
