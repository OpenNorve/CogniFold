"""Claude Code session event stream generator."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from cognifold.generator.base import BaseEventGenerator


@dataclass
class SessionProfile:
    """Defines a Claude Code session profile for event generation.

    Attributes:
        name: Profile name (e.g., "feature_development", "bug_fix").
        description: Description of the session scenario.
        project_type: Type of project being worked on.
        project_description: Brief description of the project.
        task_description: What the developer is trying to accomplish.
        common_tools: Tools frequently used in this session type.
        common_files: File patterns commonly touched.
        error_frequency: How often errors occur ("low", "medium", "high").
        interaction_style: Style of human-AI interaction.
        session_phases: Ordered phases the session typically goes through.
    """

    name: str
    description: str
    project_type: str = "python-package"
    project_description: str = "A Python application"
    task_description: str = "General development work"
    common_tools: list[str] = field(default_factory=list)
    common_files: list[str] = field(default_factory=list)
    error_frequency: str = "medium"
    interaction_style: str = "collaborative"
    session_phases: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Convert profile to prompt format."""
        return f"""## Session Profile: {self.name}
Description: {self.description}
Project Type: {self.project_type}
Project: {self.project_description}
Task: {self.task_description}

Common Tools: {", ".join(self.common_tools)}
Common Files: {", ".join(self.common_files)}
Error Frequency: {self.error_frequency}
Interaction Style: {self.interaction_style}

Session Phases:
{chr(10).join(f"- {p}" for p in self.session_phases)}"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "project_type": self.project_type,
            "project_description": self.project_description,
            "task_description": self.task_description,
            "common_tools": self.common_tools,
            "common_files": self.common_files,
            "error_frequency": self.error_frequency,
            "interaction_style": self.interaction_style,
            "session_phases": self.session_phases,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionProfile:
        """Create from dictionary."""
        return cls(**data)


# Sample session profiles
SAMPLE_SESSION_PROFILES: dict[str, SessionProfile] = {
    "feature_development": SessionProfile(
        name="feature_development",
        description="Implementing a new feature end-to-end in a Python project",
        project_type="python-package",
        project_description="A web API with FastAPI, SQLAlchemy, and pytest",
        task_description="Add a new REST endpoint with validation, business logic, and tests",
        common_tools=["Read", "Edit", "Write", "Bash", "Grep", "Glob"],
        common_files=[
            "src/**/*.py",
            "tests/**/*.py",
            "pyproject.toml",
            "README.md",
        ],
        error_frequency="medium",
        interaction_style="collaborative",
        session_phases=[
            "exploration — read existing code, understand patterns",
            "design — discuss approach, plan implementation",
            "implementation — write new code, create files",
            "testing — write tests, run pytest, fix failures",
            "commit — stage changes, commit with descriptive message",
        ],
    ),
    "bug_fix": SessionProfile(
        name="bug_fix",
        description="Diagnosing and fixing a bug reported in production",
        project_type="python-package",
        project_description="A data processing pipeline with complex state management",
        task_description="Investigate and fix a race condition causing data corruption",
        common_tools=["Read", "Grep", "Bash", "Edit", "Glob"],
        common_files=[
            "src/**/*.py",
            "tests/**/*.py",
            "logs/*.log",
        ],
        error_frequency="high",
        interaction_style="investigative",
        session_phases=[
            "reproduce — understand the bug, find reproduction steps",
            "investigate — search codebase, read logs, trace execution",
            "hypothesize — form theory about root cause",
            "fix — apply targeted code changes",
            "verify — run tests, confirm fix, check for regressions",
        ],
    ),
    "refactoring": SessionProfile(
        name="refactoring",
        description="Restructuring existing code for better maintainability",
        project_type="python-package",
        project_description="A monolithic module being split into clean sub-packages",
        task_description="Extract shared utilities and reduce code duplication across modules",
        common_tools=["Read", "Edit", "Grep", "Glob", "Bash"],
        common_files=[
            "src/**/*.py",
            "tests/**/*.py",
        ],
        error_frequency="low",
        interaction_style="methodical",
        session_phases=[
            "analysis — read code, identify duplication and coupling",
            "plan — discuss refactoring strategy with developer",
            "incremental changes — small, focused edits with tests between each",
            "test verification — run full test suite after each change",
            "cleanup — remove dead code, update imports, final commit",
        ],
    ),
}

# Alias for consistency with other modules
SAMPLE_PROFILES = SAMPLE_SESSION_PROFILES


def get_session_profile(name: str) -> SessionProfile:
    """Get a session profile by name.

    Args:
        name: Profile name.

    Returns:
        The session profile.

    Raises:
        KeyError: If profile not found.
    """
    if name not in SAMPLE_SESSION_PROFILES:
        available = ", ".join(SAMPLE_SESSION_PROFILES.keys())
        raise KeyError(f"Unknown session profile: {name}. Available: {available}")
    return SAMPLE_SESSION_PROFILES[name]


class ClaudeCodeGenerator(BaseEventGenerator):
    """Generates Claude Code session events using an LLM.

    Creates realistic sequences of coding session events including tool
    invocations, git operations, conversations, and error recovery.

    Example:
        >>> from cognifold.generator.claude_code import (
        ...     ClaudeCodeGenerator,
        ...     get_session_profile,
        ... )
        >>> profile = get_session_profile("feature_development")
        >>> generator = ClaudeCodeGenerator()
        >>> events = generator.generate(session_profile=profile, num_events=50)
    """

    source_name = "claude-code"

    def generate(
        self,
        num_events: int = 100,
        start_date: datetime | None = None,
        num_days: int = 1,
        session_profile: SessionProfile | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate a timeline of Claude Code session events.

        Args:
            num_events: Target number of events to generate.
            start_date: Starting date for events (defaults to today).
            num_days: Number of days (sessions) to span events across.
            session_profile: The session profile to generate events for (required).
            **kwargs: Additional arguments (unused).

        Returns:
            List of event dictionaries in timeline format.

        Raises:
            ValueError: If session_profile is not provided.
        """
        if session_profile is None:
            raise ValueError("session_profile is required for ClaudeCodeGenerator")

        if start_date is None:
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        all_events: list[dict[str, Any]] = []
        events_per_day = num_events // num_days

        for day_offset in range(num_days):
            current_date = start_date + timedelta(days=day_offset)
            day_events = self._generate_day(
                date=current_date,
                target_events=events_per_day,
                previous_events=all_events[-10:] if all_events else [],
                session_profile=session_profile,
            )
            all_events.extend(day_events)

        # Re-number event IDs and add source
        for i, event in enumerate(all_events):
            event["event_id"] = f"cc-{i + 1:03d}"
            event["source"] = self.source_name

        return all_events[:num_events]

    def _generate_day(
        self,
        date: datetime,
        target_events: int,
        previous_events: list[dict[str, Any]],
        session_profile: SessionProfile | None = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate Claude Code session events for a single day/session.

        Args:
            date: The date to generate events for.
            target_events: Target number of events for this session.
            previous_events: Recent events for context continuity.
            session_profile: The session profile to generate events for.
            max_retries: Maximum number of retry attempts on failure.
            **kwargs: Additional arguments (unused).

        Returns:
            List of event dictionaries for the session.
        """
        import time

        if session_profile is None:
            return []

        client = self._ensure_client()
        from google.genai import types

        day_name = date.strftime("%A")
        date_str = date.strftime("%Y-%m-%d")

        prev_context = self._build_prev_context(previous_events)

        prompt = self._build_generation_prompt(
            date_str=date_str,
            day_name=day_name,
            target_events=target_events,
            prev_context=prev_context,
            session_profile=session_profile,
        )

        gen_config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=[{"role": "user", "parts": [{"text": prompt}]}],
                    config=gen_config,
                )

                if not response.candidates:
                    raise ValueError(f"No candidates in response for {date_str}")

                candidate = response.candidates[0]

                if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                    finish_reason = str(candidate.finish_reason)
                    if "SAFETY" in finish_reason or "BLOCKED" in finish_reason:
                        raise ValueError(f"Response blocked: {finish_reason}")

                if not candidate.content or not candidate.content.parts:
                    raise ValueError(f"No content parts in response for {date_str}")

                text = candidate.content.parts[0].text
                events = self._parse_events(text, date)

                if not events:
                    preview = text[:500] if len(text) > 500 else text
                    raise ValueError(f"Failed to parse events for {date_str}. Preview: {preview!r}")

                return events

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    time.sleep(wait_time)
                else:
                    import sys

                    print(f"Error generating events for {date_str}: {e}", file=sys.stderr)
                    return []

        return []

    def _build_generation_prompt(
        self,
        date_str: str,
        day_name: str,
        target_events: int,
        prev_context: str,
        session_profile: SessionProfile | None = None,
        **kwargs: Any,
    ) -> str:
        """Build the prompt for Claude Code session event generation.

        Args:
            date_str: Date string (YYYY-MM-DD format).
            day_name: Day of week name.
            target_events: Target number of events.
            prev_context: Context from previous events.
            session_profile: The session profile to generate events for.
            **kwargs: Additional arguments (unused).

        Returns:
            The prompt string for the LLM.
        """
        if session_profile is None:
            raise ValueError("session_profile is required for prompt generation")

        return f"""You are a Claude Code session event stream generator. Generate realistic events from an AI coding assistant session.

{session_profile.to_prompt()}

## Task
Generate exactly {target_events} Claude Code session events for {day_name}, {date_str}.
The session should span 1-4 hours within work hours (9am-6pm).

## Event Types (use dot notation)
- **tool.read**: Reading a file to understand code
- **tool.edit**: Editing/modifying an existing file
- **tool.write**: Creating a new file
- **tool.bash**: Running a shell command (tests, builds, git, linting)
- **tool.grep**: Searching file contents for patterns
- **tool.glob**: Finding files by name/pattern
- **tool.web_search**: Searching the web for documentation or solutions
- **tool.web_fetch**: Fetching content from a URL
- **git.commit**: Creating a git commit
- **git.branch**: Branch operations (create, switch, delete)
- **git.push**: Pushing to remote
- **git.pr_create**: Creating a pull request
- **conversation.human_message**: Developer sending an instruction or question
- **conversation.claude_response**: AI assistant responding with explanation or plan
- **conversation.decision**: Developer making a design or implementation decision
- **conversation.correction**: Developer correcting the AI's approach
- **session.start**: Beginning of a coding session
- **session.end**: End of a coding session
- **error.tool_failure**: A tool invocation that failed (file not found, command error)
- **error.test_failure**: Test suite reporting failures
- **error.retry**: Retrying a failed operation with a different approach

## Requirements
1. Events must be temporally coherent (realistic tool sequences)
2. Follow the session phases defined in the profile
3. Timestamps must be on {date_str} within a realistic session window
4. Include realistic context in the context field
5. Show natural tool usage patterns (Read before Edit, Bash for tests after changes)
6. Include conversation events interspersed with tool usage
7. Include some errors and recovery — sessions are not always smooth
8. Start with session.start and end with session.end

## Output Format
Return a JSON array. Each event must have:
- event_id: unique ID (format: "cc-XXX")
- timestamp: ISO 8601 datetime
- event_type: one of the types above (use dot notation)
- title: short description (2-6 words)
- description: brief context (1-2 sentences)
- context: structured data specific to the event type

## Context Field Examples

tool.read:
{{"file_path": "src/app/models.py", "lines_read": 45, "purpose": "understand data model"}}

tool.edit:
{{"file_path": "src/app/views.py", "lines_changed": 12, "change_type": "add_function"}}

tool.bash:
{{"command": "pytest tests/ -x", "exit_code": 0, "output_summary": "12 passed"}}

tool.grep:
{{"pattern": "def process_event", "matches": 3, "files_searched": "src/**/*.py"}}

tool.glob:
{{"pattern": "src/**/*.py", "matches_found": 15}}

git.commit:
{{"message": "feat: add event validation endpoint", "files_changed": 3, "insertions": 45, "deletions": 8}}

conversation.human_message:
{{"message_preview": "Can you add input validation to the create endpoint?", "intent": "feature_request"}}

conversation.claude_response:
{{"message_preview": "I'll add Pydantic validation. Let me first read the current model...", "action": "plan"}}

conversation.correction:
{{"message_preview": "Actually, use the existing validator pattern from auth.py", "corrects": "cc-015"}}

error.tool_failure:
{{"tool": "Bash", "command": "pytest tests/", "error": "ModuleNotFoundError: No module named 'app'", "will_retry": true}}

error.test_failure:
{{"test_file": "tests/test_views.py", "failures": 2, "total": 10, "failure_summary": "assertion error in test_create"}}

session.start:
{{"project": "{session_profile.project_description}", "task": "{session_profile.task_description}"}}

## Example Events

{{
  "event_id": "cc-001",
  "timestamp": "{date_str}T09:30:00Z",
  "event_type": "session.start",
  "title": "Session started",
  "description": "Developer begins coding session",
  "context": {{"project": "{session_profile.project_description}", "task": "{session_profile.task_description}"}}
}}

{{
  "event_id": "cc-005",
  "timestamp": "{date_str}T09:35:00Z",
  "event_type": "tool.glob",
  "title": "Find Python source files",
  "description": "Exploring project structure to understand codebase layout",
  "context": {{"pattern": "src/**/*.py", "matches_found": 12}}
}}

{{
  "event_id": "cc-020",
  "timestamp": "{date_str}T10:15:00Z",
  "event_type": "error.test_failure",
  "title": "Test failures detected",
  "description": "Two tests failed after adding new validation logic",
  "context": {{"test_file": "tests/test_views.py", "failures": 2, "total": 10, "failure_summary": "missing required field"}}
}}
{prev_context}

Generate {target_events} events as a JSON array:"""

    def _parse_events(self, text: str, date: datetime) -> list[dict[str, Any]]:
        """Parse LLM response into event dictionaries.

        Args:
            text: Raw LLM response text.
            date: The date for these events.

        Returns:
            List of parsed event dictionaries.
        """
        # Remove markdown code blocks if present
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)

        # Extract JSON array from response
        json_match = re.search(r"\[[\s\S]*\]", text)
        if not json_match:
            return []

        json_str = json_match.group(0)

        # Fix common JSON issues
        json_str = re.sub(r",\s*]", "]", json_str)
        json_str = re.sub(r",\s*}", "}", json_str)

        try:
            events = json.loads(json_str)
        except json.JSONDecodeError:
            return []

        # Validate and normalize events
        validated_events = []
        for i, event in enumerate(events):
            if not isinstance(event, dict):
                continue

            # Ensure required fields
            if "event_id" not in event:
                event["event_id"] = f"cc-{uuid.uuid4().hex[:8]}"

            if "timestamp" not in event:
                # Generate a timestamp based on position (session hours 9am-1pm)
                hour = 9 + (i * 4 // max(len(events), 1))
                minute = (i * 15) % 60
                event["timestamp"] = date.replace(hour=hour, minute=minute).isoformat() + "Z"

            if "event_type" not in event:
                event["event_type"] = "tool.read"

            if "title" not in event:
                event["title"] = "Session activity"

            if "context" not in event:
                event["context"] = {}

            validated_events.append(event)

        return validated_events

    def save_timeline(
        self,
        events: list[dict[str, Any]],
        path: str | Path,
        timeline_id: str | None = None,
        description: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
        *,
        session_profile: SessionProfile | None = None,
        **kwargs: Any,
    ) -> Path:
        """Save generated events to a timeline JSON file.

        Args:
            events: List of event dictionaries.
            path: Path to save the timeline.
            timeline_id: Optional ID for the timeline.
            description: Optional description.
            extra_metadata: Optional extra metadata (from base class).
            session_profile: Optional session profile to include in metadata.

        Returns:
            Path to the saved file.
        """
        merged_metadata = dict(extra_metadata) if extra_metadata else {}
        if session_profile:
            merged_metadata["session_profile"] = session_profile.to_dict()

        return super().save_timeline(
            events=events,
            path=path,
            timeline_id=timeline_id,
            description=description,
            extra_metadata=merged_metadata,
        )
