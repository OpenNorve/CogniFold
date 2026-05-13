"""Computer activity event stream generator."""

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
class WorkProfile:
    """Defines a computer work style profile for event generation.

    Attributes:
        name: Profile name (e.g., "software_developer", "data_analyst").
        role: Job role description.
        primary_apps: Main applications used.
        common_websites: Frequently visited websites.
        communication_tools: IM/email tools used.
        work_patterns: Typical work patterns and habits.
        focus_areas: Main areas of focus during work.
    """

    name: str
    role: str
    primary_apps: list[str] = field(default_factory=list)
    common_websites: list[str] = field(default_factory=list)
    communication_tools: list[str] = field(default_factory=list)
    work_patterns: list[str] = field(default_factory=list)
    focus_areas: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Convert profile to prompt format."""
        return f"""## Work Profile: {self.name}
Role: {self.role}

Primary Applications: {", ".join(self.primary_apps)}
Common Websites: {", ".join(self.common_websites)}
Communication Tools: {", ".join(self.communication_tools)}

Work Patterns:
{chr(10).join(f"- {p}" for p in self.work_patterns)}

Focus Areas:
{chr(10).join(f"- {a}" for a in self.focus_areas)}"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "role": self.role,
            "primary_apps": self.primary_apps,
            "common_websites": self.common_websites,
            "communication_tools": self.communication_tools,
            "work_patterns": self.work_patterns,
            "focus_areas": self.focus_areas,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkProfile:
        """Create from dictionary."""
        return cls(**data)


# Sample work profiles
SAMPLE_WORK_PROFILES: dict[str, WorkProfile] = {
    "software_developer": WorkProfile(
        name="software_developer",
        role="Full-stack software developer working on web applications",
        primary_apps=[
            "VS Code",
            "Terminal",
            "Docker Desktop",
            "Postman",
            "Slack",
            "Chrome",
            "Firefox",
        ],
        common_websites=[
            "github.com",
            "stackoverflow.com",
            "docs.python.org",
            "developer.mozilla.org",
            "npmjs.com",
            "aws.amazon.com",
        ],
        communication_tools=["Slack", "Zoom", "Gmail", "GitHub Issues"],
        work_patterns=[
            "Morning code review and PR feedback",
            "Deep coding sessions in late morning",
            "Standup meetings around 10am",
            "Afternoon debugging and testing",
            "Documentation updates before end of day",
            "Context switching between multiple projects",
        ],
        focus_areas=[
            "Backend API development",
            "Frontend React components",
            "Database queries and optimization",
            "CI/CD pipeline maintenance",
            "Code review and mentoring",
        ],
    ),
    "data_analyst": WorkProfile(
        name="data_analyst",
        role="Data analyst working with business intelligence and reporting",
        primary_apps=[
            "Excel",
            "Tableau",
            "Jupyter Notebook",
            "Python",
            "SQL Client",
            "PowerBI",
            "Chrome",
        ],
        common_websites=[
            "kaggle.com",
            "pandas.pydata.org",
            "stackoverflow.com",
            "medium.com",
            "towardsdatascience.com",
            "google.com/sheets",
        ],
        communication_tools=["Slack", "Microsoft Teams", "Outlook", "Confluence"],
        work_patterns=[
            "Morning data pipeline checks",
            "Report generation and updates",
            "Ad-hoc analysis requests",
            "Weekly stakeholder presentations",
            "Data cleaning and preprocessing",
            "Dashboard maintenance",
        ],
        focus_areas=[
            "SQL query optimization",
            "Data visualization",
            "Statistical analysis",
            "Report automation",
            "Data quality monitoring",
        ],
    ),
    "product_manager": WorkProfile(
        name="product_manager",
        role="Product manager coordinating product development",
        primary_apps=[
            "Notion",
            "Jira",
            "Figma",
            "Slack",
            "Zoom",
            "Chrome",
            "Google Docs",
        ],
        common_websites=[
            "notion.so",
            "atlassian.com",
            "figma.com",
            "productboard.com",
            "amplitude.com",
            "mixpanel.com",
        ],
        communication_tools=["Slack", "Zoom", "Gmail", "Loom"],
        work_patterns=[
            "Morning metrics review",
            "Stakeholder meetings throughout day",
            "Sprint planning and grooming",
            "User feedback analysis",
            "Roadmap updates",
            "Cross-team coordination",
        ],
        focus_areas=[
            "Feature prioritization",
            "User research synthesis",
            "Sprint management",
            "Stakeholder communication",
            "Competitive analysis",
        ],
    ),
}


# Alias for consistency with other modules
SAMPLE_PROFILES = SAMPLE_WORK_PROFILES


def get_work_profile(name: str) -> WorkProfile:
    """Get a work profile by name.

    Args:
        name: Profile name.

    Returns:
        The work profile.

    Raises:
        KeyError: If profile not found.
    """
    if name not in SAMPLE_WORK_PROFILES:
        available = ", ".join(SAMPLE_WORK_PROFILES.keys())
        raise KeyError(f"Unknown work profile: {name}. Available: {available}")
    return SAMPLE_WORK_PROFILES[name]


class ComputerActivityGenerator(BaseEventGenerator):
    """Generates computer activity events using Gemini LLM.

    Creates realistic sequences of computer usage events including browser
    activity, application switches, file operations, and communication.

    Example:
        >>> from cognifold.generator.computer_activity import (
        ...     ComputerActivityGenerator,
        ...     get_work_profile,
        ... )
        >>> profile = get_work_profile("software_developer")
        >>> generator = ComputerActivityGenerator()
        >>> events = generator.generate(work_profile=profile, num_events=50)
    """

    source_name = "computer-activity"

    def generate(
        self,
        num_events: int = 100,
        start_date: datetime | None = None,
        num_days: int = 3,
        work_profile: WorkProfile | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate a timeline of computer activity events.

        Args:
            num_events: Target number of events to generate.
            start_date: Starting date for events (defaults to today).
            num_days: Number of days to span events across.
            work_profile: The work profile to generate events for (required).
            **kwargs: Additional arguments (unused).

        Returns:
            List of event dictionaries in timeline format.

        Raises:
            ValueError: If work_profile is not provided.
        """
        if work_profile is None:
            raise ValueError("work_profile is required for ComputerActivityGenerator")

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
                work_profile=work_profile,
            )
            all_events.extend(day_events)

        # Re-number event IDs and add source
        for i, event in enumerate(all_events):
            event["event_id"] = f"ca-{i + 1:03d}"
            event["source"] = self.source_name

        return all_events[:num_events]

    def _generate_day(
        self,
        date: datetime,
        target_events: int,
        previous_events: list[dict[str, Any]],
        work_profile: WorkProfile | None = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate computer activity events for a single day.

        Args:
            date: The date to generate events for.
            target_events: Target number of events for this day.
            previous_events: Recent events for context continuity.
            work_profile: The work profile to generate events for.
            max_retries: Maximum number of retry attempts on failure.
            **kwargs: Additional arguments (unused).

        Returns:
            List of event dictionaries for the day.
        """
        import time

        if work_profile is None:
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
            work_profile=work_profile,
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
        work_profile: WorkProfile | None = None,
        **kwargs: Any,
    ) -> str:
        """Build the prompt for computer activity event generation.

        Args:
            date_str: Date string (YYYY-MM-DD format).
            day_name: Day of week name.
            target_events: Target number of events.
            prev_context: Context from previous events.
            work_profile: The work profile to generate events for.
            **kwargs: Additional arguments (unused).

        Returns:
            The prompt string for the LLM.
        """
        if work_profile is None:
            raise ValueError("work_profile is required for prompt generation")

        return f"""You are a computer activity event stream generator. Generate realistic computer usage events for a work day.

{work_profile.to_prompt()}

## Task
Generate exactly {target_events} computer activity events for {day_name}, {date_str}.

## Event Types (use dot notation for subtypes)
- **browser.page_visit**: Visiting a web page
- **browser.search**: Search query on search engine
- **browser.tab_switch**: Switching between browser tabs
- **browser.bookmark**: Bookmarking a page
- **app.launch**: Launching an application
- **app.switch**: Switching to a different application
- **app.close**: Closing an application
- **file.open**: Opening a file
- **file.save**: Saving a file
- **file.create**: Creating a new file
- **editor.code_change**: Making code changes (for developers)
- **editor.search**: Searching in editor
- **terminal.command**: Running a terminal command
- **communication.message_send**: Sending a message (Slack, Teams, etc.)
- **communication.message_receive**: Receiving a message
- **communication.email_send**: Sending an email
- **communication.email_read**: Reading an email
- **communication.call_start**: Starting a video/audio call
- **communication.call_end**: Ending a call
- **system.login**: System login
- **system.lock**: Locking screen
- **system.notification**: Receiving a system notification
- **meeting.join**: Joining a meeting
- **meeting.leave**: Leaving a meeting

## Requirements
1. Events must be temporally coherent (realistic sequences)
2. Include natural work patterns (deep work, context switching, breaks)
3. Timestamps must be on {date_str} during work hours (roughly 8am-6pm)
4. Include realistic context in the context field
5. Show realistic application and browser usage patterns
6. Include communication events interspersed with focused work

## Output Format
Return a JSON array. Each event must have:
- event_id: unique ID (format: "ca-XXX")
- timestamp: ISO 8601 datetime
- event_type: one of the types above (use dot notation)
- title: short description (2-5 words)
- description: brief context (1 sentence)
- context: structured data specific to the event type

## Context Field Examples

browser.page_visit:
{{"url": "https://github.com/user/repo", "browser": "Chrome", "tab_id": 5, "referrer": "google.com"}}

app.switch:
{{"from_app": "VS Code", "to_app": "Slack", "window_title": "team-channel"}}

file.save:
{{"file_path": "/projects/app/src/main.py", "app": "VS Code", "file_size_kb": 45}}

communication.message_send:
{{"platform": "Slack", "channel": "#engineering", "message_preview": "PR ready for review"}}

terminal.command:
{{"command": "git push origin main", "working_dir": "/projects/app", "exit_code": 0}}

meeting.join:
{{"platform": "Zoom", "meeting_title": "Sprint Planning", "participants": 8}}

## Example Events

{{
  "event_id": "ca-001",
  "timestamp": "{date_str}T09:00:00Z",
  "event_type": "system.login",
  "title": "Morning login",
  "description": "Started work day",
  "context": {{"device": "MacBook Pro", "location": "home"}}
}}

{{
  "event_id": "ca-010",
  "timestamp": "{date_str}T09:30:00Z",
  "event_type": "browser.page_visit",
  "title": "GitHub PR review",
  "description": "Reviewing pull request from teammate",
  "context": {{"url": "https://github.com/team/project/pull/123", "browser": "Chrome", "tab_id": 3}}
}}

{{
  "event_id": "ca-020",
  "timestamp": "{date_str}T10:15:00Z",
  "event_type": "communication.message_send",
  "title": "Slack message",
  "description": "Sent code review feedback",
  "context": {{"platform": "Slack", "channel": "#code-review", "message_preview": "LGTM, minor suggestions..."}}
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
                event["event_id"] = f"ca-{uuid.uuid4().hex[:8]}"

            if "timestamp" not in event:
                # Generate a timestamp based on position (work hours 8am-6pm)
                hour = 8 + (i * 10 // len(events))
                event["timestamp"] = date.replace(hour=hour).isoformat() + "Z"

            if "event_type" not in event:
                event["event_type"] = "app.switch"

            if "title" not in event:
                event["title"] = "Activity"

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
        work_profile: WorkProfile | None = None,
        **kwargs: Any,
    ) -> Path:
        """Save generated events to a timeline JSON file.

        Args:
            events: List of event dictionaries.
            path: Path to save the timeline.
            timeline_id: Optional ID for the timeline.
            description: Optional description.
            extra_metadata: Optional extra metadata (from base class).
            work_profile: Optional work profile to include in metadata.

        Returns:
            Path to the saved file.
        """
        merged_metadata = dict(extra_metadata) if extra_metadata else {}
        if work_profile:
            merged_metadata["work_profile"] = work_profile.to_dict()

        return super().save_timeline(
            events=events,
            path=path,
            timeline_id=timeline_id,
            description=description,
            extra_metadata=merged_metadata,
        )
