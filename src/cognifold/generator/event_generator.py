"""Personal timeline event stream generator."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cognifold.generator.base import BaseEventGenerator

if TYPE_CHECKING:
    from cognifold.generator.persona import Persona


class PersonalTimelineGenerator(BaseEventGenerator):
    """Generates realistic personal timeline events using Gemini LLM.

    The generator creates temporally coherent events based on a persona's
    characteristics, routines, and lifestyle. Events follow realistic
    sequences (e.g., wake up -> breakfast -> commute -> work).

    Example:
        >>> from cognifold.generator import PersonalTimelineGenerator, Persona
        >>> persona = Persona(name="Alex", age=28, occupation="Engineer")
        >>> generator = PersonalTimelineGenerator()
        >>> events = generator.generate(persona=persona, num_events=50)
        >>> generator.save_timeline(events, "data/generated/alex_timeline.json", persona=persona)
    """

    source_name = "personal-timeline"

    def __init__(
        self,
        model_name: str = "gemini-3-flash-preview",
        temperature: float = 0.8,
        batch_size: int = 20,
    ):
        """Initialize the event generator.

        Args:
            model_name: Gemini model to use.
            temperature: Sampling temperature (higher = more creative).
            batch_size: Number of events to generate per API call.
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=8192,
        )
        self.batch_size = batch_size

    def generate(
        self,
        num_events: int = 100,
        start_date: datetime | None = None,
        num_days: int = 3,
        persona: Persona | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate a timeline of events for a persona.

        Args:
            num_events: Target number of events to generate.
            start_date: Starting date for events (defaults to today).
            num_days: Number of days to span events across.
            persona: The persona to generate events for (required).
            **kwargs: Additional arguments (unused).

        Returns:
            List of event dictionaries in timeline format.

        Raises:
            ValueError: If persona is not provided.
        """
        if persona is None:
            raise ValueError("persona is required for PersonalTimelineGenerator")

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
                persona=persona,
            )
            all_events.extend(day_events)

        # Re-number event IDs and add source
        for i, event in enumerate(all_events):
            event["event_id"] = f"e-{i + 1:03d}"
            event["source"] = self.source_name

        return all_events[:num_events]

    def _generate_day(
        self,
        date: datetime,
        target_events: int,
        previous_events: list[dict[str, Any]],
        persona: Persona | None = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate events for a single day.

        Args:
            date: The date to generate events for.
            target_events: Target number of events for this day.
            previous_events: Recent events for context continuity.
            persona: The persona to generate events for.
            max_retries: Maximum number of retry attempts on failure.
            **kwargs: Additional arguments (unused).

        Returns:
            List of event dictionaries for the day.
        """
        import time

        if persona is None:
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
            persona=persona,
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

                # Check for valid response structure
                if not response.candidates:
                    raise ValueError(f"No candidates in response for {date_str}")

                candidate = response.candidates[0]

                # Check for blocked content or other issues
                if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                    finish_reason = str(candidate.finish_reason)
                    if "SAFETY" in finish_reason or "BLOCKED" in finish_reason:
                        raise ValueError(f"Response blocked: {finish_reason}")

                if not candidate.content or not candidate.content.parts:
                    raise ValueError(f"No content parts in response for {date_str}")

                text = candidate.content.parts[0].text
                events = self._parse_events(text, date)

                if not events:
                    # Log first 500 chars of response for debugging
                    preview = text[:500] if len(text) > 500 else text
                    raise ValueError(
                        f"Failed to parse events from response for {date_str}. "
                        f"Response preview: {preview!r}"
                    )

                return events

            except Exception as e:
                if attempt < max_retries - 1:
                    # Wait before retry with exponential backoff
                    wait_time = 2**attempt
                    time.sleep(wait_time)
                else:
                    # Log the error on final failure
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
        persona: Persona | None = None,
        **kwargs: Any,
    ) -> str:
        """Build the prompt for event generation.

        Args:
            date_str: Date string (YYYY-MM-DD format).
            day_name: Day of week name.
            target_events: Target number of events.
            prev_context: Context from previous events.
            persona: The persona to generate events for.
            **kwargs: Additional arguments (unused).

        Returns:
            The prompt string for the LLM.
        """
        if persona is None:
            raise ValueError("persona is required for prompt generation")

        # Calculate a future date for deadlines
        base_date = datetime.fromisoformat(date_str)
        future_date = (base_date + timedelta(days=2)).strftime("%Y-%m-%d")

        return f"""You are an event stream generator. Generate a realistic sequence of daily events for a person.

{persona.to_prompt()}

## Task
Generate exactly {target_events} events for {day_name}, {date_str}.

## Event Types
Use these event types:
- **meal**: breakfast, lunch, dinner, snacks
- **work**: meetings, coding, writing, calls, presentations
- **study**: reading, courses, research, practice
- **exercise**: gym, yoga, running, walking
- **social**: conversations, calls, gatherings, texting
- **rest**: sleep, naps, relaxation
- **transit**: commute, travel, errands
- **entertainment**: movies, games, music, hobbies
- **planning**: scheduling, reviewing calendar, setting reminders
- **deadline**: project due dates, submission deadlines, appointment reminders

## Requirements
1. Events must be temporally coherent (realistic sequence and timing)
2. Events should reflect the persona's habits, occupation, and lifestyle
3. Include variety but stay consistent with persona
4. Timestamps must be on {date_str} in ISO 8601 format
5. Include realistic details in descriptions and metadata
6. Events should span from wake-up to sleep time

## IMPORTANT: Include Actionable Events
At least 20-30% of events should include one or more of these actionable elements:

**Scheduled meetings/appointments** (triggers TIME nodes):
- Include specific scheduled times in the future
- Examples: "Team standup at 10am", "Doctor appointment at 3pm tomorrow"
- Add `scheduled_time` in metadata for future events mentioned

**Deadlines mentioned** (triggers TIME nodes):
- Reference project due dates, submission deadlines
- Examples: "Working on report due Friday", "Preparing presentation for tomorrow"
- Add `deadline` in metadata with the deadline datetime

**Tasks to do / Follow-ups** (triggers ACTION nodes):
- Mention things that need to be done
- Examples: "Need to finish the proposal", "Should review the code before meeting"
- Add `action_needed` in metadata describing what needs to be done

**Planning/Reminder events**:
- Calendar reviews, scheduling, setting up meetings
- Examples: "Setting up meeting for next week", "Reviewing tasks for the week"
{prev_context}

## Output Format
Return a JSON array of events. Each event must have:
- event_id: unique ID (format: "e-XXX" where XXX is a number)
- timestamp: ISO 8601 datetime (must be on {date_str})
- event_type: one of the types above
- title: short description (2-5 words)
- description: detailed description (1-2 sentences, include actionable details!)
- location: where the event occurs
- duration_minutes: approximate duration
- metadata: IMPORTANT - include actionable details here

## Example Events

Regular event:
{{
  "event_id": "e-001",
  "timestamp": "{date_str}T07:30:00Z",
  "event_type": "meal",
  "title": "Breakfast at home",
  "description": "Quick oatmeal with berries and coffee while checking emails",
  "location": "home",
  "duration_minutes": 20,
  "metadata": {{"food": ["oatmeal", "berries", "coffee"]}}
}}

Event with scheduled meeting (TIME trigger):
{{
  "event_id": "e-010",
  "timestamp": "{date_str}T09:00:00Z",
  "event_type": "work",
  "title": "Morning standup meeting",
  "description": "Daily team sync to discuss progress and blockers. Need to prepare status update beforehand.",
  "location": "office",
  "duration_minutes": 30,
  "metadata": {{"meeting_type": "standup", "attendees": ["team"], "scheduled_time": "{date_str}T09:00:00Z", "action_needed": "Prepare status update"}}
}}

Event with deadline (TIME trigger):
{{
  "event_id": "e-015",
  "timestamp": "{date_str}T14:00:00Z",
  "event_type": "work",
  "title": "Working on quarterly report",
  "description": "Finishing the quarterly report that's due on Friday. Need to add the analytics section.",
  "location": "office",
  "duration_minutes": 120,
  "metadata": {{"project": "Q4 Report", "deadline": "{future_date}T17:00:00Z", "action_needed": "Complete analytics section"}}
}}

Event mentioning future task (ACTION trigger):
{{
  "event_id": "e-020",
  "timestamp": "{date_str}T16:30:00Z",
  "event_type": "social",
  "title": "Call with client",
  "description": "Discussed project requirements. Need to send follow-up email with proposal by tomorrow.",
  "location": "office",
  "duration_minutes": 45,
  "metadata": {{"client": "Acme Corp", "action_needed": "Send proposal email", "due_by": "{future_date}T12:00:00Z"}}
}}

Planning event:
{{
  "event_id": "e-025",
  "timestamp": "{date_str}T08:00:00Z",
  "event_type": "planning",
  "title": "Morning calendar review",
  "description": "Checking today's schedule. Have the design review at 2pm and need to book room for Thursday's presentation.",
  "location": "home",
  "duration_minutes": 15,
  "metadata": {{"upcoming_events": ["Design review at 2pm", "Presentation Thursday"], "action_needed": "Book meeting room for Thursday"}}
}}

Generate {target_events} events as a JSON array (remember: 20-30% should have actionable elements!):"""

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

        # Fix common JSON issues: trailing commas before ] or }
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
                event["event_id"] = f"e-{uuid.uuid4().hex[:8]}"

            if "timestamp" not in event:
                # Generate a timestamp based on position
                hour = 7 + (i * 17 // len(events))  # Spread from 7am to midnight
                event["timestamp"] = date.replace(hour=hour).isoformat() + "Z"

            if "event_type" not in event:
                event["event_type"] = "work"

            if "title" not in event:
                event["title"] = "Activity"

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
        persona: Persona | None = None,
        **kwargs: Any,
    ) -> Path:
        """Save generated events to a timeline JSON file.

        Args:
            events: List of event dictionaries.
            path: Path to save the timeline.
            timeline_id: Optional ID for the timeline.
            description: Optional description.
            extra_metadata: Optional extra metadata (from base class).
            persona: Optional persona to include in metadata.

        Returns:
            Path to the saved file.
        """
        merged_metadata = dict(extra_metadata) if extra_metadata else {}
        if persona:
            merged_metadata["persona"] = persona.to_dict()

        return super().save_timeline(
            events=events,
            path=path,
            timeline_id=timeline_id,
            description=description,
            extra_metadata=merged_metadata,
        )


# Backwards compatibility alias
EventGenerator = PersonalTimelineGenerator
