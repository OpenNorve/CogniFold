"""Base interface for event stream generators."""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any


class BaseEventGenerator(ABC):
    """Abstract base class for event stream generators.

    All domain-specific generators (personal timeline, computer activity,
    service logs, etc.) should inherit from this class and implement the
    required abstract methods.

    The base class provides:
    - Common LLM client management
    - Timeline saving/loading utilities
    - Event ID generation
    - Consistent configuration handling

    Subclasses must implement:
    - generate(): Main generation entry point
    - _generate_day(): Generate events for a single day
    - _build_generation_prompt(): Build the LLM prompt
    - _parse_events(): Parse LLM response into events

    Example:
        >>> class MyGenerator(BaseEventGenerator):
        ...     source_name = "my-domain"
        ...     def generate(self, ...): ...
        >>> generator = MyGenerator()
        >>> events = generator.generate(...)
    """

    # Subclasses should override this to identify the event source
    source_name: str = "unknown"

    def __init__(
        self,
        model_name: str = "gemini-3-flash-preview",
        temperature: float = 0.8,
        max_output_tokens: int = 8192,
    ):
        """Initialize the event generator.

        Args:
            model_name: LLM model to use.
            temperature: Sampling temperature (higher = more creative).
            max_output_tokens: Maximum tokens in LLM response.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._client: Any = None

    def _ensure_client(self) -> Any:
        """Ensure the LLM client is initialized.

        Returns:
            The initialized LLM client (Gemini or OpenAI).

        Raises:
            ValueError: If API keys are not set.
        """
        if self._client is None:
            import os

            from cognifold.service.llm_keys import get_api_key

            if self.model_name.startswith("openai:"):
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=get_api_key("OPENAI_API_KEY"),
                    base_url=os.environ.get("OPENAI_BASE_URL"),
                )
            else:
                api_key = get_api_key("GOOGLE_API_KEY") or get_api_key("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable is required. "
                        "Set it with: export GOOGLE_API_KEY='your-api-key'"
                    )

                from google import genai

                self._client = genai.Client(api_key=api_key)

        return self._client

    def _generate_text(self, prompt: str, max_retries: int = 3) -> str:
        """Generate text from LLM with retries.

        Args:
            prompt: Input prompt.
            max_retries: Retry attempts.

        Returns:
            Generated text.
        """
        import sys
        import time

        client = self._ensure_client()
        is_openai = self.model_name.startswith("openai:")
        model_name = self.model_name.replace("openai:", "") if is_openai else self.model_name

        gen_config: Any | None = None
        if not is_openai:
            from google.genai import types

            gen_config = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )

        for attempt in range(max_retries):
            try:
                if is_openai:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=self.max_output_tokens,
                    )
                    return response.choices[0].message.content or ""

                response = client.models.generate_content(
                    model=model_name,
                    contents=[{"role": "user", "parts": [{"text": prompt}]}],
                    config=gen_config,
                )

                if not response.candidates:
                    raise ValueError("No candidates in response")

                candidate = response.candidates[0]
                if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                    finish_reason = str(candidate.finish_reason)
                    if "SAFETY" in finish_reason or "BLOCKED" in finish_reason:
                        raise ValueError(f"Response blocked: {finish_reason}")

                if not candidate.content or not candidate.content.parts:
                    raise ValueError("No content parts in response")

                return candidate.content.parts[0].text
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    label = "OpenAI" if is_openai else "Gemini"
                    print(f"Error generating text ({label}): {e}", file=sys.stderr)
                    return ""

        return ""

    @abstractmethod
    def generate(
        self,
        num_events: int = 100,
        start_date: datetime | None = None,
        num_days: int = 3,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate a timeline of events.

        Args:
            num_events: Target number of events to generate.
            start_date: Starting date for events (defaults to today).
            num_days: Number of days to span events across.
            **kwargs: Domain-specific arguments (e.g., persona, service_topology).

        Returns:
            List of event dictionaries with the unified schema.
        """
        ...

    @abstractmethod
    def _generate_day(
        self,
        date: datetime,
        target_events: int,
        previous_events: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate events for a single day.

        Args:
            date: The date to generate events for.
            target_events: Target number of events for this day.
            previous_events: Recent events for context continuity.
            **kwargs: Domain-specific arguments.

        Returns:
            List of event dictionaries for the day.
        """
        ...

    @abstractmethod
    def _build_generation_prompt(
        self,
        date_str: str,
        day_name: str,
        target_events: int,
        prev_context: str,
        **kwargs: Any,
    ) -> str:
        """Build the LLM prompt for event generation.

        Args:
            date_str: Date string (YYYY-MM-DD format).
            day_name: Day of week name.
            target_events: Target number of events.
            prev_context: Context from previous events.
            **kwargs: Domain-specific arguments.

        Returns:
            The prompt string for the LLM.
        """
        ...

    @abstractmethod
    def _parse_events(self, text: str, date: datetime) -> list[dict[str, Any]]:
        """Parse LLM response into event dictionaries.

        Args:
            text: Raw LLM response text.
            date: The date for these events.

        Returns:
            List of parsed event dictionaries.
        """
        ...

    def _generate_event_id(self, prefix: str = "e") -> str:
        """Generate a unique event ID.

        Args:
            prefix: Prefix for the event ID.

        Returns:
            A unique event ID string.
        """
        return f"{prefix}-{uuid.uuid4().hex[:8]}"

    def _renumber_events(
        self, events: list[dict[str, Any]], prefix: str = "e"
    ) -> list[dict[str, Any]]:
        """Renumber event IDs to ensure sequential ordering.

        Args:
            events: List of events to renumber.
            prefix: Prefix for event IDs.

        Returns:
            Events with sequential IDs.
        """
        for i, event in enumerate(events):
            event["event_id"] = f"{prefix}-{i + 1:03d}"
        return events

    def _build_prev_context(self, previous_events: list[dict[str, Any]]) -> str:
        """Build context string from previous events.

        Args:
            previous_events: Recent events for context.

        Returns:
            Formatted context string for the prompt.
        """
        if not previous_events:
            return ""

        context = "\n\nRecent events for context continuity:\n"
        for event in previous_events[-5:]:
            context += f"- {event.get('timestamp', '')}: {event.get('title', '')}\n"
        return context

    def save_timeline(
        self,
        events: list[dict[str, Any]],
        path: str | Path,
        timeline_id: str | None = None,
        description: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Path:
        """Save generated events to a timeline JSON file.

        Args:
            events: List of event dictionaries.
            path: Path to save the timeline.
            timeline_id: Optional ID for the timeline.
            description: Optional description.
            extra_metadata: Optional extra metadata to include.

        Returns:
            Path to the saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        timeline: dict[str, Any] = {
            "timeline_id": timeline_id or f"generated-{uuid.uuid4().hex[:8]}",
            "source": self.source_name,
            "description": description or f"Generated timeline with {len(events)} events",
            "generated_at": datetime.now().isoformat(),
            "events": events,
        }

        if extra_metadata:
            timeline.update(extra_metadata)

        with open(path, "w") as f:
            json.dump(timeline, f, indent=2)

        return path

    @staticmethod
    def load_timeline(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Load a timeline from a JSON file.

        Args:
            path: Path to the timeline file.

        Returns:
            Tuple of (events list, metadata dict).

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Timeline file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        events = data.get("events", [])
        metadata = {k: v for k, v in data.items() if k != "events"}

        return events, metadata
