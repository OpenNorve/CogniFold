"""Temporal entity extraction from text.

This module extracts temporal references from text using a combination of
regex patterns and the dateparser library for natural language date parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar

# Try to import dateparser, provide fallback if not available
try:
    import dateparser

    _dateparser_available = True
except ImportError:
    dateparser = None  # type: ignore[assignment]
    _dateparser_available = False


class TemporalType(str, Enum):
    """Types of temporal references."""

    ABSOLUTE_DATE = "absolute_date"  # e.g., "January 15, 2026", "2026-01-15"
    ABSOLUTE_TIME = "absolute_time"  # e.g., "3:30 PM", "15:30"
    ABSOLUTE_DATETIME = "absolute_datetime"  # e.g., "January 15 at 3pm"
    RELATIVE_DATE = "relative_date"  # e.g., "yesterday", "last week"
    RELATIVE_TIME = "relative_time"  # e.g., "in 2 hours", "an hour ago"
    DURATION = "duration"  # e.g., "for 30 minutes", "2 hours"
    RECURRING = "recurring"  # e.g., "every Monday", "weekly"
    DEADLINE = "deadline"  # e.g., "by Friday", "due tomorrow"


@dataclass
class TemporalEntity:
    """A temporal reference extracted from text.

    Attributes:
        raw_text: The original text that was identified as temporal.
        normalized: The parsed datetime (None if parsing failed).
        temporal_type: The type of temporal reference.
        confidence: Confidence score (0.0 to 1.0).
        span: Character positions (start, end) in the source text.
        is_future: Whether this refers to a future time.
        metadata: Additional parsing metadata.
    """

    raw_text: str
    normalized: datetime | None
    temporal_type: TemporalType
    confidence: float
    span: tuple[int, int]
    is_future: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "raw_text": self.raw_text,
            "normalized": self.normalized.isoformat() if self.normalized else None,
            "temporal_type": self.temporal_type.value,
            "confidence": self.confidence,
            "span": list(self.span),
            "is_future": self.is_future,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalEntity:
        """Create from dictionary."""
        normalized = None
        if data.get("normalized"):
            normalized = datetime.fromisoformat(data["normalized"])

        return cls(
            raw_text=data["raw_text"],
            normalized=normalized,
            temporal_type=TemporalType(data["temporal_type"]),
            confidence=data["confidence"],
            span=tuple(data["span"]),  # type: ignore[arg-type]
            is_future=data.get("is_future", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExtractorConfig:
    """Configuration for temporal extraction.

    Attributes:
        min_confidence: Minimum confidence threshold for extraction.
        prefer_future: When ambiguous, prefer future dates.
        reference_time: Reference time for relative date resolution.
        languages: Languages to try for parsing (default: English).
    """

    min_confidence: float = 0.5
    prefer_future: bool = True
    reference_time: datetime | None = None
    languages: list[str] = field(default_factory=lambda: ["en"])


class TemporalExtractor:
    """Extracts temporal entities from text using regex and dateparser.

    This extractor combines:
    1. Regex patterns for common date/time formats
    2. Keyword detection for relative references
    3. dateparser library for natural language parsing

    Example:
        >>> extractor = TemporalExtractor()
        >>> entities = extractor.extract("Meeting tomorrow at 3pm")
        >>> for e in entities:
        ...     print(f"{e.raw_text} -> {e.normalized}")
    """

    # Regex patterns for common temporal expressions
    # NOTE: Order matters! More specific patterns should come first to be prioritized.
    # We use an ordered list internally to control matching priority.
    PATTERNS: ClassVar[dict[str, tuple[str, TemporalType]]] = {
        # PRIORITY 1: Multi-word specific patterns (checked first)
        # Recurring: every Monday, weekly, daily (must be before weekday)
        "recurring": (
            r"\b((?:every\s+(?:day|week|month|year|Monday|Tuesday|Wednesday|"
            r"Thursday|Friday|Saturday|Sunday))|daily|weekly|monthly|yearly|"
            r"biweekly|bi-weekly)\b",
            TemporalType.RECURRING,
        ),
        # Deadline: by Friday, due tomorrow, before 5pm (must be before weekday/relative_day)
        "deadline": (
            r"\b((?:by|due|before|until|no later than)\s+\w+)\b",
            TemporalType.DEADLINE,
        ),
        # PRIORITY 2: Date formats
        # ISO format: 2026-01-15, 2026/01/15
        "iso_date": (
            r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
            TemporalType.ABSOLUTE_DATE,
        ),
        # US format: 01/15/2026, 1/15/26
        "us_date": (
            r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b",
            TemporalType.ABSOLUTE_DATE,
        ),
        # Written dates: January 15, Jan 15th, 15 January
        "written_date": (
            r"\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
            r"Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?)\b",
            TemporalType.ABSOLUTE_DATE,
        ),
        # PRIORITY 3: Time formats
        # Time: 3:30 PM, 15:30, 3pm
        "time_12h": (
            r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM|a\.m\.|p\.m\.))\b",
            TemporalType.ABSOLUTE_TIME,
        ),
        "time_24h": (
            r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b",
            TemporalType.ABSOLUTE_TIME,
        ),
        # PRIORITY 4: Relative expressions (multi-word)
        "relative_week": (
            r"\b(this week|next week|last week|this weekend|next weekend)\b",
            TemporalType.RELATIVE_DATE,
        ),
        "relative_month": (
            r"\b(this month|next month|last month)\b",
            TemporalType.RELATIVE_DATE,
        ),
        # Days of week with modifiers (must be before plain weekday)
        "weekday_modified": (
            r"\b((?:next|this|last)\s+(?:Monday|Tuesday|Wednesday|Thursday|"
            r"Friday|Saturday|Sunday))\b",
            TemporalType.RELATIVE_DATE,
        ),
        # PRIORITY 5: Simple relative dates
        "relative_day": (
            r"\b(today|tomorrow|yesterday|the day after tomorrow|"
            r"day before yesterday)\b",
            TemporalType.RELATIVE_DATE,
        ),
        # Plain weekday (lowest priority for weekdays)
        "weekday": (
            r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
            TemporalType.RELATIVE_DATE,
        ),
        # PRIORITY 6: Duration and relative time
        # Duration: for 30 minutes, 2 hours, 3 days
        "duration": (
            r"\b((?:for |about |around |approximately )?\d+\s*"
            r"(?:minute|min|hour|hr|day|week|month|year)s?)\b",
            TemporalType.DURATION,
        ),
        # Relative time: in 2 hours, 30 minutes ago
        "relative_time": (
            r"\b((?:in )?\d+\s*(?:minute|min|hour|hr|day|week)s?\s*(?:ago|from now)?)\b",
            TemporalType.RELATIVE_TIME,
        ),
    }

    # Define pattern priority order (first = highest priority)
    PATTERN_ORDER: ClassVar[list[str]] = [
        "recurring",
        "deadline",
        "iso_date",
        "us_date",
        "written_date",
        "time_12h",
        "time_24h",
        "relative_week",
        "relative_month",
        "weekday_modified",
        "relative_day",
        "weekday",
        "duration",
        "relative_time",
    ]

    def __init__(self, config: ExtractorConfig | None = None) -> None:
        """Initialize the extractor with optional configuration.

        Args:
            config: Extraction configuration. Uses defaults if None.
        """
        self.config = config or ExtractorConfig()
        self._compiled_patterns: dict[str, re.Pattern[str]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        for name, (pattern, _) in self.PATTERNS.items():
            self._compiled_patterns[name] = re.compile(pattern, re.IGNORECASE)

    def extract(
        self,
        text: str,
        reference_time: datetime | None = None,
    ) -> list[TemporalEntity]:
        """Extract all temporal entities from text.

        Args:
            text: The text to extract temporal entities from.
            reference_time: Reference time for resolving relative dates.
                           Defaults to config.reference_time or now.

        Returns:
            List of extracted temporal entities, sorted by position.
        """
        if not text:
            return []

        ref_time = reference_time or self.config.reference_time or datetime.now()
        entities: list[TemporalEntity] = []
        seen_spans: set[tuple[int, int]] = set()

        # Extract using regex patterns in priority order
        for name in self.PATTERN_ORDER:
            if name not in self._compiled_patterns:
                continue
            compiled = self._compiled_patterns[name]
            pattern_type = self.PATTERNS[name][1]

            for match in compiled.finditer(text):
                span = (match.start(), match.end())

                # Skip if this span overlaps with an already extracted entity
                if self._spans_overlap(span, seen_spans):
                    continue

                raw_text = match.group(0)
                entity = self._create_entity(
                    raw_text=raw_text,
                    temporal_type=pattern_type,
                    span=span,
                    reference_time=ref_time,
                    pattern_name=name,
                )

                if entity.confidence >= self.config.min_confidence:
                    entities.append(entity)
                    seen_spans.add(span)

        # Try dateparser for any remaining potential dates
        if _dateparser_available:
            dateparser_entities = self._extract_with_dateparser(text, ref_time, seen_spans)
            entities.extend(dateparser_entities)

        # Sort by position in text
        entities.sort(key=lambda e: e.span[0])
        return entities

    def _spans_overlap(self, span: tuple[int, int], existing: set[tuple[int, int]]) -> bool:
        """Check if a span overlaps with any existing spans."""
        for existing_span in existing:
            # Check for overlap
            if span[0] < existing_span[1] and span[1] > existing_span[0]:
                return True
        return False

    def _create_entity(
        self,
        raw_text: str,
        temporal_type: TemporalType,
        span: tuple[int, int],
        reference_time: datetime,
        pattern_name: str,
    ) -> TemporalEntity:
        """Create a temporal entity from extracted text.

        Args:
            raw_text: The extracted text.
            temporal_type: Type of temporal reference.
            span: Character positions in source text.
            reference_time: Reference time for normalization.
            pattern_name: Name of the pattern that matched.

        Returns:
            A TemporalEntity with normalized datetime if possible.
        """
        normalized = None
        confidence = 0.7  # Default confidence for regex matches
        is_future = False

        # Try to normalize the datetime
        if _dateparser_available:
            settings = {
                "RELATIVE_BASE": reference_time,
                "PREFER_DATES_FROM": "future" if self.config.prefer_future else "past",
                "PREFER_DAY_OF_MONTH": "first",
            }

            try:
                assert dateparser is not None
                parsed = dateparser.parse(
                    raw_text,
                    languages=self.config.languages,
                    settings=settings,  # type: ignore[arg-type]
                )
                if parsed:
                    normalized = parsed
                    confidence = 0.85  # Higher confidence when dateparser succeeds
                    is_future = parsed > reference_time
            except Exception:
                pass  # dateparser failed, use None

        # Adjust confidence based on pattern type
        if temporal_type == TemporalType.ABSOLUTE_DATETIME:
            confidence = min(confidence + 0.1, 1.0)
        elif temporal_type == TemporalType.RELATIVE_DATE:
            confidence = max(confidence - 0.05, 0.5)
        elif temporal_type == TemporalType.DEADLINE:
            confidence = max(confidence - 0.1, 0.5)

        return TemporalEntity(
            raw_text=raw_text,
            normalized=normalized,
            temporal_type=temporal_type,
            confidence=confidence,
            span=span,
            is_future=is_future,
            metadata={"pattern": pattern_name},
        )

    def _extract_with_dateparser(
        self,
        text: str,
        reference_time: datetime,
        seen_spans: set[tuple[int, int]],
    ) -> list[TemporalEntity]:
        """Use dateparser to find additional temporal expressions.

        This is a fallback for expressions not caught by regex patterns.
        """
        if not _dateparser_available:
            return []

        entities: list[TemporalEntity] = []

        # Try to find date-like phrases using a simple heuristic
        # Look for sequences of words that might be dates
        potential_phrases = self._find_potential_date_phrases(text)

        settings = {
            "RELATIVE_BASE": reference_time,
            "PREFER_DATES_FROM": "future" if self.config.prefer_future else "past",
        }

        for phrase, span in potential_phrases:
            if self._spans_overlap(span, seen_spans):
                continue

            try:
                assert dateparser is not None
                parsed = dateparser.parse(
                    phrase,
                    languages=self.config.languages,
                    settings=settings,  # type: ignore[arg-type]
                )
                if parsed:
                    # Determine temporal type
                    temporal_type = self._infer_temporal_type(phrase)
                    is_future = parsed > reference_time

                    entity = TemporalEntity(
                        raw_text=phrase,
                        normalized=parsed,
                        temporal_type=temporal_type,
                        confidence=0.7,
                        span=span,
                        is_future=is_future,
                        metadata={"source": "dateparser"},
                    )

                    if entity.confidence >= self.config.min_confidence:
                        entities.append(entity)
                        seen_spans.add(span)
            except Exception:
                pass  # Skip unparseable phrases

        return entities

    def _find_potential_date_phrases(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        """Find potential date phrases in text.

        Uses heuristics to identify word sequences that might be dates.
        """
        phrases: list[tuple[str, tuple[int, int]]] = []

        # Pattern for potential date-like sequences
        # 2-4 word sequences that might be dates
        words = text.split()
        for i in range(len(words)):
            for length in range(2, 5):  # 2-4 word sequences
                if i + length > len(words):
                    break

                phrase = " ".join(words[i : i + length])

                # Simple heuristic: contains a number or month-like word
                if re.search(r"\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec", phrase, re.I):
                    # Calculate span
                    start = text.find(phrase)
                    if start >= 0:
                        phrases.append((phrase, (start, start + len(phrase))))

        return phrases

    def _infer_temporal_type(self, phrase: str) -> TemporalType:
        """Infer the temporal type from a phrase."""
        phrase_lower = phrase.lower()

        if any(word in phrase_lower for word in ["tomorrow", "yesterday", "today"]):
            return TemporalType.RELATIVE_DATE
        if any(word in phrase_lower for word in ["ago", "from now"]):
            return TemporalType.RELATIVE_TIME
        if any(word in phrase_lower for word in ["every", "daily", "weekly"]):
            return TemporalType.RECURRING
        if any(word in phrase_lower for word in ["by", "due", "before", "until"]):
            return TemporalType.DEADLINE
        if re.search(r"\d{1,2}:\d{2}", phrase):
            return TemporalType.ABSOLUTE_TIME
        if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", phrase):
            return TemporalType.ABSOLUTE_DATE

        return TemporalType.ABSOLUTE_DATE  # Default

    def extract_for_time_nodes(
        self,
        text: str,
        reference_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Extract temporal entities formatted for TIME node creation.

        This is a convenience method that returns data suitable for
        suggesting TIME node creation to the agent.

        Args:
            text: The text to extract from.
            reference_time: Reference time for resolution.

        Returns:
            List of dictionaries with TIME node suggestions.
        """
        entities = self.extract(text, reference_time)
        suggestions: list[dict[str, Any]] = []

        for entity in entities:
            if entity.normalized is None:
                continue

            suggestion = {
                "detected_text": entity.raw_text,
                "suggested_time_id": f"t-{entity.normalized.strftime('%Y%m%d%H%M')}",
                "suggested_title": f"Time: {entity.raw_text}",
                "scheduled_time": entity.normalized.isoformat(),
                "temporal_type": entity.temporal_type.value,
                "confidence": entity.confidence,
                "is_future": entity.is_future,
            }

            # Add recurrence info for recurring types
            if entity.temporal_type == TemporalType.RECURRING:
                recurrence = self._infer_recurrence(entity.raw_text)
                if recurrence is not None:
                    suggestion["recurrence"] = recurrence

            # Add deadline flag
            if entity.temporal_type == TemporalType.DEADLINE:
                suggestion["is_deadline"] = True

            suggestions.append(suggestion)

        return suggestions

    def _infer_recurrence(self, text: str) -> str | None:
        """Infer recurrence pattern from text."""
        text_lower = text.lower()

        if "daily" in text_lower or "every day" in text_lower:
            return "daily"
        if "weekly" in text_lower or "every week" in text_lower:
            return "weekly"
        if "monthly" in text_lower or "every month" in text_lower:
            return "monthly"
        if "yearly" in text_lower or "every year" in text_lower:
            return "yearly"
        if "biweekly" in text_lower or "bi-weekly" in text_lower:
            return "biweekly"

        # Check for specific day patterns
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for day in days:
            if f"every {day}" in text_lower:
                return f"weekly:{day}"

        return None
