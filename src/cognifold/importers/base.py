"""Base interface for data importers.

Importers convert external data sources (wiki files, browser history, chat logs, etc.)
into Cognifold event timelines. Unlike generators (which use LLMs to synthesize events),
importers transform existing data into the unified event schema.

Key Terminology:
- Generator: Creates synthetic events using LLMs (personal timeline, computer activity)
- Importer: Converts external data to events (wiki, browser history, logs)
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

# Type variable for importer settings
TSettings = TypeVar("TSettings")


@dataclass
class ImportResult:
    """Result of an import operation.

    Attributes:
        timeline: The generated timeline dictionary with events.
        files_scanned: Number of files examined.
        items_processed: Number of items successfully processed.
        events_emitted: Number of events created.
        skipped_items: List of items that were skipped with reasons.
        warnings: Any warnings generated during import.
    """

    timeline: dict[str, Any]
    files_scanned: int = 0
    items_processed: int = 0
    events_emitted: int = 0
    skipped_items: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class BaseImporter(ABC, Generic[TSettings]):
    """Abstract base class for data importers.

    All data importers (wiki, browser history, chat logs, etc.) should inherit
    from this class and implement the required abstract methods.

    The base class provides:
    - Common result formatting
    - Timeline saving utilities
    - Event ID generation
    - Consistent configuration handling

    Subclasses must implement:
    - import_data(): Main import entry point
    - _process_item(): Process a single item (file, record, etc.)
    - get_default_settings(): Return default settings for this importer

    Example:
        >>> class MyImporter(BaseImporter[MySettings]):
        ...     source_name = "my-source"
        ...     def import_data(self, input_path, settings=None): ...
        >>> importer = MyImporter()
        >>> result = importer.import_data("/path/to/data")
    """

    # Subclasses should override this to identify the event source
    source_name: str = "unknown"

    # Default event type for this importer
    default_event_type: str = "import_chunk"

    @abstractmethod
    def import_data(
        self,
        input_path: str | Path,
        settings: TSettings | None = None,
        specific_items: list[str] | None = None,
    ) -> ImportResult:
        """Import data from the specified source.

        Args:
            input_path: Path to the input data (directory or file).
            settings: Importer-specific settings. If None, uses defaults.
            specific_items: Optional list of specific items to import.

        Returns:
            ImportResult containing the timeline and statistics.
        """
        ...

    @abstractmethod
    def get_default_settings(self) -> TSettings:
        """Get default settings for this importer.

        Returns:
            Default settings instance for this importer type.
        """
        ...

    def _generate_event_id(self, prefix: str = "i") -> str:
        """Generate a unique event ID.

        Args:
            prefix: Prefix for the event ID (default: 'i' for import).

        Returns:
            A unique event ID string.
        """
        return f"{prefix}-{uuid.uuid4().hex[:8]}"

    def _create_event(
        self,
        event_id: str,
        timestamp: datetime,
        title: str,
        description: str = "",
        event_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a standardized event dictionary.

        Args:
            event_id: Unique event identifier.
            timestamp: Event timestamp.
            title: Short title/summary.
            description: Detailed description (often the content).
            event_type: Type of event (defaults to importer's default).
            metadata: Additional metadata.
            context: Domain-specific context data.

        Returns:
            Event dictionary following the unified schema.
        """
        return {
            "event_id": event_id,
            "timestamp": timestamp.isoformat(),
            "source": self.source_name,
            "event_type": event_type or self.default_event_type,
            "title": title,
            "description": description,
            "metadata": metadata or {},
            "context": context or {},
        }

    def _create_timeline(
        self,
        events: list[dict[str, Any]],
        timeline_id: str | None = None,
        description: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a timeline dictionary from events.

        Args:
            events: List of event dictionaries.
            timeline_id: Optional ID for the timeline.
            description: Optional description.
            extra_metadata: Optional extra metadata.

        Returns:
            Timeline dictionary with events and metadata.
        """
        timeline: dict[str, Any] = {
            "timeline_id": timeline_id or f"{self.source_name}-{uuid.uuid4().hex[:8]}",
            "source": self.source_name,
            "description": description or f"Imported from {self.source_name}",
            "imported_at": datetime.now().isoformat(),
            "events": events,
        }

        if extra_metadata:
            timeline.update(extra_metadata)

        return timeline

    def save_timeline(
        self,
        result: ImportResult,
        path: str | Path,
    ) -> Path:
        """Save import result to a timeline JSON file.

        Args:
            result: Import result containing the timeline.
            path: Path to save the timeline.

        Returns:
            Path to the saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.timeline, f, indent=2, ensure_ascii=False)

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

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        events = data.get("events", [])
        metadata = {k: v for k, v in data.items() if k != "events"}

        return events, metadata
