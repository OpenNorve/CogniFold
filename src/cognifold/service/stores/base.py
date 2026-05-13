"""Abstract base class for session persistence stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SessionStore(ABC):
    """Abstract interface for session persistence backends.

    Stores handle saving/loading serialized graph data. Sessions remain
    in-memory for active use; stores handle persistence and recovery only.
    """

    @abstractmethod
    async def save_session(self, session_id: str, data: dict[str, Any]) -> None:
        """Persist a session's serialized graph data.

        Args:
            session_id: Unique session identifier.
            data: Serialized graph data (from ``graph_to_dict``).
        """

    @abstractmethod
    async def load_session(self, session_id: str) -> dict[str, Any] | None:
        """Load a session's serialized graph data.

        Args:
            session_id: Unique session identifier.

        Returns:
            Serialized graph data, or None if not found.
        """

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session's persisted data.

        Args:
            session_id: Unique session identifier.

        Returns:
            True if the session was found and deleted.
        """

    @abstractmethod
    async def list_sessions(self) -> list[str]:
        """List all persisted session IDs.

        Returns:
            List of session IDs.
        """

    async def ping(self) -> bool:
        """Check if the store backend is reachable.

        Returns:
            True if the store is healthy, False otherwise.
        """
        return True

    async def close(self) -> None:  # noqa: B027
        """Clean up resources. Override if needed."""
