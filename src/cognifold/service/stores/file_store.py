"""File-based session persistence store."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from functools import partial
from pathlib import Path
from typing import Any

from cognifold.service.stores.base import SessionStore

logger = logging.getLogger(__name__)


class FileSessionStore(SessionStore):
    """Persists sessions as JSON files on disk.

    Directory layout::

        <persist_dir>/
        ├── <session_id>/
        │   └── graph.json
        └── ...
    """

    def __init__(self, persist_dir: str | Path) -> None:
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

    async def save_session(self, session_id: str, data: dict[str, Any]) -> None:
        """Save graph data to a JSON file."""
        session_dir = self._persist_dir / session_id
        graph_path = session_dir / "graph.json"

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, partial(self._write_sync, graph_path, data))
        logger.info("Persisted session %s to %s", session_id, graph_path)

    async def load_session(self, session_id: str) -> dict[str, Any] | None:
        """Load graph data from a JSON file."""
        graph_path = self._persist_dir / session_id / "graph.json"
        if not graph_path.exists():
            return None

        loop = asyncio.get_running_loop()
        data: dict[str, Any] = await loop.run_in_executor(
            None, partial(self._read_sync, graph_path)
        )
        return data

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session's directory."""
        session_dir = self._persist_dir / session_id
        if not session_dir.exists():
            return False

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, partial(shutil.rmtree, session_dir))
        return True

    async def list_sessions(self) -> list[str]:
        """List session IDs by scanning directories."""
        sessions: list[str] = []
        if not self._persist_dir.exists():
            return sessions
        for p in self._persist_dir.iterdir():
            if p.is_dir() and (p / "graph.json").exists():
                sessions.append(p.name)
        return sessions

    async def ping(self) -> bool:
        """Check if the persist directory is accessible and writable."""
        return self._persist_dir.exists() and os.access(self._persist_dir, os.W_OK)

    @staticmethod
    def _write_sync(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _read_sync(path: Path) -> dict[str, Any]:
        with open(path, encoding="utf-8") as f:
            result: dict[str, Any] = json.load(f)
            return result
