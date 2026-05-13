"""Session management for the Cognifold service layer."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from cognifold.config import TraceConfig
from cognifold.graph.persistence import graph_to_dict
from cognifold.graph.store import ConceptGraph
from cognifold.scoring.ranker import ContextRanker, ScoringConfig
from cognifold.service.llm_keys import llm_key_scope, metrics_scope
from cognifold.service.models import GraphStatsResponse, SessionConfig, SessionInfo
from cognifold.service.stores.base import SessionStore
from cognifold.trace.collector import TraceCollector
from cognifold.utils.budget import LLMBudget
from cognifold.utils.llm_metrics import LLMMetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """A single user session owning a graph and associated resources."""

    session_id: str
    config: SessionConfig
    graph: ConceptGraph
    ranker: ContextRanker
    llm_api_keys: dict[str, str] = field(default_factory=dict)
    user_id: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    event_count: int = 0
    checkpoints: list[dict[str, Any]] = field(default_factory=list)

    # Observability components
    trace_collector: TraceCollector = field(default_factory=lambda: TraceCollector(TraceConfig()))
    llm_metrics: LLMMetricsCollector = field(default_factory=LLMMetricsCollector)
    budget: LLMBudget = field(default_factory=LLMBudget)

    # Lazy-loaded components
    agent: Any = field(default=None, repr=False)
    query_agent: Any = field(default=None, repr=False)
    graph_sync: Any = field(default=None, repr=False)

    def touch(self) -> None:
        """Update last_accessed timestamp."""
        self.last_accessed = datetime.now()

    def get_graph_stats(self) -> GraphStatsResponse:
        """Get current graph statistics."""
        from cognifold.models.node import NodeType

        events = len(self.graph.get_nodes_by_type(NodeType.EVENT))
        concepts = len(self.graph.get_nodes_by_type(NodeType.CONCEPT))
        intents = len(self.graph.get_nodes_by_type(NodeType.INTENT))
        time_nodes = len(self.graph.get_nodes_by_type(NodeType.TIME))

        return GraphStatsResponse(
            node_count=self.graph.node_count,
            edge_count=self.graph.edge_count,
            events=events,
            concepts=concepts,
            intents=intents,
            time_nodes=time_nodes,
        )

    def to_info(self) -> SessionInfo:
        """Convert to SessionInfo response model."""
        return SessionInfo(
            session_id=self.session_id,
            user_id=self.user_id,
            created_at=self.created_at.isoformat(),
            last_accessed=self.last_accessed.isoformat(),
            config=self.config,
            graph_stats=self.get_graph_stats(),
            event_count=self.event_count,
        )

    @contextmanager
    def llm_env(self) -> Iterator[None]:
        """Context manager that makes the session's API keys visible.

        Uses thread-local storage (via ``llm_key_scope``) so concurrent
        sessions in different threads no longer serialise on a global
        lock.  All LLM client code reads keys through
        ``get_api_key()`` which checks thread-local first, then falls
        back to ``os.environ``.
        """
        with llm_key_scope(self.llm_api_keys), metrics_scope(self.llm_metrics):
            yield


class SessionManager:
    """Manages the lifecycle of sessions."""

    def __init__(
        self,
        persist_dir: str | Path = "./sessions",
        max_sessions: int = 100,
        session_ttl_hours: float = 24.0,
        store: SessionStore | None = None,
        supabase_client: Any = None,
        enable_graph_sync: bool = False,
    ) -> None:
        self._sessions: dict[str, Session] = {}
        self._persist_dir = Path(persist_dir)
        self._max_sessions = max_sessions
        self._session_ttl_hours = session_ttl_hours
        self._lock = asyncio.Lock()
        self._store = store
        self._supabase_client = supabase_client
        self._enable_graph_sync = enable_graph_sync and supabase_client is not None

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)

    async def check_store_health(self) -> bool:
        """Check if the backing store is reachable.

        Returns True if no store is configured (in-memory only) or if
        the store's ``ping()`` succeeds.
        """
        if self._store is None:
            return True
        try:
            return await self._store.ping()
        except Exception:
            return False

    async def create_session(
        self,
        config: SessionConfig | None = None,
        llm_api_keys: dict[str, str] | None = None,
        user_id: str | None = None,
        graph_data: dict[str, Any] | None = None,
    ) -> Session:
        """Create a new session with optional initial graph data."""
        async with self._lock:
            if len(self._sessions) >= self._max_sessions:
                await self._evict_oldest()

            session_id = uuid.uuid4().hex[:16]
            cfg = config or SessionConfig()

            scoring_config = ScoringConfig(
                alpha=cfg.scoring_alpha,
                beta=cfg.scoring_beta,
                gamma=cfg.scoring_gamma,
                context_window_size=cfg.max_nodes,
            )

            graph = ConceptGraph()
            if graph_data is not None:
                graph = self._load_graph_from_dict(graph_data)

            graph_sync = None
            if self._enable_graph_sync:
                from cognifold.service.stores.graph_sync import GraphSyncWriter

                graph_sync = GraphSyncWriter(self._supabase_client, session_id)

            session = Session(
                session_id=session_id,
                config=cfg,
                graph=graph,
                ranker=ContextRanker(scoring_config),
                llm_api_keys=llm_api_keys or {},
                user_id=user_id,
                graph_sync=graph_sync,
            )
            self._sessions[session_id] = session
            logger.info(f"Created session {session_id}")
            return session

    async def get_session(self, session_id: str) -> Session | None:
        """Look up a session and update last_accessed.

        If not in memory but available in the persistence store,
        recovers the session (enables worker restart without loss).
        """
        session = self._sessions.get(session_id)
        if session is not None:
            session.touch()
            return session

        # Session-miss recovery from store
        if self._store is not None:
            data = await self._store.load_session(session_id)
            if data is not None:
                logger.info("Recovering session %s from store", session_id)
                cfg, graph, user_id, created_at, event_count = self._unpack_session_data(data)
                scoring_config = ScoringConfig(
                    alpha=cfg.scoring_alpha,
                    beta=cfg.scoring_beta,
                    gamma=cfg.scoring_gamma,
                    context_window_size=cfg.max_nodes,
                )
                # Restore llm_api_keys if persisted
                llm_api_keys: dict[str, str] = data.get("llm_api_keys", {})

                graph_sync = None
                if self._enable_graph_sync and self._supabase_client:
                    from cognifold.service.stores.graph_sync import GraphSyncWriter

                    graph_sync = GraphSyncWriter(self._supabase_client, session_id)

                session = Session(
                    session_id=session_id,
                    config=cfg,
                    graph=graph,
                    ranker=ContextRanker(scoring_config),
                    llm_api_keys=llm_api_keys,
                    user_id=user_id,
                    created_at=created_at or datetime.now(),
                    event_count=event_count,
                    graph_sync=graph_sync,
                )
                self._sessions[session_id] = session
                return session

        return None

    async def delete_session(self, session_id: str) -> bool:
        """Remove a session from memory and the backing store."""
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                return False

        # Delete from the backing store so session-miss recovery
        # doesn't resurrect it.
        if self._store is not None:
            await self._store.delete_session(session_id)
        else:
            # File-based fallback: remove persisted directory
            session_dir = self._persist_dir / session_id
            if session_dir.exists():
                import shutil

                shutil.rmtree(session_dir, ignore_errors=True)

        logger.info("Deleted session %s", session_id)
        return True

    async def load_graph_into_session(
        self, session_id: str, graph_data: dict[str, Any]
    ) -> Session | None:
        """Load a saved graph into an existing session, replacing its graph."""
        session = await self.get_session(session_id)
        if session is None:
            return None

        async with session.lock:
            session.graph = self._load_graph_from_dict(graph_data)
            session.agent = None
            session.query_agent = None
            session.touch()

            # Refresh imported nodes so they are visible to scoring:
            # - Set last_accessed to now (avoids recency decay to ~0)
            # - Ensure access_count >= 1 (avoids zero access score)
            # Note: must update NetworkX attrs directly since get_node()
            # returns a new object each time.
            now_iso = datetime.now().isoformat()
            for node_id in list(session.graph.internal_graph.nodes()):
                attrs = session.graph.internal_graph.nodes[node_id]
                attrs["last_accessed"] = now_iso
                if attrs.get("access_count", 0) == 0:
                    attrs["access_count"] = 1

            # Invalidate PageRank cache so next scoring recomputes
            # against the imported graph topology.
            session.ranker.invalidate_cache()

        await self._persist_session(session)
        return session

    async def persist_session_data(self, session_id: str) -> None:
        """Persist a single session's current state to the store.

        Fire-and-forget: errors are logged but not raised, so callers
        (e.g. event processing) are not affected by persistence failures.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return
        try:
            await self._persist_session(session)
        except Exception:
            logger.warning(
                "Failed to persist session %s (fire-and-forget)",
                session_id,
                exc_info=True,
            )

    async def persist_all(self) -> None:
        """Persist all active sessions."""
        for session in list(self._sessions.values()):
            async with session.lock:
                await self._persist_session(session)

    async def cleanup_expired(self) -> int:
        """Remove sessions past TTL. Returns count removed."""
        now = datetime.now()
        to_remove: list[str] = []

        for sid, session in self._sessions.items():
            hours = (now - session.last_accessed).total_seconds() / 3600.0
            if hours > self._session_ttl_hours:
                to_remove.append(sid)

        removed = 0
        for sid in to_remove:
            if await self.delete_session(sid):
                removed += 1
        return removed

    async def _evict_oldest(self) -> None:
        """Evict the oldest session (by last_accessed). Must hold _lock."""
        if not self._sessions:
            return
        oldest_id = min(self._sessions, key=lambda s: self._sessions[s].last_accessed)
        session = self._sessions.pop(oldest_id)
        await self._persist_session(session)
        logger.info("Evicted session %s", oldest_id)

    async def _persist_session(self, session: Session) -> None:
        """Persist a session via the store (async) or fall back to disk."""
        if self._store is not None:
            data = {
                "graph": graph_to_dict(session.graph),
                "config": session.config.model_dump(),
                "user_id": session.user_id,
                "created_at": session.created_at.isoformat(),
                "llm_api_keys": session.llm_api_keys,
                "event_count": session.event_count,
            }
            await self._store.save_session(session.session_id, data)
        else:
            self._persist_session_sync(session)

    def _persist_session_sync(self, session: Session) -> None:
        """Persist a session's graph + metadata to disk (synchronous fallback)."""
        import json as _json

        session_dir = self._persist_dir / session.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "graph": graph_to_dict(session.graph),
            "config": session.config.model_dump(),
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "llm_api_keys": session.llm_api_keys,
            "event_count": session.event_count,
        }
        meta_path = session_dir / "session.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            _json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("Persisted session %s to %s", session.session_id, meta_path)

    @staticmethod
    def _load_graph_from_dict(data: dict[str, Any]) -> ConceptGraph:
        """Load a ConceptGraph from a dictionary (same format as persistence)."""
        from cognifold.graph.persistence import dict_to_graph

        return dict_to_graph(data)

    @classmethod
    def _unpack_session_data(
        cls,
        data: dict[str, Any],
    ) -> tuple[SessionConfig, ConceptGraph, str | None, datetime | None, int]:
        """Unpack persisted session data (new or legacy format).

        New format: ``{"graph": ..., "config": ..., "user_id": ..., "created_at": ...}``
        Legacy format: flat graph dict (``{"version": ..., "nodes": ..., "edges": ...}``)

        Returns:
            (config, graph, user_id, created_at, event_count)
        """
        if "graph" in data and "config" in data:
            graph = cls._load_graph_from_dict(data["graph"])
            cfg = SessionConfig(**data["config"])
            user_id: str | None = data.get("user_id")
            created_at_str = data.get("created_at")
            created_at = datetime.fromisoformat(created_at_str) if created_at_str else None
            event_count: int = data.get("event_count", 0)
            return cfg, graph, user_id, created_at, event_count

        # Legacy: flat graph dict — use defaults
        graph = cls._load_graph_from_dict(data)
        return SessionConfig(), graph, None, None, 0
