"""FastAPI application factory for Cognifold."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import Depends, FastAPI, Request, Response

from cognifold.logging import bind_contextvars, clear_contextvars
from cognifold.service.auth import APIKeyValidator
from cognifold.service.session import SessionManager
from cognifold.service.tasks import TaskTracker

logger = logging.getLogger(__name__)


@dataclass
class AppSettings:
    """Settings for the Cognifold service."""

    persist_dir: str = "./sessions"
    max_sessions: int = 100
    session_ttl_hours: float = 24.0
    api_keys: set[str] | None = None  # None = auth disabled
    session_backend: str = "file"  # "file", "redis", or "supabase"
    redis_url: str = "redis://localhost:6379/0"
    supabase_url: str = ""
    supabase_key: str = ""
    enable_graph_sync: bool = False


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = settings or AppSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup: create session store
        from cognifold.service.stores.factory import create_store

        store = create_store(
            backend=settings.session_backend,
            persist_dir=settings.persist_dir,
            redis_url=settings.redis_url,
            supabase_url=settings.supabase_url,
            supabase_key=settings.supabase_key,
        )

        # Create Supabase client for graph sync + user identity
        supabase_client = None
        if settings.supabase_url and settings.supabase_key:
            try:
                from supabase import create_client  # pyright: ignore[reportMissingImports]

                supabase_client = create_client(settings.supabase_url, settings.supabase_key)
                logger.info("Supabase client initialized")
            except ImportError:
                logger.warning("supabase package not installed, skipping Supabase features")
            except Exception:
                logger.warning("Failed to create Supabase client", exc_info=True)
        app.state.supabase_client = supabase_client

        app.state.session_manager = SessionManager(
            persist_dir=settings.persist_dir,
            max_sessions=settings.max_sessions,
            session_ttl_hours=settings.session_ttl_hours,
            store=store,
            supabase_client=supabase_client,
            enable_graph_sync=settings.enable_graph_sync,
        )
        app.state.task_tracker = TaskTracker()

        # SSE broker for real-time streaming
        from cognifold.service.sse import SSEBroker

        app.state.sse_broker = SSEBroker()

        logger.info("Cognifold service started (backend=%s)", settings.session_backend)
        yield
        # Shutdown
        await app.state.session_manager.persist_all()
        await store.close()
        logger.info("Cognifold service stopped")

    app = FastAPI(
        title="Cognifold",
        description="Dynamic concept graph API for real-time event processing",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Auth dependency
    auth_validator = APIKeyValidator(valid_keys=settings.api_keys)

    # Include API routes (with auth)
    from cognifold.service.routes import api_router

    app.include_router(api_router, dependencies=[Depends(auth_validator)])

    # Brain memory coverage (public, no auth — consumed by the showcase site)
    from cognifold.service.routes.brain import router as brain_router

    app.include_router(brain_router, prefix="/api/v1")

    # Health endpoints (no auth)
    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        return {"status": "ok"}

    @app.get("/ready", tags=["health"])
    async def ready(request: Request) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        mgr: SessionManager = request.app.state.session_manager
        store_ok = await mgr.check_store_health()
        return {
            "status": "ok" if store_ok else "degraded",
            "active_sessions": mgr.active_session_count,
            "store_healthy": store_ok,
        }

    # Request context + logging middleware
    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next: object) -> Response:  # pyright: ignore[reportUnusedFunction]
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:8])

        # Bind context for structured logging (no-op without structlog)
        clear_contextvars()
        bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        start = time.time()
        response: Response = await call_next(request)  # type: ignore[misc]
        elapsed_ms = (time.time() - start) * 1000

        logger.debug(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "elapsed_ms": round(elapsed_ms, 1),
            },
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response

    return app
