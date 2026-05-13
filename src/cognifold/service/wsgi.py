"""WSGI/ASGI entry point for Gunicorn.

Usage::

    cognifold serve --gunicorn --workers 4
"""

from __future__ import annotations

import os

from cognifold.logging import setup_logging
from cognifold.service.app import AppSettings, create_app


def _settings_from_env() -> AppSettings:
    """Build AppSettings from environment variables."""
    api_keys_raw = os.environ.get("COGNIFOLD_API_KEY", "")
    api_keys: set[str] | None = None
    if api_keys_raw:
        api_keys = {k.strip() for k in api_keys_raw.split(",") if k.strip()}

    return AppSettings(
        persist_dir=os.environ.get("COGNIFOLD_PERSIST_DIR", "./sessions"),
        max_sessions=int(os.environ.get("COGNIFOLD_MAX_SESSIONS", "100")),
        session_ttl_hours=float(os.environ.get("COGNIFOLD_SESSION_TTL_HOURS", "24")),
        api_keys=api_keys or None,
        session_backend=os.environ.get("COGNIFOLD_SESSION_BACKEND", "file"),
        redis_url=os.environ.get("COGNIFOLD_REDIS_URL", "redis://localhost:6379/0"),
        supabase_url=os.environ.get("COGNIFOLD_SUPABASE_URL", ""),
        supabase_key=os.environ.get("COGNIFOLD_SUPABASE_KEY", ""),
        enable_graph_sync=os.environ.get("COGNIFOLD_ENABLE_GRAPH_SYNC", "").lower()
        in ("1", "true", "yes"),
    )


# Configure structured logging before app creation
setup_logging()

# Create the ASGI application
app = create_app(_settings_from_env())
