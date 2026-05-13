"""Factory for creating session stores based on configuration."""

from __future__ import annotations

from cognifold.service.stores.base import SessionStore
from cognifold.service.stores.file_store import FileSessionStore


def create_store(
    backend: str = "file",
    persist_dir: str = "./sessions",
    redis_url: str = "redis://localhost:6379/0",
    ttl_seconds: int = 86400,
    supabase_url: str = "",
    supabase_key: str = "",
) -> SessionStore:
    """Create a session store based on backend type.

    Args:
        backend: Storage backend ("file", "redis", or "supabase").
        persist_dir: Directory for file-based storage.
        redis_url: Redis connection URL (for redis backend).
        ttl_seconds: TTL for Redis keys (default: 24h).
        supabase_url: Supabase project URL (for supabase backend).
        supabase_key: Supabase service-role or anon key (for supabase backend).

    Returns:
        A configured SessionStore instance.

    Raises:
        ValueError: If backend is not recognized.
    """
    if backend == "file":
        return FileSessionStore(persist_dir=persist_dir)
    elif backend == "redis":
        from cognifold.service.stores.redis_store import RedisSessionStore

        return RedisSessionStore(redis_url=redis_url, ttl_seconds=ttl_seconds)
    elif backend == "supabase":
        from cognifold.service.stores.supabase_store import SupabaseSessionStore

        return SupabaseSessionStore(supabase_url=supabase_url, supabase_key=supabase_key)
    else:
        raise ValueError(
            f"Unknown session backend: {backend!r}. Use 'file', 'redis', or 'supabase'."
        )
