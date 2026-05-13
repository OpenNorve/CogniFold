"""Pluggable session persistence stores.

Available backends:
- ``FileSessionStore`` — JSON files on disk (default)
- ``RedisSessionStore`` — Redis (requires ``cognifold[redis]``)
- ``SupabaseSessionStore`` — Supabase Postgres (requires ``cognifold[supabase]``)
"""

from cognifold.service.stores.base import SessionStore
from cognifold.service.stores.factory import create_store
from cognifold.service.stores.file_store import FileSessionStore

__all__ = ["FileSessionStore", "SessionStore", "create_store"]
