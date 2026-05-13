"""Thread-local LLM API key management.

Replaces the previous approach of setting/restoring process-global
environment variables under a ``threading.Lock``.  Each thread now
carries its own key overrides in ``threading.local()`` storage, so
concurrent sessions no longer serialise on a single lock.

Usage in the service layer (sessions / processors)::

    with llm_key_scope(session.llm_api_keys):
        # any code in this thread sees the session's keys
        client = genai.Client(api_key=get_api_key("GOOGLE_API_KEY"))

Usage in LLM client code::

    api_key = get_api_key("GOOGLE_API_KEY")   # thread-local first, then env var
    api_key = get_api_key("OPENAI_API_KEY")

CLI / non-service code that only sets env vars continues to work
because ``get_api_key`` falls back to ``os.environ.get()``.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_thread_local = threading.local()


def get_metrics_collector() -> Any:
    """Return the thread-local LLMMetricsCollector, or None."""
    return getattr(_thread_local, "llm_metrics", None)


@contextmanager
def metrics_scope(collector: Any) -> Iterator[None]:
    """Make *collector* visible to LLM call sites in this thread.

    Works like ``llm_key_scope`` — nested scopes are supported.
    """
    prev = getattr(_thread_local, "llm_metrics", None)
    _thread_local.llm_metrics = collector
    try:
        yield
    finally:
        _thread_local.llm_metrics = prev


# Canonical env-var names we manage
_PROVIDER_ENV_MAP: dict[str, str] = {
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def get_api_key(env_name: str) -> str | None:
    """Return the API key for *env_name*, checking thread-local first.

    Falls back to ``os.environ`` so CLI usage (no ``llm_key_scope``)
    keeps working unchanged.

    Args:
        env_name: The environment variable name, e.g. ``"GOOGLE_API_KEY"``.

    Returns:
        The key string, or ``None`` if not set anywhere.
    """
    # Thread-local override takes precedence
    keys: dict[str, str] | None = getattr(_thread_local, "llm_keys", None)
    if keys is not None:
        val = keys.get(env_name)
        if val:
            return val

    # Fall back to real env var (CLI, tests, etc.)
    return os.environ.get(env_name) or None


@contextmanager
def llm_key_scope(llm_api_keys: dict[str, str]) -> Iterator[None]:
    """Context manager that makes *llm_api_keys* visible to this thread.

    Keys are stored in ``threading.local()`` so they are invisible to
    other threads -- no lock required.

    Args:
        llm_api_keys: Mapping of provider name (``"google"``, ``"openai"``)
            to the actual API key value.  Provider names are normalised
            to canonical env-var names (``GOOGLE_API_KEY``, etc.).

    Example::

        with llm_key_scope({"google": "AIza...", "openai": "sk-..."}):
            key = get_api_key("GOOGLE_API_KEY")  # -> "AIza..."
    """
    # Normalise provider names to env-var names
    env_map: dict[str, str] = {}
    for provider, key in llm_api_keys.items():
        env_name = _PROVIDER_ENV_MAP.get(provider.lower())
        if env_name:
            env_map[env_name] = key

    prev: dict[str, str] | None = getattr(_thread_local, "llm_keys", None)
    # Merge with any existing scope (nested calls)
    merged = dict(prev) if prev else {}
    merged.update(env_map)
    _thread_local.llm_keys = merged
    try:
        yield
    finally:
        _thread_local.llm_keys = prev
