"""Structured LLM call logging and metrics collection.

Every LLM call (agent plan, query answer, batch enrichment, summarization,
community labeling, action proposal) should be recorded through
:class:`LLMMetricsCollector` so that downstream budget enforcement and
observability dashboards can consume usage data.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

# ---- Cost table (USD per 1 M tokens) ----
# Prices are rough estimates; update as vendors change pricing.
_COST_PER_M_TOKENS: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    # Google Gemini
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-3-flash-preview": {"input": 0.10, "output": 0.40},
    # Anthropic
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-opus-4": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
}

# Fallback for unknown models
_DEFAULT_COST_PER_M: dict[str, float] = {"input": 1.00, "output": 3.00}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Return a rough USD cost estimate for a single LLM call.

    The function performs a *prefix match* against known model names so that
    versioned variants (e.g. ``gpt-4o-2024-08-06``) are handled gracefully.
    """
    pricing = _resolve_pricing(model)
    cost_in = tokens_in * pricing["input"] / 1_000_000
    cost_out = tokens_out * pricing["output"] / 1_000_000
    return round(cost_in + cost_out, 8)


def _resolve_pricing(model: str) -> dict[str, float]:
    """Find the best-match pricing entry for *model*."""
    # Strip common prefixes that wrap actual model names
    normalized = model.replace("openai:", "").replace("google:", "")

    # Exact match first
    if normalized in _COST_PER_M_TOKENS:
        return _COST_PER_M_TOKENS[normalized]

    # Prefix match (longest prefix wins)
    best: str | None = None
    for key in _COST_PER_M_TOKENS:
        if normalized.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    if best is not None:
        return _COST_PER_M_TOKENS[best]

    return _DEFAULT_COST_PER_M


@dataclass
class LLMCallMetrics:
    """Structured record of a single LLM invocation."""

    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_estimate: float  # rough USD estimate based on model pricing
    call_type: str  # "agent_plan", "query_answer", "batch_enrichment", "summarize", etc.
    session_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class LLMMetricsCollector:
    """Thread-safe collector for LLM call metrics within a session.

    Usage::

        collector = LLMMetricsCollector()
        collector.record(LLMCallMetrics(
            model="gemini-1.5-flash",
            tokens_in=200,
            tokens_out=800,
            latency_ms=430.0,
            cost_estimate=estimate_cost("gemini-1.5-flash", 200, 800),
            call_type="summarize",
        ))
        summary = collector.get_usage_summary()
    """

    def __init__(self) -> None:
        self._calls: list[LLMCallMetrics] = []
        self._lock = threading.Lock()

    # -- mutation ----------------------------------------------------------

    def record(self, metrics: LLMCallMetrics) -> None:
        """Append a metrics record (thread-safe)."""
        with self._lock:
            self._calls.append(metrics)

    def reset(self) -> None:
        """Discard all recorded calls."""
        with self._lock:
            self._calls.clear()

    # -- queries -----------------------------------------------------------

    def get_calls(self) -> list[LLMCallMetrics]:
        """Return a shallow copy of all recorded calls."""
        with self._lock:
            return list(self._calls)

    @property
    def total_calls(self) -> int:
        with self._lock:
            return len(self._calls)

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return sum(c.tokens_in + c.tokens_out for c in self._calls)

    @property
    def total_cost(self) -> float:
        with self._lock:
            return sum(c.cost_estimate for c in self._calls)

    def get_usage_summary(self) -> dict[str, object]:
        """Aggregate usage across all recorded calls.

        Returns a dict with keys:
        - ``total_calls``: number of LLM invocations
        - ``total_tokens_in``: sum of input tokens
        - ``total_tokens_out``: sum of output tokens
        - ``total_tokens``: sum of all tokens
        - ``total_cost``: sum of cost estimates (USD)
        - ``total_latency_ms``: cumulative latency
        - ``calls_by_model``: ``{model: {calls, tokens_in, tokens_out, cost}}``
        - ``calls_by_type``: ``{call_type: count}``
        """
        with self._lock:
            calls = list(self._calls)

        tokens_in = sum(c.tokens_in for c in calls)
        tokens_out = sum(c.tokens_out for c in calls)

        by_model: dict[str, dict[str, float]] = {}
        by_type: dict[str, int] = {}

        for c in calls:
            # By model
            entry = by_model.setdefault(
                c.model, {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost": 0.0}
            )
            entry["calls"] += 1
            entry["tokens_in"] += c.tokens_in
            entry["tokens_out"] += c.tokens_out
            entry["cost"] += c.cost_estimate

            # By type
            by_type[c.call_type] = by_type.get(c.call_type, 0) + 1

        return {
            "total_calls": len(calls),
            "total_tokens_in": tokens_in,
            "total_tokens_out": tokens_out,
            "total_tokens": tokens_in + tokens_out,
            "total_cost": round(sum(c.cost_estimate for c in calls), 8),
            "total_latency_ms": round(sum(c.latency_ms for c in calls), 2),
            "calls_by_model": by_model,
            "calls_by_type": by_type,
        }
