"""LLM budget enforcement.

Provides a simple check-before-call pattern: before each LLM invocation the
caller asks :meth:`BudgetEnforcer.check` which raises
:class:`BudgetExceededError` if cumulative usage has breached any configured
limit.

A limit value of ``0`` (the default) means *unlimited* for that dimension.
"""

from __future__ import annotations

from dataclasses import dataclass

from cognifold.utils.llm_metrics import LLMMetricsCollector


class BudgetExceededError(Exception):
    """Raised when an LLM budget limit has been exceeded."""

    def __init__(self, dimension: str, limit: float, current: float) -> None:
        self.dimension = dimension
        self.limit = limit
        self.current = current
        super().__init__(f"LLM budget exceeded on '{dimension}': limit={limit}, current={current}")


@dataclass
class LLMBudget:
    """Configurable budget caps for LLM usage.

    Any field set to ``0`` (the default) means *no limit* for that dimension.
    """

    max_tokens: int = 0  # 0 = unlimited
    max_cost: float = 0.0  # 0 = unlimited (USD)
    max_calls: int = 0  # 0 = unlimited


class BudgetEnforcer:
    """Checks budget before each LLM call.

    Usage::

        enforcer = BudgetEnforcer(
            budget=LLMBudget(max_tokens=100_000, max_cost=1.0),
            collector=collector,
        )
        enforcer.check()  # raises BudgetExceededError if over
    """

    def __init__(self, budget: LLMBudget, collector: LLMMetricsCollector) -> None:
        self._budget = budget
        self._collector = collector

    @property
    def budget(self) -> LLMBudget:
        return self._budget

    def check(self) -> None:
        """Raise :class:`BudgetExceededError` if any limit is breached."""
        b = self._budget

        if b.max_calls > 0:
            current_calls = self._collector.total_calls
            if current_calls >= b.max_calls:
                raise BudgetExceededError("calls", b.max_calls, current_calls)

        if b.max_tokens > 0:
            current_tokens = self._collector.total_tokens
            if current_tokens >= b.max_tokens:
                raise BudgetExceededError("tokens", b.max_tokens, current_tokens)

        if b.max_cost > 0:
            current_cost = self._collector.total_cost
            if current_cost >= b.max_cost:
                raise BudgetExceededError("cost", b.max_cost, current_cost)

    def remaining(self) -> dict[str, float | None]:
        """Return remaining budget for each dimension.

        Returns ``None`` for a dimension when the limit is 0 (unlimited).
        """
        b = self._budget
        result: dict[str, float | None] = {}

        if b.max_tokens > 0:
            result["tokens_remaining"] = max(0, b.max_tokens - self._collector.total_tokens)
        else:
            result["tokens_remaining"] = None

        if b.max_cost > 0:
            result["cost_remaining"] = round(max(0.0, b.max_cost - self._collector.total_cost), 8)
        else:
            result["cost_remaining"] = None

        if b.max_calls > 0:
            result["calls_remaining"] = max(0, b.max_calls - self._collector.total_calls)
        else:
            result["calls_remaining"] = None

        return result
