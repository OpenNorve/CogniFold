"""Brain memory coverage: the single source of truth for how much of human
brain memory CogniFold has modeled.

Exposes :func:`get_coverage` which loads the curated taxonomy from
``memory_coverage.json`` and recomputes the overall coverage percentage.
"""

from __future__ import annotations

from cognifold.brain.coverage import get_coverage

__all__ = ["get_coverage"]
