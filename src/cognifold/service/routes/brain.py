"""Brain memory coverage endpoint.

Serves the single source of truth for how much of human brain memory CogniFold
has modeled, so the showcase site and docs render from one canonical figure.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

router = APIRouter(prefix="/brain", tags=["brain"])


@router.get("/coverage")
async def get_brain_coverage() -> dict[str, Any]:
    """Return the brain memory coverage taxonomy with the computed overall percentage."""
    from cognifold.brain.coverage import get_coverage

    return get_coverage()
