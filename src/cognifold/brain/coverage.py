"""Load and serve the brain memory coverage taxonomy.

``memory_coverage.json`` is the single source of truth for how much of human
brain memory CogniFold has modeled. This module loads it, recomputes the
weighted ``overall_coverage_pct`` from the systems list (so the published JSON
can never silently drift from the underlying scores), and exposes the result
via :func:`get_coverage`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

_PACKAGE = "cognifold.brain"
_DATA_FILE = "memory_coverage.json"


def _load_raw() -> dict[str, Any]:
    """Read the raw coverage JSON shipped alongside this module."""
    text = resources.files(_PACKAGE).joinpath(_DATA_FILE).read_text(encoding="utf-8")
    return json.loads(text)


def compute_overall_pct(systems: list[dict[str, Any]], status_scores: dict[str, float]) -> int:
    """Recompute the weighted coverage percentage from systems + status scores.

    pct = round(100 * sum(status_score(status) * weight) / sum(weight))
    """
    total_weight = sum(float(s["weight"]) for s in systems)
    if total_weight == 0:
        return 0
    weighted = sum(status_scores[s["status"]] * float(s["weight"]) for s in systems)
    return round(100 * weighted / total_weight)


@lru_cache(maxsize=1)
def get_coverage() -> dict[str, Any]:
    """Return the brain memory coverage dict with a recomputed ``overall_coverage_pct``.

    The returned value is the published JSON with ``overall_coverage_pct``
    overwritten by the freshly computed figure, guaranteeing consistency.
    """
    data = _load_raw()
    computed = compute_overall_pct(data["systems"], data["status_scores"])
    data["overall_coverage_pct"] = computed
    return data


def _self_check() -> None:
    """Assert the published JSON matches the recomputed percentage."""
    raw = _load_raw()
    computed = compute_overall_pct(raw["systems"], raw["status_scores"])
    published = raw["overall_coverage_pct"]
    assert published == computed, (
        f"memory_coverage.json overall_coverage_pct={published} "
        f"but recomputed value is {computed}; update the JSON to match."
    )


# Validate consistency at import time so a stale JSON fails loudly.
_self_check()


if __name__ == "__main__":
    cov = get_coverage()
    print(f"overall_coverage_pct = {cov['overall_coverage_pct']}")
    for system in cov["systems"]:
        print(f"  {system['status']:8} w={system['weight']:<4} {system['name']}")
