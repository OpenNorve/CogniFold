"""PageRank score caching to avoid redundant recomputation."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph
    from cognifold.scoring.ranker import ContextRanker

logger = logging.getLogger(__name__)


class PageRankCache:
    """Caches PageRank scores keyed by graph topology fingerprint.

    The fingerprint is ``(node_count, edge_count)`` — a cheap proxy for
    "has the graph changed since the last computation?"  This eliminates
    the duplicate PageRank calls that happen when ``get_context_node_ids``
    and ``score_nodes`` are called back-to-back on the same graph state.
    """

    def __init__(self) -> None:
        self._scores: dict[str, float] = {}
        self._fingerprint: tuple[int, int] = (-1, -1)

    def get_or_compute(
        self,
        graph: ConceptGraph,
        ranker: ContextRanker,
        reference_time: datetime | None = None,
    ) -> dict[str, float]:
        """Return cached PageRank scores or recompute if the graph changed.

        Args:
            graph: The concept graph.
            ranker: The ranker that owns ``compute_pagerank``.
            reference_time: Reference time for edge-weight recency.

        Returns:
            Mapping of node ID → PageRank score.
        """
        fp = (graph.node_count, graph.edge_count)
        if fp == self._fingerprint and self._scores:
            return self._scores

        self._scores = ranker.compute_pagerank(graph, reference_time)
        self._fingerprint = fp
        logger.debug("PageRank cache miss — recomputed (%d nodes, %d edges)", fp[0], fp[1])
        return self._scores

    def invalidate(self) -> None:
        """Force the next call to recompute."""
        self._scores = {}
        self._fingerprint = (-1, -1)
