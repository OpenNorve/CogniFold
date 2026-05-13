"""Result types for retrieval operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cognifold.models.node import Node


@dataclass
class RetrievalResult:
    """A single retrieval result with scores from different methods.

    Attributes:
        node_id: ID of the retrieved node.
        final_score: Combined/final score for ranking.
        bm25_score: BM25 (keyword) score (None if not computed).
        semantic_score: Semantic similarity score (None if not computed).
        bm25_rank: Rank in BM25 results (None if not computed).
        semantic_rank: Rank in semantic results (None if not computed).
        node: The actual node object (may be lazy loaded).
        metadata: Additional result metadata.
    """

    node_id: str
    final_score: float
    bm25_score: float | None = None
    semantic_score: float | None = None
    bm25_rank: int | None = None
    semantic_rank: int | None = None
    node: Node | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "final_score": self.final_score,
            "bm25_score": self.bm25_score,
            "semantic_score": self.semantic_score,
            "bm25_rank": self.bm25_rank,
            "semantic_rank": self.semantic_rank,
            "metadata": self.metadata,
        }

    @classmethod
    def from_bm25(
        cls,
        node_id: str,
        score: float,
        rank: int,
        node: Node | None = None,
    ) -> RetrievalResult:
        """Create result from BM25 search."""
        return cls(
            node_id=node_id,
            final_score=score,
            bm25_score=score,
            bm25_rank=rank,
            node=node,
        )

    @classmethod
    def from_semantic(
        cls,
        node_id: str,
        score: float,
        rank: int,
        node: Node | None = None,
    ) -> RetrievalResult:
        """Create result from semantic search."""
        return cls(
            node_id=node_id,
            final_score=score,
            semantic_score=score,
            semantic_rank=rank,
            node=node,
        )


@dataclass
class RetrievalMetrics:
    """Metrics from a retrieval operation.

    Attributes:
        total_candidates: Total number of candidate nodes.
        bm25_candidates: Number of BM25 matches.
        semantic_candidates: Number of semantic matches.
        final_results: Number of final results returned.
        strategy_used: The retrieval strategy used.
    """

    total_candidates: int = 0
    bm25_candidates: int = 0
    semantic_candidates: int = 0
    final_results: int = 0
    strategy_used: str = ""
    degraded_to_bm25: bool = False

    def to_dict(self) -> dict[str, int | str | bool]:
        """Convert to dictionary."""
        return {
            "total_candidates": self.total_candidates,
            "bm25_candidates": self.bm25_candidates,
            "semantic_candidates": self.semantic_candidates,
            "final_results": self.final_results,
            "strategy_used": self.strategy_used,
            "degraded_to_bm25": self.degraded_to_bm25,
        }
