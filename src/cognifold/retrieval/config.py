"""Configuration for retrieval strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""

    KEYWORD = "keyword"  # BM25 only
    SEMANTIC = "semantic"  # Embedding similarity only
    HYBRID = "hybrid"  # Combined BM25 + semantic with RRF
    AGENTIC = "agentic"  # Multi-round retrieval with sufficiency check


@dataclass
class RetrievalConfig:
    """Configuration for retrieval operations.

    Attributes:
        strategy: The retrieval strategy to use.
        top_k: Maximum number of results to return.
        min_score: Minimum score threshold for results.

        # BM25 parameters
        bm25_k1: BM25 term frequency saturation parameter.
        bm25_b: BM25 length normalization parameter.

        # Hybrid parameters
        semantic_weight: Weight for semantic scores in hybrid mode (0.0-1.0).
        keyword_weight: Weight for BM25 scores in hybrid mode (0.0-1.0).
        rrf_k: RRF (Reciprocal Rank Fusion) constant (typically 60).

        # Filtering
        include_node_types: Only include these node types (None = all).
        exclude_node_types: Exclude these node types.
    """

    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = 10
    min_score: float = 0.0

    # BM25 parameters
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Hybrid parameters
    semantic_weight: float = 0.5
    keyword_weight: float = 0.5
    rrf_k: int = 60

    # Agentic retrieval parameters
    agentic_max_rounds: int = 2
    agentic_max_complementary_queries: int = 3
    agentic_sufficiency_threshold: float = 0.5

    # Filtering
    include_node_types: list[str] | None = None
    exclude_node_types: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.semantic_weight + self.keyword_weight > 0:
            # Normalize weights to sum to 1.0
            total = self.semantic_weight + self.keyword_weight
            self.semantic_weight = self.semantic_weight / total
            self.keyword_weight = self.keyword_weight / total

    @classmethod
    def for_keyword_search(cls, top_k: int = 10) -> RetrievalConfig:
        """Create config for keyword-only search."""
        return cls(
            strategy=RetrievalStrategy.KEYWORD,
            top_k=top_k,
            keyword_weight=1.0,
            semantic_weight=0.0,
        )

    @classmethod
    def for_semantic_search(cls, top_k: int = 10) -> RetrievalConfig:
        """Create config for semantic-only search."""
        return cls(
            strategy=RetrievalStrategy.SEMANTIC,
            top_k=top_k,
            keyword_weight=0.0,
            semantic_weight=1.0,
        )

    @classmethod
    def for_hybrid_search(
        cls,
        top_k: int = 10,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.5,
    ) -> RetrievalConfig:
        """Create config for hybrid search."""
        return cls(
            strategy=RetrievalStrategy.HYBRID,
            top_k=top_k,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
        )

    @classmethod
    def for_agentic_search(
        cls,
        top_k: int = 10,
        max_complementary_queries: int = 3,
    ) -> RetrievalConfig:
        """Create config for agentic multi-round search."""
        return cls(
            strategy=RetrievalStrategy.AGENTIC,
            top_k=top_k,
            agentic_max_complementary_queries=max_complementary_queries,
        )
