"""Hybrid retrieval combining BM25 and semantic search."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognifold.embeddings.embedder import NodeEmbedder
    from cognifold.embeddings.search import SemanticSearch
    from cognifold.graph.store import ConceptGraph

from cognifold.retrieval.bm25 import BM25Index
from cognifold.retrieval.config import RetrievalConfig, RetrievalStrategy
from cognifold.retrieval.result import RetrievalMetrics, RetrievalResult

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever combining BM25 and semantic search.

    Uses Reciprocal Rank Fusion (RRF) to combine rankings from
    different retrieval methods into a single ranked list.

    RRF formula:
        score(d) = sum(1 / (k + rank_i(d)))

    Where k is a constant (typically 60) and rank_i(d) is the rank
    of document d in result list i.

    Example:
        >>> retriever = HybridRetriever(embedder)
        >>> results = retriever.search(graph, "exercise fitness")
    """

    def __init__(
        self,
        embedder: NodeEmbedder | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        """Initialize hybrid retriever.

        Args:
            embedder: Node embedder for semantic search.
            config: Retrieval configuration.
        """
        self.config = config or RetrievalConfig()
        self._embedder = embedder

        # BM25 index
        self._bm25_index = BM25Index()

        # Semantic search (lazy initialized)
        self._semantic_search: SemanticSearch | None = None

        # Track indexed graph
        self._indexed_graph: ConceptGraph | None = None
        # Track graph revision at index build time (ConceptGraph is mutated in-place).
        self._indexed_revision: int | None = None

    def _ensure_semantic_search(self) -> SemanticSearch | None:
        """Ensure semantic search is initialized.

        Returns:
            SemanticSearch instance or None if no embedder.
        """
        if self._embedder is None:
            return None

        if self._semantic_search is None:
            from cognifold.embeddings.search import SemanticSearch

            self._semantic_search = SemanticSearch(self._embedder)

        return self._semantic_search

    def build_index(self, graph: ConceptGraph) -> None:
        """Build both BM25 and semantic indexes.

        Args:
            graph: The concept graph to index.
        """
        self._indexed_graph = graph
        self._indexed_revision = graph.revision

        # Build BM25 index
        self._bm25_index.build(graph)

        # Build semantic index if embedder available
        semantic = self._ensure_semantic_search()
        if semantic is not None:
            semantic.build_index(graph)

    def search(
        self,
        graph: ConceptGraph,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> tuple[list[RetrievalResult], RetrievalMetrics]:
        """Search using the configured strategy.

        Args:
            graph: The concept graph to search.
            query: The search query.
            config: Optional config override.

        Returns:
            Tuple of (results list, metrics).
        """
        cfg = config or self.config

        # Ensure indexes are built
        if self._indexed_graph is not graph or self._indexed_revision != graph.revision:
            self.build_index(graph)

        metrics = RetrievalMetrics(
            total_candidates=graph.node_count,
            strategy_used=cfg.strategy.value,
        )

        # Route to appropriate search method
        if cfg.strategy == RetrievalStrategy.KEYWORD:
            results = self._keyword_search(query, cfg)
            metrics.bm25_candidates = len(results)

        elif cfg.strategy == RetrievalStrategy.SEMANTIC:
            results = self._semantic_search_only(graph, query, cfg)
            metrics.semantic_candidates = len(results)

        elif cfg.strategy == RetrievalStrategy.HYBRID:
            results, bm25_count, semantic_count = self._hybrid_search(graph, query, cfg)
            metrics.bm25_candidates = bm25_count
            metrics.semantic_candidates = semantic_count
            if semantic_count == 0 and self._ensure_semantic_search() is None:
                metrics.degraded_to_bm25 = True

        else:
            results = []

        metrics.final_results = len(results)
        return results, metrics

    def _keyword_search(
        self,
        query: str,
        config: RetrievalConfig,
    ) -> list[RetrievalResult]:
        """Perform BM25 keyword search.

        Args:
            query: The search query.
            config: Retrieval configuration.

        Returns:
            List of results.
        """
        results = self._bm25_index.search(
            query,
            top_k=config.top_k,
            min_score=config.min_score,
            include_node_types=config.include_node_types,
            exclude_node_types=config.exclude_node_types,
        )

        return results

    def _semantic_search_only(
        self,
        graph: ConceptGraph,
        query: str,
        config: RetrievalConfig,
    ) -> list[RetrievalResult]:
        """Perform semantic-only search.

        Args:
            graph: The concept graph.
            query: The search query.
            config: Retrieval configuration.

        Returns:
            List of results.
        """
        semantic = self._ensure_semantic_search()
        if semantic is None:
            return []

        from cognifold.embeddings.search import SearchConfig

        sem_config = SearchConfig(
            top_k=config.top_k,
            min_score=config.min_score,
            include_node_types=config.include_node_types,
            exclude_node_types=config.exclude_node_types,
        )

        sem_results = semantic.search(graph, query, sem_config)

        # Convert to RetrievalResult
        results: list[RetrievalResult] = []
        for rank, sr in enumerate(sem_results):
            results.append(
                RetrievalResult(
                    node_id=sr.node_id,
                    final_score=sr.score,
                    semantic_score=sr.score,
                    semantic_rank=rank + 1,
                    node=sr.node,
                )
            )

        return results

    def _hybrid_search(
        self,
        graph: ConceptGraph,
        query: str,
        config: RetrievalConfig,
    ) -> tuple[list[RetrievalResult], int, int]:
        """Perform hybrid search with RRF fusion.

        Args:
            graph: The concept graph.
            query: The search query.
            config: Retrieval configuration.

        Returns:
            Tuple of (fused results, bm25 candidate count, semantic candidate count).
        """
        # Get BM25 results (more than top_k for better fusion)
        bm25_results = self._bm25_index.search(
            query,
            top_k=config.top_k * 2,  # Get more for fusion
            include_node_types=config.include_node_types,
            exclude_node_types=config.exclude_node_types,
        )

        # Get semantic results
        semantic = self._ensure_semantic_search()
        semantic_results: list[RetrievalResult] = []
        if semantic is None:
            logger.warning(
                "Hybrid search degraded to BM25-only: no embedder configured. "
                "Set OPENAI_API_KEY or GOOGLE_API_KEY for semantic search."
            )
        if semantic is not None:
            from cognifold.embeddings.search import SearchConfig

            sem_config = SearchConfig(
                top_k=config.top_k * 2,
                include_node_types=config.include_node_types,
                exclude_node_types=config.exclude_node_types,
            )
            sem_results = semantic.search(graph, query, sem_config)

            for rank, sr in enumerate(sem_results):
                semantic_results.append(
                    RetrievalResult(
                        node_id=sr.node_id,
                        final_score=sr.score,
                        semantic_score=sr.score,
                        semantic_rank=rank + 1,
                        node=sr.node,
                    )
                )

        # Fuse results using RRF
        fused = self._rrf_fusion(
            bm25_results,
            semantic_results,
            config,
        )

        # Apply min_score filter and take top_k
        filtered = [r for r in fused if r.final_score >= config.min_score]
        return filtered[: config.top_k], len(bm25_results), len(semantic_results)

    def _rrf_fusion(
        self,
        bm25_results: list[RetrievalResult],
        semantic_results: list[RetrievalResult],
        config: RetrievalConfig,
    ) -> list[RetrievalResult]:
        """Fuse results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank_i))

        Args:
            bm25_results: Results from BM25 search.
            semantic_results: Results from semantic search.
            config: Configuration with RRF k constant.

        Returns:
            Fused and re-ranked results.
        """
        k = config.rrf_k
        scores: dict[str, float] = {}
        result_data: dict[str, RetrievalResult] = {}

        # Process BM25 results
        for result in bm25_results:
            node_id = result.node_id
            rank = result.bm25_rank or len(bm25_results) + 1

            rrf_score = config.keyword_weight / (k + rank)
            scores[node_id] = scores.get(node_id, 0.0) + rrf_score

            if node_id not in result_data:
                result_data[node_id] = RetrievalResult(
                    node_id=node_id,
                    final_score=0.0,
                    bm25_score=result.bm25_score,
                    bm25_rank=result.bm25_rank,
                    node=result.node,
                )
            else:
                result_data[node_id].bm25_score = result.bm25_score
                result_data[node_id].bm25_rank = result.bm25_rank

        # Process semantic results
        for result in semantic_results:
            node_id = result.node_id
            rank = result.semantic_rank or len(semantic_results) + 1

            rrf_score = config.semantic_weight / (k + rank)
            scores[node_id] = scores.get(node_id, 0.0) + rrf_score

            if node_id not in result_data:
                result_data[node_id] = RetrievalResult(
                    node_id=node_id,
                    final_score=0.0,
                    semantic_score=result.semantic_score,
                    semantic_rank=result.semantic_rank,
                    node=result.node,
                )
            else:
                result_data[node_id].semantic_score = result.semantic_score
                result_data[node_id].semantic_rank = result.semantic_rank
                if result_data[node_id].node is None:
                    result_data[node_id].node = result.node

        # Build final results
        fused_results: list[RetrievalResult] = []
        for node_id, score in scores.items():
            result = result_data[node_id]
            result.final_score = score
            fused_results.append(result)

        # Sort by fused score
        fused_results.sort(key=lambda r: r.final_score, reverse=True)

        return fused_results

    def invalidate_indexes(self) -> None:
        """Invalidate all indexes."""
        self._indexed_graph = None
        self._indexed_revision = None
        self._bm25_index = BM25Index()

        if self._semantic_search is not None:
            self._semantic_search.invalidate_index()

    def update_node(self, graph: ConceptGraph, node_id: str) -> None:
        """Update indexes for a single node.

        Args:
            graph: The concept graph.
            node_id: ID of the node to update.
        """
        node = graph.get_node_or_none(node_id)

        if node is None:
            # Node was deleted
            self._bm25_index.remove_document(node_id)
            if self._semantic_search is not None:
                self._semantic_search.remove_node_from_index(node_id)
        else:
            # Node was added or updated
            self._bm25_index.add_document(node)
            if self._semantic_search is not None:
                self._semantic_search.update_node_embedding(graph, node_id)
        # Keep revision in sync when indexes are updated incrementally.
        self._indexed_graph = graph
        self._indexed_revision = graph.revision

    def get_index_stats(self) -> dict[str, int]:
        """Get statistics about the indexes.

        Returns:
            Dictionary with index statistics.
        """
        stats = {
            "bm25_documents": self._bm25_index.get_document_count(),
            "bm25_vocabulary": self._bm25_index.get_vocabulary_size(),
        }

        if self._semantic_search is not None:
            stats["semantic_index_size"] = self._semantic_search.get_index_size()

        return stats
