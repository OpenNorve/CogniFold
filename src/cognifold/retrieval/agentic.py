"""Agentic multi-round retrieval with LLM sufficiency checking.

Implements a two-round retrieval pipeline inspired by EverMemOS:
- Round 1: Hybrid search + LLM sufficiency check
- Round 2 (if insufficient): LLM generates complementary queries,
  parallel hybrid search, multi-list RRF fusion

Falls back to simple hybrid search if no LLM API key is available.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Callable

from cognifold.retrieval.config import RetrievalConfig, RetrievalStrategy
from cognifold.retrieval.hybrid import HybridRetriever
from cognifold.retrieval.result import RetrievalMetrics, RetrievalResult

if TYPE_CHECKING:
    from cognifold.embeddings.embedder import NodeEmbedder
    from cognifold.graph.store import ConceptGraph

logger = logging.getLogger(__name__)


class AgenticRetriever:
    """Multi-round agentic retriever with sufficiency checking.

    Round 1: Performs hybrid (BM25 + semantic) search and asks an LLM
    whether the results are sufficient to answer the query.

    Round 2 (triggered when Round 1 is insufficient): Asks the LLM to
    generate 2-3 complementary queries, runs hybrid search for each,
    then fuses all result lists with multi-list RRF.

    Falls back to plain hybrid search when no LLM is available.

    Example:
        >>> retriever = AgenticRetriever(embedder=embedder)
        >>> results, metrics = retriever.search(graph, "exercise habits")
    """

    def __init__(
        self,
        embedder: NodeEmbedder | None = None,
        config: RetrievalConfig | None = None,
        llm_caller: Callable[[str], str] | None = None,
    ) -> None:
        """Initialize agentic retriever.

        Args:
            embedder: Node embedder for semantic search.
            config: Retrieval configuration.
            llm_caller: Optional callable(prompt) -> response for LLM calls.
                       If None, tries OpenAI/Gemini from environment.
        """
        self.config = config or RetrievalConfig.for_agentic_search()
        self._embedder = embedder
        self._llm_caller = llm_caller

        # Underlying hybrid retriever for the actual search
        hybrid_config = RetrievalConfig(
            strategy=RetrievalStrategy.HYBRID,
            top_k=self.config.top_k,
            min_score=self.config.min_score,
            bm25_k1=self.config.bm25_k1,
            bm25_b=self.config.bm25_b,
            semantic_weight=self.config.semantic_weight,
            keyword_weight=self.config.keyword_weight,
            rrf_k=self.config.rrf_k,
            include_node_types=self.config.include_node_types,
            exclude_node_types=self.config.exclude_node_types,
        )
        self._hybrid = HybridRetriever(embedder=embedder, config=hybrid_config)

    def search(
        self,
        graph: ConceptGraph,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> tuple[list[RetrievalResult], RetrievalMetrics]:
        """Execute agentic multi-round search.

        Args:
            graph: The concept graph to search.
            query: The search query.
            config: Optional config override.

        Returns:
            Tuple of (results list, metrics).
        """
        cfg = config or self.config
        metrics = RetrievalMetrics(
            total_candidates=graph.node_count,
            strategy_used="agentic",
        )

        # Round 1: Hybrid search (forward config overrides)
        hybrid_cfg = RetrievalConfig(
            strategy=RetrievalStrategy.HYBRID,
            top_k=cfg.top_k,
            min_score=cfg.min_score,
            bm25_k1=cfg.bm25_k1,
            bm25_b=cfg.bm25_b,
            semantic_weight=cfg.semantic_weight,
            keyword_weight=cfg.keyword_weight,
            rrf_k=cfg.rrf_k,
            include_node_types=cfg.include_node_types,
            exclude_node_types=cfg.exclude_node_types,
        )
        r1_results, r1_metrics = self._hybrid.search(graph, query, hybrid_cfg)
        metrics.bm25_candidates = r1_metrics.bm25_candidates
        metrics.semantic_candidates = r1_metrics.semantic_candidates

        if not r1_results:
            metrics.final_results = 0
            return [], metrics

        # Sufficiency check (skip if no LLM available)
        is_sufficient = True
        try:
            is_sufficient = self._check_sufficiency(query, r1_results, cfg)
        except Exception as e:
            logger.debug("Sufficiency check failed, assuming sufficient: %s", e)
            is_sufficient = True  # Conservative fallback

        if is_sufficient:
            metrics.final_results = len(r1_results)
            return r1_results, metrics

        # Round 2: Generate complementary queries and search
        logger.debug("Round 1 insufficient, generating complementary queries")
        try:
            complementary_queries = self._generate_complementary_queries(query, r1_results, cfg)
        except Exception as e:
            logger.debug("Complementary query generation failed: %s", e)
            metrics.final_results = len(r1_results)
            return r1_results, metrics

        if not complementary_queries:
            metrics.final_results = len(r1_results)
            return r1_results, metrics

        # Search with each complementary query
        all_result_lists: list[list[RetrievalResult]] = [r1_results]
        for cq in complementary_queries:
            try:
                cq_results, _ = self._hybrid.search(graph, cq, hybrid_cfg)
                if cq_results:
                    all_result_lists.append(cq_results)
            except Exception as e:
                logger.debug("Complementary search failed for '%s': %s", cq, e)

        # Multi-list RRF fusion
        fused = self._multi_rrf_fusion(all_result_lists, cfg)

        # Apply min_score filter and top_k
        filtered = [r for r in fused if r.final_score >= cfg.min_score]
        final = filtered[: cfg.top_k]

        metrics.final_results = len(final)
        return final, metrics

    def _check_sufficiency(
        self,
        query: str,
        results: list[RetrievalResult],
        config: RetrievalConfig,
    ) -> bool:
        """Check if Round 1 results are sufficient using LLM.

        Args:
            query: The original query.
            results: Round 1 results.
            config: Retrieval configuration.

        Returns:
            True if results are sufficient.
        """
        from cognifold.query.prompts import format_sufficiency_prompt

        result_dicts = []
        for r in results[:10]:  # Limit to top 10 for prompt
            node_data: dict[str, str] = {"title": r.node_id}
            if r.node is not None:
                node_data["title"] = r.node.data.get("title", r.node_id)
                node_data["description"] = r.node.data.get("description", "")
                node_data["node_type"] = r.node.type.value
            result_dicts.append(node_data)

        prompt = format_sufficiency_prompt(query, result_dicts)
        response = self._call_llm(prompt)

        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                sufficient = data.get("sufficient", True)
                confidence = data.get("confidence", 1.0)

                # Only treat as insufficient if LLM is confident
                return not (not sufficient and confidence >= config.agentic_sufficiency_threshold)
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug("Failed to parse sufficiency response: %s", e)

        # Conservative: assume sufficient on parse failure
        return True

    def _generate_complementary_queries(
        self,
        query: str,
        results: list[RetrievalResult],
        config: RetrievalConfig,
    ) -> list[str]:
        """Generate complementary queries using LLM.

        Args:
            query: The original query.
            results: Round 1 results.
            config: Retrieval configuration.

        Returns:
            List of complementary query strings.
        """
        from cognifold.query.prompts import format_multi_query_prompt

        # Build summary of initial results
        summaries = []
        for r in results[:5]:
            title = r.node_id
            if r.node is not None:
                title = r.node.data.get("title", r.node_id)
            summaries.append(f"- {title} (score: {r.final_score:.3f})")
        results_summary = "\n".join(summaries) if summaries else "(no results)"

        prompt = format_multi_query_prompt(query, results_summary)
        response = self._call_llm(prompt)

        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                queries = data.get("queries", [])
                # Limit to configured max
                return queries[: config.agentic_max_complementary_queries]
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug("Failed to parse multi-query response: %s", e)

        return []

    def _multi_rrf_fusion(
        self,
        result_lists: list[list[RetrievalResult]],
        config: RetrievalConfig,
    ) -> list[RetrievalResult]:
        """Fuse multiple result lists using Reciprocal Rank Fusion.

        RRF score = sum(weight_i / (k + rank_i(d))) across all lists.
        The original query list (index 0) gets higher weight.

        Args:
            result_lists: List of result lists from different queries.
            config: Retrieval configuration.

        Returns:
            Fused and re-ranked results.
        """
        k = config.rrf_k
        scores: dict[str, float] = {}
        result_data: dict[str, RetrievalResult] = {}

        for list_idx, results in enumerate(result_lists):
            # Original query gets weight 1.0, complementary get 0.7
            weight = 1.0 if list_idx == 0 else 0.7

            for rank, result in enumerate(results):
                node_id = result.node_id
                rrf_score = weight / (k + rank + 1)
                scores[node_id] = scores.get(node_id, 0.0) + rrf_score

                if node_id not in result_data:
                    result_data[node_id] = RetrievalResult(
                        node_id=node_id,
                        final_score=0.0,
                        bm25_score=result.bm25_score,
                        bm25_rank=result.bm25_rank,
                        semantic_score=result.semantic_score,
                        semantic_rank=result.semantic_rank,
                        node=result.node,
                    )
                elif result_data[node_id].node is None and result.node is not None:
                    result_data[node_id].node = result.node

        # Build final results
        fused: list[RetrievalResult] = []
        for node_id, score in scores.items():
            result = result_data[node_id]
            result.final_score = score
            fused.append(result)

        fused.sort(key=lambda r: r.final_score, reverse=True)
        return fused

    def _call_llm(self, prompt: str) -> str:
        """Call LLM for agentic retrieval decisions.

        Uses custom llm_caller if provided, otherwise delegates to the
        shared cached LLM caller in cognifold.query.llm.

        Args:
            prompt: The prompt to send.

        Returns:
            LLM response text.

        Raises:
            RuntimeError: If no LLM is available.
        """
        if self._llm_caller is not None:
            return self._llm_caller(prompt)

        from cognifold.query.llm import call_llm

        return call_llm(prompt)

    def build_index(self, graph: ConceptGraph) -> None:
        """Build indexes for the underlying hybrid retriever.

        Args:
            graph: The concept graph to index.
        """
        self._hybrid.build_index(graph)

    def invalidate_indexes(self) -> None:
        """Invalidate all indexes."""
        self._hybrid.invalidate_indexes()

    def get_index_stats(self) -> dict[str, int]:
        """Get statistics about the indexes."""
        return self._hybrid.get_index_stats()
