"""Memory Query Agent for retrieving context from the concept graph.

This module provides the main interface for querying the memory system.
It coordinates entry point selection, graph traversal, scoring, and
context assembly to produce relevant results for natural language queries.

Supports multiple retrieval backends:
- LEGACY: Original keyword matching (default, no external dependencies)
- BM25: Inverted index with BM25 scoring
- SEMANTIC: Embedding-based similarity search (requires embedder)
- HYBRID: BM25 + semantic with RRF fusion (best quality, requires embedder)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from cognifold.query.assembly import ContextAssembler
from cognifold.query.config import MAX_ENTRY_POINTS, NEIGHBOR_RELEVANCE_DISCOUNT
from cognifold.query.models import (
    NodeSummary,
    QueryConfig,
    QueryIntent,
    QueryResult,
    QueryType,
    RetrievalMode,
)
from cognifold.query.scoring import QueryScorer
from cognifold.query.strategies import EntryPointSelector, GraphTraverser

if TYPE_CHECKING:
    from cognifold.embeddings.embedder import NodeEmbedder
    from cognifold.embeddings.search import SemanticSearch
    from cognifold.graph.store import ConceptGraph
    from cognifold.scoring.ranker import ScoringConfig

logger = logging.getLogger(__name__)


class MemoryQueryAgent:
    """Agent for querying the memory system.

    The MemoryQueryAgent provides the read/query capability for the
    concept graph. It takes natural language queries and returns
    relevant context assembled from graph nodes.

    Supports multiple retrieval backends via the retrieval_mode config:
    - LEGACY: Original keyword matching (no dependencies)
    - BM25: Better keyword matching with BM25 scoring
    - SEMANTIC: Embedding-based search (requires embedder)
    - HYBRID: Best quality with BM25 + semantic fusion (requires embedder)

    Example:
        >>> from cognifold.graph.store import ConceptGraph
        >>> from cognifold.query import MemoryQueryAgent, QueryConfig, RetrievalMode
        >>>
        >>> graph = ConceptGraph()
        >>> # ... populate graph ...
        >>>
        >>> # Basic usage (legacy mode)
        >>> agent = MemoryQueryAgent(graph)
        >>> result = agent.query("What patterns exist?")
        >>>
        >>> # With hybrid retrieval (best quality)
        >>> from cognifold.embeddings.embedder import NodeEmbedder
        >>> embedder = NodeEmbedder(config)
        >>> config = QueryConfig(retrieval_mode=RetrievalMode.HYBRID)
        >>> agent = MemoryQueryAgent(graph, config=config, embedder=embedder)
        >>> result = agent.query("exercise habits")
    """

    def __init__(
        self,
        graph: ConceptGraph,
        config: QueryConfig | None = None,
        scoring_config: ScoringConfig | None = None,
        embedder: NodeEmbedder | None = None,
    ) -> None:
        """Initialize the query agent.

        Args:
            graph: The concept graph to query.
            config: Query configuration (includes retrieval_mode).
            scoring_config: Scoring configuration for ranking.
            embedder: Optional node embedder for semantic/hybrid retrieval.
                     Required if retrieval_mode is SEMANTIC or HYBRID.
        """
        self.graph = graph
        self.config = config or QueryConfig()
        self.scoring_config = scoring_config
        self._embedder = embedder

        # Validate embedder requirement for semantic modes
        if (
            self.config.retrieval_mode in (RetrievalMode.SEMANTIC, RetrievalMode.HYBRID)
            and embedder is None
        ):
            import dataclasses
            import warnings

            warnings.warn(
                f"retrieval_mode={self.config.retrieval_mode.value} requires embedder. "
                "Falling back to BM25 for keyword matching.",
                UserWarning,
                stacklevel=2,
            )
            self.config = dataclasses.replace(self.config, retrieval_mode=RetrievalMode.BM25)

        # Initialize components
        self._entry_selector = EntryPointSelector(
            graph, self.config, scoring_config, embedder=embedder
        )
        self._traverser = GraphTraverser(
            graph, config=self.config, ranker=self._entry_selector.ranker
        )
        self._scorer = QueryScorer(graph, self.config)
        self._assembler = ContextAssembler(self.config, graph)

        # Temporal extraction (lazy initialized)
        self._temporal_extractor = None

        # Cached semantic search for RAG mode (lazy initialized)
        self._semantic_search: SemanticSearch | None = None

    def invalidate_search_cache(self) -> None:
        """Invalidate cached search indexes (semantic search, hybrid retriever).

        Call this after the graph has been modified (e.g., after ingestion)
        so that the next query rebuilds indexes from the updated graph.
        """
        self._semantic_search = None
        self._entry_selector.reset_retrieval_cache()

    def _ensure_temporal_extractor(self):
        """Ensure temporal extractor is initialized.

        Returns:
            TemporalExtractor instance.
        """
        if self._temporal_extractor is None:
            from cognifold.temporal.extractor import TemporalExtractor

            self._temporal_extractor = TemporalExtractor()
        return self._temporal_extractor

    def _parse_temporal_references(
        self, query: str, reference_time: datetime
    ) -> list[dict[str, Any]]:
        """Extract temporal references from query text.

        Args:
            query: The query string.
            reference_time: Reference time for relative dates.

        Returns:
            List of temporal entity dictionaries.
        """
        extractor = self._ensure_temporal_extractor()
        entities = extractor.extract(query, reference_time)

        return [
            {
                "raw_text": e.raw_text,
                "normalized": e.normalized.isoformat() if e.normalized else None,
                "type": e.temporal_type.value,
                "confidence": e.confidence,
            }
            for e in entities
        ]

    def query(
        self,
        query: str,
        query_type: QueryType = QueryType.HYBRID,
        reference_time: datetime | None = None,
        max_nodes: int | None = None,
        max_context_chars: int | None = None,
        query_mode: str = "mergefold",
        use_intent_parsing: bool | None = None,
        use_reranking: bool | None = None,
    ) -> QueryResult:
        """Execute a query against the concept graph.

        Args:
            query: Natural language query string.
            query_type: Type of query strategy to use.
            reference_time: Reference time for temporal queries.
            max_nodes: Override max nodes to return.
            max_context_chars: Override max context characters.
            query_mode: System mode ("base", "rag", "episodic", "mergefold").
            use_intent_parsing: Override config for LLM intent parsing.
            use_reranking: Override config for LLM re-ranking.

        Returns:
            QueryResult with context, nodes, and metadata.
        """
        start_time = time.time()

        if reference_time is None:
            reference_time = datetime.now()

        # Create config override if needed
        config = self.config
        if max_nodes is not None or max_context_chars is not None:
            config = QueryConfig(
                max_nodes=max_nodes or self.config.max_nodes,
                max_context_chars=max_context_chars or self.config.max_context_chars,
                max_description_chars=self.config.max_description_chars,
                max_traversal_depth=self.config.max_traversal_depth,
                min_relevance_score=self.config.min_relevance_score,
                prefer_concepts=self.config.prefer_concepts,
                include_reasoning=self.config.include_reasoning,
                include_grounding=self.config.include_grounding,
                domain=self.config.domain,
                speaker_aware=self.config.speaker_aware,
                use_llm_rerank=self.config.use_llm_rerank,
                use_query_expansion=self.config.use_query_expansion,
                retrieval_mode=self.config.retrieval_mode,
                semantic_weight=self.config.semantic_weight,
                keyword_weight=self.config.keyword_weight,
                use_ppr=self.config.use_ppr,
                adaptive_depth=self.config.adaptive_depth,
                adaptive_depth_max=self.config.adaptive_depth_max,
                use_intent_routing=self.config.use_intent_routing,
            )
            # Update components with new config
            self._scorer = QueryScorer(self.graph, config)
            self._assembler = ContextAssembler(config, self.graph)

        # Determine if we should use LLM features
        should_parse_intent = (
            use_intent_parsing if use_intent_parsing is not None else config.use_query_expansion
        )
        should_rerank = use_reranking if use_reranking is not None else config.use_llm_rerank

        # Parse query intent if enabled
        intent: QueryIntent | None = None
        if should_parse_intent:
            try:
                intent = self.parse_query_intent(query)
                query_type = intent.query_type
                logger.debug(f"Parsed intent: {intent}")
            except Exception as e:
                logger.debug(f"Intent parsing failed, using default: {e}")

        # Extract temporal references from query
        temporal_refs = self._parse_temporal_references(query, reference_time)

        # Build query metadata
        query_metadata: dict[str, object] = {
            "query": query,
            "query_type": query_type.value,
            "reference_time": reference_time.isoformat(),
            "query_mode": query_mode,
            "intent_parsed": intent is not None,
            "reranking_enabled": should_rerank,
            "retrieval_mode": self.config.retrieval_mode.value,
            "temporal_references": temporal_refs,
        }

        # Handle empty graph
        if self.graph.node_count == 0:
            return QueryResult(
                context="The memory graph is empty. No context available.",
                nodes=[],
                traversal_path=[],
                query_metadata=query_metadata,
                total_nodes_scanned=0,
                query_time_ms=0.0,
            )

        # Handle different modes
        traversal = None
        scored_nodes = []
        nodes_scanned = 0

        if query_mode == "rag":
            # RAG Mode: Vector Search only (Top-K) using unified embeddings module
            if self._embedder is not None:
                from cognifold.embeddings.search import SearchConfig, SemanticSearch

                if self._semantic_search is None:
                    self._semantic_search = SemanticSearch(self._embedder)
                sem_config = SearchConfig(top_k=config.max_nodes * 2)
                results = self._semantic_search.search(self.graph, query, sem_config)
                matched_nodes = [r.node for r in results if r.node is not None]
                scored_nodes = self._scorer.rank_nodes_for_query(
                    matched_nodes, query_type, reference_time
                )
                nodes_scanned = len(matched_nodes)
            else:
                # Fallback to BM25 if no embedder available
                entry_points = self._entry_selector.select_entry_points(
                    query_type=query_type,
                    reference_time=reference_time,
                    max_entry_points=MAX_ENTRY_POINTS,
                    query_text=query,
                )
                traversal = self._traverser.traverse(entry_points)

        elif query_mode == "base":
            # Base Mode: Recent history (Temporal)
            # Effectively same as Temporal Query but usually just recent events
            traversal = self._traverser.traverse_temporal(
                reference_time=reference_time,
                time_window_hours=24.0 * 7,  # Last week
            )

        elif query_mode == "episodic":
            # Episodic Mode: Like MergeFold but filter for Events only?
            # Or rely on Ingestion to not produce Concepts.
            # If Ingestion allows Concepts, we should filter them out here.

            # Use standard selection/traversal
            entry_points = self._entry_selector.select_entry_points(
                query_type=query_type,
                reference_time=reference_time,
                max_entry_points=MAX_ENTRY_POINTS,
                query_text=query,
            )
            traversal = self._traverser.traverse(entry_points)

            # Filter out non-events during scoring/assembly
            # We can do this by modifying scored_nodes later or hacking traversal

        else:  # mergefold
            # Adaptive depth and entry points (Wave 5)
            max_depth = self.config.max_traversal_depth
            max_eps = MAX_ENTRY_POINTS
            if self.config.adaptive_depth and self.graph.node_count > 0:
                density = self.graph.edge_count / max(self.graph.node_count, 1)
                # Sparse graphs need deeper traversal
                if density < 2.0:
                    max_depth = min(max_depth + 1, self.config.adaptive_depth_max)
                # Large graphs need more entry points
                if self.graph.node_count > 200:
                    max_eps = min(max_eps * 2, 30)

            # Wave 7: Intent-aware edge weighting
            intent_edge_weights: dict[str, float] | None = None
            if self.config.use_intent_routing:
                try:
                    from cognifold.symbolic.intent_router import QueryIntentRouter

                    router = QueryIntentRouter()
                    intent_edge_weights = router.get_edge_weights(query)
                except Exception as e:
                    logger.debug(f"Intent routing failed: {e}")

            # Step 1: Select entry points (text search first for semantic/hybrid)
            entry_points = self._entry_selector.select_entry_points(
                query_type=query_type,
                reference_time=reference_time,
                max_entry_points=max_eps,
                query_text=query,
            )

            # Step 2: Traverse graph
            if query_type == QueryType.TEMPORAL:
                # For temporal queries, use time-based traversal.
                # Use a wide window (6 months) to support conversations
                # spanning weeks/months (e.g., LoCoMo, LongMemEval).
                # Entry point selection via BM25/semantic still narrows results.
                traversal = self._traverser.traverse_temporal(
                    reference_time=reference_time,
                    time_window_hours=4320.0,  # ~6 months
                )
            else:
                # For other queries, use BFS from entry points
                traversal = self._traverser.traverse(
                    entry_points,
                    max_depth=max_depth,
                    edge_weights=intent_edge_weights,
                )

        # Common Processing for Traversal-based modes
        if traversal:
            scored_nodes = self._scorer.score_traversal_results(
                traversal=traversal,
                query_type=query_type,
                reference_time=reference_time,
                query_text=query,
            )
            nodes_scanned = traversal.nodes_scanned

            # Filter for Episodic Mode
            if query_mode == "episodic":
                from cognifold.models.node import NodeType

                scored_nodes = [ns for ns in scored_nodes if ns.node_type == NodeType.EVENT.value]

        # Step 3.5: 1-hop neighbor expansion (adds graph-connected context)
        if scored_nodes and self.graph.edge_count > 0:
            scored_nodes = self._expand_with_neighbors(scored_nodes)
            scored_nodes = scored_nodes[: config.max_nodes]

        # Step 4: Optional LLM re-ranking
        if should_rerank and scored_nodes:
            try:
                scored_nodes = self.rerank_with_llm(query, scored_nodes, config.max_nodes)
                query_metadata["reranked"] = True
            except Exception as e:
                logger.debug(f"LLM re-ranking failed: {e}")
                query_metadata["reranked"] = False

        # Step 5: Assemble context
        query_time_ms = (time.time() - start_time) * 1000

        result = self._assembler.assemble(
            nodes=scored_nodes,
            traversal_path=traversal.traversal_path if traversal else [],
            query_metadata=query_metadata,
            total_nodes_scanned=nodes_scanned,
            query_time_ms=query_time_ms,
        )

        return result

    def query_semantic(
        self,
        query: str,
        reference_time: datetime | None = None,
    ) -> QueryResult:
        """Execute a semantic query.

        Convenience method for semantic queries that focus on
        meaning and conceptual relationships.

        Args:
            query: Natural language query.
            reference_time: Reference time.

        Returns:
            QueryResult with semantically relevant context.
        """
        return self.query(
            query=query,
            query_type=QueryType.SEMANTIC,
            reference_time=reference_time,
        )

    def query_temporal(
        self,
        query: str,
        reference_time: datetime | None = None,
    ) -> QueryResult:
        """Execute a temporal query.

        Convenience method for temporal queries that focus on
        recent or time-specific information.

        Args:
            query: Natural language query.
            reference_time: Reference time (center of time window).

        Returns:
            QueryResult with temporally relevant context.
        """
        return self.query(
            query=query,
            query_type=QueryType.TEMPORAL,
            reference_time=reference_time,
        )

    def query_structural(
        self,
        query: str,
        reference_time: datetime | None = None,
    ) -> QueryResult:
        """Execute a structural query.

        Convenience method for structural queries that focus on
        highly connected and important nodes.

        Args:
            query: Natural language query.
            reference_time: Reference time.

        Returns:
            QueryResult with structurally important context.
        """
        return self.query(
            query=query,
            query_type=QueryType.STRUCTURAL,
            reference_time=reference_time,
        )

    def get_top_concepts(
        self,
        n: int = 10,
        reference_time: datetime | None = None,
    ) -> list[NodeSummary]:
        """Get the top N most important concepts.

        Convenience method for retrieving high-level concepts
        without a specific query.

        Args:
            n: Number of concepts to return.
            reference_time: Reference time for scoring.

        Returns:
            List of top concept summaries.
        """
        from cognifold.models.node import NodeType

        if reference_time is None:
            reference_time = datetime.now()

        concepts = self.graph.get_nodes_by_type(NodeType.CONCEPT)

        if not concepts:
            return []

        # Score concepts
        summaries = self._scorer.rank_nodes_for_query(
            nodes=concepts,
            query_type=QueryType.STRUCTURAL,
            reference_time=reference_time,
        )

        return summaries[:n]

    def get_recent_intents(
        self,
        n: int = 10,
        reference_time: datetime | None = None,
    ) -> list[NodeSummary]:
        """Get the N most recent/relevant intents.

        Convenience method for retrieving goals, desires, or intentions.
        Also includes legacy "action" nodes for backward compatibility.

        Args:
            n: Number of intents to return.
            reference_time: Reference time for scoring.

        Returns:
            List of intent summaries.
        """
        from cognifold.models.node import NodeType

        if reference_time is None:
            reference_time = datetime.now()

        # Get INTENT nodes (includes legacy "action" nodes loaded via NodeType.from_string)
        intents = list(self.graph.get_nodes_by_type(NodeType.INTENT))

        if not intents:
            return []

        # Score intents with temporal emphasis
        summaries = self._scorer.rank_nodes_for_query(
            nodes=intents,
            query_type=QueryType.TEMPORAL,
            reference_time=reference_time,
        )

        return summaries[:n]

    # Backward compatibility alias
    def get_recent_actions(
        self,
        n: int = 10,
        reference_time: datetime | None = None,
    ) -> list[NodeSummary]:
        """Get the N most recent/relevant actions (deprecated).

        Use get_recent_intents() instead. This method is provided
        for backward compatibility.

        Args:
            n: Number of intents to return.
            reference_time: Reference time for scoring.

        Returns:
            List of intent summaries.
        """
        return self.get_recent_intents(n=n, reference_time=reference_time)

    def explain_node(self, node_id: str) -> NodeSummary | None:
        """Get detailed explanation of a specific node.

        Args:
            node_id: ID of the node to explain.

        Returns:
            NodeSummary with full details, or None if not found.
        """
        if not self.graph.has_node(node_id):
            return None

        node = self.graph.get_node(node_id)
        return self._scorer.node_to_summary(node, 1.0)

    # =========================================================================
    # Domain-Specific Query Methods
    # =========================================================================

    def query_for_qa(
        self,
        question: str,
        domain: str | None = None,
        reference_time: datetime | None = None,
        **kwargs: Any,
    ) -> QueryResult:
        """Query optimized for QA benchmarks with domain-specific handling.

        Args:
            question: The question to answer.
            domain: Domain hint (locomo, longmemeval, etc.).
            reference_time: Reference time for temporal queries.
            **kwargs: Additional arguments passed to query().

        Returns:
            QueryResult optimized for QA.
        """
        domain = domain or self.config.domain

        handler = {
            "locomo": self._query_locomo_qa,
            "longmemeval": self._query_longmemeval_qa,
            "msc": self._query_msc_qa,
            "babilong": self._query_babilong_qa,
            "musique": self._query_musique_qa,
            "tomi": self._query_tomi_qa,
            "qmsum": self._query_qmsum_qa,
            "narrativeqa": self._query_narrativeqa_qa,
            "rgb": self._query_rgb_qa,
            "streamingqa": self._query_streamingqa_qa,
            "socialiqa": self._query_socialiqa_qa,
            "mutual": self._query_mutual_qa,
            "safetybench": self._query_safetybench_qa,
            "futurex": self._query_futurex_qa,
        }.get(domain or "")

        if handler:
            return handler(question, reference_time, **kwargs)
        return self.query(question, reference_time=reference_time, **kwargs)

    def _query_locomo_qa(
        self,
        question: str,
        reference_time: datetime | None = None,
        **kwargs: Any,
    ) -> QueryResult:
        """LoCoMo-specific QA retrieval with speaker awareness and temporal handling.

        Uses larger context budget and temporal query detection for
        date-related questions. Speaker filtering narrows results when a speaker
        is mentioned.

        Iter1 enhancements (Zep/Mem0-inspired raw-turn preservation):
        - Lift `max_description_chars` to 2000 so each ~700-1000 char raw-turn
          event batch is preserved verbatim (concept folding loses specifics
          like "Melanie's daughter's birthday" / "Matt Patterson"; raw events
          retain them).
        - Entity-anchored event retrieval: extract entities from the question
          (Melanie, daughter, etc.) and pull matching EVENT nodes from
          `graph.entity_index`, ensuring the answer-bearing turn isn't lost
          to BM25/semantic ranking miss.
        """
        q_lower = question.lower()

        # Detect temporal questions — but always use HYBRID retrieval.
        # TEMPORAL traversal ranks by timestamp proximity to reference_time,
        # returning identical nodes regardless of question content.
        # HYBRID (BM25 + semantic) actually searches by question text.
        temporal_keywords = ("when", "what date", "what time", "how long ago", "what day")
        is_temporal = any(kw in q_lower for kw in temporal_keywords)
        query_type = QueryType.HYBRID  # Always HYBRID — TEMPORAL traversal ignores question content

        import dataclasses

        # LoCoMo graphs can be 100-600 nodes with turn-level batching.
        # Use wider context budget; temporal queries need even more.
        max_ctx = 80 if is_temporal else 60
        original_config = self.config
        self.config = dataclasses.replace(
            original_config,
            max_nodes=max_ctx,
            max_context_chars=12000,           # Iter1: lift from 8000 → 12000
            max_description_chars=2000,        # Iter1: 500 → 2000 to preserve raw turn batches
            include_reasoning=False,   # Strip metadata to fit more facts in context
            include_grounding=False,   # Strip grounding refs
        )

        try:
            # Check if question mentions a specific speaker
            speaker = self._extract_speaker_from_question(question)

            if speaker and self.config.speaker_aware:
                logger.debug(f"Speaker-aware query for: {speaker}")
                base = self._query_with_filter(
                    query=question,
                    reference_time=reference_time,
                    query_type=query_type,
                    node_filter=lambda n: self._node_mentions_speaker(n, speaker),
                    **kwargs,
                )
            else:
                base = self.query(
                    question,
                    reference_time=reference_time,
                    query_type=query_type,
                    max_nodes=max_ctx,
                    max_context_chars=12000,
                    **kwargs,
                )

            # Iter1: Entity-anchored event boost.
            # Pull all EVENT nodes whose entities overlap question entities,
            # rerank to put events FIRST (raw turn-text contains ground truth),
            # then re-assemble. This is the Zep-style raw-turn preservation +
            # Mem0-style entity-anchored recall.
            return self._boost_with_entity_events(question, base)
        finally:
            self.config = original_config

    def _boost_with_entity_events(
        self, question: str, base: QueryResult
    ) -> QueryResult:
        """Augment retrieval with entity-anchored event nodes (LoCoMo iter1).

        Uses `graph.entity_index` (built post-ingestion in run_benchmark.py)
        to fetch all event nodes mentioning entities from the question, merges
        with the base hybrid retrieval result, and prefers events over concepts
        for the LoCoMo domain (concept folding loses specific facts).

        Args:
            question: The user question.
            base: The base QueryResult from hybrid retrieval.

        Returns:
            QueryResult with an enriched context whose EVENTS section comes
            first (so the LLM reads raw conversation turns before folded
            concepts).
        """
        from cognifold.models.node import NodeType
        from cognifold.query.models import NodeSummary

        entity_index = getattr(self.graph, "entity_index", None)
        existing_ids = {n.node_id for n in base.nodes}
        event_summaries: list[NodeSummary] = []

        if entity_index is not None:
            try:
                matched_node_ids = entity_index.query_all_matches(question)
            except Exception:
                matched_node_ids = []

            for nid in matched_node_ids:
                if nid in existing_ids:
                    continue
                node = self.graph.get_node_or_none(nid)
                if node is None:
                    continue
                # Only boost EVENT nodes (raw turn text). Concept nodes
                # already arrive via normal retrieval.
                if node.type != NodeType.EVENT:
                    continue
                summary = self._scorer.node_to_summary(node, 0.85)
                event_summaries.append(summary)
                existing_ids.add(nid)
                # Cap how many we add to avoid overwhelming the budget
                if len(event_summaries) >= 12:
                    break

        # Reorder: events FIRST (highest-relevance events from base + boosted
        # events), then concepts/intents/time. Within events, keep relevance
        # order; the boosted events trail base events.
        base_events = [n for n in base.nodes if n.node_type == "event"]
        base_concepts = [
            n for n in base.nodes if n.node_type in ("concept", "intent", "fact")
        ]
        base_other = [
            n for n in base.nodes
            if n.node_type not in ("event", "concept", "intent", "fact")
        ]

        # Iter6 v2: bucketed cross-encoder reranking. Events and concepts are
        # reranked SEPARATELY against the question (BAAI/bge-reranker-v2-m3),
        # each bucket keeps its own top-k, and the buckets are concatenated.
        # Bucketing prevents events from crowding out abstraction-level
        # concepts in the final cap (Cat 3 / open-domain inference depends on
        # concepts surviving the cutoff).
        # Enabled per-profile via `qa_rerank: true` or env CF_RERANK_ENABLED.
        rerank_enabled = (
            os.environ.get("CF_RERANK_ENABLED", "0") == "1"
            or bool(getattr(self.config, "qa_rerank", False))
        )
        if rerank_enabled:
            try:
                from cognifold.retrieval.cross_encoder_rerank import (
                    rerank_nodes as _rerank_nodes,
                )

                event_top_k = int(
                    os.environ.get("CF_RERANK_EVENT_TOPK", "")
                    or getattr(self.config, "qa_rerank_event_topk", 0)
                    or 15
                )
                concept_top_k = int(
                    os.environ.get("CF_RERANK_CONCEPT_TOPK", "")
                    or getattr(self.config, "qa_rerank_concept_topk", 0)
                    or 15
                )

                all_events = base_events + event_summaries
                if len(all_events) >= 2:
                    all_events = _rerank_nodes(
                        question, all_events, top_k=event_top_k
                    )
                if len(base_concepts) >= 2:
                    base_concepts = _rerank_nodes(
                        question, base_concepts, top_k=concept_top_k
                    )
                merged = all_events + base_concepts + base_other
            except Exception as exc:
                logger.warning(f"Cross-encoder rerank skipped: {exc}")
                merged = base_events + event_summaries + base_concepts + base_other
        else:
            merged = base_events + event_summaries + base_concepts + base_other
        # Cap total nodes to config.max_nodes
        merged = merged[: self.config.max_nodes]

        # Iter1: surface raw events FIRST (Zep-style raw-turn preservation).
        # The QA prompt then encounters verbatim conversation turns before
        # folded concepts, dramatically improving fact recall on questions
        # that ask about specific entities, dates, or quoted utterances.
        new_context = self._assembler.build_context_text(
            merged,
            type_order=["event", "concept", "intent", "action", "fact", "time"],
        )

        return QueryResult(
            context=new_context,
            nodes=merged,
            traversal_path=base.traversal_path,
            query_metadata=base.query_metadata,
            total_nodes_scanned=base.total_nodes_scanned + len(event_summaries),
            query_time_ms=base.query_time_ms,
        )

    def _query_longmemeval_qa(
        self,
        question: str,
        reference_time: datetime | None = None,
        **kwargs: Any,
    ) -> QueryResult:
        """LongMemEval-specific QA retrieval with type-aware logic.

        Args:
            question: The question to answer.
            reference_time: Reference time.
            **kwargs: Additional arguments.

        Returns:
            QueryResult optimized for fact retrieval.
        """
        import dataclasses

        q_lower = question.lower()

        # Detect temporal questions
        temporal_keywords = ("when", "what date", "what time", "how long ago", "which month")
        is_temporal = any(kw in q_lower for kw in temporal_keywords)

        # Use HYBRID (BM25 + semantic) for better recall than pure semantic
        query_type = QueryType.TEMPORAL if is_temporal else QueryType.HYBRID

        # LongMemEval graphs have 150-400 concept nodes; use wider context
        original_config = self.config
        self.config = dataclasses.replace(original_config, max_nodes=40)

        try:
            return self.query(
                question,
                query_type=query_type,
                reference_time=reference_time,
                **kwargs,
            )
        finally:
            self.config = original_config

    def _query_msc_qa(
        self,
        question: str,
        reference_time: datetime | None = None,
        **kwargs: Any,
    ) -> QueryResult:
        """MSC-specific QA retrieval with speaker awareness.

        Similar to LoCoMo but focused on persona fact recall across sessions.

        Args:
            question: The question to answer.
            reference_time: Reference time.
            **kwargs: Additional arguments.

        Returns:
            QueryResult with speaker-aware filtering if applicable.
        """
        speaker = self._extract_speaker_from_question(question)

        if speaker and self.config.speaker_aware:
            logger.debug(f"MSC speaker-aware query for: {speaker}")
            return self._query_with_filter(
                query=question,
                reference_time=reference_time,
                node_filter=lambda n: self._node_mentions_speaker(n, speaker),
                **kwargs,
            )

        return self.query(question, reference_time=reference_time, **kwargs)

    def _query_babilong_qa(
        self,
        question: str,
        reference_time: datetime | None = None,
        **kwargs: Any,
    ) -> QueryResult:
        """BABILong-specific QA retrieval for entity state queries.

        Filters to concept nodes only -- event titles can contradict
        the updated concept values and confuse the QA LLM.
        """
        result = self.query(
            question,
            query_type=QueryType.HYBRID,
            reference_time=reference_time,
            **kwargs,
        )
        # Filter to concept nodes only to avoid event-concept contradictions
        concept_nodes = [n for n in result.nodes if n.node_type == "concept"]
        if concept_nodes:
            filtered_context = self._assembler.build_context_text(concept_nodes)
            return QueryResult(
                context=filtered_context,
                nodes=concept_nodes,
                traversal_path=result.traversal_path,
                query_metadata=result.query_metadata,
                total_nodes_scanned=result.total_nodes_scanned,
                query_time_ms=result.query_time_ms,
            )
        return result

    def _query_musique_qa(
        self,
        question: str,
        reference_time: datetime | None = None,
        **kwargs: Any,
    ) -> QueryResult:
        """MuSiQue-specific QA: multi-hop retrieval with PPR and expanded neighbors.

        Uses Personalized PageRank seeded from question-relevant entry points
        to boost cross-document bridge entities.
        """
        return self.query(
            question,
            query_type=QueryType.HYBRID,
            reference_time=reference_time,
            max_nodes=40,
            **kwargs,
        )

    def _query_tomi_qa(
        self,
        question: str,
        reference_time: datetime | None = None,
        **kwargs: Any,
    ) -> QueryResult:
        """ToMi-specific QA: prioritize concept nodes for belief/state tracking."""
        return self.query(
            question,
            query_type=QueryType.HYBRID,
            reference_time=reference_time,
            max_nodes=30,
            **kwargs,
        )

    def _query_qmsum_qa(
        self,
        question: str,
        reference_time: datetime | None = None,
        **kwargs: Any,
    ) -> QueryResult:
        """QMSum-specific QA: broader context for summarization."""
        return self.query(
            question,
            query_type=QueryType.SEMANTIC,
            reference_time=reference_time,
            max_nodes=40,
            max_context_chars=8000,
            **kwargs,
        )

    def _query_narrativeqa_qa(
        self,
        question: str,
        reference_time: datetime | None = None,
        **kwargs: Any,
    ) -> QueryResult:
        """NarrativeQA-specific QA: hybrid search for full story coverage."""
        return self.query(
            question,
            query_type=QueryType.HYBRID,
            reference_time=reference_time,
            max_nodes=40,
            max_context_chars=8000,
            **kwargs,
        )

    def _query_rgb_qa(
        self,
        question: str,
        reference_time: datetime | None = None,
        **kwargs: Any,
    ) -> QueryResult:
        """RGB-specific QA: precise fact retrieval, filter noise/distractors."""
        return self.query(
            question,
            query_type=QueryType.HYBRID,
            reference_time=reference_time,
            max_nodes=15,
            **kwargs,
        )

    def _query_streamingqa_qa(
        self,
        question: str,
        reference_time: datetime | None = None,
        **kwargs: Any,
    ) -> QueryResult:
        """StreamingQA-specific QA: temporal-aware retrieval over news articles.

        Uses all available nodes (graphs are small, ~3-10 nodes per example)
        and returns the full graph context for maximum factual coverage.
        """
        # StreamingQA graphs are small (per-example). Retrieve all nodes
        # to ensure we don't miss the fact-bearing node.
        node_count = self.graph.node_count
        effective_max = max(30, node_count)

        # Detect temporal questions and adjust query type
        q_lower = question.lower()
        temporal_keywords = ("when", "what year", "what date", "what time", "how long")
        query_type = (
            QueryType.TEMPORAL
            if any(kw in q_lower for kw in temporal_keywords)
            else QueryType.HYBRID
        )

        return self.query(
            question,
            query_type=query_type,
            reference_time=reference_time,
            max_nodes=effective_max,
            **kwargs,
        )

    def _query_socialiqa_qa(
        self,
        question: str,
        reference_time: datetime | None = None,
        **kwargs: Any,
    ) -> QueryResult:
        """SocialIQA-specific QA: social commonsense with semantic concept focus."""
        return self.query(
            question,
            query_type=QueryType.SEMANTIC,
            reference_time=reference_time,
            max_nodes=15,
            **kwargs,
        )

    def _query_mutual_qa(
        self,
        question: str,
        reference_time: datetime | None = None,
        **kwargs: Any,
    ) -> QueryResult:
        """MuTual-specific QA: dialogue reasoning with full turn retrieval."""
        return self.query(
            question,
            query_type=QueryType.SEMANTIC,
            reference_time=reference_time,
            max_nodes=15,
            **kwargs,
        )

    def _query_safetybench_qa(
        self,
        question: str,
        reference_time: datetime | None = None,
        **kwargs: Any,
    ) -> QueryResult:
        """SafetyBench-specific QA: minimize noise, focused retrieval."""
        return self.query(
            question,
            query_type=QueryType.SEMANTIC,
            reference_time=reference_time,
            max_nodes=5,
            **kwargs,
        )

    def _query_futurex_qa(
        self,
        question: str,
        reference_time: datetime | None = None,
        **kwargs: Any,
    ) -> QueryResult:
        """FutureX-specific QA: retrieve evidence for prediction tasks.

        FutureX tasks are self-contained prediction questions where the graph
        contains ingested research evidence. Uses larger context budget since
        prediction tasks need comprehensive evidence synthesis.
        """
        return self.query(
            question,
            query_type=QueryType.HYBRID,
            reference_time=reference_time,
            max_nodes=30,
            **kwargs,
        )

    def _query_tomi_qa(
        self,
        question: str,
        reference_time: datetime | None = None,
        **kwargs: Any,
    ) -> QueryResult:
        """ToMi-specific QA retrieval for theory-of-mind questions.

        ToMi graphs are small (8-15 nodes) and questions ask about object locations,
        beliefs, and agent movements. Returns ALL graph nodes as context since the
        graphs are small enough for the LLM to process entirely, avoiding retrieval
        misses on object/person names.

        Args:
            question: The question to answer.
            reference_time: Reference time.
            **kwargs: Additional arguments.

        Returns:
            QueryResult with all graph nodes for comprehensive ToM reasoning.
        """
        from cognifold.query.models import NodeSummary

        all_nodes = self.graph.get_all_nodes()

        if not all_nodes:
            return self.query(
                question,
                query_type=QueryType.HYBRID,
                reference_time=reference_time,
                **kwargs,
            )

        # For small ToMi graphs, return all nodes — concepts first, then events
        concepts: list[NodeSummary] = []
        events: list[NodeSummary] = []

        for node in all_nodes:
            ns = NodeSummary(
                node_id=node.id,
                node_type=node.type.value,
                title=node.data.get("title") or node.data.get("name") or node.id,
                relevance_score=0.9 if node.type.value == "concept" else 0.7,
                description=node.data.get("description"),
                data=node.data,
                reasoning=node.reasoning,
            )
            if node.type.value == "concept":
                concepts.append(ns)
            else:
                events.append(ns)

        matched = (concepts + events)[: self.config.max_nodes]

        return self._assembler.assemble(matched, [n.node_id for n in matched])

    def _query_with_filter(
        self,
        query: str,
        reference_time: datetime | None = None,
        node_filter: Callable[[NodeSummary], bool] | None = None,
        **kwargs: Any,
    ) -> QueryResult:
        """Query with optional node filtering.

        Args:
            query: The query string.
            reference_time: Reference time.
            node_filter: Function to filter nodes.
            **kwargs: Additional arguments.

        Returns:
            Filtered QueryResult.
        """
        # Get base results with higher limit to allow for filtering
        original_max = self.config.max_nodes
        result = self.query(
            query,
            reference_time=reference_time,
            max_nodes=original_max * 3,  # Get more candidates
            **kwargs,
        )

        if node_filter:
            # Filter nodes
            filtered_nodes = [n for n in result.nodes if node_filter(n)][:original_max]

            # Rebuild context with filtered nodes
            result = QueryResult(
                context=self._assembler.build_context_text(filtered_nodes),
                nodes=filtered_nodes,
                traversal_path=result.traversal_path,
                query_metadata=result.query_metadata,
                total_nodes_scanned=result.total_nodes_scanned,
                query_time_ms=result.query_time_ms,
            )

        return result

    def _extract_speaker_from_question(self, question: str) -> str | None:
        """Extract speaker reference from a question.

        Args:
            question: The question text.

        Returns:
            Speaker identifier if found (e.g., "User1", "User2", or real name), None otherwise.
        """
        question_lower = question.lower()

        # Common speaker patterns (generic)
        patterns = [
            r"\buser\s*1\b",
            r"\buser\s*2\b",
            r"\buser1\b",
            r"\buser2\b",
            r"\bspeaker\s*1\b",
            r"\bspeaker\s*2\b",
            r"\bperson\s*1\b",
            r"\bperson\s*2\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, question_lower)
            if match:
                matched = match.group().lower().replace(" ", "")
                if "1" in matched:
                    return "User1"
                elif "2" in matched:
                    return "User2"

        # Check for real speaker names from the graph
        known_speakers: set[str] = set()
        for node in self.graph.get_all_nodes():
            ctx = node.data.get("context")
            speaker = ctx.get("speaker") if isinstance(ctx, dict) else None
            if not speaker:
                speaker = node.data.get("speaker")
            if speaker and isinstance(speaker, str):
                known_speakers.add(speaker)

        for speaker in known_speakers:
            if speaker.lower() in question_lower:
                return speaker

        return None

    def _node_mentions_speaker(self, node: NodeSummary, speaker: str) -> bool:
        """Check if a node mentions a specific speaker.

        Args:
            node: The node to check.
            speaker: Speaker identifier (e.g., "User1").

        Returns:
            True if node mentions the speaker.
        """
        speaker_lower = speaker.lower()

        # Check title
        if node.title and speaker_lower in node.title.lower():
            return True

        # Check description
        if node.description and speaker_lower in node.description.lower():
            return True

        # Check data fields
        for _key, value in node.data.items():
            if isinstance(value, str) and speaker_lower in value.lower():
                return True

        return False

    # =========================================================================
    # Query Intent Parsing (LLM-based)
    # =========================================================================

    def parse_query_intent(self, query: str) -> QueryIntent:
        """Parse query intent using LLM.

        Analyzes the query to determine:
        - Query type (semantic, temporal, structural, hybrid)
        - Key topics
        - Time context
        - Scope
        - Alternative queries

        Args:
            query: The natural language query.

        Returns:
            Parsed QueryIntent.
        """
        from cognifold.query.prompts import format_intent_prompt

        prompt = format_intent_prompt(query)

        try:
            response = self._call_llm(prompt)
            return self._parse_intent_response(response, query)
        except Exception as e:
            logger.warning(f"Failed to parse query intent: {e}")
            return QueryIntent(query_type=QueryType.HYBRID, key_topics=[])

    def _parse_intent_response(self, response: str, original_query: str) -> QueryIntent:
        """Parse LLM response into QueryIntent.

        Args:
            response: LLM response text.
            original_query: Original query for fallback.

        Returns:
            Parsed QueryIntent.
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                query_type_str = data.get("query_type", "HYBRID").upper()
                query_type = QueryType(query_type_str.lower())

                return QueryIntent(
                    query_type=query_type,
                    key_topics=data.get("key_topics", []),
                    time_context=data.get("time_context"),
                    scope=data.get("scope", "focused").lower(),
                    alternative_queries=data.get("alternative_queries", []),
                    speaker_filter=self._extract_speaker_from_question(original_query),
                )
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"Failed to parse intent JSON: {e}")

        # Fallback to basic intent
        return QueryIntent(
            query_type=QueryType.HYBRID,
            key_topics=[],
            speaker_filter=self._extract_speaker_from_question(original_query),
        )

    # =========================================================================
    # Graph Expansion
    # =========================================================================

    def _expand_with_neighbors(
        self,
        scored_nodes: list[NodeSummary],
        max_expansion: int = 10,
    ) -> list[NodeSummary]:
        """Expand scored results by including 1-hop graph neighbors.

        For each retrieved node, also include its directly connected
        neighbors. This enables multi-hop reasoning when edges exist.

        Args:
            scored_nodes: Already-scored nodes from retrieval.
            max_expansion: Maximum number of neighbor nodes to add.

        Returns:
            Expanded list of NodeSummary with neighbors appended.
        """
        existing_ids = {n.node_id for n in scored_nodes}
        neighbor_nodes: list[NodeSummary] = []

        for node in scored_nodes:
            if len(neighbor_nodes) >= max_expansion:
                break

            try:
                neighbors = self.graph.get_neighbors(node.node_id)
                predecessors = self.graph.get_predecessors(node.node_id)
            except KeyError:
                continue

            for neighbor_id in list(neighbors) + list(predecessors):
                if neighbor_id in existing_ids:
                    continue
                if len(neighbor_nodes) >= max_expansion:
                    break

                neighbor = self.graph.get_node_or_none(neighbor_id)
                if neighbor is None:
                    continue

                # Create summary with discounted relevance
                summary = self._scorer.node_to_summary(
                    neighbor, node.relevance_score * NEIGHBOR_RELEVANCE_DISCOUNT
                )
                neighbor_nodes.append(summary)
                existing_ids.add(neighbor_id)

        return list(scored_nodes) + neighbor_nodes

    # =========================================================================
    # LLM Re-ranking
    # =========================================================================

    def rerank_with_llm(
        self,
        query: str,
        candidates: list[NodeSummary],
        top_k: int | None = None,
    ) -> list[NodeSummary]:
        """Re-rank candidates using LLM for relevance scoring.

        Args:
            query: The query string.
            candidates: Candidate nodes to re-rank.
            top_k: Number of top candidates to return.

        Returns:
            Re-ranked list of NodeSummary.
        """
        if not candidates:
            return []

        top_k = top_k or self.config.max_nodes

        # Limit LLM calls by only re-ranking top candidates
        candidates_to_score = candidates[: top_k * 2]

        scored: list[tuple[NodeSummary, float]] = []
        for node in candidates_to_score:
            try:
                score = self._score_node_with_llm(query, node)
                scored.append((node, score))
            except Exception as e:
                logger.debug(f"Failed to score node {node.node_id}: {e}")
                scored.append((node, node.relevance_score))

        # Sort by LLM score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Update relevance scores
        result = []
        for node, score in scored[:top_k]:
            # Create new NodeSummary with updated score
            result.append(
                NodeSummary(
                    node_id=node.node_id,
                    node_type=node.node_type,
                    title=node.title,
                    relevance_score=score,
                    description=node.description,
                    reasoning=node.reasoning,
                    grounded_in=node.grounded_in,
                    created_at=node.created_at,
                    data=node.data,
                )
            )

        return result

    def _score_node_with_llm(self, query: str, node: NodeSummary) -> float:
        """Score a single node's relevance using LLM.

        Args:
            query: The query string.
            node: Node to score.

        Returns:
            Relevance score between 0.0 and 1.0.
        """
        from cognifold.query.prompts import format_relevance_prompt

        prompt = format_relevance_prompt(
            query=query,
            node_type=node.node_type,
            title=node.title,
            description=node.description,
            reasoning=node.reasoning,
            grounded_in=node.grounded_in,
        )

        response = self._call_llm(prompt)

        # Parse score from response
        try:
            # Look for a number between 0 and 1
            match = re.search(r"(0\.\d+|1\.0|0|1)", response)
            if match:
                return float(match.group())
        except ValueError:
            pass

        # Default to mid-range score
        return 0.5

    # =========================================================================
    # LLM Helper
    # =========================================================================

    def _call_llm(self, prompt: str) -> str:
        """Call LLM for query processing.

        Delegates to the shared cached LLM caller in cognifold.query.llm.
        If a language system prompt was set (by the service route), it is
        passed as a system-level instruction for stronger language control.

        Args:
            prompt: The prompt to send.

        Returns:
            LLM response text.
        """
        from cognifold.query.llm import call_llm

        lang_prompt: str | None = getattr(self, "_language_system_prompt", None)
        return call_llm(prompt, system_prompt=lang_prompt)
