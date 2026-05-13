"""Entry point selection and graph traversal strategies.

This module provides strategies for:
- Finding entry points into the graph based on query type
- Text-based search for semantically relevant nodes
- Hybrid retrieval using BM25 + semantic search
- Traversing the graph from entry points to collect relevant nodes
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from cognifold.models.node import NodeType
from cognifold.query.config import BFS_DECAY_PER_HOP, ENTRY_POINT_BOOSTS, apply_type_boost
from cognifold.query.models import QueryConfig, QueryType, RetrievalMode
from cognifold.query.text_utils import extract_keywords
from cognifold.scoring.ranker import ContextRanker, ScoringConfig

if TYPE_CHECKING:
    from cognifold.embeddings.embedder import NodeEmbedder
    from cognifold.embeddings.search import SemanticSearch
    from cognifold.graph.store import ConceptGraph
    from cognifold.models.node import Node
    from cognifold.retrieval.hybrid import HybridRetriever


def compute_text_match_score(query_keywords: set[str], node: Node) -> float:
    """Compute how well a node matches query keywords.

    Args:
        query_keywords: Keywords extracted from the query.
        node: Node to match against.

    Returns:
        Match score (0.0 to 1.0).
    """
    if not query_keywords:
        return 0.0

    # Collect text from node
    text_parts = []
    title = node.data.get("title", "")
    if title:
        text_parts.append(str(title))
    description = node.data.get("description", "")
    if description:
        text_parts.append(str(description))
    if node.reasoning:
        text_parts.append(node.reasoning)

    node_text = " ".join(text_parts)
    if not node_text:
        return 0.0

    node_keywords = extract_keywords(node_text)
    if not node_keywords:
        return 0.0

    # Count matching keywords
    matches = query_keywords & node_keywords
    if not matches:
        return 0.0

    # Score based on proportion of query keywords found
    return len(matches) / len(query_keywords)


@dataclass
class EntryPoint:
    """An entry point into the graph for query traversal.

    Attributes:
        node_id: The node ID to start traversal from.
        score: Initial relevance score for this entry point.
        source: How this entry point was identified (pagerank, recency, type, etc.).
    """

    node_id: str
    score: float
    source: str


@dataclass
class TraversalResult:
    """Result of graph traversal from entry points.

    Attributes:
        visited_nodes: List of (node_id, depth, score) tuples.
        traversal_path: Ordered list of node IDs as they were visited.
        nodes_scanned: Total number of nodes examined.
    """

    visited_nodes: list[tuple[str, int, float]] = field(default_factory=list)
    traversal_path: list[str] = field(default_factory=list)
    nodes_scanned: int = 0


class EntryPointSelector:
    """Selects entry points for graph traversal based on query type.

    Supports multiple retrieval backends:
    - LEGACY: Original keyword matching (no external dependencies)
    - BM25: Inverted index with BM25 scoring
    - SEMANTIC: Embedding-based similarity search
    - HYBRID: BM25 + semantic with RRF fusion (best quality)
    """

    def __init__(
        self,
        graph: ConceptGraph,
        config: QueryConfig | None = None,
        scoring_config: ScoringConfig | None = None,
        embedder: NodeEmbedder | None = None,
    ) -> None:
        """Initialize the selector.

        Args:
            graph: The concept graph to query.
            config: Query configuration.
            scoring_config: Scoring configuration for PageRank.
            embedder: Optional node embedder for semantic search.
        """
        self.graph = graph
        self.config = config or QueryConfig()
        self.ranker = ContextRanker(scoring_config)
        self._embedder = embedder

        # Lazy-initialized retrieval components
        self._hybrid_retriever: HybridRetriever | None = None
        self._semantic_search: SemanticSearch | None = None

    def reset_retrieval_cache(self) -> None:
        """Reset cached retrieval components (call after graph mutation)."""
        self._hybrid_retriever = None
        self._semantic_search = None

    def _ensure_hybrid_retriever(self) -> HybridRetriever:
        """Ensure hybrid retriever is initialized.

        Returns:
            HybridRetriever instance.
        """
        if self._hybrid_retriever is None:
            from cognifold.retrieval.config import RetrievalConfig
            from cognifold.retrieval.hybrid import HybridRetriever

            retrieval_config = RetrievalConfig(
                semantic_weight=self.config.semantic_weight,
                keyword_weight=self.config.keyword_weight,
            )
            self._hybrid_retriever = HybridRetriever(
                embedder=self._embedder,
                config=retrieval_config,
            )
        return self._hybrid_retriever

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
            self._semantic_search.build_index(self.graph)

        return self._semantic_search

    def select_entry_points(
        self,
        query_type: QueryType,
        reference_time: datetime | None = None,
        max_entry_points: int = 10,
        query_text: str | None = None,
    ) -> list[EntryPoint]:
        """Select entry points based on query type and retrieval mode.

        Args:
            query_type: The type of query being executed.
            reference_time: Reference time for temporal queries.
            max_entry_points: Maximum number of entry points to return.
            query_text: Original query for text-based search.

        Returns:
            List of entry points sorted by score descending.
        """
        if self.graph.node_count == 0:
            return []

        if reference_time is None:
            reference_time = datetime.now()

        # Collect entity-index matches (supplements any retrieval mode)
        entity_entry_points: list[EntryPoint] = []
        if query_text and self.graph.entity_index is not None:
            entity_entry_points = self._select_by_entity_index(query_text, max_entry_points)

        # For semantic and hybrid queries, use retrieval backend based on mode
        if query_text and query_type in (QueryType.SEMANTIC, QueryType.HYBRID):
            retrieval_mode = self.config.retrieval_mode

            if retrieval_mode == RetrievalMode.HYBRID:
                # Use full hybrid retrieval (BM25 + semantic with RRF)
                entry_points = self._select_by_hybrid_retrieval(query_text, max_entry_points)
                if entry_points:
                    return self._merge_entry_points(
                        entity_entry_points, entry_points, max_entry_points
                    )

            elif retrieval_mode == RetrievalMode.SEMANTIC:
                # Use embedding-based semantic search only
                entry_points = self._select_by_semantic_search(query_text, max_entry_points)
                if entry_points:
                    return self._merge_entry_points(
                        entity_entry_points, entry_points, max_entry_points
                    )

            elif retrieval_mode == RetrievalMode.BM25:
                # Use BM25 keyword search only
                entry_points = self._select_by_bm25_search(query_text, max_entry_points)
                if entry_points:
                    return self._merge_entry_points(
                        entity_entry_points, entry_points, max_entry_points
                    )

            else:  # LEGACY mode
                # Use original text matching
                entry_points = self._select_by_text_search(query_text, max_entry_points)
                if entry_points:
                    return self._merge_entry_points(
                        entity_entry_points, entry_points, max_entry_points
                    )

            # If no retrieval matches but entity matches exist, return those
            if entity_entry_points:
                return entity_entry_points[:max_entry_points]

            # Fall back to structural if no matches from retrieval

        elif entity_entry_points:
            # For non-semantic query types, still use entity matches if available
            return entity_entry_points[:max_entry_points]

        if query_type == QueryType.STRUCTURAL:
            return self._select_structural_entry_points(max_entry_points)
        elif query_type == QueryType.TEMPORAL:
            return self._select_temporal_entry_points(reference_time, max_entry_points)
        elif query_type == QueryType.SEMANTIC:
            return self._select_semantic_entry_points(max_entry_points)
        else:  # HYBRID
            return self._select_hybrid_entry_points(reference_time, max_entry_points)

    def _select_by_entity_index(self, query_text: str, max_entry_points: int) -> list[EntryPoint]:
        """Select entry points using the entity index.

        Extracts entities from the query text and finds nodes that mention
        those entities. Entity matches get a high base score since they
        represent exact entity alignment.

        Args:
            query_text: The query string.
            max_entry_points: Maximum number of entry points.

        Returns:
            List of entry points from entity index matches.
        """
        entity_index = self.graph.entity_index
        if entity_index is None:
            return []

        node_ids = entity_index.query_all_matches(query_text)
        if not node_ids:
            return []

        entry_points: list[EntryPoint] = []
        for node_id in node_ids[:max_entry_points]:
            if not self.graph.has_node(node_id):
                continue
            node = self.graph.get_node(node_id)
            score = apply_type_boost(0.85, node.type, ENTRY_POINT_BOOSTS, clamp=True)
            entry_points.append(EntryPoint(node_id=node_id, score=score, source="entity_index"))

        return entry_points

    @staticmethod
    def _merge_entry_points(
        entity_points: list[EntryPoint],
        retrieval_points: list[EntryPoint],
        max_entry_points: int,
    ) -> list[EntryPoint]:
        """Merge entity-index entry points with retrieval entry points.

        Uses interleaved merging: retrieval results first (they capture
        semantic/keyword relevance), then entity matches appended for
        additional coverage.  Total budget expands by up to 50% when
        both sources contribute unique nodes.

        Args:
            entity_points: Entry points from entity index.
            retrieval_points: Entry points from BM25/hybrid/semantic.
            max_entry_points: Maximum total entry points.

        Returns:
            Merged and de-duplicated list of entry points.
        """
        if not entity_points:
            return retrieval_points[:max_entry_points]

        # Allow expanded budget when both sources contribute
        expanded_limit = min(
            max_entry_points + len(entity_points) // 2,
            int(max_entry_points * 1.5),
        )

        seen: set[str] = set()
        merged: list[EntryPoint] = []

        # Retrieval matches first (semantic/keyword relevance)
        for ep in retrieval_points:
            if ep.node_id not in seen:
                seen.add(ep.node_id)
                merged.append(ep)

        # Then entity matches for additional coverage
        for ep in entity_points:
            if ep.node_id not in seen:
                seen.add(ep.node_id)
                merged.append(ep)

        return merged[:expanded_limit]

    def _select_by_text_search(self, query_text: str, max_entry_points: int) -> list[EntryPoint]:
        """Select entry points by searching for text matches.

        This is the primary strategy for semantic queries - find nodes
        whose content matches the query keywords.

        Args:
            query_text: The query string.
            max_entry_points: Maximum number of entry points.

        Returns:
            List of entry points with matching nodes.
        """
        query_keywords = extract_keywords(query_text)
        if not query_keywords:
            return []

        # Search all nodes for text matches
        matches: list[tuple[str, float]] = []
        for node in self.graph.get_all_nodes():
            score = compute_text_match_score(query_keywords, node)
            if score > 0:
                if self.config.prefer_concepts:
                    score = apply_type_boost(score, node.type, ENTRY_POINT_BOOSTS)
                matches.append((node.id, score))

        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)

        # Return top matches as entry points
        entry_points = []
        for node_id, score in matches[:max_entry_points]:
            entry_points.append(EntryPoint(node_id=node_id, score=score, source="text_search"))

        return entry_points

    def _select_by_hybrid_retrieval(
        self, query_text: str, max_entry_points: int
    ) -> list[EntryPoint]:
        """Select entry points using hybrid retrieval (BM25 + semantic with RRF).

        This provides the best quality results by combining keyword matching
        with semantic similarity using Reciprocal Rank Fusion.

        Args:
            query_text: The query string.
            max_entry_points: Maximum number of entry points.

        Returns:
            List of entry points from hybrid retrieval.
        """
        from cognifold.retrieval.config import RetrievalConfig, RetrievalStrategy

        retriever = self._ensure_hybrid_retriever()

        config = RetrievalConfig(
            strategy=RetrievalStrategy.HYBRID,
            top_k=max_entry_points,
            semantic_weight=self.config.semantic_weight,
            keyword_weight=self.config.keyword_weight,
        )

        results, _metrics = retriever.search(self.graph, query_text, config)

        if not results:
            return []

        # Rank-based scoring preserves relative ordering from RRF
        entry_points = []
        n = len(results)
        for rank, result in enumerate(results):
            normalized = 1.0 - rank / (n + 1) if n > 1 else 1.0

            score = normalized
            if result.node is not None and self.config.prefer_concepts:
                score = apply_type_boost(score, result.node.type, ENTRY_POINT_BOOSTS, clamp=True)

            entry_points.append(
                EntryPoint(node_id=result.node_id, score=score, source="hybrid_retrieval")
            )

        return entry_points

    def _select_by_semantic_search(
        self, query_text: str, max_entry_points: int
    ) -> list[EntryPoint]:
        """Select entry points using embedding-based semantic search.

        Uses cosine similarity between query embedding and node embeddings
        to find semantically related nodes.

        Args:
            query_text: The query string.
            max_entry_points: Maximum number of entry points.

        Returns:
            List of entry points from semantic search.
        """
        semantic = self._ensure_semantic_search()
        if semantic is None:
            # Fall back to text search if no embedder
            return self._select_by_text_search(query_text, max_entry_points)

        from cognifold.embeddings.search import SearchConfig

        config = SearchConfig(top_k=max_entry_points)
        results = semantic.search(self.graph, query_text, config)

        if not results:
            return []

        # Rank-based scoring preserves relative ordering from semantic search
        entry_points = []
        n = len(results)
        for rank, result in enumerate(results):
            normalized = 1.0 - rank / (n + 1) if n > 1 else 1.0

            score = normalized
            if result.node is not None and self.config.prefer_concepts:
                score = apply_type_boost(score, result.node.type, ENTRY_POINT_BOOSTS, clamp=True)

            entry_points.append(
                EntryPoint(node_id=result.node_id, score=score, source="semantic_search")
            )

        return entry_points

    def _select_by_bm25_search(self, query_text: str, max_entry_points: int) -> list[EntryPoint]:
        """Select entry points using BM25 keyword search.

        Uses BM25 inverted index for efficient keyword matching.
        Better than simple text matching for longer documents.

        Args:
            query_text: The query string.
            max_entry_points: Maximum number of entry points.

        Returns:
            List of entry points from BM25 search.
        """
        from cognifold.retrieval.config import RetrievalConfig, RetrievalStrategy

        retriever = self._ensure_hybrid_retriever()

        config = RetrievalConfig(
            strategy=RetrievalStrategy.KEYWORD,
            top_k=max_entry_points,
        )

        results, _metrics = retriever.search(self.graph, query_text, config)

        if not results:
            return []

        # Rank-based scoring preserves relative ordering from BM25
        entry_points = []
        n = len(results)
        for rank, result in enumerate(results):
            normalized = 1.0 - rank / (n + 1) if n > 1 else 1.0

            score = normalized
            if result.node is not None and self.config.prefer_concepts:
                score = apply_type_boost(score, result.node.type, ENTRY_POINT_BOOSTS, clamp=True)

            entry_points.append(
                EntryPoint(node_id=result.node_id, score=score, source="bm25_search")
            )

        return entry_points

    def _select_structural_entry_points(self, max_entry_points: int) -> list[EntryPoint]:
        """Select entry points based on PageRank (structural importance).

        Focuses on highly connected nodes, preferring concepts over events.
        """
        pagerank_scores = self.ranker.compute_pagerank(self.graph)

        # Sort by PageRank score
        sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

        entry_points = []
        for node_id, score in sorted_nodes[:max_entry_points]:
            entry_points.append(EntryPoint(node_id=node_id, score=score, source="pagerank"))

        return entry_points

    def _select_temporal_entry_points(
        self, reference_time: datetime, max_entry_points: int
    ) -> list[EntryPoint]:
        """Select entry points based on recency.

        Focuses on recently accessed/created nodes.
        """
        nodes = self.graph.get_all_nodes()

        # Score by recency
        scored_nodes: list[tuple[str, float]] = []
        for node in nodes:
            recency = self.ranker.compute_recency_score(node, reference_time)
            scored_nodes.append((node.id, recency))

        # Sort by recency
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        entry_points = []
        for node_id, score in scored_nodes[:max_entry_points]:
            entry_points.append(EntryPoint(node_id=node_id, score=score, source="recency"))

        return entry_points

    def _select_semantic_entry_points(self, max_entry_points: int) -> list[EntryPoint]:
        """Select entry points for semantic queries.

        Focuses on concept and action nodes (higher-level abstractions)
        as they are more likely to be semantically relevant.
        """
        # Get concepts and actions first (they contain semantic meaning)
        concepts = self.graph.get_nodes_by_type(NodeType.CONCEPT)
        actions = self.graph.get_nodes_by_type(NodeType.INTENT)

        # Combine and score by PageRank
        pagerank_scores = self.ranker.compute_pagerank(self.graph)

        candidates: list[tuple[str, float]] = []
        for node in concepts + actions:
            score = pagerank_scores.get(node.id, 0.0)
            # Boost concepts slightly over actions
            if node.type == NodeType.CONCEPT:
                score *= 1.2
            candidates.append((node.id, score))

        # If not enough concepts/actions, add high-PageRank events
        if len(candidates) < max_entry_points:
            events = self.graph.get_nodes_by_type(NodeType.EVENT)
            for node in events:
                score = pagerank_scores.get(node.id, 0.0)
                candidates.append((node.id, score))

        # Sort and return top entries
        candidates.sort(key=lambda x: x[1], reverse=True)

        entry_points = []
        for node_id, score in candidates[:max_entry_points]:
            entry_points.append(EntryPoint(node_id=node_id, score=score, source="semantic"))

        return entry_points

    def _select_hybrid_entry_points(
        self, reference_time: datetime, max_entry_points: int
    ) -> list[EntryPoint]:
        """Select entry points using a hybrid strategy.

        Combines PageRank, recency, and node type preferences.
        """
        nodes = self.graph.get_all_nodes()
        pagerank_scores = self.ranker.compute_pagerank(self.graph)

        scored_nodes: list[tuple[str, float]] = []
        for node in nodes:
            # Get component scores
            structural = pagerank_scores.get(node.id, 0.0)
            recency = self.ranker.compute_recency_score(node, reference_time)

            # Type-based boost
            type_boost = 1.0
            if self.config.prefer_concepts:
                type_boost = apply_type_boost(1.0, node.type, ENTRY_POINT_BOOSTS)

            # Combine scores
            combined = (0.5 * structural + 0.5 * recency) * type_boost
            scored_nodes.append((node.id, combined))

        # Sort and return
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        entry_points = []
        for node_id, score in scored_nodes[:max_entry_points]:
            entry_points.append(EntryPoint(node_id=node_id, score=score, source="hybrid"))

        return entry_points


class GraphTraverser:
    """Traverses the graph from entry points to collect relevant nodes."""

    def __init__(
        self,
        graph: ConceptGraph,
        config: QueryConfig | None = None,
        ranker: ContextRanker | None = None,
    ) -> None:
        """Initialize the traverser.

        Args:
            graph: The concept graph to traverse.
            config: Query configuration.
            ranker: Optional context ranker for PPR blending.
        """
        self.graph = graph
        self.config = config or QueryConfig()
        self.ranker = ranker

    def traverse(
        self,
        entry_points: list[EntryPoint],
        max_depth: int | None = None,
        edge_weights: dict[str, float] | None = None,
    ) -> TraversalResult:
        """Traverse the graph from entry points using BFS.

        Collects nodes up to max_depth hops from any entry point.
        Node scores decay with distance from entry points.

        Args:
            entry_points: Starting points for traversal.
            max_depth: Maximum traversal depth (default from config).
            edge_weights: Optional edge-type weight multipliers from
                intent router. Maps edge_type -> multiplier (>1 boosts, <1 suppresses).

        Returns:
            TraversalResult with visited nodes and path.
        """
        if max_depth is None:
            max_depth = self.config.max_traversal_depth

            # Adaptive depth: allow deeper traversal for larger graphs
            # (multi-hop benchmarks like MuSiQue need depth >= 4)
            if self.graph.node_count > 50 and max_depth < 4:
                max_depth = 4

        visited: dict[str, tuple[int, float]] = {}  # node_id -> (depth, score)
        traversal_path: list[str] = []
        nodes_scanned = 0

        # Initialize queue with entry points
        queue: deque[tuple[str, int, float]] = deque()
        for ep in entry_points:
            if ep.node_id not in visited:
                queue.append((ep.node_id, 0, ep.score))
                visited[ep.node_id] = (0, ep.score)

        # BFS traversal
        while queue:
            node_id, depth, score = queue.popleft()
            nodes_scanned += 1
            traversal_path.append(node_id)

            # Stop if we've reached max depth
            if depth >= max_depth:
                continue

            # Get neighbors (both directions for undirected-like traversal)
            neighbors = set(self.graph.get_neighbors(node_id))
            predecessors = set(self.graph.get_predecessors(node_id))
            all_connected = neighbors | predecessors

            for neighbor_id in all_connected:
                # Compute edge-type weight multiplier if intent routing is active
                edge_multiplier = 1.0
                if edge_weights:
                    edge_multiplier = self._get_edge_multiplier(node_id, neighbor_id, edge_weights)

                neighbor_score = score * BFS_DECAY_PER_HOP * edge_multiplier
                if neighbor_id not in visited:
                    queue.append((neighbor_id, depth + 1, neighbor_score))
                    visited[neighbor_id] = (depth + 1, neighbor_score)
                elif visited[neighbor_id][1] < neighbor_score:
                    # Update score if we found a better path
                    visited[neighbor_id] = (depth + 1, neighbor_score)

        # --- PPR blending (Phase Wave 5) ---
        if self.config.use_ppr and self.ranker is not None and len(visited) > 1:
            seed_ids = [ep.node_id for ep in entry_points]
            ppr_scores = self.ranker.compute_personalized_pagerank(self.graph, seed_ids)
            if ppr_scores:
                # Normalize PPR scores to [0, 1]
                max_ppr = max(ppr_scores.values()) or 1.0
                for node_id in visited:
                    depth, bfs_score = visited[node_id]
                    ppr_score = ppr_scores.get(node_id, 0.0) / max_ppr
                    blended = 0.6 * bfs_score + 0.4 * ppr_score
                    visited[node_id] = (depth, blended)

        # Convert to result format and apply diversity penalty
        visited_list = [(node_id, depth, score) for node_id, (depth, score) in visited.items()]
        visited_list = self._apply_diversity_penalty(visited_list)

        return TraversalResult(
            visited_nodes=visited_list,
            traversal_path=traversal_path,
            nodes_scanned=nodes_scanned,
        )

    def _get_edge_multiplier(
        self,
        source_id: str,
        target_id: str,
        edge_weights: dict[str, float],
    ) -> float:
        """Get the intent-based weight multiplier for an edge.

        Looks up the edge type connecting source to target and returns the
        corresponding multiplier from the intent router weights.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            edge_weights: Edge-type -> multiplier mapping from intent router.

        Returns:
            Weight multiplier (default 1.0 if no edge type found).
        """
        edges = self.graph.get_edges_between(source_id, target_id)
        if not edges:
            # Try reverse direction
            edges = self.graph.get_edges_between(target_id, source_id)

        if not edges:
            return 1.0

        # Use the best (highest multiplier) edge type if multiple edges exist
        best = 1.0
        for edge in edges:
            et = edge.edge_type
            if et and et in edge_weights:
                best = max(best, edge_weights[et])
        return best

    def _apply_diversity_penalty(
        self,
        visited_list: list[tuple[str, int, float]],
    ) -> list[tuple[str, int, float]]:
        """Penalize nodes that share many neighbors with already-selected nodes.

        Sorts by score descending, then greedily penalizes nodes whose
        graph neighborhood heavily overlaps with higher-scored nodes.
        This promotes diversity in the final retrieval set.

        Args:
            visited_list: List of (node_id, depth, score) tuples.

        Returns:
            Updated list with diversity-adjusted scores.
        """
        # Only apply diversity on large result sets where redundancy is likely
        if len(visited_list) <= 20:
            return visited_list

        # Sort by score descending
        sorted_nodes = sorted(visited_list, key=lambda x: x[2], reverse=True)

        # Build neighbor sets for each visited node (cached)
        neighbor_cache: dict[str, set[str]] = {}
        for node_id, _, _ in sorted_nodes:
            try:
                neighbors = set(self.graph.get_neighbors(node_id))
                predecessors = set(self.graph.get_predecessors(node_id))
                neighbor_cache[node_id] = neighbors | predecessors
            except KeyError:
                neighbor_cache[node_id] = set()

        # Greedily select, penalizing overlap with already-selected
        selected_neighborhoods: list[set[str]] = []
        result: list[tuple[str, int, float]] = []

        for node_id, depth, score in sorted_nodes:
            node_neighbors = neighbor_cache.get(node_id, set())
            if node_neighbors and selected_neighborhoods:
                # Compute max overlap ratio with any already-selected node
                max_overlap = 0.0
                for sel_neighbors in selected_neighborhoods:
                    if sel_neighbors:
                        overlap = len(node_neighbors & sel_neighbors)
                        union = len(node_neighbors | sel_neighbors)
                        if union > 0:
                            max_overlap = max(max_overlap, overlap / union)
                # Apply mild penalty (up to 15% reduction) only for high overlap
                if max_overlap > 0.5:
                    penalty = 1.0 - (0.15 * max_overlap)
                    score = score * penalty

            selected_neighborhoods.append(node_neighbors)
            result.append((node_id, depth, score))

        return result

    def traverse_temporal(
        self,
        reference_time: datetime,
        time_window_hours: float = 24.0,
    ) -> TraversalResult:
        """Traverse nodes within a time window.

        Collects all nodes accessed/created within the time window.

        Args:
            reference_time: Center of the time window.
            time_window_hours: Size of the window in hours.

        Returns:
            TraversalResult with nodes in the time window.
        """
        nodes = self.graph.get_all_nodes()
        window_start = reference_time - timedelta(hours=time_window_hours / 2)
        window_end = reference_time + timedelta(hours=time_window_hours / 2)

        visited: list[tuple[str, int, float]] = []
        traversal_path: list[str] = []

        for node in nodes:
            # Check if node is within time window
            node_time = node.last_accessed

            # Handle timezone-aware vs naive datetime
            ref_start = window_start
            ref_end = window_end
            if node_time.tzinfo is not None:
                if ref_start.tzinfo is None:
                    ref_start = ref_start.replace(tzinfo=node_time.tzinfo)
                    ref_end = ref_end.replace(tzinfo=node_time.tzinfo)
            else:
                if ref_start.tzinfo is not None:
                    ref_start = ref_start.replace(tzinfo=None)
                    ref_end = ref_end.replace(tzinfo=None)

            if ref_start <= node_time <= ref_end:
                # Score based on proximity to reference time
                time_diff = abs((node_time - reference_time).total_seconds() / 3600)
                score = 1.0 - (time_diff / (time_window_hours / 2))
                score = max(0.0, score)

                visited.append((node.id, 0, score))
                traversal_path.append(node.id)

        return TraversalResult(
            visited_nodes=visited,
            traversal_path=traversal_path,
            nodes_scanned=len(nodes),
        )
