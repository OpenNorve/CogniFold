"""Semantic search using embeddings and cosine similarity.

Supports an optional FAISS backend for fast approximate nearest neighbor
search. Falls back to brute-force numpy if faiss-cpu is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from cognifold.embeddings.embedder import NodeEmbedder
    from cognifold.graph.store import ConceptGraph
    from cognifold.models.node import Node

logger = logging.getLogger(__name__)

# Try to import FAISS (optional dependency)
try:
    import faiss  # type: ignore[import-untyped]

    _faiss_available = True
except ImportError:
    faiss = None  # type: ignore[assignment]
    _faiss_available = False


@dataclass
class SearchResult:
    """A single search result with similarity score.

    Attributes:
        node_id: ID of the matching node.
        score: Similarity score (0.0 to 1.0 for cosine similarity).
        node: The actual node object (optional, may be lazy loaded).
    """

    node_id: str
    score: float
    node: Node | None = None

    def to_dict(self) -> dict[str, float | str]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "score": self.score,
        }


@dataclass
class SearchConfig:
    """Configuration for semantic search.

    Attributes:
        top_k: Maximum number of results to return.
        min_score: Minimum similarity score threshold.
        include_node_types: Only include these node types (None = all).
        exclude_node_types: Exclude these node types.
        boost_recent: Boost score for recently accessed nodes.
        recency_weight: Weight for recency boost (0.0 to 1.0).
        use_faiss: Use FAISS for fast ANN search if available. Default True.
    """

    top_k: int = 10
    min_score: float = 0.0
    include_node_types: list[str] | None = None
    exclude_node_types: list[str] = field(default_factory=list)
    boost_recent: bool = False
    recency_weight: float = 0.1
    use_faiss: bool = True


class _FaissIndex:
    """Wrapper around FAISS IndexIDMap for integer-keyed vector search."""

    def __init__(self, dimensions: int) -> None:
        assert faiss is not None, "faiss must be installed to use _FaissIndex"
        inner = faiss.IndexFlatIP(dimensions)
        self._index: Any = faiss.IndexIDMap(inner)
        self._id_to_node_id: dict[int, str] = {}
        self._node_id_to_id: dict[str, int] = {}
        self._next_id: int = 0
        self.dimensions = dimensions

    def add(self, node_id: str, embedding: NDArray[np.float32]) -> None:
        """Add a single embedding to the index.

        If node_id already exists, removes the old entry first to
        prevent orphaned vectors in the FAISS index.
        """
        if node_id in self._node_id_to_id:
            self.remove(node_id)
        vec = np.ascontiguousarray(embedding.reshape(1, -1), dtype=np.float32)
        # Normalize for cosine similarity via inner product
        assert faiss is not None
        faiss.normalize_L2(vec)
        int_id = self._next_id
        self._next_id += 1
        self._id_to_node_id[int_id] = node_id
        self._node_id_to_id[node_id] = int_id
        self._index.add_with_ids(vec, np.array([int_id], dtype=np.int64))

    def remove(self, node_id: str) -> None:
        """Remove a node from the index."""
        int_id = self._node_id_to_id.pop(node_id, None)
        if int_id is not None:
            self._index.remove_ids(np.array([int_id], dtype=np.int64))
            del self._id_to_node_id[int_id]

    def search(self, query: NDArray[np.float32], top_k: int) -> list[tuple[str, float]]:
        """Search for nearest neighbors. Returns list of (node_id, score)."""
        if self._index.ntotal == 0:
            return []
        vec = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        assert faiss is not None
        faiss.normalize_L2(vec)
        k = min(top_k, self._index.ntotal)
        scores, ids = self._index.search(vec, k)
        results: list[tuple[str, float]] = []
        for score, int_id in zip(scores[0], ids[0]):
            if int_id == -1:
                continue
            node_id = self._id_to_node_id.get(int(int_id))
            if node_id is not None:
                results.append((node_id, float(score)))
        return results

    @property
    def size(self) -> int:
        return int(self._index.ntotal)


class SemanticSearch:
    """Semantic search over graph nodes using embeddings.

    Uses cosine similarity between query embedding and node embeddings
    to find semantically similar nodes.

    When faiss-cpu is installed and use_faiss=True (default), uses FAISS
    IndexFlatIP for O(1)-ish search. Otherwise falls back to brute-force
    numpy cosine similarity.

    Example:
        >>> embedder = NodeEmbedder(config)
        >>> search = SemanticSearch(embedder)
        >>> results = search.search(graph, "exercise and fitness")
        >>> for r in results:
        ...     print(f"{r.node_id}: {r.score:.3f}")
    """

    def __init__(
        self,
        embedder: NodeEmbedder,
        config: SearchConfig | None = None,
    ) -> None:
        """Initialize semantic search.

        Args:
            embedder: Node embedder for generating embeddings.
            config: Search configuration.
        """
        self.embedder = embedder
        self.config = config or SearchConfig()

        # Numpy brute-force index (always available)
        self._index: dict[str, NDArray[np.float32]] | None = None
        self._indexed_graph: ConceptGraph | None = None
        self._indexed_revision: int | None = None

        # FAISS index (optional, lazy-built)
        self._faiss_index: _FaissIndex | None = None
        self._use_faiss = self.config.use_faiss and _faiss_available

        if self.config.use_faiss and not _faiss_available:
            logger.debug("FAISS not installed, falling back to numpy brute-force search")

    def build_index(self, graph: ConceptGraph) -> None:
        """Build search index from graph nodes.

        Generates embeddings for all nodes and stores them for fast search.

        Args:
            graph: The concept graph to index.
        """
        self._index = self.embedder.embed_graph(graph)
        self._indexed_graph = graph
        self._indexed_revision = graph.revision

        # Build FAISS index if available
        if self._use_faiss and self._index:
            dims = self.embedder.config.dimensions
            self._faiss_index = _FaissIndex(dims)
            for node_id, embedding in self._index.items():
                self._faiss_index.add(node_id, embedding)
            logger.debug("Built FAISS index with %d vectors", self._faiss_index.size)

    def _ensure_index(self, graph: ConceptGraph) -> dict[str, NDArray[np.float32]]:
        """Ensure index is built for the graph.

        Args:
            graph: The concept graph.

        Returns:
            The embedding index.
        """
        if (
            self._index is None
            or self._indexed_graph is not graph
            or self._indexed_revision != graph.revision
        ):
            self.build_index(graph)

        return self._index  # type: ignore[return-value]

    def search(
        self,
        graph: ConceptGraph,
        query: str,
        config: SearchConfig | None = None,
    ) -> list[SearchResult]:
        """Search for nodes semantically similar to a query.

        Args:
            graph: The concept graph to search.
            query: The search query text.
            config: Optional config override for this search.

        Returns:
            List of SearchResult sorted by score (descending).
        """
        cfg = config or self.config
        index = self._ensure_index(graph)
        query_embedding = self.embedder.embed_query(query)

        # Fast path: use FAISS if available
        if self._faiss_index is not None:
            return self._search_faiss(graph, query_embedding, cfg)

        # Slow path: brute-force numpy
        return self._search_numpy(graph, query_embedding, index, cfg)

    def _search_faiss(
        self,
        graph: ConceptGraph,
        query_embedding: NDArray[np.float32],
        cfg: SearchConfig,
    ) -> list[SearchResult]:
        """FAISS-accelerated search with post-hoc filtering."""
        assert self._faiss_index is not None
        # Over-fetch to handle type filtering
        fetch_k = cfg.top_k * 4 if (cfg.include_node_types or cfg.exclude_node_types) else cfg.top_k
        raw_results = self._faiss_index.search(query_embedding, fetch_k)

        results: list[SearchResult] = []
        for node_id, score in raw_results:
            if score < cfg.min_score:
                continue
            node = graph.get_node_or_none(node_id)
            if node is None:
                continue
            if cfg.include_node_types is not None and node.type.value not in cfg.include_node_types:
                continue
            if node.type.value in cfg.exclude_node_types:
                continue
            results.append(SearchResult(node_id=node_id, score=score, node=node))
            if len(results) >= cfg.top_k:
                break

        # Post-hoc recency boost (applied after FAISS search)
        if cfg.boost_recent:
            for result in results:
                if result.node is not None:
                    recency_score = self._calculate_recency_score(result.node)
                    result.score = (
                        1 - cfg.recency_weight
                    ) * result.score + cfg.recency_weight * recency_score
            # Re-sort after recency adjustment
            results.sort(key=lambda r: r.score, reverse=True)

        return results

    def _search_numpy(
        self,
        graph: ConceptGraph,
        query_embedding: NDArray[np.float32],
        index: dict[str, NDArray[np.float32]],
        cfg: SearchConfig,
    ) -> list[SearchResult]:
        """Brute-force numpy cosine similarity search."""
        results: list[SearchResult] = []

        for node_id, node_embedding in index.items():
            node = graph.get_node_or_none(node_id)
            if node is None:
                continue
            if cfg.include_node_types is not None and node.type.value not in cfg.include_node_types:
                continue
            if node.type.value in cfg.exclude_node_types:
                continue

            score = self._cosine_similarity(query_embedding, node_embedding)

            if cfg.boost_recent:
                recency_score = self._calculate_recency_score(node)
                score = (1 - cfg.recency_weight) * score + cfg.recency_weight * recency_score

            if score >= cfg.min_score:
                results.append(SearchResult(node_id=node_id, score=float(score), node=node))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[: cfg.top_k]

    def search_similar_to_node(
        self,
        graph: ConceptGraph,
        node_id: str,
        config: SearchConfig | None = None,
    ) -> list[SearchResult]:
        """Find nodes similar to a given node.

        Args:
            graph: The concept graph to search.
            node_id: ID of the node to find similar nodes for.
            config: Optional config override.

        Returns:
            List of SearchResult (excluding the query node).
        """
        cfg = config or self.config
        index = self._ensure_index(graph)

        if node_id not in index:
            node = graph.get_node_or_none(node_id)
            if node is None:
                return []
            query_embedding = self.embedder.embed_node(node)
        else:
            query_embedding = index[node_id]

        # Use FAISS fast path
        if self._faiss_index is not None:
            # Fetch one extra to account for excluding self
            raw_results = self._faiss_index.search(query_embedding, cfg.top_k + 1)
            results: list[SearchResult] = []
            for nid, score in raw_results:
                if nid == node_id:
                    continue
                if score < cfg.min_score:
                    continue
                node = graph.get_node_or_none(nid)
                if node is None:
                    continue
                if (
                    cfg.include_node_types is not None
                    and node.type.value not in cfg.include_node_types
                ):
                    continue
                if node.type.value in cfg.exclude_node_types:
                    continue
                results.append(SearchResult(node_id=nid, score=score, node=node))
                if len(results) >= cfg.top_k:
                    break

            # Post-hoc recency boost (applied after FAISS search)
            if cfg.boost_recent:
                for result in results:
                    if result.node is not None:
                        recency_score = self._calculate_recency_score(result.node)
                        result.score = (
                            1 - cfg.recency_weight
                        ) * result.score + cfg.recency_weight * recency_score
                # Re-sort after recency adjustment
                results.sort(key=lambda r: r.score, reverse=True)

            return results

        # Brute-force fallback
        results = []
        for other_id, other_embedding in index.items():
            if other_id == node_id:
                continue
            node = graph.get_node_or_none(other_id)
            if node is None:
                continue
            if cfg.include_node_types is not None and node.type.value not in cfg.include_node_types:
                continue
            if node.type.value in cfg.exclude_node_types:
                continue
            score = self._cosine_similarity(query_embedding, other_embedding)
            if score >= cfg.min_score:
                results.append(SearchResult(node_id=other_id, score=float(score), node=node))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[: cfg.top_k]

    def search_by_embedding(
        self,
        graph: ConceptGraph,
        embedding: NDArray[np.float32],
        config: SearchConfig | None = None,
    ) -> list[SearchResult]:
        """Search using a pre-computed embedding.

        Args:
            graph: The concept graph to search.
            embedding: The query embedding vector.
            config: Optional config override.

        Returns:
            List of SearchResult sorted by score.
        """
        cfg = config or self.config
        self._ensure_index(graph)

        # FAISS fast path
        if self._faiss_index is not None:
            return self._search_faiss(graph, embedding, cfg)

        # Numpy fallback
        index = self._index or {}
        return self._search_numpy(graph, embedding, index, cfg)

    def _cosine_similarity(
        self,
        a: NDArray[np.float32],
        b: NDArray[np.float32],
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

    def _calculate_recency_score(self, node: Node) -> float:
        """Calculate a recency score for a node."""
        from datetime import datetime

        now = datetime.now()
        last_accessed = node.last_accessed

        # Handle timezone-aware vs naive datetime mismatch
        if last_accessed.tzinfo is not None and now.tzinfo is None:
            now = now.replace(tzinfo=last_accessed.tzinfo)
        elif last_accessed.tzinfo is None and now.tzinfo is not None:
            last_accessed = last_accessed.replace(tzinfo=now.tzinfo)

        hours_since_access = (now - last_accessed).total_seconds() / 3600.0
        recency = 0.5 ** (hours_since_access / 24.0)
        return float(recency)

    def invalidate_index(self) -> None:
        """Invalidate the search index."""
        self._index = None
        self._indexed_graph = None
        self._indexed_revision = None
        self._faiss_index = None

    def update_node_embedding(
        self,
        graph: ConceptGraph,
        node_id: str,
    ) -> None:
        """Update embedding for a single node."""
        if self._index is None:
            return

        node = graph.get_node_or_none(node_id)
        if node is None:
            self._index.pop(node_id, None)
            if self._faiss_index is not None:
                self._faiss_index.remove(node_id)
        else:
            new_embedding = self.embedder.embed_node(node)
            self._index[node_id] = new_embedding
            if self._faiss_index is not None:
                self._faiss_index.remove(node_id)
                self._faiss_index.add(node_id, new_embedding)

    def remove_node_from_index(self, node_id: str) -> None:
        """Remove a node from the search index."""
        if self._index is not None:
            self._index.pop(node_id, None)
        if self._faiss_index is not None:
            self._faiss_index.remove(node_id)
        self.embedder.remove_from_cache(node_id)

    def get_index_size(self) -> int:
        """Get the number of nodes in the index."""
        if self._faiss_index is not None:
            return self._faiss_index.size
        if self._index is None:
            return 0
        return len(self._index)
