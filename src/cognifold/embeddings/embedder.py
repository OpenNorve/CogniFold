"""Node embedder for generating and caching node embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from cognifold.embeddings.config import EmbeddingConfig
    from cognifold.embeddings.providers import EmbeddingProvider
    from cognifold.graph.store import ConceptGraph
    from cognifold.models.node import Node


class NodeEmbedder:
    """Generates and manages embeddings for graph nodes.

    The NodeEmbedder creates text representations of nodes and generates
    embeddings using the configured provider. Embeddings can be cached
    in memory or persisted with the graph.

    Attributes:
        config: Embedding configuration.
        provider: The embedding provider instance.
        cache: In-memory cache of node embeddings.
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        """Initialize the node embedder.

        Args:
            config: Embedding configuration.
            provider: Optional pre-created provider. If None, creates one.
        """
        self.config = config
        self._cache: dict[str, NDArray[np.float32]] = {}

        if provider is not None:
            self.provider = provider
        else:
            from cognifold.embeddings.providers import create_provider

            self.provider = create_provider(config)

    def get_node_text(self, node: Node) -> str:
        """Extract text representation from a node for embedding.

        Combines title, description, reasoning, and other relevant fields
        into a single text representation.

        Args:
            node: The node to extract text from.

        Returns:
            Text representation of the node.
        """
        parts: list[str] = []

        # Get title from data (different fields depending on node type)
        title = node.data.get("title") or node.data.get("name") or ""
        if title:
            parts.append(f"Title: {title}")

        # Get description
        description = node.data.get("description") or ""
        if description:
            parts.append(f"Description: {description}")

        # Add reasoning for non-event nodes
        if node.reasoning:
            parts.append(f"Reasoning: {node.reasoning}")

        # Add type context
        parts.append(f"Type: {node.type.value}")

        # Add event-specific fields
        if node.type.value == "event":
            event_type = node.data.get("event_type")
            if event_type:
                parts.append(f"Event type: {event_type}")

            context = node.data.get("context")
            if context:
                parts.append(f"Context: {context}")

        # Add concept-specific fields
        elif node.type.value == "concept":
            temporal_pattern = node.data.get("temporal_pattern")
            if temporal_pattern:
                parts.append(f"Pattern: {temporal_pattern}")

        # Add intent-specific fields
        elif node.type.value == "intent":
            status = node.data.get("status")
            if status:
                parts.append(f"Status: {status}")

            priority = node.data.get("priority")
            if priority:
                parts.append(f"Priority: {priority}")

        # Combine all parts
        return " | ".join(parts) if parts else node.id

    def embed_node(self, node: Node) -> NDArray[np.float32]:
        """Generate or retrieve embedding for a single node.

        Args:
            node: The node to embed.

        Returns:
            Embedding vector for the node.
        """
        # Check cache first
        if self.config.cache_embeddings and node.id in self._cache:
            return self._cache[node.id]

        # Generate embedding
        text = self.get_node_text(node)
        embedding = self.provider.embed_text(text)

        # Cache if enabled
        if self.config.cache_embeddings:
            self._cache[node.id] = embedding

        return embedding

    def embed_nodes(self, nodes: list[Node]) -> dict[str, NDArray[np.float32]]:
        """Generate embeddings for multiple nodes.

        Uses batch processing for efficiency.

        Args:
            nodes: List of nodes to embed.

        Returns:
            Dictionary mapping node IDs to embeddings.
        """
        result: dict[str, NDArray[np.float32]] = {}

        # Separate cached and uncached nodes
        uncached_nodes: list[Node] = []
        for node in nodes:
            if self.config.cache_embeddings and node.id in self._cache:
                result[node.id] = self._cache[node.id]
            else:
                uncached_nodes.append(node)

        # Generate embeddings for uncached nodes in batch
        if uncached_nodes:
            texts = [self.get_node_text(node) for node in uncached_nodes]
            embeddings = self.provider.embed_batch(texts)

            for node, embedding in zip(uncached_nodes, embeddings):
                result[node.id] = embedding
                if self.config.cache_embeddings:
                    self._cache[node.id] = embedding

        return result

    def embed_graph(self, graph: ConceptGraph) -> dict[str, NDArray[np.float32]]:
        """Generate embeddings for all nodes in a graph.

        Args:
            graph: The concept graph to embed.

        Returns:
            Dictionary mapping node IDs to embeddings.
        """
        nodes = graph.get_all_nodes()
        return self.embed_nodes(nodes)

    def embed_query(self, query: str) -> NDArray[np.float32]:
        """Generate embedding for a query string.

        Args:
            query: The query text to embed.

        Returns:
            Embedding vector for the query.
        """
        return self.provider.embed_text(query)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    def remove_from_cache(self, node_id: str) -> None:
        """Remove a specific node from the cache.

        Args:
            node_id: ID of the node to remove.
        """
        self._cache.pop(node_id, None)

    def get_cached_count(self) -> int:
        """Get the number of cached embeddings.

        Returns:
            Number of embeddings in cache.
        """
        return len(self._cache)

    def is_cached(self, node_id: str) -> bool:
        """Check if a node's embedding is cached.

        Args:
            node_id: ID of the node to check.

        Returns:
            True if embedding is cached.
        """
        return node_id in self._cache

    def export_embeddings(self) -> dict[str, list[float]]:
        """Export cached embeddings as serializable dict.

        Returns:
            Dictionary mapping node IDs to embedding lists.
        """
        return {node_id: embedding.tolist() for node_id, embedding in self._cache.items()}

    def import_embeddings(self, embeddings: dict[str, list[float]]) -> None:
        """Import embeddings from serialized format.

        Args:
            embeddings: Dictionary mapping node IDs to embedding lists.
        """
        for node_id, embedding_list in embeddings.items():
            self._cache[node_id] = np.array(embedding_list, dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        """Serialize embedder state for persistence.

        Returns:
            Dictionary with embeddings and config.
        """
        return {
            "embeddings": self.export_embeddings(),
            "config": {
                "provider": self.config.provider.value,
                "model": self.config.model,
                "dimensions": self.config.dimensions,
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        config: EmbeddingConfig | None = None,
    ) -> NodeEmbedder:
        """Create embedder from serialized state.

        Args:
            data: Serialized embedder data.
            config: Optional config override.

        Returns:
            NodeEmbedder with restored embeddings.
        """
        if config is None:
            from cognifold.embeddings.config import EmbeddingConfig, EmbeddingProviderType

            saved_config = data.get("config", {})
            config = EmbeddingConfig(
                provider=EmbeddingProviderType(saved_config.get("provider", "mock")),
                model=saved_config.get("model", "mock-model"),
                dimensions=saved_config.get("dimensions", 768),
            )

        embedder = cls(config)
        embeddings = data.get("embeddings", {})
        embedder.import_embeddings(embeddings)

        return embedder
