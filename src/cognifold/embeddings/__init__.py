"""Embeddings module for semantic search and similarity.

This module provides tools for generating and managing embeddings for graph nodes,
enabling semantic similarity search beyond keyword matching.
"""

from cognifold.embeddings.config import EmbeddingConfig
from cognifold.embeddings.embedder import NodeEmbedder
from cognifold.embeddings.providers import (
    EmbeddingProvider,
    GeminiEmbeddingProvider,
    MockEmbeddingProvider,
)
from cognifold.embeddings.search import SemanticSearch

__all__ = [
    "EmbeddingConfig",
    "EmbeddingProvider",
    "GeminiEmbeddingProvider",
    "MockEmbeddingProvider",
    "NodeEmbedder",
    "SemanticSearch",
]
