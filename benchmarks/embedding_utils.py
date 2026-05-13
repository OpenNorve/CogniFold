"""Shared embedding utilities for benchmark runners.

Provides a factory function to create an embedder based on available API keys.
All benchmark runners should use this to enable hybrid (BM25 + semantic) retrieval.

Usage in benchmark runners:
    try:
        from benchmarks.embedding_utils import create_embedder
    except ImportError:
        create_embedder = None

    # Then when creating MemoryQueryAgent:
    embedder = create_embedder() if create_embedder else None
    query_agent = MemoryQueryAgent(graph, config=query_config, embedder=embedder)
"""

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognifold.embeddings.embedder import NodeEmbedder

logger = logging.getLogger(__name__)


def create_embedder() -> "NodeEmbedder | None":
    """Create an embedder based on available API keys.

    Tries providers in order: OpenAI > Gemini > None (fallback to BM25).

    Returns:
        NodeEmbedder instance or None if no provider available.
    """
    try:
        from cognifold.embeddings.config import EmbeddingConfig
        from cognifold.embeddings.embedder import NodeEmbedder

        if os.environ.get("OPENAI_API_KEY"):
            config = EmbeddingConfig.for_openai()
            logger.info("Using OpenAI embeddings for hybrid retrieval")
            return NodeEmbedder(config)
        elif os.environ.get("GOOGLE_API_KEY"):
            config = EmbeddingConfig.for_gemini()
            logger.info("Using Gemini embeddings for hybrid retrieval")
            return NodeEmbedder(config)
        else:
            logger.warning(
                "No embedding API key found (OPENAI_API_KEY or GOOGLE_API_KEY). "
                "Falling back to BM25-only retrieval."
            )
            return None
    except Exception as e:
        logger.warning(f"Failed to initialize embedder: {e}. Using BM25 only.")
        return None
