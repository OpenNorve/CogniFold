"""Embedding service for vectorizing text.

DEPRECATED: Use the unified ``cognifold.embeddings`` module instead.
This module is kept for backward compatibility and will be removed
in a future release.

Migration guide:
    from cognifold.embeddings import EmbeddingConfig, NodeEmbedder
    config = EmbeddingConfig.for_openai()
    embedder = NodeEmbedder(config)
"""

from __future__ import annotations

import logging
import warnings

logger = logging.getLogger(__name__)

warnings.warn(
    "cognifold.utils.embeddings is deprecated. Use cognifold.embeddings instead.",
    DeprecationWarning,
    stacklevel=2,
)


class EmbeddingService:
    """Service for generating text embeddings."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        """Initialize the embedding service.

        Args:
            model: The model name to use.
        """
        from cognifold.service.llm_keys import get_api_key

        self.model = model
        self.client = None

        # Initialize OpenAI client if API key is present
        if get_api_key("OPENAI_API_KEY"):
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))
            except ImportError:
                logger.warning("OpenAI package not found. Embeddings will not work.")
        else:
            logger.warning("OPENAI_API_KEY not found. Embeddings will not work.")

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text string.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        if not text:
            return []

        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not self.client:
            logger.error("Embedding service not initialized properly.")
            return [[] for _ in texts]

        if not texts:
            return []

        try:
            # Clean texts
            cleaned_texts = [t.replace("\n", " ") for t in texts]

            response = self.client.embeddings.create(
                input=cleaned_texts,
                model=self.model,
            )

            return [data.embedding for data in response.data]

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return empty embeddings on failure
            return [[] for _ in texts]


# Global instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service instance.

    Re-creates the service if the current instance has no client but
    an API key is now available (e.g., inside an llm_key_scope context).
    """
    from cognifold.service.llm_keys import get_api_key

    global _embedding_service
    if _embedding_service is None or (
        _embedding_service.client is None and get_api_key("OPENAI_API_KEY")
    ):
        _embedding_service = EmbeddingService()
    return _embedding_service


def reset_embedding_service() -> None:
    """Reset the global embedding service (e.g., after API key changes)."""
    global _embedding_service
    _embedding_service = None
