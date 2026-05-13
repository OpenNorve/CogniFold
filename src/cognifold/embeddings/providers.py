"""Embedding providers for generating text embeddings."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from cognifold.embeddings.config import EmbeddingConfig


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Embedding providers generate vector representations of text that capture
    semantic meaning. Different providers use different models and APIs.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """Initialize the provider with configuration.

        Args:
            config: Embedding configuration.
        """
        self.config = config

    @abstractmethod
    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as numpy array.
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[NDArray[np.float32]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        pass

    def normalize(self, embedding: NDArray[np.float32]) -> NDArray[np.float32]:
        """L2-normalize an embedding vector.

        Args:
            embedding: The embedding to normalize.

        Returns:
            Normalized embedding with unit length.
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return (embedding / norm).astype(np.float32)
        return embedding


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing.

    Generates deterministic embeddings based on text hash.
    """

    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Generate a deterministic mock embedding from text hash.

        Args:
            text: The text to embed.

        Returns:
            Mock embedding vector.
        """
        # Use hash to seed a random generator for deterministic embeddings
        hash_digest = hashlib.sha256(text.encode()).hexdigest()
        seed = int(hash_digest[:8], 16)  # Use first 8 hex chars as seed

        # Create deterministic random generator
        rng = np.random.Generator(np.random.PCG64(seed))

        # Generate embedding from normal distribution
        embedding = rng.standard_normal(self.config.dimensions).astype(np.float32)

        if self.config.normalize:
            embedding = self.normalize(embedding)

        return embedding

    def embed_batch(self, texts: list[str]) -> list[NDArray[np.float32]]:
        """Generate mock embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of mock embedding vectors.
        """
        return [self.embed_text(text) for text in texts]


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using Google's Gemini API.

    Uses the google-genai library for embedding generation.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """Initialize the Gemini provider.

        Args:
            config: Embedding configuration.

        Raises:
            ImportError: If google-genai is not installed.
            ValueError: If API key is not provided.
        """
        super().__init__(config)

        # Get API key from config, thread-local scope, or environment
        from cognifold.service.llm_keys import get_api_key

        self.api_key = config.api_key or get_api_key("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GOOGLE_API_KEY env var or pass api_key in config."
            )

        # Import and configure the client
        try:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
            self._genai = genai
        except ImportError as e:
            raise ImportError(
                "google-genai package required for Gemini embeddings. "
                "Install with: pip install google-genai"
            ) from e

    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Generate embedding using Gemini API.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector.
        """
        from google.genai import types as _genai_types

        result = self._client.models.embed_content(
            model=self.config.model,
            contents=text,
            config=_genai_types.EmbedContentConfig(
                output_dimensionality=self.config.dimensions,
            ),
        )

        if result.embeddings is None:
            raise ValueError("Gemini API returned no embeddings")
        embedding = np.array(result.embeddings[0].values, dtype=np.float32)

        if self.config.normalize:
            embedding = self.normalize(embedding)

        return embedding

    def embed_batch(self, texts: list[str]) -> list[NDArray[np.float32]]:
        """Generate embeddings for multiple texts using Gemini API.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        embeddings: list[NDArray[np.float32]] = []

        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]

            from google.genai import types as _genai_types

            # Gemini API can handle multiple texts at once
            result = self._client.models.embed_content(
                model=self.config.model,
                contents=batch,  # type: ignore[arg-type]
                config=_genai_types.EmbedContentConfig(
                    output_dimensionality=self.config.dimensions,
                ),
            )

            if result.embeddings is None:
                continue
            for emb in result.embeddings:
                embedding = np.array(emb.values, dtype=np.float32)
                if self.config.normalize:
                    embedding = self.normalize(embedding)
                embeddings.append(embedding)

        return embeddings


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using OpenAI's API.

    Uses the openai library for embedding generation.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """Initialize the OpenAI provider.

        Args:
            config: Embedding configuration.

        Raises:
            ImportError: If openai is not installed.
            ValueError: If API key is not provided.
        """
        super().__init__(config)

        # Get API key from config, thread-local scope, or environment
        from cognifold.service.llm_keys import get_api_key

        self.api_key = config.api_key or get_api_key("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key in config."
            )

        # Import and configure the client
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        except ImportError as e:
            raise ImportError(
                "openai package required for OpenAI embeddings. Install with: pip install openai"
            ) from e

    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Generate embedding using OpenAI API.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector.
        """
        response = self._client.embeddings.create(
            model=self.config.model,
            input=text,
        )

        embedding = np.array(response.data[0].embedding, dtype=np.float32)

        if self.config.normalize:
            embedding = self.normalize(embedding)

        return embedding

    def embed_batch(self, texts: list[str]) -> list[NDArray[np.float32]]:
        """Generate embeddings for multiple texts using OpenAI API.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        embeddings: list[NDArray[np.float32]] = []

        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]

            response = self._client.embeddings.create(
                model=self.config.model,
                input=batch,
            )

            for data in response.data:
                embedding = np.array(data.embedding, dtype=np.float32)
                if self.config.normalize:
                    embedding = self.normalize(embedding)
                embeddings.append(embedding)

        return embeddings


def create_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    """Factory function to create an embedding provider.

    Args:
        config: Embedding configuration.

    Returns:
        An EmbeddingProvider instance.

    Raises:
        ValueError: If provider type is unknown.
    """
    from cognifold.embeddings.config import EmbeddingProviderType

    if config.provider == EmbeddingProviderType.MOCK:
        return MockEmbeddingProvider(config)
    elif config.provider == EmbeddingProviderType.GEMINI:
        return GeminiEmbeddingProvider(config)
    elif config.provider == EmbeddingProviderType.OPENAI:
        return OpenAIEmbeddingProvider(config)
    else:
        raise ValueError(f"Unknown embedding provider: {config.provider}")
