"""Configuration for embedding generation and storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EmbeddingProviderType(str, Enum):
    """Supported embedding providers."""

    GEMINI = "gemini"
    OPENAI = "openai"
    MOCK = "mock"  # For testing


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation.

    Attributes:
        provider: The embedding provider to use.
        model: The embedding model name (provider-specific).
        dimensions: Output embedding dimensions.
        batch_size: Maximum batch size for embedding requests.
        cache_embeddings: Whether to cache embeddings in nodes.
        lazy_generation: If True, generate embeddings on first query.
                        If False, generate on node creation.
        api_key: API key for the provider (optional, uses env var if not set).
        normalize: Whether to L2-normalize embeddings.
        extra_config: Provider-specific configuration.
    """

    provider: EmbeddingProviderType = EmbeddingProviderType.GEMINI
    model: str = "gemini-embedding-001"
    dimensions: int = 768
    batch_size: int = 100
    cache_embeddings: bool = True
    lazy_generation: bool = True
    api_key: str | None = None
    normalize: bool = True
    extra_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_testing(cls) -> EmbeddingConfig:
        """Create a config suitable for testing (mock provider)."""
        return cls(
            provider=EmbeddingProviderType.MOCK,
            model="mock-model",
            dimensions=128,  # Smaller for faster tests
            batch_size=10,
            cache_embeddings=True,
            lazy_generation=False,
            normalize=True,
        )

    @classmethod
    def for_gemini(
        cls,
        api_key: str | None = None,
        model: str = "gemini-embedding-001",
    ) -> EmbeddingConfig:
        """Create a config for Gemini embeddings."""
        return cls(
            provider=EmbeddingProviderType.GEMINI,
            model=model,
            dimensions=768,
            api_key=api_key,
        )

    @classmethod
    def for_openai(
        cls,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
    ) -> EmbeddingConfig:
        """Create a config for OpenAI embeddings."""
        return cls(
            provider=EmbeddingProviderType.OPENAI,
            model=model,
            dimensions=1536,
            api_key=api_key,
        )
