"""Shared utilities for benchmark runners."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Add src to python path
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src",
    )
)

if TYPE_CHECKING:
    from cognifold.embeddings.embedder import NodeEmbedder
    from cognifold.query.models import RetrievalMode


def load_embedding_config(profile_path: Path, benchmark_name: str) -> str:
    """Load embedding model from profile YAML.

    Args:
        profile_path: Path to the profile YAML file.
        benchmark_name: Name of the benchmark in the profile.

    Returns:
        Embedding model string, e.g. "openai:text-embedding-3-small" or "none".
    """
    if not profile_path.exists():
        return "none"
    try:
        import yaml

        with open(profile_path) as f:
            raw = yaml.safe_load(f)
        bench_raw = raw.get("profiles", {}).get(benchmark_name, {})
        embedding_cfg = bench_raw.get("embedding", {})
        if isinstance(embedding_cfg, dict):
            return embedding_cfg.get("model", "none")
        return "none"
    except Exception:
        return "none"


def create_embedder(
    embedding: str,
) -> tuple[NodeEmbedder | None, RetrievalMode]:
    """Create an embedder based on the embedding model string.

    The embedding string format is "provider:model_name" (e.g. "openai:text-embedding-3-small")
    or "none" to disable embeddings and use BM25.

    Args:
        embedding: Embedding model string from CLI or config.
            - "none" → BM25 only
            - "openai:text-embedding-3-small" → OpenAI embeddings
            - "gemini:text-embedding-004" → Gemini embeddings

    Returns:
        Tuple of (embedder, retrieval_mode).

    Raises:
        RuntimeError: If the required API key is not set.
        ValueError: If the provider is unknown.
    """
    from cognifold.embeddings.config import EmbeddingConfig
    from cognifold.embeddings.embedder import NodeEmbedder
    from cognifold.query.models import RetrievalMode

    if embedding == "none":
        return None, RetrievalMode.BM25

    # Parse "provider:model_name" format
    if ":" not in embedding:
        raise ValueError(
            f"Invalid embedding format: '{embedding}'. "
            "Expected 'provider:model_name' (e.g. 'openai:text-embedding-3-small') or 'none'."
        )

    provider, model_name = embedding.split(":", 1)

    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                f"Embedding '{embedding}' requires OPENAI_API_KEY environment variable"
            )
        config = EmbeddingConfig.for_openai(model=model_name)
    elif provider == "gemini":
        if not os.environ.get("GOOGLE_API_KEY"):
            raise RuntimeError(
                f"Embedding '{embedding}' requires GOOGLE_API_KEY environment variable"
            )
        config = EmbeddingConfig.for_gemini(model=model_name)
    else:
        raise ValueError(f"Unknown embedding provider: '{provider}'. Supported: openai, gemini")

    embedder = NodeEmbedder(config)

    # Validate: test embedding call to catch auth/model errors early
    try:
        embedder.provider.embed_text("test")
    except Exception as e:
        raise RuntimeError(
            f"Embedding '{embedding}' failed validation: {e}"
        ) from e

    return embedder, RetrievalMode.HYBRID


def resolve_embedding(cli_embedding: str | None, profile_path: Path, benchmark_name: str) -> str:
    """Resolve embedding model: CLI arg overrides config file.

    Args:
        cli_embedding: Value from --embedding CLI arg, or None if not specified.
        profile_path: Path to the profile YAML file.
        benchmark_name: Name of the benchmark.

    Returns:
        Resolved embedding model string.
    """
    if cli_embedding is not None:
        return cli_embedding
    return load_embedding_config(profile_path, benchmark_name)
