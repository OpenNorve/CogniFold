"""Retrieval module for hybrid search combining BM25 and semantic search.

This module provides tools for retrieving graph nodes using:
- BM25 (lexical/keyword matching)
- Semantic search (embedding-based similarity)
- Hybrid retrieval (combining both with RRF fusion)
- Agentic retrieval (multi-round with LLM sufficiency checking)
"""

from cognifold.retrieval.agentic import AgenticRetriever
from cognifold.retrieval.bm25 import BM25Index
from cognifold.retrieval.config import RetrievalConfig, RetrievalStrategy
from cognifold.retrieval.hybrid import HybridRetriever
from cognifold.retrieval.result import RetrievalResult

__all__ = [
    "AgenticRetriever",
    "BM25Index",
    "HybridRetriever",
    "RetrievalConfig",
    "RetrievalResult",
    "RetrievalStrategy",
]
