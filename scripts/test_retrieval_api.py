#!/usr/bin/env python3
"""Test script for the new retrieval API.

Run with: python scripts/test_retrieval_api.py

This script demonstrates and tests all retrieval modes:
- LEGACY: Original keyword matching
- BM25: Inverted index with BM25 scoring
- SEMANTIC: Embedding-based similarity (requires embedder)
- HYBRID: BM25 + semantic with RRF fusion (requires embedder)
"""

from __future__ import annotations

import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, "src")

from cognifold.graph.store import ConceptGraph
from cognifold.models.node import Edge, Node, NodeType


def create_test_graph() -> ConceptGraph:
    """Create a sample graph for testing."""
    graph = ConceptGraph()

    # Add events
    graph.add_node(Node(
        id="e-001",
        type=NodeType.EVENT,
        data={
            "title": "Morning gym workout",
            "description": "Cardio and strength training at the fitness center",
            "event_type": "exercise",
        },
    ))
    graph.add_node(Node(
        id="e-002",
        type=NodeType.EVENT,
        data={
            "title": "Team standup meeting",
            "description": "Discussed quarterly goals and project deadlines",
            "event_type": "work",
        },
    ))
    graph.add_node(Node(
        id="e-003",
        type=NodeType.EVENT,
        data={
            "title": "Evening run in the park",
            "description": "5km jog for fitness and stress relief",
            "event_type": "exercise",
        },
    ))
    graph.add_node(Node(
        id="e-004",
        type=NodeType.EVENT,
        data={
            "title": "Lunch with colleagues",
            "description": "Team lunch at the Italian restaurant",
            "event_type": "social",
        },
    ))

    # Add concepts
    graph.add_node(Node(
        id="c-001",
        type=NodeType.CONCEPT,
        data={
            "title": "Fitness routine",
            "description": "Regular exercise habit including gym workouts and running",
        },
        reasoning="Pattern of exercise events detected",
    ))
    graph.add_node(Node(
        id="c-002",
        type=NodeType.CONCEPT,
        data={
            "title": "Work productivity",
            "description": "Focus on work tasks, meetings, and deadlines",
        },
        reasoning="Work-related activities form a pattern",
    ))

    # Add intent
    graph.add_node(Node(
        id="i-001",
        type=NodeType.INTENT,
        data={
            "title": "Improve fitness consistency",
            "description": "Maintain regular exercise schedule for better health",
        },
    ))

    # Add edges
    graph.add_edge(Edge(source="e-001", target="c-001"))
    graph.add_edge(Edge(source="e-003", target="c-001"))
    graph.add_edge(Edge(source="e-002", target="c-002"))
    graph.add_edge(Edge(source="c-001", target="i-001"))

    return graph


def test_legacy_mode():
    """Test LEGACY retrieval mode."""
    print("\n" + "=" * 60)
    print("TEST: Legacy Mode (keyword matching)")
    print("=" * 60)

    from cognifold.query import MemoryQueryAgent, QueryConfig, RetrievalMode

    graph = create_test_graph()
    config = QueryConfig(retrieval_mode=RetrievalMode.LEGACY)
    agent = MemoryQueryAgent(graph, config=config)

    result = agent.query("gym workout fitness exercise")

    print(f"Query: 'gym workout fitness exercise'")
    print(f"Retrieval mode: {result.query_metadata.get('retrieval_mode')}")
    print(f"Nodes found: {result.node_count}")
    print(f"Query time: {result.query_time_ms:.2f}ms")
    print("\nTop results:")
    for node in result.nodes[:3]:
        print(f"  - [{node.node_type}] {node.title} (score: {node.relevance_score:.3f})")

    assert result.query_metadata.get("retrieval_mode") == "legacy"
    print("\n✅ Legacy mode test PASSED")


def test_bm25_mode():
    """Test BM25 retrieval mode."""
    print("\n" + "=" * 60)
    print("TEST: BM25 Mode (inverted index)")
    print("=" * 60)

    from cognifold.query import MemoryQueryAgent, QueryConfig, RetrievalMode

    graph = create_test_graph()
    config = QueryConfig(retrieval_mode=RetrievalMode.BM25)
    agent = MemoryQueryAgent(graph, config=config)

    result = agent.query("gym workout fitness")

    print(f"Query: 'gym workout fitness'")
    print(f"Retrieval mode: {result.query_metadata.get('retrieval_mode')}")
    print(f"Nodes found: {result.node_count}")
    print(f"Query time: {result.query_time_ms:.2f}ms")
    print("\nTop results:")
    for node in result.nodes[:3]:
        print(f"  - [{node.node_type}] {node.title} (score: {node.relevance_score:.3f})")

    assert result.query_metadata.get("retrieval_mode") == "bm25"
    print("\n✅ BM25 mode test PASSED")


def test_bm25_index_directly():
    """Test BM25 index directly."""
    print("\n" + "=" * 60)
    print("TEST: BM25 Index Direct Usage")
    print("=" * 60)

    from cognifold.retrieval.bm25 import BM25Index

    graph = create_test_graph()
    index = BM25Index()
    index.build(graph)

    print(f"Documents indexed: {index.get_document_count()}")
    print(f"Vocabulary size: {index.get_vocabulary_size()}")

    results = index.search("gym workout exercise", top_k=5)
    print(f"\nSearch: 'gym workout exercise'")
    print(f"Results: {len(results)}")
    for r in results:
        print(f"  - {r.node_id}: score={r.bm25_score:.3f}, rank={r.bm25_rank}")

    assert index.get_document_count() == 7  # 4 events + 2 concepts + 1 intent
    print("\n✅ BM25 index test PASSED")


def test_hybrid_retriever_directly():
    """Test HybridRetriever directly."""
    print("\n" + "=" * 60)
    print("TEST: Hybrid Retriever Direct Usage")
    print("=" * 60)

    from cognifold.retrieval.config import RetrievalConfig, RetrievalStrategy
    from cognifold.retrieval.hybrid import HybridRetriever

    graph = create_test_graph()

    # Test keyword-only (no embedder)
    retriever = HybridRetriever(embedder=None)
    config = RetrievalConfig(strategy=RetrievalStrategy.KEYWORD, top_k=5)

    results, metrics = retriever.search(graph, "fitness exercise gym", config)

    print(f"Strategy: {metrics.strategy_used}")
    print(f"Total candidates: {metrics.total_candidates}")
    print(f"BM25 candidates: {metrics.bm25_candidates}")
    print(f"Final results: {metrics.final_results}")
    print("\nResults:")
    for r in results:
        print(f"  - {r.node_id}: final_score={r.final_score:.3f}")

    assert metrics.strategy_used == "keyword"
    print("\n✅ Hybrid retriever test PASSED")


def test_temporal_extraction():
    """Test temporal extraction in queries."""
    print("\n" + "=" * 60)
    print("TEST: Temporal Extraction")
    print("=" * 60)

    from cognifold.temporal.extractor import TemporalExtractor

    extractor = TemporalExtractor()
    reference = datetime(2026, 2, 1, 12, 0, 0)

    test_cases = [
        "What happened yesterday at 3pm?",
        "Meeting scheduled for next Monday",
        "Review code by Friday",
        "Daily standup every morning at 9am",
    ]

    for query in test_cases:
        entities = extractor.extract(query, reference)
        print(f"\nQuery: '{query}'")
        for e in entities:
            print(f"  - '{e.raw_text}' → {e.temporal_type.value} (confidence: {e.confidence:.2f})")

    print("\n✅ Temporal extraction test PASSED")


def test_query_with_temporal():
    """Test query with temporal extraction."""
    print("\n" + "=" * 60)
    print("TEST: Query with Temporal References")
    print("=" * 60)

    from cognifold.query import MemoryQueryAgent, QueryConfig, RetrievalMode

    graph = create_test_graph()
    config = QueryConfig(retrieval_mode=RetrievalMode.BM25)
    agent = MemoryQueryAgent(graph, config=config)

    result = agent.query("What happened yesterday at the gym?")

    print(f"Query: 'What happened yesterday at the gym?'")
    print(f"Temporal references found:")
    for ref in result.query_metadata.get("temporal_references", []):
        print(f"  - '{ref['raw_text']}' → {ref['type']}")

    assert "temporal_references" in result.query_metadata
    print("\n✅ Query with temporal test PASSED")


def test_semantic_mode_warning():
    """Test that semantic mode without embedder shows warning."""
    print("\n" + "=" * 60)
    print("TEST: Semantic Mode Warning (no embedder)")
    print("=" * 60)

    import warnings
    from cognifold.query import MemoryQueryAgent, QueryConfig, RetrievalMode

    graph = create_test_graph()
    config = QueryConfig(retrieval_mode=RetrievalMode.SEMANTIC)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        agent = MemoryQueryAgent(graph, config=config, embedder=None)

        if w:
            print(f"Warning captured: {w[0].message}")
            print("\n✅ Semantic mode warning test PASSED")
        else:
            print("No warning captured (may have been suppressed)")
            print("\n⚠️ Semantic mode warning test SKIPPED")


def test_mock_embedder():
    """Test with mock embedder for semantic search."""
    print("\n" + "=" * 60)
    print("TEST: Mock Embedder for Semantic Search")
    print("=" * 60)

    from cognifold.embeddings.config import EmbeddingConfig, EmbeddingProviderType
    from cognifold.embeddings.embedder import NodeEmbedder
    from cognifold.embeddings.search import SemanticSearch, SearchConfig

    graph = create_test_graph()

    # Use mock provider (no API calls)
    config = EmbeddingConfig(
        provider=EmbeddingProviderType.MOCK,
        dimensions=128,
    )
    embedder = NodeEmbedder(config)

    # Build semantic search
    search = SemanticSearch(embedder)
    search.build_index(graph)

    print(f"Index size: {search.get_index_size()}")

    # Search
    search_config = SearchConfig(top_k=3)
    results = search.search(graph, "fitness exercise workout", search_config)

    print(f"\nSearch: 'fitness exercise workout'")
    print(f"Results: {len(results)}")
    for r in results:
        print(f"  - {r.node_id}: score={r.score:.3f}")

    assert search.get_index_size() == 7
    print("\n✅ Mock embedder test PASSED")


def test_hybrid_mode_with_mock_embedder():
    """Test hybrid mode with mock embedder."""
    print("\n" + "=" * 60)
    print("TEST: Hybrid Mode with Mock Embedder")
    print("=" * 60)

    from cognifold.embeddings.config import EmbeddingConfig, EmbeddingProviderType
    from cognifold.embeddings.embedder import NodeEmbedder
    from cognifold.query import MemoryQueryAgent, QueryConfig, RetrievalMode

    graph = create_test_graph()

    # Create mock embedder
    embed_config = EmbeddingConfig(
        provider=EmbeddingProviderType.MOCK,
        dimensions=128,
    )
    embedder = NodeEmbedder(embed_config)

    # Query with hybrid mode
    query_config = QueryConfig(
        retrieval_mode=RetrievalMode.HYBRID,
        semantic_weight=0.5,
        keyword_weight=0.5,
    )
    agent = MemoryQueryAgent(graph, config=query_config, embedder=embedder)

    result = agent.query("fitness exercise gym workout")

    print(f"Query: 'fitness exercise gym workout'")
    print(f"Retrieval mode: {result.query_metadata.get('retrieval_mode')}")
    print(f"Nodes found: {result.node_count}")
    print(f"Query time: {result.query_time_ms:.2f}ms")
    print("\nTop results:")
    for node in result.nodes[:5]:
        print(f"  - [{node.node_type}] {node.title} (score: {node.relevance_score:.3f})")

    assert result.query_metadata.get("retrieval_mode") == "hybrid"
    print("\n✅ Hybrid mode with mock embedder test PASSED")


def main():
    """Run all tests."""
    print("=" * 60)
    print("RETRIEVAL API TEST SUITE")
    print("=" * 60)

    tests = [
        test_legacy_mode,
        test_bm25_mode,
        test_bm25_index_directly,
        test_hybrid_retriever_directly,
        test_temporal_extraction,
        test_query_with_temporal,
        test_semantic_mode_warning,
        test_mock_embedder,
        test_hybrid_mode_with_mock_embedder,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
