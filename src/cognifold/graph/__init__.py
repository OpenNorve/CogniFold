"""Graph operations for Cognifold."""

from cognifold.graph.consolidation import merge_similar_concepts, prune_orphan_concepts
from cognifold.graph.edge_inference import EdgeInferenceEngine
from cognifold.graph.entity_index import EntityIndex
from cognifold.graph.fact_extraction import extract_facts
from cognifold.graph.metrics import MetricsCollector, QualityMetrics
from cognifold.graph.persistence import load_graph, save_graph
from cognifold.graph.projection import (
    GraphProjection,
    GraphSnapshot,
    NetworkXProjection,
    graph_to_snapshot,
)
from cognifold.graph.store import ConceptGraph
from cognifold.graph.validator import (
    GraphValidator,
    IntegrityIssue,
    IntegrityLevel,
    ValidationReport,
)

__all__ = [
    "ConceptGraph",
    "EdgeInferenceEngine",
    "EntityIndex",
    "GraphProjection",
    "GraphSnapshot",
    "GraphValidator",
    "IntegrityIssue",
    "IntegrityLevel",
    "MetricsCollector",
    "NetworkXProjection",
    "QualityMetrics",
    "ValidationReport",
    "extract_facts",
    "graph_to_snapshot",
    "load_graph",
    "merge_similar_concepts",
    "prune_orphan_concepts",
    "save_graph",
]
