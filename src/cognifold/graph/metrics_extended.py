"""Extended metrics for Symbolic Emergence evaluation."""

import numpy as np

from cognifold.graph.store import ConceptGraph
from cognifold.models.node import NodeType


class EmergenceMetrics:
    """Calculates metrics related to symbolic emergence and memory consolidation."""

    def __init__(self, graph: ConceptGraph):
        self.graph = graph

    def calculate_compression_ratio(self) -> float:
        """Calculate compression ratio (Events / Concepts)."""
        events = len(self.graph.get_nodes_by_type(NodeType.EVENT))
        concepts = len(self.graph.get_nodes_by_type(NodeType.CONCEPT))
        if concepts == 0:
            return float(events)  # Infinite compression? Or just return events count as fallback
        return events / concepts

    def calculate_concept_purity(self) -> float:
        """Calculate average internal semantic coherence of concepts.

        Returns:
            Average pairwise cosine similarity of events grounded in each concept.
            Range: [-1.0, 1.0], higher is better (purer).
        """
        concepts = self.graph.get_nodes_by_type(NodeType.CONCEPT)
        if not concepts:
            return 0.0

        total_purity = 0.0
        valid_concepts = 0

        for concept in concepts:
            # Get grounded events
            event_ids = concept.grounded_in
            embeddings = []
            for eid in event_ids:
                if self.graph.has_node(eid):
                    node = self.graph.get_node(eid)
                    if node.embedding:
                        embeddings.append(node.embedding)

            if len(embeddings) < 2:
                # Trivial purity for singletons or empty
                continue

            try:
                # Calculate average pairwise similarity
                mat = np.array(embeddings)

                # Normalize rows
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                # Avoid division by zero
                norms[norms == 0] = 1.0
                mat_norm = mat / norms

                # Dot product (Cosine Similarity)
                sims = np.dot(mat_norm, mat_norm.T)

                # Exclude self-similarity (diagonal)
                np.fill_diagonal(sims, 0)

                avg_sim = np.sum(sims) / (len(embeddings) * (len(embeddings) - 1))
                total_purity += avg_sim
                valid_concepts += 1
            except Exception:
                continue

        if valid_concepts == 0:
            return 0.0

        return total_purity / valid_concepts

    def calculate_graph_stats(self) -> dict[str, float]:
        """Return a dictionary of all metrics."""
        return {
            "compression_ratio": self.calculate_compression_ratio(),
            "concept_purity": self.calculate_concept_purity(),
            "node_count": self.graph.node_count,
            "edge_count": self.graph.edge_count,
            "concept_count": len(self.graph.get_nodes_by_type(NodeType.CONCEPT)),
            "event_count": len(self.graph.get_nodes_by_type(NodeType.EVENT)),
        }
