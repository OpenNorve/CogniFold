"""Post-ingestion edge inference using stored node embeddings.

When LLM-generated UpdatePlans fail to create edges (e.g. referencing
titles instead of IDs), the graph ends up with 0 edges. This module
provides a general-purpose edge inference engine that creates RELATED_TO
edges between nodes with high embedding similarity.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from cognifold.models.node import BaseEdgeType, Edge

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph

logger = logging.getLogger(__name__)


class EdgeInferenceEngine:
    """Infer edges between nodes based on embedding similarity.

    **Opt-in only** -- this engine is NOT auto-invoked during normal event
    ingestion or plan execution.  Callers must explicitly instantiate it and
    call :meth:`infer_edges`.

    *Why opt-in?*  The kNN-based approach connects each node to its top-k
    most similar neighbors by cosine similarity.  On small graphs (< ~100
    nodes) this tends to over-connect unrelated nodes, causing regressions
    in retrieval precision and benchmark scores.  For the common case of
    reconnecting orphaned nodes created by faulty LLM plans, the executor's
    built-in orphan detection (``executor/runner.py``,
    :meth:`PlanExecutor._detect_orphan_nodes`) is preferred: it uses the
    deterministic ``grounded_in`` references on each node to create GROUNDS
    edges, which is cheaper and more precise than similarity-based inference.

    Use this engine when you need dense inter-node connectivity on larger
    graphs where similarity-based edges add meaningful signal (e.g. after
    bulk wiki ingestion with hundreds of chunks).

    Uses a k-nearest-neighbor approach: for each node, connect it to its
    top-k most similar neighbors (by cosine similarity), subject to a
    minimum similarity floor. This works even when absolute similarities
    are moderate (e.g. multi-hop QA paragraphs about diverse topics).

    Args:
        graph: The concept graph to add edges to.
        similarity_threshold: Minimum cosine similarity to create an edge.
        max_edges_per_node: Maximum inferred edges per node (kNN k value).
    """

    def __init__(
        self,
        graph: ConceptGraph,
        similarity_threshold: float = 0.30,
        max_edges_per_node: int = 3,
        source_types: list[str] | None = None,
        target_types: list[str] | None = None,
    ) -> None:
        self.graph = graph
        self.similarity_threshold = similarity_threshold
        self.max_edges_per_node = max_edges_per_node
        self.source_types = source_types
        self.target_types = target_types

    def infer_edges(self) -> list[Edge]:
        """For each node, connect to its top-k nearest neighbors by cosine similarity.

        Returns:
            List of newly created Edge objects.
        """
        nodes = self.graph.get_all_nodes()

        # Collect nodes that have embeddings (with optional type filtering)
        allowed_types = set()
        if self.source_types:
            allowed_types.update(self.source_types)
        if self.target_types:
            allowed_types.update(self.target_types)

        embedded_nodes: list[tuple[str, np.ndarray]] = []
        for node in nodes:
            if allowed_types and node.type.value not in allowed_types:
                continue
            if node.embedding:
                vec = np.array(node.embedding)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    embedded_nodes.append((node.id, vec / norm))

        if len(embedded_nodes) < 2:
            logger.debug(
                "EdgeInference: fewer than 2 nodes with embeddings (%d), skipping",
                len(embedded_nodes),
            )
            return []

        logger.info(
            "EdgeInference: computing kNN edges for %d nodes (k=%d, min_sim=%.2f)",
            len(embedded_nodes),
            self.max_edges_per_node,
            self.similarity_threshold,
        )

        # Build full similarity matrix (vectorized for efficiency)
        n = len(embedded_nodes)
        vecs = np.stack([v for _, v in embedded_nodes])  # (n, dim)
        sim_matrix = vecs @ vecs.T  # (n, n) cosine similarity matrix

        # Track edges created per node
        edge_count: dict[str, int] = {nid: 0 for nid, _ in embedded_nodes}
        created_edges: list[Edge] = []

        # For each node, find top-k neighbors
        for i in range(n):
            node_a_id = embedded_nodes[i][0]
            if edge_count[node_a_id] >= self.max_edges_per_node:
                continue

            # Get similarities to all other nodes, sorted descending
            sims = sim_matrix[i].copy()
            sims[i] = -1.0  # exclude self
            ranked_indices = np.argsort(sims)[::-1]

            slots = self.max_edges_per_node - edge_count[node_a_id]
            for j_idx in ranked_indices[: slots * 2]:  # check extra in case of filters
                j = int(j_idx)
                similarity = float(sims[j])

                if similarity < self.similarity_threshold:
                    break  # sorted, so no better candidates

                node_b_id = embedded_nodes[j][0]
                if edge_count[node_b_id] >= self.max_edges_per_node:
                    continue

                # Skip if edge already exists (either direction)
                if self.graph.has_edge(node_a_id, node_b_id):
                    continue
                if self.graph.has_edge(node_b_id, node_a_id):
                    continue

                edge = Edge.create(
                    source=node_a_id,
                    target=node_b_id,
                    edge_type=BaseEdgeType.RELATED_TO.value,
                    weight=round(similarity, 3),
                )

                try:
                    self.graph.add_edge(edge)
                    created_edges.append(edge)
                    edge_count[node_a_id] += 1
                    edge_count[node_b_id] += 1
                except (KeyError, ValueError) as exc:
                    logger.debug(
                        "EdgeInference: skipped edge %s→%s: %s",
                        node_a_id,
                        node_b_id,
                        exc,
                    )

                if edge_count[node_a_id] >= self.max_edges_per_node:
                    break

        logger.info("EdgeInference: created %d edges", len(created_edges))
        return created_edges
