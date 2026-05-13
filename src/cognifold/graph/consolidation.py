"""Post-ingestion concept consolidation.

Provides utilities to clean up the concept graph after ingestion:
- merge_similar_concepts: merge concept nodes with near-identical titles
- prune_orphan_concepts: tag isolated concept nodes as low-confidence
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph

logger = logging.getLogger(__name__)


def _normalize_title(title: str) -> str:
    """Normalize a title for comparison (lowercase, strip whitespace)."""
    return title.strip().lower()


def _title_similarity(a: str, b: str) -> float:
    """Compute normalized string similarity between two titles.

    Uses SequenceMatcher which handles insertions, deletions, and
    substitutions — better than edit distance for natural language titles.

    Args:
        a: First title.
        b: Second title.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    na = _normalize_title(a)
    nb = _normalize_title(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def merge_similar_concepts(
    graph: ConceptGraph,
    threshold: float = 0.85,
) -> int:
    """Merge concept nodes with very similar titles.

    For each pair of concepts above the similarity threshold, keeps the
    one with more edges and transfers all edges from the other before
    removing it.

    Args:
        graph: The concept graph to consolidate.
        threshold: Minimum title similarity to trigger a merge (0.0-1.0).

    Returns:
        Number of merges performed.
    """
    from cognifold.models.node import NodeType

    concepts = graph.get_nodes_by_type(NodeType.CONCEPT)
    if len(concepts) < 2:
        return 0

    # Build list of (node_id, title) for comparison
    concept_info: list[tuple[str, str]] = []
    for c in concepts:
        title = c.data.get("title", "")
        if isinstance(title, str) and title.strip():
            concept_info.append((c.id, title))

    if len(concept_info) < 2:
        return 0

    # Find merge candidates (greedy: mark nodes to remove)
    merged_into: dict[str, str] = {}  # removed_id -> keeper_id
    removed: set[str] = set()

    for i in range(len(concept_info)):
        if concept_info[i][0] in removed:
            continue
        for j in range(i + 1, len(concept_info)):
            if concept_info[j][0] in removed:
                continue

            id_a, title_a = concept_info[i]
            id_b, title_b = concept_info[j]

            sim = _title_similarity(title_a, title_b)
            if sim >= threshold:
                # Keep the node with more edges
                edges_a = len(graph.get_neighbors(id_a)) + len(graph.get_predecessors(id_a))
                edges_b = len(graph.get_neighbors(id_b)) + len(graph.get_predecessors(id_b))

                if edges_a >= edges_b:
                    keeper_id, remove_id = id_a, id_b
                else:
                    keeper_id, remove_id = id_b, id_a

                merged_into[remove_id] = keeper_id
                removed.add(remove_id)
                logger.debug(
                    "Consolidation: merging '%s' into '%s' (sim=%.2f)",
                    title_b if remove_id == id_b else title_a,
                    title_a if keeper_id == id_a else title_b,
                    sim,
                )

    # Execute merges
    merge_count = 0
    for remove_id, keeper_id in merged_into.items():
        if not graph.has_node(remove_id) or not graph.has_node(keeper_id):
            continue

        try:
            _transfer_edges(graph, remove_id, keeper_id)
            graph.remove_node(remove_id)
            merge_count += 1
        except Exception as e:
            logger.debug("Consolidation: failed to merge %s: %s", remove_id, e)

    if merge_count > 0:
        logger.info("Consolidation: merged %d similar concept pairs", merge_count)

    return merge_count


def _transfer_edges(graph: ConceptGraph, from_id: str, to_id: str) -> None:
    """Transfer edges from one node to another.

    Incoming edges to from_id become incoming to to_id.
    Outgoing edges from from_id become outgoing from to_id.
    Skips self-loops and duplicate edges.

    Args:
        graph: The concept graph.
        from_id: Node being removed.
        to_id: Node receiving edges.
    """
    from cognifold.models.node import Edge

    # Transfer outgoing edges
    for neighbor_id in list(graph.get_neighbors(from_id)):
        if neighbor_id in (to_id, from_id):
            continue
        for edge in graph.get_edges_between(from_id, neighbor_id):
            if not graph.has_edge(to_id, neighbor_id, edge.edge_type):
                try:
                    new_edge = Edge.create(
                        source=to_id,
                        target=neighbor_id,
                        edge_type=edge.edge_type,
                        weight=edge.weight,
                    )
                    graph.add_edge(new_edge)
                except (KeyError, ValueError):
                    pass

    # Transfer incoming edges
    for pred_id in list(graph.get_predecessors(from_id)):
        if pred_id in (to_id, from_id):
            continue
        for edge in graph.get_edges_between(pred_id, from_id):
            if not graph.has_edge(pred_id, to_id, edge.edge_type):
                try:
                    new_edge = Edge.create(
                        source=pred_id,
                        target=to_id,
                        edge_type=edge.edge_type,
                        weight=edge.weight,
                    )
                    graph.add_edge(new_edge)
                except (KeyError, ValueError):
                    pass


def prune_orphan_concepts(
    graph: ConceptGraph,
    min_edges: int = 0,
) -> int:
    """Tag concept nodes with no incoming edges as low-confidence.

    Concepts with zero incoming edges were likely created by the LLM
    without grounding evidence.  Rather than deleting them (which could
    lose valid information), we mark them with low confidence so
    downstream scoring can deprioritize them.

    Args:
        graph: The concept graph.
        min_edges: Minimum total edges (in+out) required.  Nodes with
                   fewer edges are tagged as low-confidence.

    Returns:
        Number of concepts tagged.
    """
    from cognifold.models.node import NodeType

    concepts = graph.get_nodes_by_type(NodeType.CONCEPT)
    tagged = 0

    for concept in concepts:
        incoming = len(graph.get_predecessors(concept.id))
        outgoing = len(graph.get_neighbors(concept.id))
        total_edges = incoming + outgoing

        if total_edges <= min_edges:
            try:
                graph.update_node(concept.id, {"_low_confidence": True})
                tagged += 1
            except KeyError:
                pass

    if tagged > 0:
        logger.info("Consolidation: tagged %d orphan concepts as low-confidence", tagged)

    return tagged
