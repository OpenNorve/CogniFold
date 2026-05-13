"""CogEval-Bench Track B: Relationship Topology Evaluator.

Evaluates whether the system discovers multi-hop connections and builds
meaningful graph structure. Metrics: chain discovery rate, modularity,
clustering coefficient, small-world sigma, edge type entropy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import networkx as nx


@dataclass
class ChainResult:
    """Result for a single planted chain evaluation."""

    chain_id: str
    expected_hops: int
    discovered: bool = False
    discovered_path_length: int = 0
    accuracy: float = 0.0


@dataclass
class TopologyEvalResult:
    """Results from topology evaluation."""

    chain_discovery_rate: float = 0.0
    chain_results: list[ChainResult] = field(default_factory=list)

    modularity: float = 0.0
    clustering_coefficient: float = 0.0
    small_world_sigma: float = 0.0
    edge_type_entropy: float = 0.0

    n_communities: int = 0
    avg_path_length: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_discovery_rate": round(self.chain_discovery_rate, 4),
            "chain_results": [
                {
                    "chain_id": cr.chain_id,
                    "expected_hops": cr.expected_hops,
                    "discovered": cr.discovered,
                    "discovered_path_length": cr.discovered_path_length,
                    "accuracy": round(cr.accuracy, 4),
                }
                for cr in self.chain_results
            ],
            "modularity": round(self.modularity, 4),
            "clustering_coefficient": round(self.clustering_coefficient, 4),
            "small_world_sigma": round(self.small_world_sigma, 4),
            "edge_type_entropy": round(self.edge_type_entropy, 4),
            "n_communities": self.n_communities,
            "avg_path_length": round(self.avg_path_length, 4),
        }


def evaluate_chain_discovery(
    graph_nx: nx.Graph,
    planted_chains: list[dict],
    node_content_map: dict[str, str],
) -> tuple[float, list[ChainResult]]:
    """Check if planted multi-hop chains were discovered in the graph.

    For each planted chain, check if there is a path in the graph connecting
    nodes whose content matches the chain's events (via semantic similarity).

    Args:
        graph_nx: NetworkX graph (undirected for path finding).
        planted_chains: Gold chain definitions from the scenario.
        node_content_map: {node_id: content_text} for all nodes.

    Returns:
        (discovery_rate, list of ChainResult)
    """
    if not planted_chains:
        return 0.0, []

    results = []
    for chain in planted_chains:
        cr = ChainResult(chain_id=chain["id"], expected_hops=chain["hops"])

        chain_steps = chain["steps"]
        step_node_candidates = []
        for step in chain_steps:
            event_text = step["event"].lower()
            entity = step.get("entity", "").lower()
            candidates = []
            for nid, content in node_content_map.items():
                content_lower = content.lower()
                if entity and entity.replace("_", " ") in content_lower:
                    candidates.append(nid)
                elif _keyword_overlap(event_text, content_lower) >= 0.3:
                    candidates.append(nid)
            step_node_candidates.append(candidates)

        n_steps = len(step_node_candidates)
        if all(step_node_candidates) and n_steps >= 2:
            best_coverage = 0
            best_path_len = 0
            for start in step_node_candidates[0]:
                for end in step_node_candidates[-1]:
                    try:
                        path = nx.shortest_path(graph_nx, start, end)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue

                    path_set = set(path)
                    path_neighbors: set = set()
                    for p in path:
                        path_neighbors.update(graph_nx.neighbors(p))
                    path_and_neighbors = path_set | path_neighbors

                    steps_covered = 2
                    for mid_idx in range(1, n_steps - 1):
                        if any(
                            c in path_and_neighbors
                            for c in step_node_candidates[mid_idx]
                        ):
                            steps_covered += 1

                    if steps_covered > best_coverage or (
                        steps_covered == best_coverage
                        and len(path) - 1 > best_path_len
                    ):
                        best_coverage = steps_covered
                        best_path_len = len(path) - 1

            if best_coverage >= 2:
                cr.discovered = True
                cr.discovered_path_length = best_path_len

                coverage_score = best_coverage / n_steps
                expected = chain["hops"]
                length_score = 1.0 - abs(best_path_len - expected) / max(expected, 1)
                length_score = max(0.0, length_score)
                cr.accuracy = 0.5 * coverage_score + 0.5 * length_score

        results.append(cr)

    discovery_rate = (
        sum(1 for r in results if r.discovered) / len(results) if results else 0.0
    )
    return discovery_rate, results


_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at",
        "for", "to", "and", "or", "it",
    }
)


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """Simple keyword overlap ratio between two texts."""
    words_a = set(text_a.split())
    words_b = set(text_b.split())
    stop = set() | _STOPWORDS
    words_a -= stop
    words_b -= stop
    if not words_a or not words_b:
        return 0.0
    overlap = words_a & words_b
    return len(overlap) / min(len(words_a), len(words_b))


def evaluate_graph_structure(graph_nx: nx.Graph) -> dict[str, float]:
    """Compute structural metrics on the graph.

    Returns dict with: modularity, clustering_coefficient, small_world_sigma,
    avg_path_length, n_communities.
    """
    metrics: dict[str, Any] = {}

    if graph_nx.number_of_nodes() < 3:
        return {
            "modularity": 0.0,
            "clustering_coefficient": 0.0,
            "small_world_sigma": 0.0,
            "avg_path_length": 0.0,
            "n_communities": 0,
        }

    G = graph_nx.to_undirected() if graph_nx.is_directed() else graph_nx

    metrics["clustering_coefficient"] = nx.average_clustering(G)

    try:
        communities = list(nx.community.greedy_modularity_communities(G))
        metrics["modularity"] = nx.community.modularity(G, communities)
        metrics["n_communities"] = len(communities)
    except Exception:
        metrics["modularity"] = 0.0
        metrics["n_communities"] = 0

    try:
        if nx.is_connected(G):
            metrics["avg_path_length"] = nx.average_shortest_path_length(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            metrics["avg_path_length"] = nx.average_shortest_path_length(subgraph)
    except Exception:
        metrics["avg_path_length"] = 0.0

    try:
        n = G.number_of_nodes()
        m = G.number_of_edges()
        if n > 10 and m > 0:
            p = (2 * m) / (n * (n - 1)) if n > 1 else 0
            C_rand = p
            L_rand = (
                math.log(n) / math.log(max(n * p, 2)) if n * p > 1 else float("inf")
            )
            C = metrics["clustering_coefficient"]
            L = metrics["avg_path_length"]
            if C_rand > 0 and L_rand > 0 and L > 0:
                metrics["small_world_sigma"] = (C / C_rand) / (L / L_rand)
            else:
                metrics["small_world_sigma"] = 0.0
        else:
            metrics["small_world_sigma"] = 0.0
    except Exception:
        metrics["small_world_sigma"] = 0.0

    return metrics


def compute_edge_type_entropy(edge_types: list[str]) -> float:
    """Compute Shannon entropy of edge type distribution.

    Higher entropy = more diverse relationship types (good).
    """
    if not edge_types:
        return 0.0

    from collections import Counter

    counts = Counter(edge_types)
    total = len(edge_types)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    if max_entropy > 0:
        return entropy / max_entropy
    return 0.0


def evaluate_topology(
    graph_nx: nx.Graph,
    planted_chains: list[dict],
    node_content_map: dict[str, str],
    edge_types: list[str],
) -> TopologyEvalResult:
    """Run full topology evaluation.

    Args:
        graph_nx: NetworkX graph.
        planted_chains: Gold chain definitions.
        node_content_map: {node_id: content_text}.
        edge_types: List of all edge types in the graph.

    Returns:
        TopologyEvalResult with all metrics.
    """
    result = TopologyEvalResult()

    chain_discovery_rate, chain_results = evaluate_chain_discovery(
        graph_nx, planted_chains, node_content_map
    )
    result.chain_discovery_rate = chain_discovery_rate
    result.chain_results = chain_results

    struct = evaluate_graph_structure(graph_nx)
    result.modularity = struct.get("modularity", 0.0)
    result.clustering_coefficient = struct.get("clustering_coefficient", 0.0)
    result.small_world_sigma = struct.get("small_world_sigma", 0.0)
    result.avg_path_length = struct.get("avg_path_length", 0.0)
    result.n_communities = int(struct.get("n_communities", 0))

    result.edge_type_entropy = compute_edge_type_entropy(edge_types)

    return result
