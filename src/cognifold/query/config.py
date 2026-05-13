"""Shared configuration constants for the query subsystem.

Centralizes type-boost multipliers, scoring parameters, and retrieval
constants that were previously scattered as magic numbers across
strategies.py, scoring.py, assembly.py, and agent.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from cognifold.models.node import NodeType

# ---------------------------------------------------------------------------
# Retrieval constants (strategies.py, agent.py)
# ---------------------------------------------------------------------------

# Maximum number of entry points when falling back to structural/type search.
# Keeps traversal fast without missing important starting nodes.
MAX_ENTRY_POINTS: int = 10

# BFS traversal decay factor per hop.  A value of 0.7 means a neighbor's
# score is 70% of its parent's — far enough hops yield negligible weight.
BFS_DECAY_PER_HOP: float = 0.7

# Discount applied to 1-hop neighbor relevance when expanding scored nodes
# in the agentic retrieval path.  Neighbors are contextually relevant but
# less directly than the matched node itself.
NEIGHBOR_RELEVANCE_DISCOUNT: float = 0.6


# ---------------------------------------------------------------------------
# Scoring constants (scoring.py)
# ---------------------------------------------------------------------------

# Penalty multiplier for non-matching nodes found via traversal (connected
# to a matching node but not matching the query keywords themselves).
NON_MATCH_PENALTY: float = 0.6

# Per-depth score decay applied on top of the traversal decay.  A mild
# exponential (0.95^depth) gently penalizes distant nodes without zeroing
# them out — keeps 2-hop nodes at ~90% of their traversal score.
DEPTH_PENALTY_FACTOR: float = 0.95


# ---------------------------------------------------------------------------
# Assembly constants (assembly.py)
# ---------------------------------------------------------------------------

# Maximum characters for a node description before truncation.
# Keeps context concise for downstream LLM consumption.
MAX_DESCRIPTION_CHARS: int = 500


# ---------------------------------------------------------------------------
# Executor constants (executor/runner.py)
# ---------------------------------------------------------------------------

# Number of hex characters used when generating fallback node IDs from UUIDs.
UUID_HEX_LENGTH: int = 8


@dataclass(frozen=True)
class TypeBoosts:
    """NodeType-based score multipliers.

    Higher-level node types (concepts, intents) are more informative for
    retrieval than raw events, so they receive score boosts. Two preset
    configurations are provided:

    - ENTRY_POINT: Aggressive boosts for entry-point selection where we want
      concepts/intents to surface as starting nodes for traversal.
    - RELEVANCE: Moderate boosts for final relevance scoring where we refine
      an already-filtered candidate set.
    """

    concept: float = 1.0
    intent: float = 1.0
    time: float = 1.0
    event: float = 1.0


# Boosts for entry-point selection (strategies.py) — more aggressive because
# we want high-level nodes as traversal starting points.
ENTRY_POINT_BOOSTS = TypeBoosts(concept=1.5, intent=1.3)

# Boosts for final relevance scoring (scoring.py) — more moderate because
# we're refining an already-filtered candidate set.
RELEVANCE_BOOSTS = TypeBoosts(concept=1.3, intent=1.2, time=1.1)


def apply_type_boost(
    score: float,
    node_type: NodeType,
    boosts: TypeBoosts,
    *,
    clamp: bool = False,
) -> float:
    """Apply a NodeType-based multiplier to a score.

    Args:
        score: The base score to boost.
        node_type: The type of node being scored.
        boosts: Which boost configuration to use.
        clamp: If True, clamp result to [0.0, 1.0].

    Returns:
        Boosted score.
    """
    multiplier = {
        NodeType.CONCEPT: boosts.concept,
        NodeType.INTENT: boosts.intent,
        NodeType.TIME: boosts.time,
        NodeType.EVENT: boosts.event,
    }.get(node_type, 1.0)

    result = score * multiplier
    if clamp:
        result = min(1.0, max(0.0, result))
    return result
