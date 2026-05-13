"""Node and Edge models for the concept graph."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class BaseEdgeType(str, Enum):
    """Fixed base edge types available in all domains.

    These are semantic relationship types that capture how nodes relate to each other.
    Domain-specific types can be registered at runtime (e.g., CALLS, DEPENDS_ON for service-logs).
    """

    CAUSES = "causes"  # Event A causes Event B
    PART_OF = "part_of"  # Event/Concept is part of Concept
    REINFORCES = "reinforces"  # Event reinforces Concept
    TRIGGERS = "triggers"  # Concept triggers Intent
    DEADLINE_FOR = "deadline_for"  # Time is deadline for Intent
    RELATED_TO = "related_to"  # Generic relationship (default for new edges)
    DERIVED_FROM = "derived_from"  # Concept derived from another Concept
    GROUNDS = "grounds"  # Event grounds a Concept/Intent
    USER_FEEDBACK = "user_feedback"  # Feedback event → Intent


# Type-based default weights for edges
# Agent can override these defaults based on context
EDGE_TYPE_DEFAULT_WEIGHTS: dict[str, float] = {
    BaseEdgeType.CAUSES.value: 0.9,  # Strong causal link
    BaseEdgeType.GROUNDS.value: 0.9,  # Direct evidence
    BaseEdgeType.TRIGGERS.value: 0.8,  # Clear activation
    BaseEdgeType.REINFORCES.value: 0.7,  # Supporting evidence
    BaseEdgeType.PART_OF.value: 0.7,  # Structural relationship
    BaseEdgeType.DERIVED_FROM.value: 0.6,  # Indirect derivation
    BaseEdgeType.DEADLINE_FOR.value: 0.6,  # Temporal constraint
    BaseEdgeType.RELATED_TO.value: 0.5,  # Generic/weak connection
    BaseEdgeType.USER_FEEDBACK.value: 0.8,  # Direct user signal
}

# Default weight for unknown/custom edge types
DEFAULT_EDGE_WEIGHT: float = 0.5

# Recommended type constraints (soft warnings only, not enforced)
# Format: {edge_type: {"source_types": [...], "target_types": [...]}}
EDGE_TYPE_CONSTRAINTS: dict[str, dict[str, list[str]]] = {
    BaseEdgeType.GROUNDS.value: {
        "source_types": ["event"],
        "target_types": ["concept", "intent"],
    },
    BaseEdgeType.CAUSES.value: {
        "source_types": ["event"],
        "target_types": ["event"],
    },
    BaseEdgeType.DEADLINE_FOR.value: {
        "source_types": ["time"],
        "target_types": ["intent"],
    },
    BaseEdgeType.TRIGGERS.value: {
        "source_types": ["concept", "event"],
        "target_types": ["intent"],
    },
    BaseEdgeType.REINFORCES.value: {
        "source_types": ["event"],
        "target_types": ["concept"],
    },
    BaseEdgeType.PART_OF.value: {
        "source_types": ["event", "concept"],
        "target_types": ["concept"],
    },
    BaseEdgeType.DERIVED_FROM.value: {
        "source_types": ["concept"],
        "target_types": ["concept"],
    },
    BaseEdgeType.USER_FEEDBACK.value: {
        "source_types": ["event"],
        "target_types": ["intent"],
    },
}


def get_default_weight_for_type(edge_type: str | None) -> float:
    """Get the default weight for an edge type.

    Args:
        edge_type: The edge type string, or None for legacy edges.

    Returns:
        The default weight for the edge type.
    """
    if edge_type is None:
        return 1.0  # Legacy edges have weight 1.0
    return EDGE_TYPE_DEFAULT_WEIGHTS.get(edge_type, DEFAULT_EDGE_WEIGHT)


class NodeType(str, Enum):
    """Types of nodes in the concept graph."""

    EVENT = "event"
    CONCEPT = "concept"
    INTENT = "intent"  # Goals/desires (formerly "action")
    TIME = "time"  # Temporal anchors: deadlines, scheduled times, recurring periods

    @classmethod
    def from_string(cls, value: str) -> NodeType:
        """Convert a string to NodeType, handling backward compatibility.

        Args:
            value: The node type string (e.g., "event", "concept", "intent", "action").

        Returns:
            The corresponding NodeType enum value.

        Note:
            "action" is deprecated and maps to INTENT for backward compatibility.
        """
        # Handle backward compatibility: "action" -> "intent"
        if value == "action":
            return cls.INTENT
        return cls(value)


class IntentStatus(str, Enum):
    """Status of an intent node in its lifecycle.

    Intent lifecycle:
    - PENDING: Intent created, no actions generated yet
    - ACTION_SCHEDULED: Actions have been generated and queued
    - RESOLVED: Intent has been fulfilled via action execution
    - REJECTED: User explicitly rejected this intent
    - DEFERRED: User deferred this intent (not now, maybe later)
    """

    PENDING = "pending"
    ACTION_SCHEDULED = "action_scheduled"
    RESOLVED = "resolved"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class UpdateHistoryEntry(BaseModel):
    """A record of a node update for explainability.

    Tracks what changed and why, providing a full audit trail
    of how a node evolved over time.
    """

    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the update occurred"
    )
    update_reasoning: str = Field(..., description="Explanation of why this update was made")
    changes: dict[str, Any] = Field(
        default_factory=dict, description="Fields that were changed (old -> new values)"
    )


class Node(BaseModel):
    """A node in the concept graph.

    Nodes represent events, concepts, intents, or time anchors. They track metadata
    for relevance scoring including creation time, last access, and access count.

    Intent nodes (formerly "action" nodes) represent goals or desires that may be
    converted into concrete, schedulable actions by the Intent-to-Action agent.

    Time nodes are special temporal anchors that represent:
    - Deadlines (e.g., "project due Friday")
    - Scheduled events (e.g., "meeting at 2pm")
    - Recurring times (e.g., "weekly standup")

    Connecting other nodes to time nodes provides temporal context for urgency.

    Explainability fields:
    - reasoning: Why this node was created (required for non-event nodes)
    - grounded_in: List of event IDs that justify this node's existence
    - update_history: Audit trail of changes with reasoning
    """

    id: str = Field(..., description="Unique identifier for the node")
    type: NodeType = Field(..., description="Category of the node")
    data: dict[str, Any] = Field(default_factory=dict, description="Node payload data")
    created_at: datetime = Field(default_factory=datetime.now, description="When node was created")
    last_accessed: datetime = Field(
        default_factory=datetime.now, description="Last time node was accessed"
    )
    access_count: int = Field(default=0, ge=0, description="Number of times node was accessed")

    # Explainability fields
    reasoning: str | None = Field(
        default=None, description="Why this node was created (1-2 sentences)"
    )
    grounded_in: list[str] = Field(
        default_factory=list,
        description="List of event/node IDs that ground this node's existence",
    )
    update_history: list[UpdateHistoryEntry] = Field(
        default_factory=list, description="History of updates with reasoning"
    )
    embedding: list[float] | None = Field(
        default=None, description="Vector embedding for semantic search"
    )

    def touch(self) -> Node:
        """Return a new node with updated access time and count."""
        return self.model_copy(
            update={
                "last_accessed": datetime.now(),
                "access_count": self.access_count + 1,
            }
        )

    def add_update_history(self, update_reasoning: str, changes: dict[str, Any]) -> Node:
        """Return a new node with an update history entry added.

        Args:
            update_reasoning: Explanation of why this update was made.
            changes: Dictionary of field changes.

        Returns:
            New Node with the update history entry appended.
        """
        entry = UpdateHistoryEntry(update_reasoning=update_reasoning, changes=changes)
        new_history = [*self.update_history, entry]
        return self.model_copy(update={"update_history": new_history})


class Edge(BaseModel):
    """An edge connecting two nodes in the concept graph.

    Edges can be typed and weighted to capture semantic relationships.
    Legacy edges (edge_type=None) maintain backward compatibility.

    Multiple edges between the same node pair are allowed if they have
    different edge types.

    Attributes:
        source: ID of the source node.
        target: ID of the target node.
        edge_type: Semantic relationship type (None for legacy edges).
        weight: Relationship strength (0.0 to 1.0). Defaults based on edge_type.
        created_at: When the edge was created.
        metadata: Additional edge metadata (optional).
    """

    source: str = Field(..., description="ID of the source node")
    target: str = Field(..., description="ID of the target node")
    edge_type: str | None = Field(
        default=None,
        description="Semantic relationship type (None for legacy untyped edges)",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Relationship strength (0.0 to 1.0)",
    )
    created_at: datetime = Field(default_factory=datetime.now, description="When edge was created")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional edge metadata")

    model_config = {"frozen": True}

    @classmethod
    def create(
        cls,
        source: str,
        target: str,
        edge_type: str | None = None,
        weight: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Edge:
        """Create an edge with default weight based on edge type.

        Args:
            source: ID of the source node.
            target: ID of the target node.
            edge_type: Semantic relationship type (optional).
            weight: Override weight (if None, uses type-based default).
            metadata: Additional metadata (optional).

        Returns:
            A new Edge instance.
        """
        if weight is None:
            weight = get_default_weight_for_type(edge_type)

        return cls(
            source=source,
            target=target,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata or {},
        )

    @property
    def edge_key(self) -> tuple[str, str, str | None]:
        """Get a unique key for this edge (source, target, edge_type).

        This key allows multiple edges between the same node pair
        with different types.
        """
        return (self.source, self.target, self.edge_type)


def validate_edge_type_constraints(edge: Edge, source_node: Node, target_node: Node) -> list[str]:
    """Validate edge type constraints and return warnings (soft validation).

    Args:
        edge: The edge to validate.
        source_node: The source node.
        target_node: The target node.

    Returns:
        List of warning messages (empty if no constraint violations).
    """
    warnings: list[str] = []

    if edge.edge_type is None:
        return warnings  # No constraints for legacy edges

    if edge.edge_type not in EDGE_TYPE_CONSTRAINTS:
        return warnings  # No constraints defined for this type

    constraint = EDGE_TYPE_CONSTRAINTS[edge.edge_type]

    source_types = constraint.get("source_types", [])
    if source_types and source_node.type.value not in source_types:
        warnings.append(
            f"Edge type '{edge.edge_type}' typically has source type "
            f"{source_types}, got '{source_node.type.value}'"
        )

    target_types = constraint.get("target_types", [])
    if target_types and target_node.type.value not in target_types:
        warnings.append(
            f"Edge type '{edge.edge_type}' typically has target type "
            f"{target_types}, got '{target_node.type.value}'"
        )

    return warnings
