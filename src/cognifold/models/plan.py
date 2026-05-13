"""UpdatePlan model for graph modifications."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class OperationType(str, Enum):
    """Types of operations that can be performed on the graph."""

    ADD_NODE = "ADD_NODE"
    UPDATE_NODE = "UPDATE_NODE"
    REMOVE_NODE = "REMOVE_NODE"
    ADD_EDGE = "ADD_EDGE"
    REMOVE_EDGE = "REMOVE_EDGE"
    MERGE_NODES = "MERGE_NODES"


class Operation(BaseModel):
    """A single operation in an update plan.

    Different operation types require different parameters:
    - ADD_NODE: node_type, data, reasoning (required for non-event), grounded_in (required for non-event)
    - UPDATE_NODE: node_id, data, update_reasoning (required)
    - REMOVE_NODE: node_id
    - ADD_EDGE: source_id, target_id, edge_type (optional), weight (optional)
    - REMOVE_EDGE: source_id, target_id, edge_type (optional - removes specific type or all)
    - MERGE_NODES: node_ids, merged_data, reasoning

    Explainability fields:
    - reasoning: Why this node is being created (for ADD_NODE)
    - update_reasoning: Why this update is being made (for UPDATE_NODE)
    - grounded_in: List of event/node IDs that justify this operation
    """

    op: OperationType = Field(..., description="Type of operation")
    node_type: str | None = Field(default=None, description="Type for ADD_NODE")
    node_id: str | None = Field(default=None, description="Target node for UPDATE/REMOVE")
    data: dict[str, Any] | None = Field(default=None, description="Data for ADD/UPDATE")
    source_id: str | None = Field(default=None, description="Source for edge operations")
    target_id: str | None = Field(default=None, description="Target for edge operations")
    node_ids: list[str] | None = Field(default=None, description="Nodes to merge")
    merged_data: dict[str, Any] | None = Field(default=None, description="Data for merged node")

    # Edge-specific fields (Phase 9.1)
    edge_type: str | None = Field(
        default=None,
        description="Semantic relationship type for edges (e.g., grounds, causes, triggers)",
    )
    weight: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override edge weight (0.0-1.0), uses type-based default if omitted",
    )

    # Explainability fields
    reasoning: str | None = Field(
        default=None, description="Why this node is being created (for ADD_NODE)"
    )
    update_reasoning: str | None = Field(
        default=None, description="Why this update is being made (for UPDATE_NODE)"
    )
    grounded_in: list[str] | None = Field(
        default=None, description="Node IDs that justify this operation"
    )

    @model_validator(mode="after")
    def validate_operation_params(self) -> Operation:
        """Validate that required parameters are present for each operation type."""
        op = self.op

        if op == OperationType.ADD_NODE:
            if not self.node_type or self.data is None:
                raise ValueError("ADD_NODE requires node_type and data")

        elif op == OperationType.UPDATE_NODE:
            if not self.node_id or self.data is None:
                raise ValueError("UPDATE_NODE requires node_id and data")

        elif op == OperationType.REMOVE_NODE:
            if not self.node_id:
                raise ValueError("REMOVE_NODE requires node_id")

        elif op in (OperationType.ADD_EDGE, OperationType.REMOVE_EDGE):
            if not self.source_id or not self.target_id:
                raise ValueError(f"{op} requires source_id and target_id")

        elif op == OperationType.MERGE_NODES:
            if not self.node_ids or len(self.node_ids) < 2:
                raise ValueError("MERGE_NODES requires at least 2 node_ids")
            if self.merged_data is None:
                raise ValueError("MERGE_NODES requires merged_data")

        return self


class UpdatePlan(BaseModel):
    """A plan containing operations to execute atomically on the graph.

    The agent generates update plans in response to events. The executor
    validates and applies all operations atomically.
    """

    plan_id: str = Field(..., description="Unique identifier for the plan")
    trigger_event_id: str = Field(..., description="Event that triggered this update")
    reasoning: str = Field(..., description="Agent's explanation for the changes")
    operations: list[Operation] = Field(
        default_factory=list, description="List of operations to execute"
    )
    symbolic_actions: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Structured state-change actions extracted by the LLM. "
            "Types: STATE_CHANGE, PRESENCE_CHANGE, FACT_ASSERTION."
        ),
    )

    model_config = {"frozen": True}
