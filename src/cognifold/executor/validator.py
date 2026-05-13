"""Validation for UpdatePlans before execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph
    from cognifold.models.plan import Operation, UpdatePlan


class IssueSeverity(str, Enum):
    """Severity level of validation issues."""

    ERROR = "error"  # Plan cannot be executed
    WARNING = "warning"  # Plan can execute but may have problems


@dataclass
class ValidationIssue:
    """A single validation issue found in an UpdatePlan."""

    severity: IssueSeverity
    operation_index: int
    message: str


@dataclass
class ValidationResult:
    """Result of validating an UpdatePlan."""

    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Get only error-severity issues."""
        return [i for i in self.issues if i.severity == IssueSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get only warning-severity issues."""
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]


class PlanValidator:
    """Validates UpdatePlans before execution.

    Checks for issues like:
    - Referenced nodes don't exist (for UPDATE/REMOVE)
    - Duplicate node IDs in ADD operations
    - Edge operations with missing endpoints
    - Grounding requirements (non-event nodes must be grounded)
    - Explainability requirements (reasoning required for non-event nodes)
    """

    def __init__(
        self,
        graph: ConceptGraph,
        require_grounding: bool = True,
        require_reasoning: bool = True,
    ):
        """Initialize with a reference to the graph.

        Args:
            graph: The graph to validate against.
            require_grounding: If True, non-event nodes must have grounded_in.
            require_reasoning: If True, non-event ADD_NODE must have reasoning.
        """
        self._graph = graph
        self._require_grounding = require_grounding
        self._require_reasoning = require_reasoning

    def validate(self, plan: UpdatePlan) -> ValidationResult:
        """Validate an update plan against the current graph state.

        Args:
            plan: The plan to validate.

        Returns:
            ValidationResult with is_valid flag and any issues found.
        """
        from cognifold.models.plan import OperationType

        issues: list[ValidationIssue] = []

        # Track nodes that will exist after prior operations
        existing_nodes = {n.id for n in self._graph.get_all_nodes()}
        pending_adds: set[str] = set()
        pending_removes: set[str] = set()

        for i, op in enumerate(plan.operations):
            op_issues = self._validate_operation(
                op, i, existing_nodes, pending_adds, pending_removes
            )
            issues.extend(op_issues)

            # Update tracking sets based on operation
            if op.op == OperationType.ADD_NODE:
                # Check explicit node_id first, then data ID fields
                node_id = op.node_id
                if not node_id and op.data:
                    node_id = (
                        op.data.get("event_id")
                        or op.data.get("concept_id")
                        or op.data.get("action_id")
                        or op.data.get("intent_id")
                        or op.data.get("id")
                    )
                if node_id:
                    pending_adds.add(node_id)

            elif op.op == OperationType.REMOVE_NODE and op.node_id:
                pending_removes.add(op.node_id)

            elif op.op == OperationType.MERGE_NODES and op.node_ids:
                for node_id in op.node_ids[1:]:  # First ID survives
                    pending_removes.add(node_id)

        has_errors = any(i.severity == IssueSeverity.ERROR for i in issues)

        return ValidationResult(is_valid=not has_errors, issues=issues)

    def _validate_operation(
        self,
        op: Operation,
        index: int,
        existing_nodes: set[str],
        pending_adds: set[str],
        pending_removes: set[str],
    ) -> list[ValidationIssue]:
        """Validate a single operation."""
        from cognifold.models.plan import OperationType

        issues: list[ValidationIssue] = []

        def node_will_exist(node_id: str) -> bool:
            """Check if node will exist after prior operations."""
            if node_id in pending_removes:
                return False
            return node_id in existing_nodes or node_id in pending_adds

        if op.op == OperationType.ADD_NODE:
            issues.extend(self._validate_add_node(op, index, existing_nodes, pending_adds))

        elif op.op == OperationType.UPDATE_NODE:
            if op.node_id and not node_will_exist(op.node_id):
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        operation_index=index,
                        message=f"UPDATE_NODE: Node '{op.node_id}' does not exist",
                    )
                )
            # Validate explainability for updates
            issues.extend(self._validate_update_explainability(op, index))

        elif op.op == OperationType.REMOVE_NODE:
            if op.node_id and not node_will_exist(op.node_id):
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        operation_index=index,
                        message=f"REMOVE_NODE: Node '{op.node_id}' does not exist",
                    )
                )

        elif op.op in (OperationType.ADD_EDGE, OperationType.REMOVE_EDGE):
            if op.source_id and not node_will_exist(op.source_id):
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        operation_index=index,
                        message=f"{op.op}: Source node '{op.source_id}' does not exist",
                    )
                )
            if op.target_id and not node_will_exist(op.target_id):
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        operation_index=index,
                        message=f"{op.op}: Target node '{op.target_id}' does not exist",
                    )
                )

        elif op.op == OperationType.MERGE_NODES and op.node_ids:
            for node_id in op.node_ids:
                if not node_will_exist(node_id):
                    issues.append(
                        ValidationIssue(
                            severity=IssueSeverity.ERROR,
                            operation_index=index,
                            message=f"MERGE_NODES: Node '{node_id}' does not exist",
                        )
                    )

        return issues

    def _validate_add_node(
        self,
        op: Operation,
        index: int,
        existing_nodes: set[str],
        pending_adds: set[str],
    ) -> list[ValidationIssue]:
        """Validate ADD_NODE operation."""
        issues: list[ValidationIssue] = []

        # Check explicit node_id first, then data ID fields
        node_id = op.node_id
        if not node_id and op.data:
            node_id = (
                op.data.get("event_id")
                or op.data.get("concept_id")
                or op.data.get("action_id")
                or op.data.get("intent_id")
                or op.data.get("id")
            )
        if not node_id:
            # ID will be auto-generated, no conflict possible
            pass

        if node_id and node_id in existing_nodes:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    operation_index=index,
                    message=f"ADD_NODE: Node '{node_id}' already exists in graph",
                )
            )

        if node_id and node_id in pending_adds:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    operation_index=index,
                    message=f"ADD_NODE: Duplicate node ID '{node_id}' in plan",
                )
            )

        # Validate explainability for non-event nodes
        is_event = op.node_type == "event"
        if not is_event:
            issues.extend(self._validate_explainability(op, index, existing_nodes, pending_adds))

        return issues

    def _validate_explainability(
        self,
        op: Operation,
        index: int,
        existing_nodes: set[str],
        pending_adds: set[str],
    ) -> list[ValidationIssue]:
        """Validate explainability requirements for non-event nodes.

        Rules:
        - concept/action/time nodes require reasoning
        - concept/action/time nodes require grounded_in references
        - grounded_in references must point to valid nodes
        """
        issues: list[ValidationIssue] = []

        # Check reasoning requirement
        if self._require_reasoning and not op.reasoning:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    operation_index=index,
                    message=f"ADD_NODE ({op.node_type}): Missing reasoning for node creation",
                )
            )

        # Check grounding requirement
        if self._require_grounding:
            if not op.grounded_in or len(op.grounded_in) == 0:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        operation_index=index,
                        message=f"ADD_NODE ({op.node_type}): Missing grounded_in references",
                    )
                )
            else:
                # Validate grounding references exist
                for ref_id in op.grounded_in:
                    if ref_id not in existing_nodes and ref_id not in pending_adds:
                        issues.append(
                            ValidationIssue(
                                severity=IssueSeverity.ERROR,
                                operation_index=index,
                                message=f"ADD_NODE ({op.node_type}): grounded_in reference '{ref_id}' does not exist",
                            )
                        )

        return issues

    def _validate_update_explainability(self, op: Operation, index: int) -> list[ValidationIssue]:
        """Validate explainability requirements for UPDATE_NODE operations."""
        issues: list[ValidationIssue] = []

        if self._require_reasoning and not op.update_reasoning:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    operation_index=index,
                    message=f"UPDATE_NODE: Missing update_reasoning for node '{op.node_id}'",
                )
            )

        return issues
