"""Graph integrity validation.

This module provides validation for the graph state itself,
checking integrity rules like orphan nodes, connectivity, grounding, and reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph


class IntegrityLevel(str, Enum):
    """Severity level of integrity issues."""

    ERROR = "error"  # Critical issue that should be fixed
    WARNING = "warning"  # Issue that may indicate a problem


@dataclass
class IntegrityIssue:
    """A single integrity issue found in the graph."""

    level: IntegrityLevel
    node_id: str
    rule: str
    message: str
    suggestion: str | None = None


@dataclass
class ValidationReport:
    """Complete validation report for a graph."""

    orphan_nodes: list[str] = field(default_factory=list)
    ungrounded_nodes: list[str] = field(default_factory=list)
    nodes_missing_reasoning: list[str] = field(default_factory=list)
    connectivity_violations: list[str] = field(default_factory=list)
    issues: list[IntegrityIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if the graph passes all integrity checks."""
        return len(self.orphan_nodes) == 0 and len(self.connectivity_violations) == 0

    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return len([i for i in self.issues if i.level == IntegrityLevel.ERROR])

    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return len([i for i in self.issues if i.level == IntegrityLevel.WARNING])

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = []
        lines.append(f"Validation Report: {'PASS' if self.is_valid else 'FAIL'}")
        lines.append(f"  Errors: {self.error_count}, Warnings: {self.warning_count}")
        if self.orphan_nodes:
            lines.append(f"  Orphan nodes: {len(self.orphan_nodes)}")
        if self.ungrounded_nodes:
            lines.append(f"  Ungrounded nodes: {len(self.ungrounded_nodes)}")
        if self.nodes_missing_reasoning:
            lines.append(f"  Missing reasoning: {len(self.nodes_missing_reasoning)}")
        if self.connectivity_violations:
            lines.append(f"  Connectivity violations: {len(self.connectivity_violations)}")
        return "\n".join(lines)


class GraphValidator:
    """Validates graph integrity and quality.

    Checks include:
    - No orphan nodes (except events which can be standalone)
    - Connectivity rules by node type
    - Grounding requirements (non-event nodes should be grounded)
    - Reasoning requirements (non-event nodes should have reasoning)
    """

    def __init__(self, graph: ConceptGraph) -> None:
        """Initialize with a reference to the graph.

        Args:
            graph: The graph to validate.
        """
        self._graph = graph

    def validate_no_orphans(self) -> list[str]:
        """Find nodes with no edges (orphan nodes).

        Events are allowed to be orphans (they are sources of truth).
        Concept, intent (formerly "action"), and time nodes must have at least one edge.

        Returns:
            List of orphan node IDs (excluding events).
        """
        orphans = []

        for node in self._graph.get_all_nodes():
            # Events can be standalone
            if node.type.value == "event":
                continue

            # Check if node has any edges (incoming or outgoing)
            neighbors = self._graph.get_neighbors(node.id)
            predecessors = self._graph.get_predecessors(node.id)

            if len(neighbors) == 0 and len(predecessors) == 0:
                orphans.append(node.id)

        return orphans

    def validate_connectivity_rules(self) -> list[str]:
        """Validate connectivity requirements by node type.

        Rules:
        - event: Can be standalone (source of truth)
        - concept: Must connect to at least 1 event or concept
        - intent: Must connect to at least 1 concept or event (formerly "action")
        - time: Must connect to at least 1 intent or event

        Returns:
            List of node IDs that violate connectivity rules.
        """
        violations = []

        for node in self._graph.get_all_nodes():
            node_type = node.type.value

            # Events can be standalone
            if node_type == "event":
                continue

            # Get all connected nodes (both directions)
            neighbors = self._graph.get_neighbors(node.id)
            predecessors = self._graph.get_predecessors(node.id)
            all_connected = set(neighbors + predecessors)

            if len(all_connected) == 0:
                violations.append(node.id)
                continue

            # Get types of connected nodes
            connected_types = set()
            for connected_id in all_connected:
                try:
                    connected_node = self._graph.get_node(connected_id)
                    connected_types.add(connected_node.type.value)
                except KeyError:
                    pass  # Node may have been removed

            # Check connectivity rules based on node type
            if node_type == "concept":
                # Must connect to event or concept
                if not (connected_types & {"event", "concept"}):
                    violations.append(node.id)

            elif node_type in ("intent", "action"):
                # Must connect to concept or event
                # ("action" is legacy, "intent" is current)
                if not (connected_types & {"concept", "event"}):
                    violations.append(node.id)

            elif node_type == "time" and not (connected_types & {"intent", "action", "event"}):
                # Must connect to intent or event
                # Also accept "action" for backward compatibility
                violations.append(node.id)

        return violations

    def validate_grounding(self) -> list[str]:
        """Find nodes without grounding references.

        Non-event nodes should have grounded_in references pointing
        to the events/nodes that justify their existence.

        Returns:
            List of ungrounded node IDs (excluding events).
        """
        ungrounded = []

        for node in self._graph.get_all_nodes():
            # Events don't need grounding (they ARE the ground truth)
            if node.type.value == "event":
                continue

            # Check for grounded_in references
            if not node.grounded_in or len(node.grounded_in) == 0:
                ungrounded.append(node.id)

        return ungrounded

    def validate_reasoning(self) -> list[str]:
        """Find nodes without reasoning.

        Non-event nodes should have reasoning explaining why they exist.

        Returns:
            List of node IDs missing reasoning (excluding events).
        """
        missing_reasoning = []

        for node in self._graph.get_all_nodes():
            # Events don't need reasoning
            if node.type.value == "event":
                continue

            # Check for reasoning
            if not node.reasoning or len(node.reasoning.strip()) == 0:
                missing_reasoning.append(node.id)

        return missing_reasoning

    def validate_all(self) -> ValidationReport:
        """Run all validation checks and return a complete report.

        Returns:
            ValidationReport with all issues found.
        """
        report = ValidationReport()

        # Run all validations
        report.orphan_nodes = self.validate_no_orphans()
        report.connectivity_violations = self.validate_connectivity_rules()
        report.ungrounded_nodes = self.validate_grounding()
        report.nodes_missing_reasoning = self.validate_reasoning()

        # Build detailed issues list
        for node_id in report.orphan_nodes:
            report.issues.append(
                IntegrityIssue(
                    level=IntegrityLevel.ERROR,
                    node_id=node_id,
                    rule="no_orphans",
                    message=f"Node '{node_id}' has no edges (orphan)",
                    suggestion="Connect this node to related events or concepts, or remove it",
                )
            )

        for node_id in report.connectivity_violations:
            if node_id not in report.orphan_nodes:  # Avoid duplicate issues
                node = self._graph.get_node(node_id)
                report.issues.append(
                    IntegrityIssue(
                        level=IntegrityLevel.ERROR,
                        node_id=node_id,
                        rule="connectivity",
                        message=f"Node '{node_id}' ({node.type.value}) violates connectivity rules",
                        suggestion=f"Connect to appropriate node types for {node.type.value}",
                    )
                )

        for node_id in report.ungrounded_nodes:
            report.issues.append(
                IntegrityIssue(
                    level=IntegrityLevel.WARNING,
                    node_id=node_id,
                    rule="grounding",
                    message=f"Node '{node_id}' has no grounded_in references",
                    suggestion="Add grounded_in references to source events",
                )
            )

        for node_id in report.nodes_missing_reasoning:
            report.issues.append(
                IntegrityIssue(
                    level=IntegrityLevel.WARNING,
                    node_id=node_id,
                    rule="reasoning",
                    message=f"Node '{node_id}' has no reasoning",
                    suggestion="Add reasoning explaining why this node exists",
                )
            )

        return report

    def get_repair_suggestions(self, report: ValidationReport) -> list[str]:
        """Generate repair suggestions for issues in the report.

        Args:
            report: ValidationReport from validate_all().

        Returns:
            List of suggested actions to fix issues.
        """
        suggestions = []

        if report.orphan_nodes:
            suggestions.append(
                f"Remove {len(report.orphan_nodes)} orphan node(s): {', '.join(report.orphan_nodes[:5])}"
                + ("..." if len(report.orphan_nodes) > 5 else "")
            )

        if report.ungrounded_nodes:
            suggestions.append(f"Add grounding to {len(report.ungrounded_nodes)} node(s)")

        if report.nodes_missing_reasoning:
            suggestions.append(f"Add reasoning to {len(report.nodes_missing_reasoning)} node(s)")

        return suggestions
