"""Replay player for reconstructing graph states from logs.

This module provides functionality to parse logs and reconstruct the graph
state at any point in time, generating keyframes for visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cognifold.replay.logger import LogEntry, LogEntryType, load_log


@dataclass
class Keyframe:
    """A snapshot of graph state at a specific step.

    Attributes:
        step: Step number (0 = initial, 1+ = after event).
        event_id: Event ID that triggered this state (None for step 0).
        event_title: Title of the event.
        event_type: Type of event.
        nodes: List of node data dicts.
        edges: List of edge data dicts (source, target).
        context_node_ids: Node IDs in context window.
        scores: Node scores at this step.
        operations: Operations applied in this step.
        reasoning: Agent's reasoning for this step.
        added_nodes: Node IDs added in this step.
        removed_nodes: Node IDs removed in this step.
        added_edges: Edges added in this step (source, target tuples).
        removed_edges: Edges removed in this step.
        intents_selected: Intents selected for action generation in this step.
        actions_generated: Actions generated from intents in this step.
        actions_executed: Actions executed in this step.
        action_results: Action result events processed in this step.
    """

    step: int
    event_id: str | None
    event_title: str
    event_type: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]  # source, target, edge_type, weight
    context_node_ids: list[str]
    scores: dict[str, float]
    operations: list[dict[str, Any]]
    reasoning: str | None
    added_nodes: list[str] = field(default_factory=list)
    removed_nodes: list[str] = field(default_factory=list)
    added_edges: list[tuple[str, str]] = field(default_factory=list)
    removed_edges: list[tuple[str, str]] = field(default_factory=list)
    intents_selected: list[dict[str, Any]] = field(default_factory=list)
    actions_generated: list[dict[str, Any]] = field(default_factory=list)
    actions_executed: list[dict[str, Any]] = field(default_factory=list)
    action_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step": self.step,
            "event_id": self.event_id,
            "event_title": self.event_title,
            "event_type": self.event_type,
            "nodes": self.nodes,
            "edges": self.edges,
            "context_node_ids": self.context_node_ids,
            "scores": self.scores,
            "operations": self.operations,
            "reasoning": self.reasoning,
            "added_nodes": self.added_nodes,
            "removed_nodes": self.removed_nodes,
            "added_edges": self.added_edges,
            "removed_edges": self.removed_edges,
            "intents_selected": self.intents_selected,
            "actions_generated": self.actions_generated,
            "actions_executed": self.actions_executed,
            "action_results": self.action_results,
        }


@dataclass
class ReplayPlayer:
    """Player for reconstructing and navigating graph evolution.

    This class parses log files and generates keyframes representing
    the graph state after each event is processed.

    Attributes:
        entries: Raw log entries.
        keyframes: Generated keyframes.
        metadata: Run metadata (timeline path, config, etc.).
    """

    entries: list[LogEntry] = field(default_factory=list)
    keyframes: list[Keyframe] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Internal state for reconstruction
    _nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    _edges: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_log(cls, path: str | Path) -> ReplayPlayer:
        """Create a player from a log file.

        Args:
            path: Path to the JSONL log file.

        Returns:
            ReplayPlayer with parsed entries and generated keyframes.
        """
        entries = load_log(path)
        player = cls(entries=entries)
        player._generate_keyframes()
        return player

    def _generate_keyframes(self) -> None:
        """Generate keyframes from log entries."""
        self._nodes = {}
        self._edges = []

        current_step = 0
        current_event_id: str | None = None
        current_event_title = ""
        current_event_type = ""
        current_context: list[str] = []
        current_scores: dict[str, float] = {}
        current_operations: list[dict[str, Any]] = []
        current_reasoning: str | None = None
        step_added_nodes: list[str] = []
        step_removed_nodes: list[str] = []
        step_added_edges: list[tuple[str, str]] = []
        step_removed_edges: list[tuple[str, str]] = []
        # Intent/action tracking (Phase 8)
        step_intents_selected: list[dict[str, Any]] = []
        step_actions_generated: list[dict[str, Any]] = []
        step_actions_executed: list[dict[str, Any]] = []
        step_action_results: list[dict[str, Any]] = []

        for entry in self.entries:
            if entry.entry_type == LogEntryType.RUN_START:
                self.metadata = entry.data
                # Create initial keyframe (empty graph)
                self.keyframes.append(
                    Keyframe(
                        step=0,
                        event_id=None,
                        event_title="Initial State",
                        event_type="",
                        nodes=[],
                        edges=[],
                        context_node_ids=[],
                        scores={},
                        operations=[],
                        reasoning=None,
                    )
                )

            elif entry.entry_type == LogEntryType.EVENT_START:
                current_step = entry.step
                current_event_id = entry.data.get("event_id")
                current_event_title = entry.data.get("title", "")
                current_event_type = entry.data.get("event_type", "")
                current_operations = []
                step_added_nodes = []
                step_removed_nodes = []
                step_added_edges = []
                step_removed_edges = []
                # Reset intent/action tracking for new step
                step_intents_selected = []
                step_actions_generated = []
                step_actions_executed = []
                step_action_results = []

            elif entry.entry_type == LogEntryType.OPERATION:
                op_type = entry.data.get("op_type", "")
                op_data = entry.data.get("op_data", {})
                success = entry.data.get("success", True)

                if success:
                    current_operations.append({"op": op_type, **op_data})
                    self._apply_operation(
                        op_type,
                        op_data,
                        step_added_nodes,
                        step_removed_nodes,
                        step_added_edges,
                        step_removed_edges,
                    )

            elif entry.entry_type == LogEntryType.CONTEXT_WINDOW:
                current_context = entry.data.get("context_node_ids", [])

            elif entry.entry_type == LogEntryType.SCORES:
                current_scores = entry.data.get("scores", {})

            # Intent/action flow entries (Phase 8)
            elif entry.entry_type == LogEntryType.INTENT_SELECTED:
                step_intents_selected.append(entry.data)

            elif entry.entry_type == LogEntryType.ACTION_GENERATED:
                step_actions_generated.append(entry.data)

            elif entry.entry_type == LogEntryType.ACTION_EXECUTED:
                step_actions_executed.append(entry.data)

            elif entry.entry_type == LogEntryType.ACTION_RESULT_EVENT:
                step_action_results.append(entry.data)

            elif entry.entry_type == LogEntryType.EVENT_END:
                current_reasoning = entry.data.get("reasoning")

                # Create keyframe for this step
                self.keyframes.append(
                    Keyframe(
                        step=current_step,
                        event_id=current_event_id,
                        event_title=current_event_title,
                        event_type=current_event_type,
                        nodes=list(self._nodes.values()),
                        edges=list(self._edges),
                        context_node_ids=current_context,
                        scores=dict(current_scores),
                        operations=current_operations,
                        reasoning=current_reasoning,
                        added_nodes=step_added_nodes,
                        removed_nodes=step_removed_nodes,
                        added_edges=step_added_edges,
                        removed_edges=step_removed_edges,
                        intents_selected=step_intents_selected,
                        actions_generated=step_actions_generated,
                        actions_executed=step_actions_executed,
                        action_results=step_action_results,
                    )
                )

            elif entry.entry_type == LogEntryType.RUN_END:
                self.metadata["final_stats"] = entry.data

    def _apply_operation(
        self,
        op_type: str,
        op_data: dict[str, Any],
        added_nodes: list[str],
        removed_nodes: list[str],
        added_edges: list[tuple[str, str]],
        removed_edges: list[tuple[str, str]],
    ) -> None:
        """Apply an operation to the internal graph state.

        Args:
            op_type: Operation type.
            op_data: Operation data.
            added_nodes: List to append added node IDs.
            removed_nodes: List to append removed node IDs.
            added_edges: List to append added edges.
            removed_edges: List to append removed edges.
        """
        if op_type == "ADD_NODE":
            node_type = op_data.get("node_type", "event")
            data = op_data.get("data", {})
            node_id = (
                data.get("event_id")
                or data.get("concept_id")
                or data.get("action_id")
                or data.get("intent_id")
                or data.get("id")
            )
            # Extract reasoning and grounded_in from op_data (Phase 5.5 explainability)
            reasoning = op_data.get("reasoning")
            grounded_in = op_data.get("grounded_in", [])
            if node_id:
                self._nodes[node_id] = {
                    "id": node_id,
                    "type": node_type,
                    "data": data,
                    "reasoning": reasoning,
                    "grounded_in": grounded_in,
                }
                added_nodes.append(node_id)

        elif op_type == "UPDATE_NODE":
            node_id = op_data.get("node_id")
            update_data = op_data.get("data", {})
            if node_id and node_id in self._nodes:
                self._nodes[node_id]["data"].update(update_data)

        elif op_type == "REMOVE_NODE":
            node_id = op_data.get("node_id")
            if node_id and node_id in self._nodes:
                del self._nodes[node_id]
                removed_nodes.append(node_id)
                # Remove associated edges
                edges_to_remove = [
                    e for e in self._edges if e["source"] == node_id or e["target"] == node_id
                ]
                for edge in edges_to_remove:
                    self._edges.remove(edge)
                    removed_edges.append((edge["source"], edge["target"]))

        elif op_type == "ADD_EDGE":
            source = op_data.get("source_id")
            target = op_data.get("target_id")
            edge_type = op_data.get("edge_type")
            weight = op_data.get("weight")
            if source and target:
                edge = {
                    "source": source,
                    "target": target,
                    "edge_type": edge_type,
                    "weight": weight,
                }
                # Check if this exact edge doesn't already exist
                existing = [
                    e
                    for e in self._edges
                    if e["source"] == source
                    and e["target"] == target
                    and e.get("edge_type") == edge_type
                ]
                if not existing:
                    self._edges.append(edge)
                    added_edges.append((source, target))

        elif op_type == "REMOVE_EDGE":
            source = op_data.get("source_id")
            target = op_data.get("target_id")
            edge_type = op_data.get("edge_type")
            if source and target:
                # Find matching edge (with or without edge_type)
                matching = [
                    e
                    for e in self._edges
                    if e["source"] == source
                    and e["target"] == target
                    and (edge_type is None or e.get("edge_type") == edge_type)
                ]
                for edge in matching:
                    self._edges.remove(edge)
                    removed_edges.append((source, target))

        elif op_type == "MERGE_NODES":
            node_ids = op_data.get("node_ids", [])
            merged_data = op_data.get("merged_data", {})
            if len(node_ids) >= 2:
                # Keep first node, remove others
                primary_id = node_ids[0]
                if primary_id in self._nodes:
                    self._nodes[primary_id]["data"].update(merged_data)

                for node_id in node_ids[1:]:
                    if node_id in self._nodes:
                        del self._nodes[node_id]
                        removed_nodes.append(node_id)
                        # Redirect edges to primary node
                        for edge in self._edges:
                            if edge["source"] == node_id:
                                edge["source"] = primary_id
                            if edge["target"] == node_id:
                                edge["target"] = primary_id

    def get_keyframe(self, step: int) -> Keyframe | None:
        """Get keyframe at a specific step.

        Args:
            step: Step number (0 = initial state).

        Returns:
            Keyframe at that step, or None if not found.
        """
        for kf in self.keyframes:
            if kf.step == step:
                return kf
        return None

    def get_keyframe_range(
        self,
        start_step: int | None = None,
        end_step: int | None = None,
    ) -> list[Keyframe]:
        """Get keyframes in a range.

        Args:
            start_step: Starting step (inclusive).
            end_step: Ending step (inclusive).

        Returns:
            List of keyframes in the range.
        """
        result = []
        for kf in self.keyframes:
            if start_step is not None and kf.step < start_step:
                continue
            if end_step is not None and kf.step > end_step:
                continue
            result.append(kf)
        return result

    @property
    def total_steps(self) -> int:
        """Total number of steps (excluding initial state)."""
        if not self.keyframes:
            return 0
        return max(kf.step for kf in self.keyframes)

    @property
    def timeline_path(self) -> str:
        """Path to the original timeline."""
        return self.metadata.get("timeline_path", "")
