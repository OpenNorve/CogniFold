"""Simulator CLI for stepping through events and visualizing the graph."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cognifold.graph.store import ConceptGraph
from cognifold.models.event import Event
from cognifold.models.node import Edge, Node, NodeType
from cognifold.models.plan import Operation, OperationType, UpdatePlan
from cognifold.scoring.ranker import ContextRanker, ScoringConfig
from cognifold.simulator.timeline import Timeline, load_timeline
from cognifold.simulator.visualizer import GraphVisualizer, VisualizerConfig

if TYPE_CHECKING:
    from cognifold.agent.config import AgentConfig
    from cognifold.agent.prompt_profile import PromptProfile
    from cognifold.intent.queue import ActionQueue
    from cognifold.replay.logger import GraphLogger

logger = logging.getLogger(__name__)


@dataclass
class SimulatorState:
    """Current state of the simulation.

    Attributes:
        graph: The concept graph.
        timeline: The event timeline.
        current_step: Current step index (0-based).
        history: List of applied update plans.
    """

    graph: ConceptGraph = field(default_factory=ConceptGraph)
    timeline: Timeline | None = None
    current_step: int = 0
    history: list[UpdatePlan] = field(default_factory=list)

    @property
    def current_event(self) -> Event | None:
        """Get the current event, or None if timeline exhausted."""
        if self.timeline is None or self.current_step >= len(self.timeline):
            return None
        return self.timeline[self.current_step]

    @property
    def is_complete(self) -> bool:
        """Check if simulation has processed all events."""
        return self.timeline is None or self.current_step >= len(self.timeline)


class Simulator:
    """Interactive simulator for the concept graph.

    Supports stepping through events, applying update plans,
    and visualizing the evolving graph.

    Action Mode:
        When action_mode is enabled, the simulator will:
        1. Process each event normally
        2. Check for actionable intents and generate actions
        3. Execute due actions between events
        4. Process action result events
    """

    def __init__(
        self,
        scoring_config: ScoringConfig | None = None,
        visualizer_config: VisualizerConfig | None = None,
        graph_logger: GraphLogger | None = None,
        agent_config: AgentConfig | None = None,
        prompt_profile: PromptProfile | None = None,
        action_mode: bool = False,
        action_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the simulator.

        Args:
            scoring_config: Configuration for context window scoring.
            visualizer_config: Configuration for graph visualization.
            graph_logger: Optional logger for replay functionality.
            agent_config: Configuration for the LLM agent.
            prompt_profile: Prompt profile for the agent.
            action_mode: Enable action execution during simulation.
            action_config: Configuration for action mode (optional).
        """
        self.state = SimulatorState()
        self.ranker = ContextRanker(scoring_config)
        self.visualizer = GraphVisualizer(visualizer_config)
        self.graph_logger = graph_logger
        self.agent_config = agent_config
        self.prompt_profile = prompt_profile

        # Action mode settings
        self.action_mode = action_mode
        self.action_config = action_config or {}

        # Action mode components (lazy-loaded)
        self._action_queue: ActionQueue | None = None
        self._intent_selector: Any = None
        self._intent_agent: Any = None
        self._action_executor: Any = None

        # Track action results for visualization
        self._action_results: list[Event] = []

        # Optional run configuration metadata (set externally by CLI)
        self.run_config: dict[str, Any] | None = None

    def load_timeline(self, path: str | Path) -> None:
        """Load a timeline from a JSON file.

        Args:
            path: Path to the timeline JSON file.
        """
        self.state.timeline = load_timeline(path)
        self.state.current_step = 0
        self.state.graph = ConceptGraph()
        self.state.history = []

        # Log run start
        if self.graph_logger:
            self.graph_logger.log_run_start(
                timeline_path=str(path),
                total_events=len(self.state.timeline),
                config=getattr(self, "run_config", None),
            )

    def get_context_window(self, reference_time: datetime | None = None) -> list[str]:
        """Get current context window node IDs.

        Args:
            reference_time: Reference time for recency scoring.

        Returns:
            List of node IDs in the context window.
        """
        return self.ranker.get_context_node_ids(self.state.graph, reference_time)

    def get_node_scores(self, reference_time: datetime | None = None) -> dict[str, float]:
        """Get composite scores for all nodes.

        Args:
            reference_time: Reference time for recency scoring.

        Returns:
            Dictionary mapping node IDs to composite scores.
        """
        scores = self.ranker.score_nodes(self.state.graph, reference_time)
        return {s.node_id: s.composite_score for s in scores}

    def create_default_plan(self, event: Event) -> UpdatePlan:
        """Create a default update plan for an event.

        This creates a simple plan that adds the event as a node.
        In real use, the agent would generate more sophisticated plans.

        Args:
            event: The event to process.

        Returns:
            An UpdatePlan that adds the event as a node.
        """
        return UpdatePlan(
            plan_id=f"plan-{event.event_id}",
            trigger_event_id=event.event_id,
            reasoning=f"Default plan: add event '{event.title}' as node",
            operations=[
                Operation(
                    op=OperationType.ADD_NODE,
                    node_type="event",
                    data={
                        "event_id": event.event_id,
                        "title": event.title,
                        "event_type": event.event_type,
                        "timestamp": event.timestamp.isoformat(),
                        "description": event.description,
                        "location": event.location,
                    },
                )
            ],
        )

    def apply_plan(self, plan: UpdatePlan, strict: bool = False) -> list[str]:
        """Apply an update plan to the graph.

        Args:
            plan: The update plan to execute.
            strict: If True, raise on first error. If False, skip failed ops.

        Returns:
            List of error messages for failed operations.

        Raises:
            ValueError: If strict=True and an operation fails.
        """
        errors: list[str] = []
        for i, op in enumerate(plan.operations):
            try:
                # For ADD_EDGE, infer edge_type/weight BEFORE execution and logging
                # so both get the correct values
                inferred_edge_type = op.edge_type
                inferred_weight = op.weight
                if op.op == OperationType.ADD_EDGE and op.source_id and op.target_id:
                    if inferred_edge_type is None:
                        inferred_edge_type = self._infer_edge_type(op.source_id, op.target_id)
                    if inferred_weight is None:
                        # Default weights by edge type
                        weight_defaults = {
                            "grounds": 0.9,
                            "triggers": 0.8,
                            "deadline_for": 0.7,
                            "related_to": 0.5,
                        }
                        inferred_weight = weight_defaults.get(inferred_edge_type, 0.5)

                self._execute_operation(op, inferred_edge_type, inferred_weight)
                # Log successful operation
                if self.graph_logger:
                    self.graph_logger.log_operation(
                        step=self.state.current_step + 1,
                        op_type=op.op.value,
                        op_data={
                            "node_type": op.node_type,
                            "node_id": op.node_id,
                            "data": op.data,
                            "source_id": op.source_id,
                            "target_id": op.target_id,
                            "node_ids": op.node_ids,
                            "merged_data": op.merged_data,
                            # Explainability fields (Phase 5.5)
                            "reasoning": op.reasoning,
                            "update_reasoning": op.update_reasoning,
                            "grounded_in": op.grounded_in,
                            # Edge type fields (Phase 9.1) - use inferred values
                            "edge_type": inferred_edge_type,
                            "weight": inferred_weight,
                        },
                        success=True,
                    )
            except Exception as e:
                error_msg = f"Operation {i} ({op.op}): {e}"
                # Log failed operation
                if self.graph_logger:
                    self.graph_logger.log_operation(
                        step=self.state.current_step + 1,
                        op_type=op.op.value,
                        op_data={
                            "node_type": op.node_type,
                            "node_id": op.node_id,
                            "data": op.data,
                            "source_id": op.source_id,
                            "target_id": op.target_id,
                        },
                        success=False,
                        error=str(e),
                    )
                if strict:
                    raise ValueError(error_msg) from e
                errors.append(error_msg)

        self.state.history.append(plan)
        return errors

    def _execute_operation(
        self,
        op: Operation,
        inferred_edge_type: str | None = None,
        inferred_weight: float | None = None,
    ) -> None:
        """Execute a single operation.

        Args:
            op: Operation to execute.
            inferred_edge_type: Pre-computed edge type (for ADD_EDGE).
            inferred_weight: Pre-computed weight (for ADD_EDGE).
        """
        if op.op == OperationType.ADD_NODE:
            # Extract ID from appropriate field based on node type
            node_id = None
            if op.data:
                node_id = (
                    op.data.get("event_id")
                    or op.data.get("concept_id")
                    or op.data.get("action_id")
                    or op.data.get("intent_id")
                    or op.data.get("id")
                )
            if not node_id:
                # Generate an ID for nodes without explicit ID
                import uuid

                node_id = f"{op.node_type}-{uuid.uuid4().hex[:8]}"

            node = Node(
                id=node_id,
                type=NodeType(op.node_type) if op.node_type else NodeType.EVENT,
                data=op.data or {},
            )
            self.state.graph.add_node(node)

        elif op.op == OperationType.UPDATE_NODE:
            if op.node_id and op.data:
                self.state.graph.update_node(op.node_id, op.data)

        elif op.op == OperationType.REMOVE_NODE:
            if op.node_id:
                self.state.graph.remove_node(op.node_id)

        elif op.op == OperationType.ADD_EDGE:
            if op.source_id and op.target_id:
                # Use pre-computed inferred values (or original values from operation)
                edge_type = inferred_edge_type or op.edge_type
                weight = inferred_weight if inferred_weight is not None else op.weight

                # Use Edge.create() for proper default weight handling
                edge = Edge.create(
                    source=op.source_id,
                    target=op.target_id,
                    edge_type=edge_type,
                    weight=weight,
                )
                self.state.graph.add_edge(edge)

        elif op.op == OperationType.REMOVE_EDGE:
            if op.source_id and op.target_id:
                self.state.graph.remove_edge(op.source_id, op.target_id)

        elif op.op == OperationType.MERGE_NODES:
            if op.node_ids and len(op.node_ids) >= 2 and op.merged_data:
                # Create merged node with first ID
                merged_id = op.node_ids[0]

                # Collect all edges
                incoming: set[str] = set()
                outgoing: set[str] = set()
                for node_id in op.node_ids:
                    incoming.update(self.state.graph.get_predecessors(node_id))
                    outgoing.update(self.state.graph.get_neighbors(node_id))

                # Remove old nodes
                for node_id in op.node_ids:
                    self.state.graph.remove_node(node_id)

                # Create merged node
                merged_node = Node(
                    id=merged_id,
                    type=NodeType.CONCEPT,
                    data=op.merged_data,
                )
                self.state.graph.add_node(merged_node)

                # Reconnect edges
                for source in incoming:
                    if source not in op.node_ids and self.state.graph.has_node(source):
                        self.state.graph.add_edge(Edge(source=source, target=merged_id))
                for target in outgoing:
                    if target not in op.node_ids and self.state.graph.has_node(target):
                        self.state.graph.add_edge(Edge(source=merged_id, target=target))

    def _infer_edge_type(self, source_id: str, target_id: str) -> str:
        """Infer edge type from source and target node types.

        Uses heuristics based on common patterns:
        - Event → Concept: "grounds" (event provides evidence for concept)
        - Event → Intent: "triggers" (event activates an intent)
        - Concept → Intent: "triggers" (pattern suggests action)
        - Concept → Concept: "related_to" (generic relationship)
        - Time → Intent: "deadline_for" (temporal constraint)

        Args:
            source_id: Source node ID.
            target_id: Target node ID.

        Returns:
            Inferred edge type string.
        """
        # Get node types (if nodes exist)
        source_type = None
        target_type = None

        try:
            source_node = self.state.graph.get_node(source_id)
            source_type = source_node.type
        except KeyError:
            pass

        try:
            target_node = self.state.graph.get_node(target_id)
            target_type = target_node.type
        except KeyError:
            pass

        # Infer based on type combinations
        if source_type == NodeType.EVENT:
            if target_type == NodeType.CONCEPT:
                return "grounds"
            elif target_type == NodeType.INTENT:
                return "triggers"
            elif target_type == NodeType.TIME:
                return "related_to"
        elif source_type == NodeType.CONCEPT:
            if target_type == NodeType.INTENT:
                return "triggers"
            elif target_type == NodeType.CONCEPT:
                return "related_to"
        elif source_type == NodeType.TIME:
            if target_type == NodeType.INTENT:
                return "deadline_for"
        elif source_type == NodeType.INTENT and target_type == NodeType.CONCEPT:
            return "related_to"

        # Default fallback
        return "related_to"

    def step(self, plan: UpdatePlan | None = None) -> bool:
        """Process the next event in the timeline.

        Args:
            plan: Optional custom update plan. If None, uses default plan.

        Returns:
            True if an event was processed, False if timeline exhausted.
        """
        event = self.state.current_event
        if event is None:
            return False

        step_num = self.state.current_step + 1

        # Log event start
        if self.graph_logger:
            self.graph_logger.log_event_start(
                step=step_num,
                event_id=event.event_id,
                event_type=event.event_type,
                title=event.title,
                timestamp=event.timestamp.isoformat(),
                event_data={
                    "description": event.description,
                    "location": event.location,
                    "duration_minutes": event.duration_minutes,
                },
            )

            # Log context window
            context_ids = self.get_context_window(event.timestamp)
            self.graph_logger.log_context_window(step=step_num, context_node_ids=context_ids)

            # Log scores
            scores = self.get_node_scores(event.timestamp)
            self.graph_logger.log_scores(step=step_num, scores=scores)

        if plan is None:
            plan = self.create_default_plan(event)

        self.apply_plan(plan)

        # Log event end
        if self.graph_logger:
            self.graph_logger.log_event_end(
                step=step_num,
                event_id=event.event_id,
                operations_count=len(plan.operations),
                reasoning=plan.reasoning,
            )

        self.state.current_step += 1

        return True

    def step_with_json_plan(self, plan_json: str) -> bool:
        """Process next event with a JSON-provided plan.

        Args:
            plan_json: JSON string containing the update plan.

        Returns:
            True if event was processed, False if timeline exhausted.
        """
        event = self.state.current_event
        if event is None:
            return False

        plan_data = json.loads(plan_json)
        plan = self._parse_plan(plan_data, event.event_id)
        return self.step(plan)

    def _parse_plan(self, data: dict[str, Any], trigger_event_id: str) -> UpdatePlan:
        """Parse an UpdatePlan from dictionary data."""
        operations = []
        for op_data in data.get("operations", []):
            operations.append(
                Operation(
                    op=OperationType(op_data["op"]),
                    node_type=op_data.get("node_type"),
                    node_id=op_data.get("node_id"),
                    data=op_data.get("data"),
                    source_id=op_data.get("source_id"),
                    target_id=op_data.get("target_id"),
                    node_ids=op_data.get("node_ids"),
                    merged_data=op_data.get("merged_data"),
                )
            )

        return UpdatePlan(
            plan_id=data.get("plan_id", f"plan-{trigger_event_id}"),
            trigger_event_id=trigger_event_id,
            reasoning=data.get("reasoning", "User-provided plan"),
            operations=operations,
        )

    def visualize(
        self,
        output_path: str | Path,
        title: str | None = None,
    ) -> Path:
        """Render current graph state to HTML.

        Args:
            output_path: Path for the output HTML file.
            title: Optional title for the visualization.

        Returns:
            Path to the generated HTML file.
        """
        event = self.state.current_event
        reference_time = event.timestamp if event else None

        context_ids = self.get_context_window(reference_time)
        scores = self.get_node_scores(reference_time)

        if title is None:
            title = f"Step {self.state.current_step}"
            if event:
                title = f"{title}: {event.title}"

        return self.visualizer.render(
            graph=self.state.graph,
            output_path=output_path,
            context_node_ids=context_ids,
            node_scores=scores,
            title=title,
        )

    def run_all(self, output_dir: str | Path) -> list[Path]:
        """Run all events and generate visualizations for each step.

        Args:
            output_dir: Directory for output HTML files.

        Returns:
            List of paths to generated HTML files.
        """
        output_dir = Path(output_dir)
        output_paths = []

        while not self.state.is_complete:
            event = self.state.current_event
            step = self.state.current_step

            # Visualize before stepping (shows state when event arrives)
            path = self.visualizer.render_step(
                graph=self.state.graph,
                output_dir=output_dir,
                step_number=step,
                context_node_ids=self.get_context_window(event.timestamp if event else None),
                node_scores=self.get_node_scores(event.timestamp if event else None),
                event_title=event.title if event else "",
            )
            output_paths.append(path)

            self.step()

        # Final visualization
        path = self.visualizer.render_step(
            graph=self.state.graph,
            output_dir=output_dir,
            step_number=self.state.current_step,
            context_node_ids=self.get_context_window(),
            node_scores=self.get_node_scores(),
            event_title="Final State",
        )
        output_paths.append(path)

        return output_paths

    def step_with_agent(self) -> bool:
        """Process the next event using the LLM agent.

        The agent analyzes the event and context to generate an UpdatePlan
        that may create concepts, actions, and edges beyond just adding
        the event.

        Returns:
            True if an event was processed, False if timeline exhausted.

        Raises:
            ValueError: If GOOGLE_API_KEY is not set.
        """
        event = self.state.current_event
        if event is None:
            return False

        step_num = self.state.current_step + 1

        # Log event start
        if self.graph_logger:
            self.graph_logger.log_event_start(
                step=step_num,
                event_id=event.event_id,
                event_type=event.event_type,
                title=event.title,
                timestamp=event.timestamp.isoformat(),
                event_data={
                    "description": event.description,
                    "location": event.location,
                    "duration_minutes": event.duration_minutes,
                },
            )

        # Lazy-load agent
        if not hasattr(self, "_agent"):
            from cognifold.agent import CognifoldAgent

            self._agent = CognifoldAgent(
                config=self.agent_config,
                prompt_profile=self.prompt_profile,
            )

        # Get context window
        context_ids = self.get_context_window(event.timestamp)
        node_scores = self.get_node_scores(event.timestamp)

        # Log context window and scores
        if self.graph_logger:
            self.graph_logger.log_context_window(step=step_num, context_node_ids=context_ids)
            self.graph_logger.log_scores(step=step_num, scores=node_scores)

        # Generate plan using agent
        plan = self._agent.process_event(
            event=event,
            graph=self.state.graph,
            context_node_ids=context_ids,
            node_scores=node_scores,
        )

        # Validate plan and filter out invalid operations
        from cognifold.executor import PlanValidator

        validator = PlanValidator(self.state.graph)
        validation = validator.validate(plan)

        if not validation.is_valid:
            # Get indices of invalid operations
            invalid_indices = {i.operation_index for i in validation.errors}
            error_msgs = [f"{i.message}" for i in validation.errors]
            print(f"Validation failed for some operations: {error_msgs}")

            # Filter out invalid operations, keeping valid ones
            valid_ops = [op for i, op in enumerate(plan.operations) if i not in invalid_indices]

            if valid_ops:
                # Create filtered plan with only valid operations
                plan = UpdatePlan(
                    plan_id=plan.plan_id,
                    trigger_event_id=plan.trigger_event_id,
                    reasoning=f"{plan.reasoning} (filtered: {len(plan.operations) - len(valid_ops)} invalid ops removed)",
                    operations=valid_ops,
                )
            else:
                # All operations invalid, fall back to default
                print("All operations invalid, using default plan")
                plan = self.create_default_plan(event)

        # Execute plan (errors are logged but don't abort)
        errors = self.apply_plan(plan)
        if errors:
            print(f"  Execution warnings: {errors}")

        # Log event end
        if self.graph_logger:
            self.graph_logger.log_event_end(
                step=step_num,
                event_id=event.event_id,
                operations_count=len(plan.operations),
                reasoning=plan.reasoning,
            )

        self.state.current_step += 1
        return True

    def get_status(self) -> dict[str, Any]:
        """Get current simulation status.

        Returns:
            Dictionary with simulation state information.
        """
        event = self.state.current_event
        status = {
            "current_step": self.state.current_step,
            "total_events": len(self.state.timeline) if self.state.timeline else 0,
            "is_complete": self.state.is_complete,
            "current_event": {
                "id": event.event_id,
                "title": event.title,
                "type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
            }
            if event
            else None,
            "graph": {
                "node_count": self.state.graph.node_count,
                "edge_count": self.state.graph.edge_count,
            },
            "context_window_size": len(self.get_context_window()),
            "plans_applied": len(self.state.history),
        }

        # Add action mode status
        if self.action_mode and self._action_queue:
            status["action_mode"] = {
                "enabled": True,
                "queued_actions": self._action_queue.queued_count,
                "total_actions": self._action_queue.size,
                "action_results_processed": len(self._action_results),
            }

        return status

    # =========================================================================
    # Action Mode Methods (Phase 8.3)
    # =========================================================================

    def _init_action_mode(self) -> None:
        """Initialize action mode components if not already done."""
        if self._action_queue is not None:
            return

        from cognifold.intent import (
            ActionQueue,
            IntentSelector,
            IntentToActionAgent,
            SimulatedActionExecutor,
        )

        self._action_queue = ActionQueue()
        self._intent_selector = IntentSelector(
            self.state.graph,
            scoring_config=self.action_config.get("scoring", {}),
        )
        self._intent_agent = IntentToActionAgent(
            llm_provider=self.action_config.get("llm_provider", "mock"),
        )
        self._action_executor = SimulatedActionExecutor(
            time_compression=self.action_config.get("time_compression", 1.0),
        )
        self._action_results = []

        logger.info("Action mode initialized")

    @property
    def action_queue(self) -> ActionQueue | None:
        """Get the action queue (None if action mode not initialized)."""
        return self._action_queue

    def step_with_actions(self, plan: UpdatePlan | None = None) -> bool:
        """Process the next event with action mode enabled.

        This method:
        1. Processes the current event
        2. Checks for actionable intents and generates actions
        3. Executes actions scheduled before the next event
        4. Processes action result events

        Args:
            plan: Optional custom update plan. If None, uses default plan.

        Returns:
            True if an event was processed, False if timeline exhausted.
        """
        if not self.action_mode:
            return self.step(plan)

        # Initialize action mode if needed
        self._init_action_mode()

        event = self.state.current_event
        if event is None:
            return False

        current_time = event.timestamp

        # 1. Process the event normally
        if not self.step(plan):
            return False

        # 2. Check for actionable intents and generate actions
        self._process_actionable_intents(current_time)

        # 3. Get the next event time (or use a far future time if no more events)
        next_event = self.state.current_event
        if next_event:
            next_time = next_event.timestamp
        else:
            # No more events, use end of day + 24 hours
            from datetime import timedelta

            next_time = current_time + timedelta(hours=24)

        # 4. Execute actions scheduled between current and next event
        self._execute_due_actions(current_time, next_time)

        return True

    def _process_actionable_intents(self, current_time: datetime) -> None:
        """Check for actionable intents and generate actions.

        Args:
            current_time: Current simulation time.
        """
        if self._intent_selector is None or self._intent_agent is None:
            return

        # Update selector's graph reference
        self._intent_selector.graph = self.state.graph

        # Get actionable intents
        min_urgency = self.action_config.get("min_urgency", 0.3)
        max_intents = self.action_config.get("max_intents_per_step", 3)

        actionable = self._intent_selector.select_actionable_intents(
            current_time=current_time,
            min_urgency=min_urgency,
            max_intents=max_intents,
        )

        if not actionable:
            return

        logger.info(f"Found {len(actionable)} actionable intents")

        # Generate actions for each intent
        max_actions = self.action_config.get("max_actions_per_intent", 3)

        for intent in actionable:
            # Log intent selection
            if self.graph_logger:
                self.graph_logger.log_intent_selected(
                    step=self.state.current_step + 1,
                    intent_id=intent.id,
                    intent_title=intent.data.get("title", ""),
                    urgency_score=intent.data.get("priority", 0.5),
                    status=intent.data.get("status", "pending"),
                )

            # Get context for this intent
            context_ids = self.state.graph.get_neighbors(
                intent.id
            ) + self.state.graph.get_predecessors(intent.id)
            context = [
                self.state.graph.get_node(nid)
                for nid in context_ids
                if self.state.graph.has_node(nid)
            ]

            # Generate actions
            actions = self._intent_agent.generate_actions(
                intent=intent,
                context=context,
                current_time=current_time,
                max_actions=max_actions,
            )

            if actions and self._action_queue is not None:
                # Enqueue actions
                self._action_queue.enqueue_many(actions)

                # Log each generated action
                for action in actions:
                    if self.graph_logger:
                        self.graph_logger.log_action_generated(
                            step=self.state.current_step + 1,
                            action_id=action.action_id,
                            intent_id=action.intent_id,
                            action_title=action.title,
                            scheduled_time=action.scheduled_time.isoformat(),
                            urgency=action.metadata.urgency if action.metadata else "medium",
                        )

                # Update intent status to action_scheduled
                self._mark_intent_scheduled(intent.id)

                logger.info(f"Generated {len(actions)} actions for intent {intent.id}")

    def _mark_intent_scheduled(self, intent_id: str) -> None:
        """Mark an intent as having actions scheduled.

        Args:
            intent_id: ID of the intent to update.
        """
        try:
            intent = self.state.graph.get_node(intent_id)
            updated_data = dict(intent.data)
            updated_data["status"] = "action_scheduled"
            self.state.graph.update_node(intent_id, updated_data)
            logger.debug(f"Marked intent {intent_id} as action_scheduled")
        except Exception as e:
            logger.warning(f"Failed to mark intent {intent_id}: {e}")

    def _execute_due_actions(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        """Execute actions scheduled in the time window.

        Args:
            start_time: Start of time window.
            end_time: End of time window.
        """
        if self._action_queue is None or self._action_executor is None:
            return

        # Get actions due in this window
        due_actions = self._action_queue.get_actions_between(start_time, end_time)

        if not due_actions:
            return

        logger.info(f"Executing {len(due_actions)} actions between {start_time} and {end_time}")

        # Sort by scheduled time
        due_actions.sort(key=lambda a: a.scheduled_time)

        for action in due_actions:
            # Execute the action
            result_event, updated_action = self._action_executor.execute(
                action,
                action.scheduled_time,
            )

            # Log action execution
            if self.graph_logger:
                self.graph_logger.log_action_executed(
                    step=self.state.current_step + 1,
                    action_id=action.action_id,
                    intent_id=action.intent_id,
                    action_title=action.title,
                    execution_time=action.scheduled_time.isoformat(),
                    result_event_id=result_event.event_id,
                )

            # Update action in queue
            self._action_queue.update_action(updated_action)

            # Process the result event
            self._process_action_result_event(result_event)

            # Track for reporting
            self._action_results.append(result_event)

            logger.info(f"Executed action {action.action_id}: {action.title}")

    def _process_action_result_event(self, event: Event) -> None:
        """Process an action result event through the agent pipeline.

        The action result is processed like a normal event, allowing the agent to:
        - Connect it to relevant concepts
        - Create new intents based on the outcome
        - Link it to the original intent

        Args:
            event: The action result event.
        """
        step_num = self.state.current_step + 1
        intent_id = event.metadata.get("intent_id")

        # Log event start
        if self.graph_logger:
            self.graph_logger.log_event_start(
                step=step_num,
                event_id=event.event_id,
                event_type=event.event_type,
                title=event.title,
                timestamp=event.timestamp.isoformat(),
                event_data={
                    "description": event.description,
                    "metadata": event.metadata,
                },
            )

        # Try to use the agent if available, otherwise use simple plan
        plan = None
        use_agent = hasattr(self, "_agent") or self.agent_config is not None

        if use_agent:
            try:
                # Lazy-load agent if not already loaded
                if not hasattr(self, "_agent"):
                    from cognifold.agent import CognifoldAgent

                    self._agent = CognifoldAgent(
                        config=self.agent_config,
                        prompt_profile=self.prompt_profile,
                    )

                # Get context window - include the original intent in context
                context_ids = self.get_context_window(event.timestamp)
                if intent_id and intent_id not in context_ids:
                    context_ids.insert(0, intent_id)  # Prioritize the related intent

                node_scores = self.get_node_scores(event.timestamp)

                # Log context window and scores
                if self.graph_logger:
                    self.graph_logger.log_context_window(
                        step=step_num, context_node_ids=context_ids
                    )
                    self.graph_logger.log_scores(step=step_num, scores=node_scores)

                # Generate plan using agent
                plan = self._agent.process_event(
                    event=event,
                    graph=self.state.graph,
                    context_node_ids=context_ids,
                    node_scores=node_scores,
                )
            except ValueError as e:
                # API key not set - fall back to simple plan
                logger.debug(f"Agent not available for action result: {e}")
                use_agent = False

        # Fallback: create simple plan with edge to intent
        if not use_agent or plan is None:
            operations = [
                Operation(
                    op=OperationType.ADD_NODE,
                    node_type="event",
                    data={
                        "event_id": event.event_id,
                        "title": event.title,
                        "event_type": event.event_type,
                        "timestamp": event.timestamp.isoformat(),
                        "description": event.description,
                        "metadata": event.metadata,
                    },
                )
            ]
            # Add edge to the original intent if it exists
            if intent_id and self.state.graph.has_node(intent_id):
                operations.append(
                    Operation(
                        op=OperationType.ADD_EDGE,
                        source_id=event.event_id,
                        target_id=intent_id,
                    )
                )
            plan = UpdatePlan(
                plan_id=f"action-result-{event.event_id}",
                trigger_event_id=event.event_id,
                reasoning=f"Action result: {event.title}",
                operations=operations,
            )

        # Validate plan and filter out invalid operations (only for agent-generated plans)
        if use_agent and plan:
            from cognifold.executor import PlanValidator

            validator = PlanValidator(self.state.graph)
            validation = validator.validate(plan)

            if not validation.is_valid:
                invalid_indices = {i.operation_index for i in validation.errors}
                error_msgs = [f"{i.message}" for i in validation.errors]
                logger.warning(f"Validation failed for action result: {error_msgs}")

                # Filter out invalid operations
                valid_ops = [op for i, op in enumerate(plan.operations) if i not in invalid_indices]
                if valid_ops:
                    plan = UpdatePlan(
                        plan_id=plan.plan_id,
                        trigger_event_id=plan.trigger_event_id,
                        reasoning=plan.reasoning
                        + f" (filtered: {len(invalid_indices)} invalid ops removed)",
                        operations=valid_ops,
                    )
                else:
                    logger.warning(f"All operations invalid for action result {event.event_id}")
                    plan = None

        # Apply the plan
        if plan:
            self.apply_plan(plan, strict=False)

            # Log event end
            if self.graph_logger:
                self.graph_logger.log_event_end(
                    step=step_num,
                    event_id=event.event_id,
                    operations_count=len(plan.operations),
                    reasoning=plan.reasoning,
                )

        # Check if we should resolve the intent
        action_id = event.metadata.get("action_id")
        outcome = event.metadata.get("outcome", "unknown")
        intent_resolved = False

        if intent_id and outcome == "success":
            self._maybe_resolve_intent(intent_id)
            # Check if intent was actually resolved
            try:
                intent = self.state.graph.get_node(intent_id)
                intent_resolved = intent.data.get("status") == "resolved"
            except Exception:
                pass

        # Log the action result event
        if self.graph_logger and action_id:
            self.graph_logger.log_action_result_event(
                step=step_num,
                result_event_id=event.event_id,
                action_id=action_id,
                intent_id=intent_id or "",
                outcome=outcome,
                intent_resolved=intent_resolved,
            )

    def _maybe_resolve_intent(self, intent_id: str) -> None:
        """Check if an intent should be marked as resolved.

        An intent is resolved when all its scheduled actions are completed.

        Args:
            intent_id: ID of the intent to check.
        """
        if self._action_queue is None:
            return

        # Get all actions for this intent
        intent_actions = self._action_queue.get_actions_for_intent(intent_id)

        # Check if all are completed
        from cognifold.intent.models import ActionStatus

        all_completed = all(a.status == ActionStatus.COMPLETED for a in intent_actions)

        if all_completed and intent_actions:
            try:
                intent = self.state.graph.get_node(intent_id)
                updated_data = dict(intent.data)
                updated_data["status"] = "resolved"
                self.state.graph.update_node(intent_id, updated_data)
                logger.info(f"Resolved intent {intent_id}")
            except Exception as e:
                logger.warning(f"Failed to resolve intent {intent_id}: {e}")

    def run_all_with_actions(self, output_dir: str | Path) -> list[Path]:
        """Run all events with action mode and generate visualizations.

        Args:
            output_dir: Directory for output HTML files.

        Returns:
            List of paths to generated HTML files.
        """
        if not self.action_mode:
            logger.warning("Action mode not enabled, using regular run_all")
            return self.run_all(output_dir)

        output_dir = Path(output_dir)
        output_paths = []

        self._init_action_mode()

        while not self.state.is_complete:
            event = self.state.current_event
            step = self.state.current_step

            # Visualize before stepping
            path = self.visualizer.render_step(
                graph=self.state.graph,
                output_dir=output_dir,
                step_number=step,
                context_node_ids=self.get_context_window(event.timestamp if event else None),
                node_scores=self.get_node_scores(event.timestamp if event else None),
                event_title=event.title if event else "",
            )
            output_paths.append(path)

            # Step with actions
            self.step_with_actions()

        # Final visualization
        path = self.visualizer.render_step(
            graph=self.state.graph,
            output_dir=output_dir,
            step_number=self.state.current_step,
            context_node_ids=self.get_context_window(),
            node_scores=self.get_node_scores(),
            event_title="Final State",
        )
        output_paths.append(path)

        # Log summary
        if self._action_queue:
            logger.info(f"Simulation complete. Actions: {self._action_queue.summary()}")

        return output_paths

    def get_action_summary(self) -> dict[str, Any]:
        """Get a summary of action mode execution.

        Returns:
            Dictionary with action mode statistics.
        """
        if not self.action_mode or self._action_queue is None:
            return {"action_mode": False}

        # Count actions by status
        status_counts = {}
        for action in self._action_queue.actions:
            status = action.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "action_mode": True,
            "total_actions": self._action_queue.size,
            "status_counts": status_counts,
            "action_results_processed": len(self._action_results),
            "intents_processed": len({a.intent_id for a in self._action_queue.actions}),
        }
