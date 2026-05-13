"""End-to-end pipeline for Cognifold."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cognifold.config import CognifoldConfig
    from cognifold.models.event import Event
    from cognifold.models.plan import UpdatePlan


@dataclass
class PipelineResult:
    """Result of processing an event through the pipeline."""

    event: Event
    plan: UpdatePlan
    success: bool
    error: str | None = None
    execution_time_ms: float = 0.0
    nodes_added: int = 0
    edges_added: int = 0
    concepts_created: list[str] = field(default_factory=list)
    actions_created: list[str] = field(default_factory=list)


@dataclass
class PipelineStats:
    """Statistics for a pipeline run."""

    events_processed: int = 0
    events_failed: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    concepts_created: int = 0
    actions_created: int = 0
    total_time_ms: float = 0.0


class Pipeline:
    """End-to-end pipeline for processing events through Cognifold.

    The pipeline orchestrates:
    1. Event ingestion
    2. Context window selection
    3. Agent-based plan generation (or default plans)
    4. Plan validation
    5. Plan execution
    6. Graph updates

    Example:
        >>> from cognifold.pipeline import Pipeline
        >>> from cognifold.config import CognifoldConfig
        >>>
        >>> config = CognifoldConfig.load("config.yaml")
        >>> pipeline = Pipeline(config)
        >>> pipeline.load_timeline("data/events.json")
        >>>
        >>> # Process all events
        >>> stats = pipeline.run()
        >>> print(f"Processed {stats.events_processed} events")
        >>>
        >>> # Save results
        >>> pipeline.save_graph("output/graph.json")
        >>> pipeline.visualize("output/graph.html")
    """

    def __init__(self, config: CognifoldConfig | None = None):
        """Initialize the pipeline.

        Args:
            config: Configuration for the pipeline. Uses defaults if not provided.
        """
        from cognifold.config import CognifoldConfig
        from cognifold.graph.store import ConceptGraph
        from cognifold.logging import get_logger, setup_logging
        from cognifold.scoring.ranker import ContextRanker, ScoringConfig

        self._config = config or CognifoldConfig.load()

        # Setup logging
        setup_logging(self._config.logging)
        self._logger = get_logger("pipeline")

        # Initialize components
        self._graph = ConceptGraph()

        scoring_config = ScoringConfig(
            alpha=self._config.scoring.alpha,
            beta=self._config.scoring.beta,
            gamma=self._config.scoring.gamma,
            decay_rate=self._config.scoring.decay_rate,
            context_window_size=self._config.context.max_nodes,
            min_score_threshold=self._config.context.min_score_threshold,
        )
        self._ranker = ContextRanker(scoring_config)

        # Agent is lazy-loaded
        self._agent: Any = None
        self._use_agent = bool(self._config.api_key)

        # Timeline
        self._timeline: Any = None
        self._current_index = 0

        # Stats
        self._results: list[PipelineResult] = []

    def load_timeline(self, path: str | Path) -> int:
        """Load a timeline of events.

        Args:
            path: Path to the timeline JSON file.

        Returns:
            Number of events loaded.
        """
        from cognifold.simulator.timeline import load_timeline

        self._timeline = load_timeline(path)
        self._current_index = 0
        self._results = []

        self._logger.info(f"Loaded {len(self._timeline)} events from {path}")
        return len(self._timeline)

    def process_event(self, event: Event) -> PipelineResult:
        """Process a single event through the pipeline.

        Args:
            event: The event to process.

        Returns:
            PipelineResult with details of the processing.
        """
        from cognifold.executor.runner import PlanExecutor
        from cognifold.executor.validator import PlanValidator
        from cognifold.logging import log_event_processing
        from cognifold.models.plan import OperationType

        start_time = datetime.now()

        with log_event_processing(event.event_id, event.title):
            # Score once, derive both context IDs and score dict
            scored = self._ranker.score_nodes(self._graph, event.timestamp)
            context_ids = [
                s.node_id
                for s in scored[: self._config.context.max_nodes]
                if s.composite_score >= self._config.context.min_score_threshold
            ]
            node_scores = {s.node_id: s.composite_score for s in scored}

            # Generate plan
            plan: UpdatePlan
            if self._use_agent:
                try:
                    plan = self._get_agent_plan(event, context_ids, node_scores)
                except Exception as e:
                    self._logger.warning(f"Agent error, using default plan: {e}")
                    plan = self._create_default_plan(event)
            else:
                plan = self._create_default_plan(event)

            # Validate plan
            validator = PlanValidator(self._graph)
            validation = validator.validate(plan)

            if not validation.is_valid:
                errors = [i.message for i in validation.errors]
                self._logger.warning(f"Validation failed: {errors}")
                plan = self._create_default_plan(event)

            # Execute plan
            executor = PlanExecutor(self._graph)
            execution = executor.execute(plan)

            # Calculate stats
            elapsed = (datetime.now() - start_time).total_seconds() * 1000

            # Find created concepts and actions
            concepts: list[str] = []
            actions: list[str] = []
            nodes_added = 0
            edges_added = 0

            for op in plan.operations:
                if op.op == OperationType.ADD_NODE:
                    nodes_added += 1
                    if op.node_type == "concept" and op.data:
                        concepts.append(op.data.get("id", op.data.get("title", "unknown")))
                    elif op.node_type == "action" and op.data:
                        actions.append(op.data.get("id", op.data.get("title", "unknown")))
                elif op.op == OperationType.ADD_EDGE:
                    edges_added += 1

            result = PipelineResult(
                event=event,
                plan=plan,
                success=execution.success,
                error=execution.error,
                execution_time_ms=elapsed,
                nodes_added=nodes_added,
                edges_added=edges_added,
                concepts_created=concepts,
                actions_created=actions,
            )

            self._results.append(result)
            return result

    def step(self) -> PipelineResult | None:
        """Process the next event in the timeline.

        Returns:
            PipelineResult or None if timeline is exhausted.
        """
        if self._timeline is None or self._current_index >= len(self._timeline):
            return None

        event = self._timeline[self._current_index]
        self._current_index += 1

        return self.process_event(event)

    def run(self, max_events: int | None = None) -> PipelineStats:
        """Run the pipeline on all remaining events.

        Args:
            max_events: Maximum number of events to process.

        Returns:
            PipelineStats with summary statistics.
        """
        count = 0
        while True:
            if max_events and count >= max_events:
                break

            result = self.step()
            if result is None:
                break

            count += 1

            if result.concepts_created:
                self._logger.info(f"Created concepts: {result.concepts_created}")
            if result.actions_created:
                self._logger.info(f"Created actions: {result.actions_created}")

        return self.get_stats()

    def get_stats(self) -> PipelineStats:
        """Get statistics for the current run.

        Returns:
            PipelineStats with summary information.
        """
        concepts = sum(len(r.concepts_created) for r in self._results)
        actions = sum(len(r.actions_created) for r in self._results)

        return PipelineStats(
            events_processed=len(self._results),
            events_failed=sum(1 for r in self._results if not r.success),
            total_nodes=self._graph.node_count,
            total_edges=self._graph.edge_count,
            concepts_created=concepts,
            actions_created=actions,
            total_time_ms=sum(r.execution_time_ms for r in self._results),
        )

    def save_graph(self, path: str | Path) -> None:
        """Save the current graph to a JSON file.

        Args:
            path: Path to save the graph.
        """
        from cognifold.graph.persistence import save_graph

        save_graph(self._graph, path)
        self._logger.info(f"Graph saved to {path}")

    def visualize(self, path: str | Path, title: str = "Cognifold Graph") -> Path:
        """Generate a visualization of the current graph.

        Args:
            path: Path for the output HTML file.
            title: Title for the visualization.

        Returns:
            Path to the generated HTML file.
        """
        from cognifold.simulator.visualizer import GraphVisualizer

        visualizer = GraphVisualizer()
        context_ids = self._ranker.get_context_node_ids(self._graph)
        node_scores = {s.node_id: s.composite_score for s in self._ranker.score_nodes(self._graph)}

        return visualizer.render(
            graph=self._graph,
            output_path=path,
            context_node_ids=context_ids,
            node_scores=node_scores,
            title=title,
        )

    @property
    def graph(self) -> Any:
        """Get the current graph."""
        return self._graph

    @property
    def results(self) -> list[PipelineResult]:
        """Get all processing results."""
        return self._results.copy()

    def _get_agent_plan(
        self,
        event: Event,
        context_ids: list[str],
        node_scores: dict[str, float],
    ) -> UpdatePlan:
        """Get a plan from the LLM agent."""
        if self._agent is None:
            from cognifold.agent import AgentConfig, CognifoldAgent

            agent_config = AgentConfig(
                model_name=self._config.model.name,
                temperature=self._config.model.temperature,
                max_tokens=self._config.model.max_tokens,
                max_exploration_steps=self._config.model.max_exploration_steps,
            )
            self._agent = CognifoldAgent(config=agent_config)

        return self._agent.process_event(
            event=event,
            graph=self._graph,
            context_node_ids=context_ids,
            node_scores=node_scores,
        )

    def _create_default_plan(self, event: Event) -> UpdatePlan:
        """Create a default plan that just adds the event."""
        from cognifold.models.plan import Operation, OperationType, UpdatePlan

        return UpdatePlan(
            plan_id=f"default-{event.event_id}",
            trigger_event_id=event.event_id,
            reasoning="Default plan: add event as node",
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
