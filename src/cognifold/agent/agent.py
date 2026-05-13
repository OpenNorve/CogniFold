"""CognifoldAgent orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cognifold.agent.config import AgentConfig
    from cognifold.agent.prompt_profile import PromptProfile
    from cognifold.graph.store import ConceptGraph
    from cognifold.models.event import Event
    from cognifold.models.plan import UpdatePlan


class CognifoldAgent:
    """Orchestrates the LangGraph agent for event processing.

    The agent:
    1. Receives an event and context window
    2. Analyzes patterns in the graph
    3. Generates an UpdatePlan for graph modifications
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        prompt_profile: PromptProfile | None = None,
    ):
        """Initialize the agent.

        Args:
            config: Agent configuration. Uses defaults if not provided.
        """
        from cognifold.agent.config import AgentConfig

        self._config = config or AgentConfig()
        self._prompt_profile: PromptProfile | None = prompt_profile
        self._graph: Any = None  # Lazy-loaded LangGraph
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure the agent is initialized (lazy initialization)."""
        if self._initialized:
            return

        from cognifold.service.llm_keys import get_api_key

        # Verify API key is set (thread-local or env var)
        api_key = get_api_key("GOOGLE_API_KEY")
        openai_key = get_api_key("OPENAI_API_KEY")
        if not api_key and not openai_key:
            raise ValueError(
                "API Key is missing. "
                "Set GOOGLE_API_KEY for Gemini models or OPENAI_API_KEY for OpenAI models."
            )

        # Build the LangGraph (google.genai client created in graph.py)
        from cognifold.agent.graph import build_agent_graph

        self._graph = build_agent_graph()
        self._initialized = True

    def process_event(
        self,
        event: Event,
        graph: ConceptGraph,
        context_node_ids: list[str],
        node_scores: dict[str, float] | None = None,
    ) -> UpdatePlan:
        """Process an event and generate an update plan.

        Args:
            event: The new event to process.
            graph: The concept graph.
            context_node_ids: IDs of nodes in the context window.
            node_scores: Optional scores for context nodes.

        Returns:
            An UpdatePlan with operations to apply.

        Raises:
            ValueError: If API key is not set.
            RuntimeError: If agent fails to generate a valid plan.
        """
        self._ensure_initialized()

        # Build context
        from cognifold.agent.context import AgentContext
        from cognifold.agent.state import create_initial_state

        context = AgentContext.build(
            event=event,
            graph=graph,
            context_node_ids=context_node_ids,
            node_scores=node_scores or {},
        )

        # Create initial state
        initial_state = create_initial_state(
            context=context,
            config=self._config,
            prompt_profile=self._prompt_profile,
            domain=self._config.domain,
            max_exploration_steps=self._config.max_exploration_steps,
        )

        # Run the graph
        result = self._graph.invoke(initial_state)

        # Check for errors
        if result.get("error"):
            # Fall back to default plan (just add the event)
            return self._create_fallback_plan(event, result.get("error", "Unknown error"))

        # Return the plan
        plan = result.get("update_plan")
        if plan:
            return plan

        # No plan generated, create fallback
        return self._create_fallback_plan(event, "No plan generated")

    def _create_fallback_plan(self, event: Event, error: str) -> UpdatePlan:
        """Create a fallback plan that just adds the event.

        Args:
            event: The event to add.
            error: The error that caused the fallback.

        Returns:
            A minimal UpdatePlan that adds the event.
        """
        from cognifold.models.plan import Operation, OperationType, UpdatePlan

        return UpdatePlan(
            plan_id=f"fallback-{event.event_id}",
            trigger_event_id=event.event_id,
            reasoning=f"Fallback plan due to error: {error}",
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
                        "duration_minutes": event.duration_minutes,
                    },
                )
            ],
        )

    @property
    def config(self) -> AgentConfig:
        """Get the agent configuration."""
        return self._config
