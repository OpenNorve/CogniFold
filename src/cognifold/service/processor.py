"""Event processing logic for the service layer."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from typing import Any

from cognifold.service.models import (
    ContextEntry,
    EventInput,
    IngestEventResponse,
    OperationSummary,
)

logger = logging.getLogger(__name__)


def _flatten_context(context: dict[str, ContextEntry] | None) -> dict[str, Any]:
    """Convert structured ContextEntry dict to a flat dict for the pipeline.

    The existing pipeline expects ``context`` to be a plain dict. We store
    ContextEntry weight in metadata for now but flatten for compatibility.
    """
    if not context:
        return {}
    flat: dict[str, Any] = {}
    for key, entry in context.items():
        flat[key] = entry.value
    return flat


def process_event_sync(
    event_input: EventInput,
    session: Any,  # service.session.Session — avoid circular import
    include_diff: bool = False,
) -> IngestEventResponse:
    """Process an event synchronously using the Pipeline components.

    This wraps ``Pipeline.process_event()`` while accepting explicit
    dependencies from the session object.
    """
    from cognifold.executor.runner import PlanExecutor
    from cognifold.executor.validator import PlanValidator
    from cognifold.models.event import Event

    start = datetime.now()

    # Build an Event from EventInput (server generates event_id)
    event_id = f"evt-{uuid.uuid4().hex[:12]}"
    event = Event(
        event_id=event_id,
        timestamp=event_input.timestamp or datetime.now(),
        source=event_input.source or session.config.domain,
        event_type=event_input.event_type,
        title=event_input.title,
        description=event_input.description,
        location=event_input.location,
        duration_minutes=event_input.duration_minutes,
        context=_flatten_context(event_input.context),
        metadata=event_input.metadata,
    )

    graph = session.graph
    ranker = session.ranker

    # 1. Context window — score once, derive both context IDs and score dict
    scored = ranker.score_nodes(graph, event.timestamp)
    context_ids = [
        s.node_id for s in scored[: session.config.max_nodes] if s.composite_score >= 0.01
    ]
    node_scores = {s.node_id: s.composite_score for s in scored}

    # 2. Generate plan (agent or default)
    plan = _generate_plan(event, session, context_ids, node_scores)

    # 3. Ensure event ADD_NODE ops preserve original event fields.
    #    LLM-generated plans may omit description/location/duration_minutes.
    from cognifold.models.plan import OperationType

    for op in plan.operations:
        if op.op == OperationType.ADD_NODE and op.node_type == "event" and op.data:
            if event.description and not op.data.get("description"):
                op.data["description"] = event.description
            if event.location and not op.data.get("location"):
                op.data["location"] = event.location
            if event.duration_minutes and not op.data.get("duration_minutes"):
                op.data["duration_minutes"] = event.duration_minutes

    # 4. Validate
    validator = PlanValidator(graph)
    validation = validator.validate(plan)
    if not validation.is_valid:
        plan = _default_plan(event)

    # 5. Execute (inside llm_env so embeddings have API keys)
    executor = PlanExecutor(graph, graph_sync=getattr(session, "graph_sync", None))
    with session.llm_env():
        execution = executor.execute(plan)

    if not execution.success:
        logger.warning(
            f"Plan execution failed at op {execution.failed_at_operation}: {execution.error}"
        )

    # 5b. Record cognitive trace
    if hasattr(session, "trace_collector") and session.trace_collector.enabled:
        from cognifold.trace.collector import trace_from_plan

        trace_entry = trace_from_plan(plan, event_id)
        session.trace_collector.record(trace_entry)

    elapsed_ms = (datetime.now() - start).total_seconds() * 1000

    # 5. Build response
    ops_summary: list[OperationSummary] | None = None
    if include_diff:
        ops_summary = []
        for op in plan.operations:
            ops_summary.append(
                OperationSummary(
                    op=op.op.value,
                    node_type=op.node_type,
                    node_id=op.node_id,
                    source_id=op.source_id,
                    target_id=op.target_id,
                )
            )

    # Increment event count on successful processing
    if execution.success:
        session.event_count += 1

    stats = session.get_graph_stats()

    return IngestEventResponse(
        event_id=event_id,
        plan_id=plan.plan_id,
        reasoning=plan.reasoning,
        operations_completed=execution.operations_completed,
        success=execution.success,
        execution_time_ms=round(elapsed_ms, 2),
        graph_stats=stats,
        operations=ops_summary,
        error=execution.error,
    )


def _generate_plan(
    event: Any,
    session: Any,
    context_ids: list[str],
    node_scores: dict[str, float],
) -> Any:
    """Generate an update plan, using the agent if API keys are available."""
    from cognifold.models.plan import UpdatePlan
    from cognifold.service.llm_keys import get_api_key

    has_keys = bool(session.llm_api_keys) or bool(
        get_api_key("GOOGLE_API_KEY") or get_api_key("OPENAI_API_KEY")
    )

    if has_keys:
        # Budget check before LLM call
        if hasattr(session, "budget") and hasattr(session, "llm_metrics"):
            from cognifold.utils.budget import BudgetEnforcer, BudgetExceededError

            enforcer = BudgetEnforcer(budget=session.budget, collector=session.llm_metrics)
            try:
                enforcer.check()
            except BudgetExceededError as exc:
                logger.warning("LLM budget exceeded for session: %s", exc)
                return _default_plan(event)

        max_retries = 3
        base_delay = 2.0
        for attempt in range(max_retries):
            try:
                with session.llm_env():
                    if session.agent is None:
                        from cognifold.agent import AgentConfig, CognifoldAgent

                        agent_config = AgentConfig(
                            model_name=session.config.model_name,
                            temperature=session.config.temperature,
                            domain=session.config.domain,
                            language=session.config.language,
                            intent_density=session.config.intent_density,
                        )
                        session.agent = CognifoldAgent(config=agent_config)

                    plan: UpdatePlan = session.agent.process_event(
                        event=event,
                        graph=session.graph,
                        context_node_ids=context_ids,
                        node_scores=node_scores,
                    )
                    return plan
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = base_delay * (2**attempt)
                    logger.warning(
                        f"Rate limited (attempt {attempt + 1}/{max_retries}), waiting {wait}s"
                    )
                    time.sleep(wait)
                    continue
                logger.warning(f"Agent failed, using default plan: {e}")
                break

    return _default_plan(event)


def _default_plan(event: Any) -> Any:
    """Create a default plan that adds the event as a node.

    Explicitly sets ``node_id`` on the ADD_NODE operation so that the
    OperationSummary returned to clients includes the resolved ID
    (needed by playback / node-tracking features).
    """
    from cognifold.models.plan import Operation, OperationType, UpdatePlan

    return UpdatePlan(
        plan_id=f"default-{event.event_id}",
        trigger_event_id=event.event_id,
        reasoning="Default plan: add event as node",
        operations=[
            Operation(
                op=OperationType.ADD_NODE,
                node_type="event",
                node_id=event.event_id,
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
