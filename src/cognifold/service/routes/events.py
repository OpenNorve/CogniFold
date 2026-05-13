"""Event ingestion endpoints."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Union

from fastapi import APIRouter, HTTPException, Query, Request

from cognifold.service.models import (
    AsyncTaskResponse,
    BatchIngestRequest,
    BatchIngestResponse,
    IngestEventRequest,
    IngestEventResponse,
    IngestionMode,
    LayeredBatchRequest,
    LayeredBatchResponse,
    LayerResult,
    TaskStatus,
    TaskStatusResponse,
)
from cognifold.service.processor import process_event_sync

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions/{session_id}", tags=["events"])


def _preserve_source_text(session: Any, event_id: str, description: str) -> None:
    """Write original event description into the event node's data.source_text.

    Must be called while holding session.lock.
    """
    from cognifold.models.node import NodeType

    node = session.graph.get_node(event_id)
    if node is not None and node.type == NodeType.EVENT:
        session.graph.update_node(event_id, {"source_text": description})


@router.post("/events", response_model=Union[IngestEventResponse, AsyncTaskResponse])
async def ingest_event(
    session_id: str,
    body: IngestEventRequest,
    request: Request,
    include_diff: bool = Query(default=False, description="Include operation details"),
) -> IngestEventResponse | AsyncTaskResponse:
    """Ingest a single event into a session."""
    mgr = request.app.state.session_manager
    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    if body.mode == IngestionMode.ASYNC:
        tracker = request.app.state.task_tracker
        task = tracker.create_task(session_id)
        _bg_task = asyncio.create_task(
            _process_event_async(task.task_id, session, body, include_diff, tracker, mgr)
        )
        # Store reference on app state to prevent GC
        if not hasattr(request.app.state, "background_tasks"):
            request.app.state.background_tasks = set()
        request.app.state.background_tasks.add(_bg_task)
        _bg_task.add_done_callback(request.app.state.background_tasks.discard)
        return AsyncTaskResponse(task_id=task.task_id)

    # Sync mode — run in thread pool to avoid blocking the event loop
    async with session.lock:
        result = await asyncio.to_thread(
            process_event_sync, body.event, session, include_diff=include_diff
        )
        # Preserve original description as source_text on the event node
        if result.success and body.event.description:
            _preserve_source_text(session, result.event_id, body.event.description)
    if result.success:
        await mgr.persist_session_data(session_id)
        # Flush graph sync writes after session row is in Supabase (FK safety)
        graph_sync = getattr(session, "graph_sync", None)
        if graph_sync is not None:
            await asyncio.to_thread(graph_sync.flush)
        await _publish_sse_events(request, session_id, result)
    return result


async def _process_event_async(
    task_id: str,
    session: Any,
    body: IngestEventRequest,
    include_diff: bool,
    tracker: Any,
    session_manager: Any = None,
) -> None:
    """Background coroutine for async event processing."""
    tracker.set_running(task_id)
    try:
        async with session.lock:
            result = await asyncio.to_thread(
                process_event_sync, body.event, session, include_diff=include_diff
            )
        if result.success and session_manager is not None:
            await session_manager.persist_session_data(session.session_id)
            graph_sync = getattr(session, "graph_sync", None)
            if graph_sync is not None:
                await asyncio.to_thread(graph_sync.flush)
        tracker.complete_task(task_id, result)
    except Exception as e:
        logger.exception(f"Async task {task_id} failed")
        tracker.fail_task(task_id, str(e))


@router.post("/events/batch", response_model=BatchIngestResponse)
async def ingest_batch(
    session_id: str,
    body: BatchIngestRequest,
    request: Request,
    include_diff: bool = Query(default=False),
) -> BatchIngestResponse:
    """Ingest a batch of events."""
    mgr = request.app.state.session_manager
    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    results: list[IngestEventResponse] = []
    succeeded = 0
    failed = 0

    async with session.lock:
        for event_input in body.events:
            try:
                result = await asyncio.to_thread(
                    process_event_sync, event_input, session, include_diff=include_diff
                )
                results.append(result)
                if result.success:
                    succeeded += 1
                    # Preserve original description as source_text
                    if event_input.description:
                        _preserve_source_text(session, result.event_id, event_input.description)
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Batch event failed: {e}")
                failed += 1

    if succeeded > 0:
        await mgr.persist_session_data(session_id)
        graph_sync = getattr(session, "graph_sync", None)
        if graph_sync is not None:
            await asyncio.to_thread(graph_sync.flush)

    return BatchIngestResponse(
        results=results,
        total=len(body.events),
        succeeded=succeeded,
        failed=failed,
    )


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(session_id: str, task_id: str, request: Request) -> TaskStatusResponse:
    """Poll async task status."""
    tracker = request.app.state.task_tracker
    record = tracker.get_task(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    if record.session_id != session_id:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found in session")

    return TaskStatusResponse(
        task_id=record.task_id,
        status=TaskStatus(record.status),
        result=record.result,
        error=record.error,
        created_at=record.created_at.isoformat(),
        completed_at=record.completed_at.isoformat() if record.completed_at else None,
    )


@router.post("/events/batch/layered", response_model=LayeredBatchResponse)
async def ingest_batch_layered(
    session_id: str,
    body: LayeredBatchRequest,
    request: Request,
) -> LayeredBatchResponse:
    """Ingest events using the layered (fast) pipeline.

    Layer 1: Add all events as nodes (no LLM, <30s for 1000+ events)
    Layer 2: Batched LLM enrichment (concepts, intents, edges)
    Layer 3: Batch embeddings + FAISS index
    """
    import uuid
    from datetime import datetime as dt

    from cognifold.models.event import Event
    from cognifold.pipeline.layered import LayeredPipeline
    from cognifold.pipeline.progress import LayerProgress
    from cognifold.simulator.timeline import Timeline

    mgr = request.app.state.session_manager
    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # Convert EventInput list to Event objects
    events: list[Event] = []
    for ei in body.events:
        events.append(
            Event(
                event_id=f"evt-{uuid.uuid4().hex[:12]}",
                timestamp=ei.timestamp or dt.now(),
                source=ei.source or session.config.domain,
                event_type=ei.event_type,
                title=ei.title,
                description=ei.description,
                location=ei.location,
                duration_minutes=ei.duration_minutes,
                context={},
                metadata=ei.metadata or {},
            )
        )

    timeline = Timeline(
        timeline_id=f"layered-{session_id}",
        description="Layered batch ingest",
        events=events,
    )

    # Build agent config from session if Layer 2 is requested
    agent_config = None
    if 2 in body.layers and session.agent:
        agent_config = session.agent.config

    # Use a no-op progress callback for the API
    class _NoOpProgress:
        def on_layer_start(self, progress: LayerProgress) -> None:
            pass

        def on_layer_progress(self, progress: LayerProgress) -> None:
            pass

        def on_layer_complete(self, progress: LayerProgress) -> None:
            pass

    async with session.lock:
        pipeline = LayeredPipeline(
            graph=session.graph,
            agent_config=agent_config,
            progress=_NoOpProgress(),
            batch_size=body.batch_size,
            graph_sync=getattr(session, "graph_sync", None),
        )

        layers_completed: list[LayerResult] = []

        if 1 in body.layers:
            pipeline.run_layer1(timeline)
            layers_completed.append(
                LayerResult(
                    layer=1,
                    completed=pipeline.stats.layer1_events,
                    errors=0,
                    time_ms=pipeline.stats.layer1_time_ms,
                )
            )

        if 2 in body.layers:
            with session.llm_env():
                pipeline.timeline = timeline  # ensure timeline is set
                pipeline.run_layer2()
            layers_completed.append(
                LayerResult(
                    layer=2,
                    completed=pipeline.stats.layer2_plans,
                    errors=len(pipeline.stats.errors),
                    time_ms=pipeline.stats.layer2_time_ms,
                )
            )

        if 3 in body.layers:
            with session.llm_env():
                pipeline.run_layer3()
            layers_completed.append(
                LayerResult(
                    layer=3,
                    completed=pipeline.stats.layer3_nodes_embedded,
                    errors=0,
                    time_ms=pipeline.stats.layer3_time_ms,
                )
            )

    if layers_completed:
        await mgr.persist_session_data(session_id)
        # Flush graph sync writes after session row is in Supabase (FK safety)
        graph_sync = getattr(session, "graph_sync", None)
        if graph_sync is not None:
            await asyncio.to_thread(graph_sync.flush)

    graph_stats = session.get_graph_stats()

    return LayeredBatchResponse(
        total_events=len(events),
        layers_completed=layers_completed,
        graph_stats=graph_stats,
        total_time_ms=pipeline.stats.total_time_ms,
    )


async def _publish_sse_events(
    request: Request, session_id: str, result: IngestEventResponse
) -> None:
    """Publish SSE events after a successful event ingestion."""
    broker = getattr(request.app.state, "sse_broker", None)
    if broker is None:
        return

    from cognifold.service.sse import SSEEvent

    # Always publish graph_updated
    await broker.publish(
        SSEEvent(
            event_type="graph_updated",
            data={
                "session_id": session_id,
                "node_count": result.graph_stats.node_count,
                "edge_count": result.graph_stats.edge_count,
            },
            session_id=session_id,
        )
    )

    # Check for intent_emerged (intent ADD_NODE in operations)
    if result.operations:
        for op in result.operations:
            if op.op == "ADD_NODE" and op.node_type == "intent":
                await broker.publish(
                    SSEEvent(
                        event_type="intent_emerged",
                        data={
                            "session_id": session_id,
                            "intent_id": op.node_id or "",
                        },
                        session_id=session_id,
                    )
                )
