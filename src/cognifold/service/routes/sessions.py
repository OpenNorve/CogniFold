"""Session management endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field

from cognifold.service.models import (
    CreateSessionRequest,
    LoadGraphRequest,
    SessionInfo,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])


class CheckpointResponse(BaseModel):
    """Response for checkpoint creation."""

    checkpoint_index: int
    timestamp: str
    event_count: int


class RestoreResponse(BaseModel):
    """Response for checkpoint restore."""

    restored_checkpoint_index: int
    timestamp: str
    event_count: int
    graph_stats: dict[str, Any] = Field(default_factory=dict)


@router.post("", response_model=SessionInfo, status_code=status.HTTP_201_CREATED)
async def create_session(body: CreateSessionRequest, request: Request) -> SessionInfo:
    """Create a new session."""
    mgr = request.app.state.session_manager
    session = await mgr.create_session(
        config=body.config,
        llm_api_keys=body.llm_api_keys,
        user_id=body.user_id,
        graph_data=body.graph_data,
    )
    return session.to_info()


@router.get("/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str, request: Request) -> SessionInfo:
    """Get session info/status."""
    mgr = request.app.state.session_manager
    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session.to_info()


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, request: Request) -> Response:
    """Persist and teardown session."""
    mgr = request.app.state.session_manager
    deleted = await mgr.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{session_id}/load", response_model=SessionInfo)
async def load_graph(session_id: str, body: LoadGraphRequest, request: Request) -> SessionInfo:
    """Load a saved graph into an existing session."""
    mgr = request.app.state.session_manager
    session = await mgr.load_graph_into_session(session_id, body.graph_data)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session.to_info()


@router.post("/{session_id}/checkpoint", response_model=CheckpointResponse)
async def create_checkpoint(session_id: str, request: Request) -> CheckpointResponse:
    """Save current graph state as a checkpoint for rollback."""
    from cognifold.graph.persistence import graph_to_dict

    mgr = request.app.state.session_manager
    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    now = datetime.now(timezone.utc).isoformat()
    async with session.lock:
        checkpoint: dict[str, Any] = {
            "timestamp": now,
            "event_count": session.event_count,
            "graph_data": graph_to_dict(session.graph),
        }
        session.checkpoints.append(checkpoint)
        idx = len(session.checkpoints) - 1

    return CheckpointResponse(
        checkpoint_index=idx,
        timestamp=now,
        event_count=session.event_count,
    )


@router.post("/{session_id}/restore", response_model=RestoreResponse)
async def restore_checkpoint(
    session_id: str,
    request: Request,
    checkpoint_index: int = Query(default=-1, description="Checkpoint index (-1 for latest)"),
) -> RestoreResponse:
    """Restore graph state from a checkpoint."""
    from cognifold.graph.persistence import dict_to_graph

    mgr = request.app.state.session_manager
    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    async with session.lock:
        if not session.checkpoints:
            raise HTTPException(status_code=400, detail="No checkpoints available")

        if checkpoint_index < 0:
            checkpoint_index = len(session.checkpoints) + checkpoint_index
        if checkpoint_index < 0 or checkpoint_index >= len(session.checkpoints):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid checkpoint index: {checkpoint_index} "
                f"(available: 0-{len(session.checkpoints) - 1})",
            )

        checkpoint = session.checkpoints[checkpoint_index]
        session.graph = dict_to_graph(checkpoint["graph_data"])
        session.event_count = checkpoint["event_count"]
        # Reset lazy-loaded components since graph changed
        session.agent = None
        session.query_agent = None
        session.ranker.invalidate_cache()

    stats = session.get_graph_stats()

    return RestoreResponse(
        restored_checkpoint_index=checkpoint_index,
        timestamp=checkpoint["timestamp"],
        event_count=session.event_count,
        graph_stats={
            "node_count": stats.node_count,
            "edge_count": stats.edge_count,
            "events": stats.events,
            "concepts": stats.concepts,
            "intents": stats.intents,
        },
    )


# ---------------------------------------------------------------------------
# Traces endpoint
# ---------------------------------------------------------------------------


class TraceEntryResponse(BaseModel):
    """Serialized trace entry."""

    event_id: str
    timestamp: str
    activated_concepts: list[str] = Field(default_factory=list)
    new_edges: list[list[str]] = Field(default_factory=list)
    removed_nodes: list[str] = Field(default_factory=list)
    merged_nodes: list[Any] = Field(default_factory=list)
    operation_count: int = 0
    plan_reasoning: str = ""


class TracesResponse(BaseModel):
    """Response for GET /sessions/{id}/traces."""

    entries: list[TraceEntryResponse]
    count: int
    total: int


@router.get("/{session_id}/traces", response_model=TracesResponse)
async def get_traces(
    session_id: str,
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    event_id: str | None = Query(default=None),
) -> TracesResponse:
    """Get cognitive trace entries for a session."""
    mgr = request.app.state.session_manager
    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    collector = getattr(session, "trace_collector", None)
    if collector is None:
        return TracesResponse(entries=[], count=0, total=0)

    entries = collector.get_entries(limit=limit, event_id=event_id)
    total = collector.count

    response_entries = [
        TraceEntryResponse(
            event_id=e.event_id,
            timestamp=e.timestamp.isoformat(),
            activated_concepts=e.activated_concepts,
            new_edges=[list(edge) for edge in e.new_edges],
            removed_nodes=e.removed_nodes,
            merged_nodes=e.merged_nodes,
            operation_count=e.operation_count,
            plan_reasoning=e.plan_reasoning,
        )
        for e in entries
    ]

    return TracesResponse(entries=response_entries, count=len(response_entries), total=total)


# ---------------------------------------------------------------------------
# Usage endpoint
# ---------------------------------------------------------------------------


class UsageResponse(BaseModel):
    """Response for GET /sessions/{id}/usage."""

    total_calls: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    calls_by_model: dict[str, Any] = Field(default_factory=dict)
    calls_by_type: dict[str, int] = Field(default_factory=dict)
    budget: dict[str, Any] = Field(default_factory=dict)


@router.get("/{session_id}/usage", response_model=UsageResponse)
async def get_usage(session_id: str, request: Request) -> UsageResponse:
    """Get LLM usage statistics and budget status for a session."""
    mgr = request.app.state.session_manager
    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    metrics = getattr(session, "llm_metrics", None)
    if metrics is None:
        return UsageResponse()

    summary = metrics.get_usage_summary()

    # Budget info
    budget_info: dict[str, Any] = {}
    budget_obj = getattr(session, "budget", None)
    if budget_obj is not None:
        from cognifold.utils.budget import BudgetEnforcer

        enforcer = BudgetEnforcer(budget=budget_obj, collector=metrics)
        budget_info = {
            "limits": {
                "max_tokens": budget_obj.max_tokens,
                "max_cost": budget_obj.max_cost,
                "max_calls": budget_obj.max_calls,
            },
            "remaining": enforcer.remaining(),
        }

    return UsageResponse(
        total_calls=int(summary.get("total_calls", 0)),
        total_tokens_in=int(summary.get("total_tokens_in", 0)),
        total_tokens_out=int(summary.get("total_tokens_out", 0)),
        total_tokens=int(summary.get("total_tokens", 0)),
        total_cost=float(summary.get("total_cost", 0.0)),
        total_latency_ms=float(summary.get("total_latency_ms", 0.0)),
        calls_by_model=summary.get("calls_by_model", {}),  # type: ignore[arg-type]
        calls_by_type=summary.get("calls_by_type", {}),  # type: ignore[arg-type]
        budget=budget_info,
    )


# ---------------------------------------------------------------------------
# Concept quality stats endpoint
# ---------------------------------------------------------------------------


class ConceptQualityResponse(BaseModel):
    """Response for GET /sessions/{id}/concept-quality."""

    total_concepts: int = 0
    total_events: int = 0
    concepts_per_event: float = 0.0
    orphan_concepts: int = 0
    orphan_rate: float = 0.0


@router.get("/{session_id}/concept-quality", response_model=ConceptQualityResponse)
async def get_concept_quality(session_id: str, request: Request) -> ConceptQualityResponse:
    """Get concept extraction quality stats for a session."""
    mgr = request.app.state.session_manager
    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    stats = session.graph.get_concept_quality_stats()
    return ConceptQualityResponse(**stats)
