"""Intent feedback and calibration endpoints (Phase 14.1)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from cognifold.intent.calibrator import IntentCalibrator
from cognifold.intent.feedback_store import FeedbackStore
from cognifold.intent.personalization import FeedbackType

router = APIRouter(prefix="/sessions/{session_id}/intents", tags=["intents"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SubmitFeedbackRequest(BaseModel):
    """Request body for submitting intent feedback."""

    feedback_type: FeedbackType = Field(..., description="accept | reject | defer | modify")
    user_comment: str | None = Field(default=None, description="Optional comment")
    modified_priority: str | None = Field(default=None, description="New priority (for modify)")
    modified_description: str | None = Field(
        default=None, description="Adjusted description (for modify)"
    )
    category_tags: list[str] = Field(default_factory=list, description="Optional category tags")


class FeedbackResponse(BaseModel):
    """Response after feedback submission."""

    feedback_id: str
    intent_id: str
    feedback_type: str
    intent_status: str | None = None


class CalibrationResponse(BaseModel):
    """Calibration profile response."""

    category_weights: dict[str, float]
    priority_bias: float
    acceptance_rate: float
    rejection_patterns: list[str]
    preferred_patterns: list[str]
    total_feedback: int


class PendingIntentResponse(BaseModel):
    """A pending intent suitable for user review."""

    intent_id: str
    title: str
    priority: str
    status: str
    description: str | None = None
    score_multiplier: float = 1.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/{intent_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    session_id: str,
    intent_id: str,
    body: SubmitFeedbackRequest,
    request: Request,
) -> FeedbackResponse:
    """Submit user feedback on an intent."""
    mgr = request.app.state.session_manager
    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    async with session.lock:
        if not session.graph.has_node(intent_id):
            raise HTTPException(status_code=404, detail=f"Intent {intent_id} not found")

        store = FeedbackStore(session.graph)
        fb = store.make_feedback(
            intent_id=intent_id,
            feedback_type=body.feedback_type,
            user_comment=body.user_comment,
            modified_priority=body.modified_priority,
            modified_description=body.modified_description,
            category_tags=body.category_tags,
        )
        store.add_feedback(fb)

        # Read back intent status
        intent_status: str | None = None
        if session.graph.has_node(intent_id):
            node = session.graph.get_node(intent_id)
            intent_status = node.data.get("status")

    return FeedbackResponse(
        feedback_id=fb.feedback_id,
        intent_id=intent_id,
        feedback_type=fb.feedback_type.value,
        intent_status=intent_status,
    )


@router.get("/calibration", response_model=CalibrationResponse)
async def get_calibration(session_id: str, request: Request) -> CalibrationResponse:
    """Get the current intent calibration profile for this session."""
    mgr = request.app.state.session_manager
    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    async with session.lock:
        store = FeedbackStore(session.graph)
        calibrator = IntentCalibrator(store)
        profile = calibrator.compute_profile()
        stats = store.get_stats()

    return CalibrationResponse(
        category_weights=profile.category_weights,
        priority_bias=profile.priority_bias,
        acceptance_rate=profile.acceptance_rate,
        rejection_patterns=profile.rejection_patterns,
        preferred_patterns=profile.preferred_patterns,
        total_feedback=stats.total_count,
    )


@router.get("/pending", response_model=list[PendingIntentResponse])
async def get_pending_intents(
    session_id: str,
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[PendingIntentResponse]:
    """Get pending intents that need user review, with calibrated scores."""
    from cognifold.models.node import NodeType

    mgr = request.app.state.session_manager
    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    async with session.lock:
        store = FeedbackStore(session.graph)
        calibrator = IntentCalibrator(store)
        calibrator.compute_profile()

        intents = list(session.graph.get_nodes_by_type(NodeType.INTENT))
        pending = [n for n in intents if n.data.get("status", "pending") == "pending"]
        pending.sort(key=lambda n: n.created_at, reverse=True)
        pending = pending[:limit]

        results: list[PendingIntentResponse] = []
        for node in pending:
            multiplier = calibrator.get_score_multiplier(node)
            results.append(
                PendingIntentResponse(
                    intent_id=node.id,
                    title=node.data.get("title", node.id),
                    priority=node.data.get("priority", "medium"),
                    status=node.data.get("status", "pending"),
                    description=node.data.get("description"),
                    score_multiplier=round(multiplier, 3),
                )
            )

    return results
