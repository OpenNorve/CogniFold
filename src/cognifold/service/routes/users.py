"""User identity endpoints (basic, no auth)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from cognifold.service.models import (
    CreateUserRequest,
    UserInfo,
    UserSessionInfo,
    UserSessionsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])


def _get_supabase(request: Request) -> Any:
    """Get Supabase client from app state, or raise 503."""
    client = getattr(request.app.state, "supabase_client", None)
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Supabase not configured. Set COGNIFOLD_SUPABASE_URL and COGNIFOLD_SUPABASE_KEY.",
        )
    return client


@router.post("", response_model=UserInfo)
async def create_user(body: CreateUserRequest, request: Request) -> UserInfo:
    """Create or upsert a user (no auth check)."""
    client = _get_supabase(request)

    row: dict[str, Any] = {
        "user_id": body.user_id,
        "display_name": body.display_name,
        "metadata": body.metadata,
    }
    resp = client.table("users").upsert(row, on_conflict="user_id").execute()
    data = resp.data[0] if resp.data else row

    return UserInfo(
        user_id=data.get("user_id", body.user_id),
        display_name=data.get("display_name"),
        metadata=data.get("metadata", {}),
        created_at=data.get("created_at"),
    )


@router.get("/{user_id}", response_model=UserInfo)
async def get_user(user_id: str, request: Request) -> UserInfo:
    """Get user info."""
    client = _get_supabase(request)

    resp = client.table("users").select("*").eq("user_id", user_id).maybe_single().execute()
    if resp is None or resp.data is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    data = resp.data
    return UserInfo(
        user_id=data["user_id"],
        display_name=data.get("display_name"),
        metadata=data.get("metadata", {}),
        created_at=data.get("created_at"),
    )


@router.get("/{user_id}/sessions", response_model=UserSessionsResponse)
async def list_user_sessions(user_id: str, request: Request) -> UserSessionsResponse:
    """List all sessions belonging to a user."""
    client = _get_supabase(request)

    resp = (
        client.table("sessions")
        .select("session_id, domain, created_at, updated_at")
        .eq("user_id", user_id)
        .execute()
    )

    sessions = [
        UserSessionInfo(
            session_id=row["session_id"],
            domain=row.get("domain"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )
        for row in (resp.data or [])
    ]

    return UserSessionsResponse(user_id=user_id, sessions=sessions)
