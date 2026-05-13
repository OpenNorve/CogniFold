"""Route handlers for the Cognifold service layer."""

from __future__ import annotations

from fastapi import APIRouter

from cognifold.service.routes.domains import router as domains_router
from cognifold.service.routes.events import router as events_router
from cognifold.service.routes.graph import router as graph_router
from cognifold.service.routes.intents import router as intents_router
from cognifold.service.routes.query import router as query_router
from cognifold.service.routes.sessions import router as sessions_router
from cognifold.service.routes.stream import router as stream_router
from cognifold.service.routes.users import router as users_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(sessions_router)
api_router.include_router(events_router)
api_router.include_router(query_router)
api_router.include_router(graph_router)
api_router.include_router(intents_router)
api_router.include_router(domains_router)
api_router.include_router(stream_router)
api_router.include_router(users_router)
