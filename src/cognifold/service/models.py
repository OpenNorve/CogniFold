"""Request/response Pydantic models for the Cognifold service layer."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Context entry schema
# ---------------------------------------------------------------------------


class ContextEntry(BaseModel):
    """A structured context entry with optional description and weight."""

    value: Any = Field(..., description="The context value")
    description: str | None = Field(default=None, description="Human-readable description")
    weight: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Importance weight (0.0-1.0)"
    )


# ---------------------------------------------------------------------------
# Session models
# ---------------------------------------------------------------------------


class SessionConfig(BaseModel):
    """Configuration for a session."""

    model_name: str = Field(
        default="gemini-3-flash-preview", min_length=1, description="LLM model name"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    max_nodes: int = Field(default=20, ge=1, le=1000, description="Max nodes in context window")
    domain: str = Field(default="personal-timeline", min_length=1, description="Event domain")
    language: str = Field(default="auto", description="Response language: auto, en, zh")
    scoring_alpha: float = Field(default=0.4, description="PageRank weight")
    scoring_beta: float = Field(default=0.4, description="Recency weight")
    scoring_gamma: float = Field(default=0.2, description="Access frequency weight")
    intent_density: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Intent generation aggressiveness (0.0=never, 1.0=maximum)",
    )


class CreateSessionRequest(BaseModel):
    """Request body for creating a new session."""

    config: SessionConfig = Field(default_factory=SessionConfig)
    llm_api_keys: dict[str, str] = Field(
        default_factory=dict,
        description="LLM API keys, e.g. {'google': '...', 'openai': '...'}",
    )
    user_id: str | None = Field(default=None, description="Optional user identifier")
    graph_data: dict[str, Any] | None = Field(
        default=None, description="Optional JSON graph to initialize from"
    )


class SessionInfo(BaseModel):
    """Response model for session information."""

    session_id: str
    user_id: str | None = None
    created_at: str
    last_accessed: str
    config: SessionConfig
    graph_stats: GraphStatsResponse | None = None
    event_count: int = 0

    model_config = {"from_attributes": True}


class LoadGraphRequest(BaseModel):
    """Request body for loading a graph into a session."""

    graph_data: dict[str, Any] = Field(..., description="JSON graph payload")


# ---------------------------------------------------------------------------
# Event ingestion models
# ---------------------------------------------------------------------------


class IngestionMode(str, Enum):
    """Event ingestion mode."""

    SYNC = "sync"
    ASYNC = "async"


class EventInput(BaseModel):
    """Flexible event input — server fills defaults for missing fields."""

    event_type: str = Field(..., description="Type of event (e.g. meal, work, exercise)")
    title: str = Field(..., description="Short description of the event")
    timestamp: datetime | None = Field(
        default=None, description="When the event occurred (default: now)"
    )
    source: str | None = Field(default=None, description="Event source/domain")
    description: str | None = Field(default=None, description="Detailed description")
    location: str | None = Field(default=None, description="Where the event occurred")
    duration_minutes: int | None = Field(default=None, ge=0, description="Duration in minutes")
    context: dict[str, ContextEntry] | None = Field(
        default=None, description="Structured context entries"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class IngestEventRequest(BaseModel):
    """Request body for ingesting a single event."""

    event: EventInput
    mode: IngestionMode = Field(default=IngestionMode.SYNC, description="Ingestion mode")


class OperationSummary(BaseModel):
    """Summary of a single graph operation."""

    op: str
    node_type: str | None = None
    node_id: str | None = None
    source_id: str | None = None
    target_id: str | None = None


class IngestEventResponse(BaseModel):
    """Response for a successfully ingested event (sync mode)."""

    event_id: str
    plan_id: str
    reasoning: str
    operations_completed: int
    success: bool
    execution_time_ms: float
    graph_stats: GraphStatsResponse
    operations: list[OperationSummary] | None = None
    error: str | None = None


class AsyncTaskResponse(BaseModel):
    """Response for an async ingestion request."""

    task_id: str
    status: str = "pending"
    message: str = "Task created, poll for status"


class TaskStatus(str, Enum):
    """Status of an async task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatusResponse(BaseModel):
    """Response for task status polling."""

    task_id: str
    status: TaskStatus
    result: IngestEventResponse | None = None
    error: str | None = None
    created_at: str
    completed_at: str | None = None


class BatchIngestRequest(BaseModel):
    """Request body for batch event ingestion."""

    events: list[EventInput] = Field(..., min_length=1, max_length=100)
    mode: IngestionMode = Field(default=IngestionMode.SYNC, description="Ingestion mode")


class BatchIngestResponse(BaseModel):
    """Response for batch event ingestion."""

    results: list[IngestEventResponse]
    total: int
    succeeded: int
    failed: int


class LayeredBatchRequest(BaseModel):
    """Request body for layered (fast) batch ingestion."""

    events: list[EventInput] = Field(..., min_length=1, max_length=5000)
    layers: list[int] = Field(
        default=[1],
        description="Which layers to run: 1=fast ingest, 2=LLM enrichment, 3=embeddings",
    )
    batch_size: int = Field(default=10, ge=1, le=100, description="Events per LLM batch (Layer 2)")


class LayerResult(BaseModel):
    """Result for a single pipeline layer."""

    layer: int
    completed: int
    errors: int
    time_ms: float


class LayeredBatchResponse(BaseModel):
    """Response for layered batch ingestion."""

    total_events: int
    layers_completed: list[LayerResult]
    graph_stats: GraphStatsResponse
    total_time_ms: float


# ---------------------------------------------------------------------------
# Query models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for querying the graph."""

    query: str = Field(..., min_length=1, description="Natural language query")
    max_nodes: int | None = Field(default=None, ge=1, description="Override max nodes")
    max_context_chars: int | None = Field(
        default=None, ge=100, description="Override max context chars"
    )
    query_mode: str = Field(default="mergefold", description="Query mode")
    generate_answer: bool = Field(
        default=False, description="Generate an LLM answer from retrieved context"
    )


class QueryNodeResponse(BaseModel):
    """A node in the query response."""

    node_id: str
    node_type: str
    title: str
    relevance_score: float
    description: str | None = None
    reasoning: str | None = None
    grounded_in: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """Response for a graph query."""

    context: str
    nodes: list[QueryNodeResponse]
    traversal_path: list[str]
    query_metadata: dict[str, Any]
    query_time_ms: float
    answer: str | None = Field(default=None, description="LLM-generated answer from context")


# ---------------------------------------------------------------------------
# Graph state models
# ---------------------------------------------------------------------------


class ExpandedNode(BaseModel):
    """A node in an expansion response, with its depth from the root."""

    node_id: str
    node_type: str
    data: dict[str, Any]
    depth: int = Field(description="Distance from root node (0 = root)")


class EdgeResponse(BaseModel):
    """An edge in an expansion response."""

    source_id: str
    target_id: str
    edge_type: str | None = None
    weight: float = 1.0


class NodeExpansionResponse(BaseModel):
    """Response for a node expansion (BFS subgraph)."""

    root_node_id: str
    layers: int
    direction: str
    nodes: list[ExpandedNode]
    edges: list[EdgeResponse]
    total_nodes: int = Field(description="Total reachable nodes before truncation")
    truncated: bool


class GraphStatsResponse(BaseModel):
    """Graph statistics."""

    node_count: int
    edge_count: int
    concepts: int = 0
    events: int = 0
    intents: int = 0
    time_nodes: int = 0


class NodeResponse(BaseModel):
    """Full node detail response."""

    node_id: str
    node_type: str
    data: dict[str, Any]
    created_at: str
    last_accessed: str
    access_count: int
    reasoning: str | None = None
    grounded_in: list[str] = Field(default_factory=list)
    neighbors: list[str] = Field(default_factory=list)
    predecessors: list[str] = Field(default_factory=list)


class GraphStateResponse(BaseModel):
    """Full or partial graph state."""

    stats: GraphStatsResponse
    nodes: list[NodeResponse]
    edges: list[EdgeResponse] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# User models
# ---------------------------------------------------------------------------


class CreateUserRequest(BaseModel):
    """Request body for creating/updating a user."""

    user_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_\-:.]+$",
        description="Unique user identifier",
    )
    display_name: str | None = Field(default=None, description="Display name")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class UserInfo(BaseModel):
    """Response model for user information."""

    user_id: str
    display_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class UserSessionInfo(BaseModel):
    """Summary of a session in user session list."""

    session_id: str
    domain: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class UserSessionsResponse(BaseModel):
    """Response for listing a user's sessions."""

    user_id: str
    sessions: list[UserSessionInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Error model
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    error_code: str | None = None


# Forward reference update
SessionInfo.model_rebuild()
