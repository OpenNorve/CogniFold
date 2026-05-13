"""Graph state endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from cognifold.service.models import (
    EdgeResponse,
    ExpandedNode,
    GraphStateResponse,
    GraphStatsResponse,
    NodeExpansionResponse,
    NodeResponse,
    QueryNodeResponse,
)

router = APIRouter(prefix="/sessions/{session_id}/graph", tags=["graph"])


@router.get("/export")
async def export_graph(session_id: str, request: Request) -> dict[str, Any]:
    """Export full graph in persistence format (version, nodes, edges).

    Used by NeoLearn to persist graph_data for session restore.
    """
    from cognifold.graph.persistence import graph_to_dict

    mgr = request.app.state.session_manager
    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    async with session.lock:
        return graph_to_dict(session.graph)


@router.get("", response_model=GraphStateResponse)
async def get_graph_state(
    session_id: str,
    request: Request,
    max_nodes: int = Query(default=200, ge=1, le=2000),
) -> GraphStateResponse:
    """Get full graph state (bounded by max_nodes)."""
    mgr = request.app.state.session_manager
    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    async with session.lock:
        stats = session.get_graph_stats()
        all_nodes = session.graph.get_all_nodes()

        # When truncation is needed, score nodes so the most relevant
        # are kept.  PageRankCache avoids redundant recomputation.
        score_map: dict[str, float] = {}
        if len(all_nodes) > max_nodes:
            from datetime import datetime

            node_scores = session.ranker.score_nodes(session.graph, datetime.now())
            score_map = {ns.node_id: ns.composite_score for ns in node_scores}
            all_nodes.sort(key=lambda n: score_map.get(n.id, 0.0), reverse=True)

        node_list = all_nodes[:max_nodes]

        # Attach composite_score to node data so frontend can map to
        # visual properties (size, opacity, etc.)
        if score_map:
            for node in node_list:
                node.data["composite_score"] = round(score_map.get(node.id, 0.0), 4)

        node_ids = {n.id for n in node_list}

        # Collect edges where both endpoints are in the returned node set
        edges_resp: list[EdgeResponse] = []
        for src, tgt, _key, attrs in session.graph.internal_graph.edges(keys=True, data=True):
            if src in node_ids and tgt in node_ids:
                edges_resp.append(
                    EdgeResponse(
                        source_id=src,
                        target_id=tgt,
                        edge_type=attrs.get("edge_type"),
                        weight=attrs.get("weight", 1.0),
                    )
                )

    nodes_resp = [_node_to_response(n, session) for n in node_list]

    return GraphStateResponse(stats=stats, nodes=nodes_resp, edges=edges_resp)


@router.get("/stats", response_model=GraphStatsResponse)
async def get_graph_stats(session_id: str, request: Request) -> GraphStatsResponse:
    """Get graph statistics."""
    mgr = request.app.state.session_manager
    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    async with session.lock:
        return session.get_graph_stats()


@router.get("/concepts", response_model=list[QueryNodeResponse])
async def get_top_concepts(
    session_id: str,
    request: Request,
    top: int = Query(default=10, ge=1, le=100),
) -> list[QueryNodeResponse]:
    """Get top N concepts by importance."""
    from cognifold.query.agent import MemoryQueryAgent

    mgr = request.app.state.session_manager
    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    async with session.lock:
        agent = MemoryQueryAgent(graph=session.graph)
        summaries = agent.get_top_concepts(n=top)

    return [
        QueryNodeResponse(
            node_id=s.node_id,
            node_type=s.node_type,
            title=s.title,
            relevance_score=s.relevance_score,
            description=s.description,
            reasoning=s.reasoning,
            grounded_in=s.grounded_in,
        )
        for s in summaries
    ]


@router.get("/intents", response_model=list[QueryNodeResponse])
async def get_recent_intents(
    session_id: str,
    request: Request,
    recent: int = Query(default=10, ge=1, le=100),
) -> list[QueryNodeResponse]:
    """Get recent N intents."""
    from cognifold.query.agent import MemoryQueryAgent

    mgr = request.app.state.session_manager
    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    async with session.lock:
        agent = MemoryQueryAgent(graph=session.graph)
        summaries = agent.get_recent_intents(n=recent)

    return [
        QueryNodeResponse(
            node_id=s.node_id,
            node_type=s.node_type,
            title=s.title,
            relevance_score=s.relevance_score,
            description=s.description,
            reasoning=s.reasoning,
            grounded_in=s.grounded_in,
        )
        for s in summaries
    ]


@router.get("/events", response_model=list[QueryNodeResponse])
async def get_recent_events(
    session_id: str,
    request: Request,
    recent: int = Query(default=10, ge=1, le=100),
) -> list[QueryNodeResponse]:
    """Get recent N events."""
    from cognifold.models.node import NodeType
    from cognifold.query.models import QueryType
    from cognifold.query.scoring import QueryScorer

    mgr = request.app.state.session_manager
    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    async with session.lock:
        events = session.graph.get_nodes_by_type(NodeType.EVENT)
        # Sort by created_at descending
        events.sort(key=lambda n: n.created_at, reverse=True)  # pyright: ignore[reportUnknownLambdaType]
        events = events[:recent]

        scorer = QueryScorer(session.graph)
        summaries = scorer.rank_nodes_for_query(
            nodes=events,
            query_type=QueryType.TEMPORAL,
        )

    return [
        QueryNodeResponse(
            node_id=s.node_id,
            node_type=s.node_type,
            title=s.title,
            relevance_score=s.relevance_score,
            description=s.description,
            reasoning=s.reasoning,
            grounded_in=s.grounded_in,
        )
        for s in summaries[:recent]
    ]


@router.get("/nodes/{node_id}", response_model=NodeResponse)
async def get_node(session_id: str, node_id: str, request: Request) -> NodeResponse:
    """Get a single node by ID."""
    mgr = request.app.state.session_manager
    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    async with session.lock:
        if not session.graph.has_node(node_id):
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
        node = session.graph.get_node(node_id)
        neighbors = session.graph.get_neighbors(node_id)
        predecessors = session.graph.get_predecessors(node_id)

    return NodeResponse(
        node_id=node.id,
        node_type=node.type.value,
        data=node.data,
        created_at=node.created_at.isoformat(),
        last_accessed=node.last_accessed.isoformat(),
        access_count=node.access_count,
        reasoning=node.reasoning,
        grounded_in=node.grounded_in,
        neighbors=neighbors,
        predecessors=predecessors,
    )


@router.get("/nodes/{node_id}/expand", response_model=NodeExpansionResponse)
async def expand_node(
    session_id: str,
    node_id: str,
    request: Request,
    layers: int = Query(default=1, ge=1, le=5),
    direction: str = Query(default="both", pattern=r"^(outgoing|incoming|both)$"),
    max_nodes: int = Query(default=50, ge=1, le=500),
) -> NodeExpansionResponse:
    """Expand from a node by N layers, returning a subgraph."""
    mgr = request.app.state.session_manager
    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    async with session.lock:
        if not session.graph.has_node(node_id):
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

        nodes_with_depth, edges = session.graph.expand_from_node(
            node_id=node_id,
            max_depth=layers,
            direction=direction,
            max_nodes=max_nodes,
        )

        # Build response nodes with full data
        expanded_nodes: list[ExpandedNode] = []
        for nid, depth in nodes_with_depth:
            node = session.graph.get_node(nid)
            expanded_nodes.append(
                ExpandedNode(
                    node_id=nid,
                    node_type=node.type.value,
                    data=node.data,
                    depth=depth,
                )
            )

    edge_responses = [
        EdgeResponse(source_id=s, target_id=t, edge_type=et, weight=w) for s, t, et, w in edges
    ]

    total_nodes = len(expanded_nodes)
    truncated = total_nodes >= max_nodes

    return NodeExpansionResponse(
        root_node_id=node_id,
        layers=layers,
        direction=direction,
        nodes=expanded_nodes,
        edges=edge_responses,
        total_nodes=total_nodes,
        truncated=truncated,
    )


def _node_to_response(node: Any, session: Any) -> NodeResponse:
    """Convert a Node model to NodeResponse."""
    from cognifold.models.node import Node

    n: Node = node
    graph = session.graph
    return NodeResponse(
        node_id=n.id,
        node_type=n.type.value,
        data=n.data,
        created_at=n.created_at.isoformat(),
        last_accessed=n.last_accessed.isoformat(),
        access_count=n.access_count,
        reasoning=n.reasoning,
        grounded_in=n.grounded_in,
        neighbors=graph.get_neighbors(n.id) if graph.has_node(n.id) else [],
        predecessors=graph.get_predecessors(n.id) if graph.has_node(n.id) else [],
    )
