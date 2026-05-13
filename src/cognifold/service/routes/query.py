"""Query endpoint."""

from __future__ import annotations

import asyncio
import logging
from functools import partial

from fastapi import APIRouter, HTTPException, Request

from cognifold.service.models import QueryNodeResponse, QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions/{session_id}", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_graph(session_id: str, body: QueryRequest, request: Request) -> QueryResponse:
    """Query the concept graph."""
    from cognifold.query.agent import MemoryQueryAgent
    from cognifold.query.models import QueryConfig, RetrievalMode

    mgr = request.app.state.session_manager
    session = await mgr.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    async with session.lock:
        # Auto-select retrieval mode
        has_embeddings = bool(session.llm_api_keys)
        retrieval_mode = RetrievalMode.HYBRID if has_embeddings else RetrievalMode.BM25

        query_config = QueryConfig(
            max_nodes=body.max_nodes or session.config.max_nodes,
            max_context_chars=body.max_context_chars or 6000,
            retrieval_mode=retrieval_mode,
        )

        if session.query_agent is None:
            session.query_agent = MemoryQueryAgent(
                graph=session.graph,
                config=query_config,
            )

        agent: MemoryQueryAgent = session.query_agent

        # Build language system prompt for LLM calls within the query agent
        language = getattr(session.config, "language", "auto")
        from cognifold.agent.prompt_sections import get_language_section

        _, lang_content = get_language_section(language)
        agent._language_system_prompt = lang_content  # type: ignore[attr-defined]

        # Keep query text clean for better search matching;
        # also prepend language hint as fallback for non-LLM paths
        query_text = body.query
        if language == "zh":
            query_text = f"[请用中文回答] {query_text}"
        elif language == "en":
            query_text = f"[Please respond in English] {query_text}"

        with session.llm_env():
            result = await asyncio.to_thread(
                partial(
                    agent.query,
                    query=query_text,
                    max_nodes=body.max_nodes,
                    max_context_chars=body.max_context_chars,
                    query_mode=body.query_mode,
                )
            )

    nodes = [
        QueryNodeResponse(
            node_id=n.node_id,
            node_type=n.node_type,
            title=n.title,
            relevance_score=n.relevance_score,
            description=n.description,
            reasoning=n.reasoning,
            grounded_in=n.grounded_in,
        )
        for n in result.nodes
    ]

    # Generate LLM answer if requested and we have context
    answer: str | None = None
    if body.generate_answer and result.context and result.nodes:
        try:
            from cognifold.query.llm import call_llm
            from cognifold.query.prompts import format_chat_answer_prompt

            answer_prompt = format_chat_answer_prompt(query=body.query, context=result.context)
            lang_prompt = lang_content if lang_content else None
            with session.llm_env():
                answer = call_llm(answer_prompt, system_prompt=lang_prompt)
        except Exception as e:
            logger.warning("Failed to generate LLM answer: %s", e)
            # Fallback: answer stays None, client uses context

    return QueryResponse(
        context=result.context,
        nodes=nodes,
        traversal_path=result.traversal_path,
        query_metadata=result.query_metadata,
        query_time_ms=round(result.query_time_ms, 2),
        answer=answer,
    )
