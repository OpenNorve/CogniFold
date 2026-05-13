"""Query-specific prompts for the Memory Query Interface.

This module provides prompt templates for:
- Query intent parsing
- Node relevance assessment
- Context summarization
"""

from __future__ import annotations

# Query intent parsing prompt
QUERY_INTENT_PROMPT = """Analyze the following query and extract its intent.

Query: {query}

Determine:
1. Query Type: What kind of information is being sought?
   - SEMANTIC: Looking for meaning, relationships, or concepts
   - TEMPORAL: Looking for time-based information (recent, date ranges)
   - STRUCTURAL: Looking for important/connected information
   - HYBRID: Combination of the above

2. Key Topics: What are the main subjects or concepts being queried?

3. Time Context: Is there a specific time reference? (e.g., "yesterday", "last week", "this morning")

4. Scope: How broad or narrow is the query?
   - BROAD: General overview needed
   - FOCUSED: Specific information needed
   - DEEP: Detailed exploration needed

Respond in the following JSON format:
{{
    "query_type": "SEMANTIC|TEMPORAL|STRUCTURAL|HYBRID",
    "key_topics": ["topic1", "topic2"],
    "time_context": "description or null",
    "scope": "BROAD|FOCUSED|DEEP",
    "reasoning": "brief explanation of analysis"
}}
"""

# Node relevance assessment prompt
NODE_RELEVANCE_PROMPT = """Given the query and a node from the concept graph, assess how relevant this node is.

Query: {query}

Node:
- Type: {node_type}
- Title: {title}
- Description: {description}
- Reasoning: {reasoning}
- Connected to events: {grounded_in}

Rate the relevance from 0.0 (not relevant) to 1.0 (highly relevant).

Consider:
1. Does the node directly address the query?
2. Does the node provide context that helps answer the query?
3. Is the node connected to concepts/events relevant to the query?

Respond with a single number between 0.0 and 1.0.
"""

# Context summarization prompt
CONTEXT_SUMMARY_PROMPT = """Summarize the following context from a concept graph in response to a query.

Query: {query}

Retrieved Context:
{context}

Create a concise summary that:
1. Directly addresses the query
2. Highlights the most relevant information
3. Notes any patterns or connections between nodes
4. Identifies any gaps in the information

Keep the summary under 500 words.
"""

# Query refinement prompt
QUERY_REFINEMENT_PROMPT = """The initial query returned limited results. Suggest refinements.

Original Query: {query}
Results Found: {result_count} nodes

Based on the available graph structure, suggest:
1. Alternative phrasings that might match more nodes
2. Related concepts that could be explored
3. Broader or narrower scopes that might help

Respond in JSON format:
{{
    "alternative_queries": ["query1", "query2"],
    "related_concepts": ["concept1", "concept2"],
    "scope_suggestions": ["suggestion1", "suggestion2"]
}}
"""


def format_intent_prompt(query: str) -> str:
    """Format the query intent parsing prompt.

    Args:
        query: The user's natural language query.

    Returns:
        Formatted prompt string.
    """
    return QUERY_INTENT_PROMPT.format(query=query)


def format_relevance_prompt(
    query: str,
    node_type: str,
    title: str,
    description: str | None = None,
    reasoning: str | None = None,
    grounded_in: list[str] | None = None,
) -> str:
    """Format the node relevance assessment prompt.

    Args:
        query: The user's query.
        node_type: Type of the node.
        title: Node title.
        description: Node description.
        reasoning: Why the node exists.
        grounded_in: Source event IDs.

    Returns:
        Formatted prompt string.
    """
    return NODE_RELEVANCE_PROMPT.format(
        query=query,
        node_type=node_type,
        title=title,
        description=description or "Not available",
        reasoning=reasoning or "Not provided",
        grounded_in=", ".join(grounded_in) if grounded_in else "None",
    )


def format_summary_prompt(query: str, context: str) -> str:
    """Format the context summarization prompt.

    Args:
        query: The user's query.
        context: Retrieved context text.

    Returns:
        Formatted prompt string.
    """
    return CONTEXT_SUMMARY_PROMPT.format(query=query, context=context)


def format_refinement_prompt(query: str, result_count: int) -> str:
    """Format the query refinement prompt.

    Args:
        query: The original query.
        result_count: Number of results found.

    Returns:
        Formatted prompt string.
    """
    return QUERY_REFINEMENT_PROMPT.format(query=query, result_count=result_count)


# =========================================================================
# Agentic retrieval prompts (W2)
# =========================================================================

# Sufficiency check prompt - determines if Round 1 results answer the query
SUFFICIENCY_CHECK_PROMPT = """You are a retrieval quality evaluator. Given a query and retrieved results,
determine whether the results are SUFFICIENT to answer the query.

Query: {query}

Retrieved Results ({result_count} items):
{results_text}

Evaluate:
1. Do the results contain information directly relevant to the query?
2. Is there enough detail to provide a complete answer?
3. Are there obvious gaps in the information?

Respond with ONLY a JSON object:
{{
    "sufficient": true or false,
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}}
"""

# Multi-query generation prompt - generates complementary queries for Round 2
MULTI_QUERY_PROMPT = """You are a query expansion expert. The initial retrieval for the following query
returned insufficient results. Generate 2-3 complementary queries that approach
the information need from different angles.

Original Query: {query}

Initial Results Summary:
{results_summary}

Generate queries that:
1. Use different keywords or phrasings to find the same information
2. Break down the original query into sub-questions
3. Search for related concepts that might contain the answer

Respond with ONLY a JSON object:
{{
    "queries": ["query1", "query2", "query3"]
}}
"""


def format_sufficiency_prompt(
    query: str,
    results: list[dict[str, str]],
) -> str:
    """Format the sufficiency check prompt.

    Args:
        query: The original query.
        results: List of result dicts with 'title' and 'description' keys.

    Returns:
        Formatted prompt string.
    """
    results_text = "\n".join(
        f"- [{r.get('node_type', 'unknown')}] {r.get('title', 'Untitled')}: "
        f"{r.get('description', 'No description')}"
        for r in results
    )
    return SUFFICIENCY_CHECK_PROMPT.format(
        query=query,
        result_count=len(results),
        results_text=results_text or "(no results)",
    )


# =========================================================================
# Chat answer generation prompt
# =========================================================================

CHAT_ANSWER_PROMPT = """You are a knowledgeable assistant. Answer the user's question based ONLY on the provided context from a knowledge graph. Be direct, informative, and well-structured.

User's Question: {query}

Retrieved Context:
{context}

Rules:
1. Answer based ONLY on the provided context. If the context doesn't contain enough information, say so.
2. Use markdown formatting for clarity (headers, lists, bold for key terms).
3. Reference specific concepts or events from the context when relevant.
4. Keep the answer concise but comprehensive — aim for 2-4 paragraphs.
5. If the context contains multiple perspectives or topics, organize them clearly.
"""


def format_chat_answer_prompt(query: str, context: str) -> str:
    """Format the chat answer generation prompt.

    Args:
        query: The user's natural language query.
        context: Retrieved context from the knowledge graph.

    Returns:
        Formatted prompt string.
    """
    return CHAT_ANSWER_PROMPT.format(query=query, context=context)


def format_multi_query_prompt(
    query: str,
    results_summary: str,
) -> str:
    """Format the multi-query generation prompt.

    Args:
        query: The original query.
        results_summary: Summary of initial results.

    Returns:
        Formatted prompt string.
    """
    return MULTI_QUERY_PROMPT.format(
        query=query,
        results_summary=results_summary or "(no results)",
    )
