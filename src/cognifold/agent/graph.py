"""LangGraph state graph definition for the Cognifold agent."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

# Note: AgentState must be imported unconditionally because LangGraph
# inspects function signatures at runtime when building the graph.
from cognifold.agent.state import AgentState

if TYPE_CHECKING:
    pass


def analyze_node(state: AgentState) -> dict[str, Any]:
    """Initial analysis node - sends event to LLM for analysis.

    This node:
    1. Formats the system and user prompts
    2. Sends to the LLM
    3. Returns messages to add to state

    Args:
        state: Current agent state.

    Returns:
        State updates with new messages.
    """
    import contextlib

    from cognifold.agent.domain import PERSONAL_TIMELINE_DOMAIN, DomainConfig, get_domain_config
    from cognifold.agent.prompts import format_system_prompt_for_domain, format_user_prompt

    context = state["context"]
    config = state["config"]
    profile = state.get("prompt_profile")

    domain = PERSONAL_TIMELINE_DOMAIN
    domain_name = None
    if profile and profile.domain:
        domain_name = profile.domain
    elif state.get("domain"):
        domain_name = state["domain"]
    if domain_name:
        with contextlib.suppress(KeyError):
            domain = get_domain_config(domain_name)

    mode = profile.mode if profile else None
    time_guidelines = getattr(config, "time_guidelines", ())
    if profile and profile.features.get("enable_time_nodes") is False:
        time_guidelines = ()

    # Merge section config: profile overrides domain
    disabled = domain.disabled_sections
    extras = dict(domain.extra_sections)
    extras_pos = domain.extra_section_position

    intent_density = getattr(config, "intent_density", 0.3)

    if profile:
        if profile.disabled_sections is not None:
            disabled = profile.disabled_sections
        if profile.extra_sections is not None:
            extras = dict(profile.extra_sections)

    # When intent_density is 0.0, disable all intent prompt sections.
    # Applied AFTER profile override so both sets are merged.
    if intent_density <= 0.0:
        disabled = frozenset(disabled | {"intents"})

    domain_for_prompt = DomainConfig(
        name=domain.name,
        description=domain.description,
        event_description=domain.event_description,
        node_type_descriptions=dict(domain.node_type_descriptions),
        concept_examples=list(domain.concept_examples),
        action_examples=list(domain.action_examples),
        time_examples=list(domain.time_examples),
        pattern_types=list(domain.pattern_types),
        hierarchy_examples=list(domain.hierarchy_examples),
        concept_guidelines=config.concept_guidelines,
        action_guidelines=config.action_guidelines,
        time_guidelines=time_guidelines,
        disabled_sections=disabled,
        extra_sections=extras,
        extra_section_position=extras_pos,
    )

    # Format prompts
    system_prompt = format_system_prompt_for_domain(
        domain_for_prompt,
        mode=mode,
        template=profile.system_prompt_template if profile else None,
    )

    # Inject intent density guidance based on config (skip when density <= 0)
    from cognifold.agent.prompt_sections import get_intent_density_section, get_language_section

    if intent_density > 0.0:
        _, density_content = get_intent_density_section(intent_density)
        system_prompt += density_content

    # Inject language section so LLM outputs match the session language
    language = config.language if hasattr(config, "language") else "auto"

    _, lang_content = get_language_section(language)
    system_prompt += "\n" + lang_content
    if profile and profile.user_prompt_template:
        # Extract event context fields for template formatting
        event_context = context.event.context if context.event and context.event.context else {}
        template_vars = {
            "event_details": context.format_event_for_prompt(),
            "context_window": context.format_context_for_prompt(),
            "graph_stats": "",
            "speaker": event_context.get("speaker", "Unknown"),
            "session_id": event_context.get("session_id", ""),
            "timestamp": context.event.timestamp.isoformat() if context.event else "",
        }
        user_prompt = profile.user_prompt_template.format(**template_vars)
    else:
        user_prompt = format_user_prompt(
            event_details=context.format_event_for_prompt(),
            context_window=context.format_context_for_prompt(),
            mode=mode,
        )

    # Build messages
    messages = [
        {"role": "system", "content": system_prompt, "tool_calls": None, "tool_call_id": None},
        {"role": "user", "content": user_prompt, "tool_calls": None, "tool_call_id": None},
    ]

    return {"messages": messages}


def call_llm_node(state: AgentState) -> dict[str, Any]:
    """Call the LLM with current messages.

    Args:
        state: Current agent state.

    Returns:
        State updates with LLM response.
    """
    from cognifold.service.llm_keys import get_api_key

    config = state["config"]
    messages = state["messages"]

    # Handle OpenAI models
    if config.model_name.startswith("openai:"):
        return _call_openai_node(state)

    from google import genai
    from google.genai import types

    api_key = get_api_key("GOOGLE_API_KEY") or get_api_key("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    # Convert to Gemini format
    gemini_contents: Any = _convert_to_gemini_format(cast(Sequence[dict[str, Any]], messages))

    # Get tool definitions for new SDK
    tools = (
        _get_tool_definitions_new()
        if state["exploration_steps"] < state["max_exploration_steps"]
        else None
    )

    # Build generation config with automatic function calling for gemini-3 thought signatures
    gen_config_kwargs: dict[str, Any] = {
        "temperature": config.temperature,
        "max_output_tokens": config.max_tokens,
        "tools": tools,
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(
            disable=False,
            maximum_remote_calls=3,
        )
        if tools
        else None,
    }

    # Only use response_mime_type if no tools are present, as they are mutually exclusive in some API versions
    if not tools:
        gen_config_kwargs["response_mime_type"] = "application/json"

    try:
        gen_config = types.GenerateContentConfig(**gen_config_kwargs)
    except TypeError:
        gen_config_kwargs.pop("response_mime_type", None)
        gen_config = types.GenerateContentConfig(**gen_config_kwargs)

    # Call Gemini
    import time as _time

    _t0 = _time.monotonic()
    response: Any = client.models.generate_content(
        model=config.model_name,
        contents=gemini_contents,
        config=gen_config,
    )
    _latency_ms = (_time.monotonic() - _t0) * 1000

    # Record LLM metrics if a collector is active
    _usage = getattr(response, "usage_metadata", None)
    if _usage:
        from cognifold.service.llm_keys import get_metrics_collector
        from cognifold.utils.llm_metrics import LLMCallMetrics, estimate_cost

        _collector = get_metrics_collector()
        if _collector is not None:
            _tin = getattr(_usage, "prompt_token_count", 0) or 0
            _tout = getattr(_usage, "candidates_token_count", 0) or 0
            _collector.record(
                LLMCallMetrics(
                    model=config.model_name,
                    tokens_in=_tin,
                    tokens_out=_tout,
                    latency_ms=_latency_ms,
                    cost_estimate=estimate_cost(config.model_name, _tin, _tout),
                    call_type="agent_plan",
                )
            )

    # Process response
    if (
        getattr(response, "candidates", None)
        and response.candidates[0]
        and getattr(response.candidates[0], "content", None)
        and getattr(response.candidates[0].content, "parts", None)
    ):
        parts = response.candidates[0].content.parts

        # Check for tool calls
        tool_calls = []
        text_content = ""

        for part in parts:
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                tool_calls.append(
                    {
                        "id": f"call_{len(tool_calls)}",
                        "name": fc.name,
                        "arguments": dict(fc.args) if fc.args else {},
                    }
                )
            elif hasattr(part, "text") and part.text:
                text_content += part.text

        # Preserve raw Gemini parts so thought_signature is retained
        # for subsequent API calls (required by gemini-3).
        raw_gemini_parts = list(parts)

        if tool_calls:
            # LLM wants to use tools
            new_message = {
                "role": "assistant",
                "content": text_content,
                "tool_calls": tool_calls,
                "tool_call_id": None,
                "_gemini_parts": raw_gemini_parts,
            }
            return {
                "messages": state["messages"] + [new_message],
                "raw_response": text_content,
            }
        else:
            # LLM provided final response
            new_message = {
                "role": "assistant",
                "content": text_content,
                "tool_calls": None,
                "tool_call_id": None,
            }
            return {
                "messages": state["messages"] + [new_message],
                "raw_response": text_content,
            }

    return {"error": "No response from LLM"}


def _call_openai_node(state: AgentState) -> dict[str, Any]:
    """Call OpenAI-compatible LLM."""
    import os

    from openai import OpenAI

    from cognifold.service.llm_keys import get_api_key

    config = state["config"]
    messages = state["messages"]
    model_name = config.model_name.replace("openai:", "")

    # Create client
    client = OpenAI(
        api_key=get_api_key("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    # Convert messages
    openai_messages = []
    for msg in messages:
        if msg["role"] == "tool":
            openai_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": msg["tool_call_id"],
                    "content": msg["content"],
                }
            )
        elif msg["role"] == "assistant" and msg.get("tool_calls"):
            tool_calls_for_msg = msg.get("tool_calls") or []
            openai_messages.append(
                {
                    "role": "assistant",
                    "content": msg.get("content"),
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in tool_calls_for_msg
                    ],
                }
            )
        else:
            openai_messages.append(
                {
                    "role": msg["role"],
                    "content": msg["content"] or "",
                }
            )

    # Get tools
    tools = []
    if state["exploration_steps"] < state["max_exploration_steps"]:
        for tool in _get_tool_definitions()[0]["function_declarations"]:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                }
            )

    # Call API
    create_kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": openai_messages,
        "temperature": config.temperature,
    }

    # Handle o1/o3 models that use max_completion_tokens
    if model_name.startswith("o1") or model_name.startswith("o3") or "gpt-5" in model_name:
        create_kwargs["max_completion_tokens"] = config.max_tokens
    else:
        create_kwargs["max_tokens"] = config.max_tokens

    if tools:
        create_kwargs["tools"] = tools

    import time as _time

    _t0 = _time.monotonic()
    response = client.chat.completions.create(**create_kwargs)
    _latency_ms = (_time.monotonic() - _t0) * 1000

    # Record LLM metrics if a collector is active
    _usage = getattr(response, "usage", None)
    if _usage:
        from cognifold.service.llm_keys import get_metrics_collector
        from cognifold.utils.llm_metrics import LLMCallMetrics, estimate_cost

        _collector = get_metrics_collector()
        if _collector is not None:
            _tin = getattr(_usage, "prompt_tokens", 0) or 0
            _tout = getattr(_usage, "completion_tokens", 0) or 0
            _collector.record(
                LLMCallMetrics(
                    model=model_name,
                    tokens_in=_tin,
                    tokens_out=_tout,
                    latency_ms=_latency_ms,
                    cost_estimate=estimate_cost(model_name, _tin, _tout),
                    call_type="agent_plan",
                )
            )

    message: Any = response.choices[0].message

    if message.tool_calls:
        tool_calls = []
        for tc in message.tool_calls or []:
            tc = cast(Any, tc)
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }
            )

        new_message = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": tool_calls,
            "tool_call_id": None,
        }
    else:
        new_message = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": None,
            "tool_call_id": None,
        }

    return {
        "messages": state["messages"] + [new_message],
        "raw_response": message.content or "",
    }


def execute_tools_node(state: AgentState) -> dict[str, Any]:
    """Execute tool calls from the LLM.

    Args:
        state: Current agent state.

    Returns:
        State updates with tool results.
    """
    from cognifold.agent.tools import GraphTools

    messages = state["messages"]
    last_message = messages[-1]

    tool_calls_to_run = last_message.get("tool_calls") or []
    if not tool_calls_to_run:
        return {}

    # Execute tools
    tools = GraphTools(state["context"].graph)
    new_messages = list(messages)

    for tool_call in tool_calls_to_run:
        tool_name = tool_call["name"]
        arguments = tool_call["arguments"]

        try:
            result = tools.call_tool(tool_name, arguments)
            result_str = json.dumps(result, default=str)
        except Exception as e:
            result_str = json.dumps({"error": str(e)})

        new_messages.append(
            {
                "role": "tool",
                "content": result_str,
                "tool_calls": None,
                "tool_call_id": tool_call["id"],
                "_tool_name": tool_name,
            }
        )

    return {
        "messages": new_messages,
        "exploration_steps": state["exploration_steps"] + 1,
    }


def parse_response_node(state: AgentState) -> dict[str, Any]:
    """Parse the LLM response into an UpdatePlan.

    Args:
        state: Current agent state.

    Returns:
        State updates with parsed plan or error.
    """
    import logging

    from cognifold.models.plan import Operation, OperationType, UpdatePlan

    logger = logging.getLogger(__name__)

    raw_response = state.get("raw_response", "")
    if not raw_response:
        return {"error": "No response to parse"}

    # Strip common Gemini 3 preamble/thinking markers before extraction
    cleaned = _strip_gemini_preamble(raw_response)

    # Try to extract JSON from response
    json_str = _extract_json(cleaned) or _extract_json_balanced(cleaned)
    if not json_str:
        # Fallback: try the original unstripped text
        json_str = _extract_json(raw_response) or _extract_json_balanced(raw_response)
    if not json_str:
        logger.warning(
            "Could not extract JSON from LLM response (first 500 chars): %s",
            raw_response[:500],
        )
        return {
            "error": f"Could not extract JSON from response: {raw_response[:200]}",
            "parse_attempts": state["parse_attempts"] + 1,
        }

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        repaired = _extract_json_balanced(json_str) or _extract_json_balanced(raw_response)
        if repaired:
            try:
                data = json.loads(repaired)
            except json.JSONDecodeError as e2:
                return {
                    "error": f"Invalid JSON: {e2}",
                    "parse_attempts": state["parse_attempts"] + 1,
                }
        else:
            return {
                "error": "Invalid JSON: could not repair",
                "parse_attempts": state["parse_attempts"] + 1,
            }

    # Build UpdatePlan
    try:
        operations = []
        # Support both "operations" (legacy) and "plan" (new profiles) keys
        ops_list = data.get("operations") or data.get("plan") or []
        for op_data in ops_list:
            op_type = OperationType(op_data["op"])
            operations.append(
                Operation(
                    op=op_type,
                    node_type=op_data.get("node_type"),
                    node_id=op_data.get("node_id"),
                    data=op_data.get("data"),
                    source_id=op_data.get("source_id"),
                    target_id=op_data.get("target_id"),
                    node_ids=op_data.get("node_ids"),
                    merged_data=op_data.get("merged_data"),
                    # Explainability fields (Phase 5.5)
                    reasoning=op_data.get("reasoning"),
                    update_reasoning=op_data.get("update_reasoning"),
                    grounded_in=op_data.get("grounded_in"),
                    # Edge type fields (Phase 9.1)
                    edge_type=op_data.get("edge_type"),
                    weight=op_data.get("weight"),
                )
            )

        event_id = state["context"].event.event_id

        # Extract symbolic_actions if LLM provided them
        symbolic_actions = data.get("symbolic_actions") or []

        plan = UpdatePlan(
            plan_id=data.get("plan_id", f"plan-{event_id}"),
            trigger_event_id=event_id,
            reasoning=data.get("reasoning", ""),
            operations=operations,
            symbolic_actions=symbolic_actions,
        )

        return {"update_plan": plan, "error": None}

    except Exception as e:
        return {
            "error": f"Failed to build UpdatePlan: {e}",
            "parse_attempts": state["parse_attempts"] + 1,
        }


def should_continue(state: AgentState) -> Literal["execute_tools", "parse_response", "end"]:
    """Determine the next step based on current state.

    Args:
        state: Current agent state.

    Returns:
        Next node to execute.
    """
    # Check for errors
    if state.get("error"):
        return "end"

    # Check if we have a plan
    if state.get("update_plan"):
        return "end"

    # Check messages for tool calls
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        if last_message.get("tool_calls"):
            # Check exploration limit
            if state["exploration_steps"] < state["max_exploration_steps"]:
                return "execute_tools"
            # At limit, parse what we have
            return "parse_response"
        elif last_message.get("role") == "assistant":
            return "parse_response"

    return "end"


def should_continue_after_tools(
    state: AgentState,
) -> Literal["call_llm", "parse_response", "end"]:
    """Determine next step after tool execution.

    Args:
        state: Current agent state.

    Returns:
        Next node to execute.
    """
    if state.get("error"):
        return "end"

    if state["exploration_steps"] >= state["max_exploration_steps"]:
        return "parse_response"

    return "call_llm"


def _get_tool_definitions() -> list[dict[str, Any]]:
    """Get Gemini-compatible tool definitions (legacy SDK)."""
    return [
        {
            "function_declarations": [
                {
                    "name": "get_node",
                    "description": "Retrieve full details of a node by ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_id": {
                                "type": "string",
                                "description": "The ID of the node to retrieve",
                            }
                        },
                        "required": ["node_id"],
                    },
                },
                {
                    "name": "get_neighbors",
                    "description": "Get nodes connected to the specified node",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_id": {
                                "type": "string",
                                "description": "The node to find neighbors for",
                            },
                            "direction": {
                                "type": "string",
                                "enum": ["outgoing", "incoming", "both"],
                                "description": "Direction of edges to follow",
                            },
                        },
                        "required": ["node_id"],
                    },
                },
                {
                    "name": "find_nodes_by_type",
                    "description": "Find all nodes of a specific type (event, concept, or action)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node_type": {
                                "type": "string",
                                "enum": ["event", "concept", "action"],
                                "description": "The type of nodes to find",
                            }
                        },
                        "required": ["node_type"],
                    },
                },
                {
                    "name": "search_nodes",
                    "description": "Search nodes by keyword in title and data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword": {
                                "type": "string",
                                "description": "Search term to match",
                            }
                        },
                        "required": ["keyword"],
                    },
                },
                {
                    "name": "get_graph_stats",
                    "description": "Get overview statistics of the graph",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            ]
        }
    ]


def _get_tool_definitions_new() -> list[Any]:
    """Get tool definitions for new google.genai SDK."""
    from google.genai import types

    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="get_node",
                    description="Retrieve full details of a node by ID",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "node_id": types.Schema(
                                type=types.Type.STRING,
                                description="The ID of the node to retrieve",
                            ),
                        },
                        required=["node_id"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="get_neighbors",
                    description="Get nodes connected to the specified node",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "node_id": types.Schema(
                                type=types.Type.STRING,
                                description="The node to find neighbors for",
                            ),
                            "direction": types.Schema(
                                type=types.Type.STRING,
                                description="Direction of edges to follow (outgoing, incoming, or both)",
                            ),
                        },
                        required=["node_id"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="find_nodes_by_type",
                    description="Find all nodes of a specific type (event, concept, or action)",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "node_type": types.Schema(
                                type=types.Type.STRING,
                                description="The type of nodes to find (event, concept, or action)",
                            ),
                        },
                        required=["node_type"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="search_nodes",
                    description="Search nodes by keyword in title and data",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "keyword": types.Schema(
                                type=types.Type.STRING,
                                description="Search term to match",
                            ),
                        },
                        required=["keyword"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="get_graph_stats",
                    description="Get overview statistics of the graph",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={},
                    ),
                ),
            ]
        )
    ]


def _convert_to_gemini_format(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert internal message format to Gemini format.

    Args:
        messages: Internal format messages.

    Returns:
        Gemini-compatible messages.
    """
    gemini_messages = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            # Gemini doesn't have system role, prepend to first user message
            gemini_messages.append(
                {
                    "role": "user",
                    "parts": [{"text": f"[System Instructions]\n{content}"}],
                }
            )
        elif role == "user":
            gemini_messages.append(
                {
                    "role": "user",
                    "parts": [{"text": content}],
                }
            )
        elif role == "assistant":
            # If raw Gemini parts were preserved (with thought_signature),
            # reuse them directly to satisfy gemini-3 requirements.
            raw_parts = msg.get("_gemini_parts")
            if raw_parts:
                gemini_messages.append(
                    {
                        "role": "model",
                        "parts": raw_parts,
                    }
                )
            else:
                parts = []
                if content:
                    parts.append({"text": content})
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        parts.append(
                            {
                                "function_call": {
                                    "name": tc["name"],
                                    "args": tc["arguments"],
                                }
                            }
                        )
                gemini_messages.append(
                    {
                        "role": "model",
                        "parts": parts,
                    }
                )
        elif role == "tool":
            # Use the actual tool name from the tool_call_id context
            # so Gemini can match responses to calls.
            tool_name = msg.get("_tool_name", "tool_result")
            gemini_messages.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "name": tool_name,
                                "response": {"result": content},
                            }
                        }
                    ],
                }
            )

    return gemini_messages


def _strip_gemini_preamble(text: str) -> str:
    """Strip thinking/preamble markers that Gemini 3 may wrap around JSON."""
    # Remove <think>...</think> blocks (Gemini 3 thinking mode)
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.DOTALL)
    # Remove leading markdown prose before the first { or ```
    first_brace = text.find("{")
    first_fence = text.find("```")
    if first_brace == -1 and first_fence == -1:
        return text
    # Pick whichever comes first
    if first_fence == -1:
        start = first_brace
    elif first_brace == -1:
        start = first_fence
    else:
        start = min(first_brace, first_fence)
    return text[start:]


def _extract_json(text: str) -> str | None:
    """Extract JSON from LLM response text.

    Args:
        text: Raw LLM response.

    Returns:
        Extracted JSON string or None.
    """
    # Strip ALL code fences (handle multiple or nested fences)
    fence_open = re.search(r"```(?:json|JSON)?\s*\n?", text)
    if fence_open:
        inner = text[fence_open.end() :]
        fence_close = inner.rfind("```")
        text = inner[:fence_close] if fence_close != -1 else inner

    # Try to find raw JSON object (greedy — outermost braces)
    json_pattern = r"\{[\s\S]*\}"
    match = re.search(json_pattern, text)
    if match:
        return match.group(0)

    # Fallback: if JSON appears truncated (has opening { but no closing }),
    # try to repair by closing open braces/brackets
    first_brace = text.find("{")
    if first_brace != -1:
        fragment = text[first_brace:]
        # Count unclosed braces and brackets
        depth_brace = 0
        depth_bracket = 0
        in_string = False
        escape = False
        for ch in fragment:
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth_brace += 1
            elif ch == "}":
                depth_brace -= 1
            elif ch == "[":
                depth_bracket += 1
            elif ch == "]":
                depth_bracket -= 1
        if depth_brace > 0:
            # Truncated JSON — close brackets then braces
            repaired = fragment
            if in_string:
                repaired += '"'
            repaired += "]" * max(depth_bracket, 0)
            repaired += "}" * depth_brace
            return repaired

    return None


def _extract_json_balanced(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    in_string = False
    escape = False
    depth = 0

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def build_agent_graph() -> Any:
    """Build the LangGraph state graph.

    Returns:
        Compiled LangGraph StateGraph.
    """
    import importlib

    lg = importlib.import_module("langgraph.graph")
    END = lg.END  # noqa: N806 - constant from langgraph
    StateGraph = lg.StateGraph  # noqa: N806 - class from langgraph

    # Create the graph (AgentState imported at top of module)
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("call_llm", call_llm_node)
    workflow.add_node("execute_tools", execute_tools_node)
    workflow.add_node("parse_response", parse_response_node)

    # Set entry point
    workflow.set_entry_point("analyze")

    # Add edges
    workflow.add_edge("analyze", "call_llm")

    # Conditional edges after LLM call
    workflow.add_conditional_edges(
        "call_llm",
        should_continue,
        {
            "execute_tools": "execute_tools",
            "parse_response": "parse_response",
            "end": END,
        },
    )

    # After tool execution
    workflow.add_conditional_edges(
        "execute_tools",
        should_continue_after_tools,
        {
            "call_llm": "call_llm",
            "parse_response": "parse_response",
            "end": END,
        },
    )

    # Parse response goes to end
    workflow.add_edge("parse_response", END)

    return workflow.compile()
