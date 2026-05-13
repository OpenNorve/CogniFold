"""LoCoMo Benchmark Runner for Cognifold.

This script evaluates the Cognifold memory system using the LoCoMo
(Long-term Conversational Memory) benchmark dataset.
"""

import argparse
import dataclasses
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

# Add src and project root to python path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_project_root, "src"))
sys.path.append(_project_root)

try:
    from cognifold.agent.agent import CognifoldAgent
    from cognifold.agent.config import AgentConfig
    from cognifold.agent.prompt_profile import load_prompt_profiles
    from cognifold.executor.runner import PlanExecutor
    from cognifold.graph.store import ConceptGraph
    from cognifold.models.event import Event
    from cognifold.query.agent import MemoryQueryAgent
    from cognifold.query.models import QueryConfig, RetrievalMode
    from cognifold.replay.logger import GraphLogger
    from cognifold.replay.player import ReplayPlayer
    from cognifold.replay.renderer import ReplayRenderer
except ImportError as e:
    print(f"Error importing Cognifold modules: {e}")
    print("Please ensure you are running from project root or set PYTHONPATH.")
    sys.exit(1)

# Analysis utils for enriched wrong-case reporting
try:
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, _project_root)
    from benchmarks.analysis_utils import enrich_eval_result, save_wrong_cases
except ImportError:
    enrich_eval_result = None  # type: ignore[assignment]
    save_wrong_cases = None  # type: ignore[assignment]

from benchmarks.shared.base_runner import (
    _call_llm_text,
    generate_answer_with_llm as _generate_answer,
)

DATASET_FILE = "locomo10.json"


def _decompose_locomo_question(question: str, llm_model: str) -> list[str]:
    """Decompose chained-fact memory questions for retrieval (Mem0/EverMemOS style).

    For questions that depend on a chain of lookups (e.g. "When is Melanie's
    daughter's birthday?" needs "Who is Melanie's daughter?" then
    "When is [name]'s birthday?"), break into 1-3 atomic sub-questions
    so each can independently match its bridge fact via BM25/dense retrieval.

    Single-fact questions return [question] alone (1 LLM call overhead).
    """
    prompt = (
        "You are helping a memory-retrieval system answer questions about a long "
        "conversation between two friends. Some questions chain through MULTIPLE "
        "FACTS that must be looked up in sequence.\n\n"
        "RULES:\n"
        "- If the question depends on a chain (e.g. 'When is X's daughter's "
        "birthday?' needs 'Who is X's daughter?' THEN 'When is [name]'s birthday?'), "
        "decompose it into 2-3 atomic sub-questions, simplest first.\n"
        "- If the question is already a single-fact lookup ('Where does Melanie "
        "live?'), output ONLY the original question.\n"
        "- Always end with the original question on the last line.\n"
        "- No numbering, no explanation, no preface. Just the questions.\n\n"
        f"Question: {question}\n\n"
        "Sub-questions (1-3 lines, ending with the original):"
    )
    try:
        raw = _call_llm_text(
            model=llm_model, user_prompt=prompt, temperature=0.0, max_tokens=200
        )
        sub_qs = [line.strip().lstrip("-*0123456789. )") for line in raw.split("\n")]
        sub_qs = [s for s in sub_qs if s and "?" in s]
        if not sub_qs:
            return [question]
        # Ensure the original question is included (final lookup uses verbatim Q).
        if question not in sub_qs:
            sub_qs.append(question)
        return sub_qs[:3]
    except Exception:
        return [question]
OUTPUT_DIR = "benchmarks/locomo/output"
PROFILE_PATH = Path(__file__).parents[2] / "configs" / "locomo_profile.yaml"


def load_data(limit: Optional[int] = None) -> list[dict]:
    """Load the LoCoMo dataset."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, DATASET_FILE)

    if not os.path.exists(file_path):
        print(f"Dataset not found at {file_path}. Please run download_data.py first.")
        sys.exit(1)

    with open(file_path) as f:
        data = json.load(f)

    if limit:
        return data[:limit]
    return data


def parse_locomo_timestamp(ts_str: str) -> datetime:
    """Parse LoCoMo timestamp format."""
    # Example: "1:56 pm on 8 May, 2023"
    try:
        return datetime.strptime(ts_str, "%I:%M %p on %d %B, %Y")
    except ValueError:
        print(f"Warning: Could not parse timestamp '{ts_str}', using current time.")
        return datetime.now()


def check_api_keys() -> bool:
    """Check if API keys are set in the environment."""
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("\nERROR: No API keys found in environment variables.")
        print("Please set OPENAI_API_KEY or GOOGLE_API_KEY to run the benchmark.")
        print("Example: export OPENAI_API_KEY=sk-...")
        return False
    return True


def _resolve_temporal_references(graph: "ConceptGraph", conv_data: dict[str, Any]) -> None:
    """Resolve relative time references in graph nodes to absolute dates.

    Scans node descriptions for patterns like 'yesterday', 'last week', etc.
    and appends the computed absolute date based on the session timestamp.
    This helps QA retrieval find temporal facts.
    """
    # Build session_id → datetime mapping
    session_dates: dict[str, datetime] = {}
    for key in conv_data:
        if key.endswith("_date_time"):
            session_id = key.replace("_date_time", "")
            session_dates[session_id] = parse_locomo_timestamp(conv_data[key])

    relative_patterns = {
        "yesterday": timedelta(days=-1),
        "the day before": timedelta(days=-2),
        "last week": timedelta(weeks=-1),
        "last month": timedelta(days=-30),
        "last year": timedelta(days=-365),
        "two days ago": timedelta(days=-2),
        "a few days ago": timedelta(days=-3),
        "this morning": timedelta(hours=0),
        "last night": timedelta(days=-1),
        "the other day": timedelta(days=-2),
    }

    resolved_count = 0
    for node in graph.get_all_nodes():
        desc = str(node.data.get("description", "")).lower()
        title = str(node.data.get("title", "")).lower()
        text = f"{title} {desc}"

        # Find which session this node belongs to
        ctx = node.data.get("context", {})
        session_id = ctx.get("session_id", "") if isinstance(ctx, dict) else ""
        base_date = session_dates.get(session_id)
        if not base_date:
            continue

        for pattern, delta in relative_patterns.items():
            if pattern in text:
                resolved_date = base_date + delta
                date_str = resolved_date.strftime("%d %B, %Y")
                # Append resolved date to description
                current_desc = str(node.data.get("description", ""))
                if date_str not in current_desc:
                    node.data["description"] = f"{current_desc} [Resolved date: {date_str}]"
                    resolved_count += 1
                break

    if resolved_count:
        print(f"    Temporal resolution: {resolved_count} nodes enriched with dates")


def call_llm_for_eval(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call LLM for evaluation (OpenAI or Gemini)."""
    try:
        if model.startswith("gemini"):
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print("ERROR: GOOGLE_API_KEY not set for Gemini model")
                return ""
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            config = types.GenerateContentConfig(temperature=0.0, max_output_tokens=200)
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            text = getattr(response, "text", None)
            return text.strip() if isinstance(text, str) else ""

        # Default: OpenAI
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            print("ERROR: OPENAI_API_KEY not set")
            return ""
        from openai import OpenAI

        client = OpenAI()
        token_kwargs: dict[str, int] = {}
        if model.startswith("gpt-5"):
            token_kwargs["max_completion_tokens"] = 200
        else:
            token_kwargs["max_tokens"] = 200
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            **token_kwargs,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"LLM eval failed: {e}")
        return ""


def evaluate_with_llm(
    question: str,
    expected: Any,
    context: str,
    generated: str = "",
    model: str = "gpt-4o-mini",
) -> tuple[str, str]:
    """Evaluate answer using LLM (3-way: CORRECT/PARTIAL/INCORRECT).

    Returns:
        Tuple of (result: CORRECT/PARTIAL/INCORRECT, explanation)
    """
    # Handle adversarial/unanswerable questions
    if expected is None:
        no_info_signals = [
            "don't have this information",
            "not mentioned",
            "unknown",
            "no relevant context",
            "does not provide",
            "does not contain",
            "cannot be determined",
            "no information",
            "not available",
            "i'm not sure",
        ]
        context_lower = (str(context) + " " + str(generated)).lower()
        if any(signal in context_lower for signal in no_info_signals):
            return "CORRECT", "Adversarial question: correctly identified as unanswerable"
        return "INCORRECT", "Adversarial question: system should not have found an answer"

    prompt = f"""Evaluate if the context correctly answers the question.

Question: {question}
Expected Answer: {expected}
Retrieved Context: {context[:1500]}
Generated Answer: {generated if generated else "(using context directly)"}

Evaluation criteria:
1. Does the context contain the key information from the expected answer?
2. Would someone reading this context be able to answer the question correctly?

Reply with EXACTLY one of: CORRECT, PARTIAL, or INCORRECT
Then on a new line, provide a brief explanation (1 sentence).

Example responses:
CORRECT
The context mentions that User1 works at Google, matching the expected answer.

INCORRECT
The context does not contain any information about the user's workplace.
"""
    try:
        response = call_llm_for_eval(prompt, model=model)
        lines = response.strip().split("\n", 1)
        result = lines[0].strip().upper()
        explanation = lines[1].strip() if len(lines) > 1 else ""

        # Normalize result
        if "CORRECT" in result and "INCORRECT" not in result:
            return "CORRECT", explanation
        elif "PARTIAL" in result:
            return "PARTIAL", explanation
        else:
            return "INCORRECT", explanation
    except Exception as e:
        return "ERROR", str(e)


def evaluate_with_jscore(
    question: str,
    expected: Any,
    generated: str,
    model: str = "gpt-4o-mini",
) -> tuple[bool, str]:
    """Standard LoCoMo J-score evaluation (binary CORRECT/WRONG).

    Uses the same generous prompt as Backboard/Mem0/MAGMA evaluations.
    Adversarial questions (expected=None) handled by keyword detection.

    Returns:
        Tuple of (is_correct: bool, reasoning: str)
    """
    if expected is None:
        no_info_signals = [
            "don't have this information", "not mentioned", "unknown",
            "no relevant context", "does not provide", "cannot be determined",
            "no information", "not available", "i'm not sure",
        ]
        gen_lower = generated.lower()
        is_correct = any(s in gen_lower for s in no_info_signals)
        reason = "Adversarial: correctly declined" if is_correct else "Adversarial: hallucinated"
        return is_correct, reason

    prompt = f"""Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given:
(1) a question (posed by one user to another user),
(2) a 'gold' (ground truth) answer,
(3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer. The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might use relative time references (like "last Tuesday" or "next month"), but you should be generous - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Question: {question}
Gold answer: {expected}
Generated answer: {generated}

First provide a one-sentence reasoning, then finish with CORRECT or WRONG on a new line.
Do NOT include both CORRECT and WRONG in your response."""

    try:
        response = call_llm_for_eval(prompt, model=model)
        upper = response.upper()
        # Find last occurrence of CORRECT or WRONG
        correct_pos = upper.rfind("CORRECT")
        wrong_pos = upper.rfind("WRONG")
        if correct_pos > wrong_pos:
            return True, response.strip()
        return False, response.strip()
    except Exception as e:
        return False, f"Judge error: {e}"


def run_benchmark(
    limit_conversations: Optional[int] = None,
    limit_sessions: Optional[int] = None,
    visualize: bool = False,
    disable_concepts: bool = False,
    query_mode: str = "mergefold",
    use_llm_eval: bool = True,
    use_profile: bool = True,
    embedding: Optional[str] = None,
    event_stream: bool = False,
    model: Optional[str] = None,
    judge_model: Optional[str] = None,
) -> None:
    """Run the LoCoMo benchmark.

    Args:
        model: Agent backbone model (e.g. "openai:gpt-4o-mini" or
            "openai:gpt-4.1-mini"). Takes precedence over profile config.
            Examples: Mem0 / MemOS baselines use gpt-4o-mini;
            EverMemOS headline uses gpt-4.1-mini.
        judge_model: LLM-as-judge model (e.g. "gpt-4o", "gpt-4o-mini").
            Decoupled from the agent so you can run a stronger judge
            (gpt-4o) with a different agent backbone. When None, falls
            back to the same model as ``model`` — preserves the historical
            behavior of sharing one model for agent and judge.
    """
    if not check_api_keys():
        return

    # Resolve embedding config: CLI --embedding overrides profile YAML
    from benchmarks._utils import create_embedder, resolve_embedding

    resolved_embedding = resolve_embedding(embedding, PROFILE_PATH, "locomo")
    embedder, retrieval_mode = create_embedder(resolved_embedding)
    if embedder:
        print(f"Using embedding: {resolved_embedding}")
    else:
        print("Using retrieval: BM25 (no embedding)")

    data = load_data(limit_conversations)
    print(f"Loaded {len(data)} conversations.")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load prompt profile if available
    prompt_profile = None
    profile_templates: dict[str, str] = {}
    llm_model = "openai:gpt-4o-mini" if os.environ.get("OPENAI_API_KEY") else "gemini-2.5-flash"
    if use_profile and PROFILE_PATH.exists():
        try:
            profiles = load_prompt_profiles(PROFILE_PATH)
            prompt_profile = profiles.get("locomo")
            if prompt_profile:
                print(f"Using profile: locomo from {PROFILE_PATH}")
            import yaml

            with open(PROFILE_PATH) as _pf:
                _raw = yaml.safe_load(_pf)
            _bench_raw = _raw.get("profiles", {}).get("locomo", {})
            profile_templates = _bench_raw.get("templates", {})
            _raw_model = _bench_raw.get("model", {}).get("name", "")
            if _raw_model:
                llm_model = _raw_model
        except Exception as e:
            print(f"Warning: Could not load profile: {e}")

    # CLI --model overrides both profile and default
    if model:
        llm_model = model
        print(f"Agent model override: {llm_model}")

    # Judge model: explicit --judge-model wins; else derive from agent model
    # (historical behavior of sharing one model for agent + judge).
    if judge_model:
        judge_model_name = (
            judge_model.split(":", 1)[-1] if ":" in judge_model else judge_model
        )
        print(f"Judge model override: {judge_model_name}")
    else:
        judge_model_name = (
            llm_model.split(":", 1)[-1] if ":" in llm_model else llm_model
        )

    results = []
    all_qa_results = []

    for i, conversation_sample in enumerate(data):
        sample_id = conversation_sample.get("sample_id", "unknown")
        print(f"\nProcessing Conversation {i + 1}/{len(data)} (ID: {sample_id})")

        # Initialize fresh graph
        graph = ConceptGraph()

        # Configure agent with domain-specific settings
        if prompt_profile:
            config = prompt_profile.to_agent_config()
            if disable_concepts:
                config = dataclasses.replace(config, disable_concepts=True)
            if model:
                config = dataclasses.replace(config, model_name=model)
            agent = CognifoldAgent(config=config, prompt_profile=prompt_profile)
        else:
            config = AgentConfig(
                model_name=model or "openai:gpt-4o-mini", temperature=0.0
            )
            if disable_concepts:
                config = dataclasses.replace(config, disable_concepts=True)
            agent = CognifoldAgent(config=config)

        # Initialize executor
        executor = PlanExecutor(graph)

        # Initialize query agent with domain config
        query_config = QueryConfig(
            domain="locomo",
            speaker_aware=True,
            max_nodes=20,
            include_reasoning=True,
            retrieval_mode=retrieval_mode,
        )
        query_agent = MemoryQueryAgent(graph, config=query_config, embedder=embedder)

        # Initialize logger if visualizing
        graph_logger = None
        log_path_str: str | None = None
        if visualize:
            log_path_str = os.path.join(OUTPUT_DIR, f"replay_{sample_id}.jsonl")
            graph_logger = GraphLogger(log_path=Path(log_path_str))
            graph_logger.log_run_start(
                timeline_path=f"locomo_{sample_id}",
                total_events=0,
                config={"limit_sessions": limit_sessions, "disable_concepts": disable_concepts},
            )

        conv_data = conversation_sample.get("conversation", {})

        # Extract and sort sessions
        session_keys = [k for k in conv_data if re.match(r"^session_\d+$", k)]
        session_keys.sort(key=lambda x: int(x.split("_")[1]))

        if limit_sessions:
            session_keys = session_keys[:limit_sessions]

        print(f"  Ingesting {len(session_keys)} sessions...")

        total_turns = 0
        step = 1
        turn_batch_size = 4  # Small batches: finer-grained fact extraction per dialogue segment

        for session_idx, session_key in enumerate(session_keys):
            turns = conv_data[session_key]
            ts_key = f"{session_key}_date_time"
            session_ts_str = conv_data.get(ts_key, "")
            base_timestamp = parse_locomo_timestamp(session_ts_str)

            # Turn-level batching: split session into small chunks for
            # more granular concept extraction and better retrieval
            for batch_start in range(0, len(turns), turn_batch_size):
                batch = turns[batch_start : batch_start + turn_batch_size]
                batch_offset = timedelta(minutes=batch_start)

                # Build batch text with speaker attribution and turn IDs
                batch_lines = []
                speakers_in_batch = set()
                dia_ids = []
                for t in batch:
                    speaker = t.get("speaker", "Unknown")
                    text = t.get("text", "")
                    dia_id = t.get("dia_id", "")
                    speakers_in_batch.add(speaker)
                    dia_ids.append(dia_id)
                    batch_lines.append(f"{speaker}: {text}")

                batch_text = "\n".join(batch_lines)
                speakers_str = " & ".join(sorted(speakers_in_batch))
                title = f"{session_key} turns {batch_start + 1}-{batch_start + len(batch)} ({speakers_str})"

                # Pre-resolve relative dates in batch text so both raw event nodes
                # and LLM-extracted concepts contain absolute dates.
                # "yesterday" → "yesterday (2022-11-14)", "last week" → "last week (week of 2022-10-25)"
                event_date = base_timestamp + batch_offset
                date_str = event_date.strftime("%Y-%m-%d")
                for rel, resolved in [
                    ("yesterday", f"yesterday ({(event_date - timedelta(days=1)).strftime('%Y-%m-%d')})"),
                    ("today", f"today ({date_str})"),
                    ("last week", f"last week (week of {(event_date - timedelta(weeks=1)).strftime('%Y-%m-%d')})"),
                    ("last weekend", f"last weekend ({(event_date - timedelta(days=event_date.weekday() + 2)).strftime('%Y-%m-%d')})"),
                    ("this morning", f"this morning ({date_str})"),
                    ("tonight", f"tonight ({date_str})"),
                    ("last night", f"last night ({(event_date - timedelta(days=1)).strftime('%Y-%m-%d')})"),
                ]:
                    if rel in batch_text.lower():
                        batch_text = re.sub(
                            re.escape(rel), resolved, batch_text, flags=re.IGNORECASE
                        )

                # Prepend session date header so LLM and retrieval always see the date
                batch_text = f"[Date: {session_ts_str or date_str}]\n{batch_text}"

                event = Event(
                    event_id=str(uuid.uuid4()),
                    timestamp=base_timestamp + batch_offset,
                    source="locomo-benchmark",
                    event_type="conversation.turns",
                    title=title,
                    description=batch_text,
                    context={
                        "session_id": session_key,
                        "session_date": session_ts_str,
                        "turn_count": len(batch),
                        "speakers": list(speakers_in_batch),
                        "dia_ids": dia_ids,
                        "batch_start": batch_start,
                    },
                )

                if graph_logger:
                    graph_logger.log_event_start(
                        step=step,
                        event_id=event.event_id,
                        event_type=event.event_type,
                        title=event.title,
                        timestamp=event.timestamp.isoformat(),
                        event_data=event.model_dump(mode="json"),
                    )

                # Always add raw event node first — ensures graph is never empty
                # even if LLM processing fails (API quota, rate limit, etc.)
                from cognifold.models.node import Node, NodeType

                raw_node = Node(
                    id=event.event_id,
                    type=NodeType.EVENT,
                    data={
                        "title": title,
                        "description": batch_text,
                        "source": "locomo-benchmark",
                        "session_id": session_key,
                        "session_date": session_ts_str,
                        "speakers": list(speakers_in_batch),
                    },
                )
                try:
                    graph.add_node(raw_node)
                except Exception:
                    pass  # Node may already exist

                try:
                    # Retrieve context before processing
                    retrieval = query_agent.query_semantic(batch_text[:200])
                    context_node_ids = [n.node_id for n in retrieval.nodes[:10]]

                    # Process event with context — LLM enriches the raw node
                    plan = agent.process_event(
                        event=event,
                        graph=graph,
                        context_node_ids=context_node_ids,
                        node_scores={},
                    )

                    # Execute plan
                    executor.execute(plan)

                    if graph_logger and plan.operations:
                        for op in plan.operations:
                            graph_logger.log_operation(
                                step=step,
                                op_type=op.op.value,
                                op_data=op.model_dump(mode="json"),
                                success=True,
                            )

                    if graph_logger:
                        graph_logger.log_event_end(
                            step=step,
                            event_id=event.event_id,
                            operations_count=len(plan.operations),
                            reasoning=plan.reasoning,
                        )

                    time.sleep(0.3)  # Rate limit protection

                except Exception as e:
                    print(f"    Error processing {session_key} batch {batch_start}: {e}")
                    if "429" in str(e):
                        print("    Rate limit hit, sleeping for 10s...")
                        time.sleep(10)

                step += 1

            total_turns += len(turns)
            if total_turns % 10 == 0:
                print(f"    Processed {total_turns} turns...", end="\r")

            # Event-stream protocol: inter-session consolidation
            # Between sessions, run concept merging and edge decay to simulate
            # the passage of real time. This tests whether the graph's evolution
            # mechanisms (merge, decay, reconnect) remain effective over time.
            if event_stream and session_idx < len(session_keys) - 1:
                try:
                    from cognifold.graph.consolidation import (
                        merge_similar_concepts,
                        prune_orphan_concepts,
                    )

                    # Ablation hook: COGNIFOLD_ABLATE_MERGE=1 skips merges.
                    # Used by ablation benchmark runs to isolate MERGE_NODES'
                    # causal contribution; production runs leave it unset.
                    ablate_merge = os.environ.get("COGNIFOLD_ABLATE_MERGE") == "1"
                    merges = 0 if ablate_merge else merge_similar_concepts(graph)
                    orphans = prune_orphan_concepts(graph)
                    if merges or orphans or ablate_merge:
                        tag = " [ABLATED]" if ablate_merge else ""
                        print(
                            f"    Inter-session consolidation{tag}: {merges} merges, {orphans} orphans"
                        )
                except Exception:
                    pass

        print(f"    Processed {total_turns} turns total. Graph: {graph.node_count} nodes")

        # Post-ingestion: temporal resolution, fact extraction, entity indexing
        try:
            # Resolve relative time references in node descriptions
            _resolve_temporal_references(graph, conv_data)
        except Exception as e:
            print(f"    Warning: temporal resolution failed: {e}")

        try:
            from cognifold.graph.entity_index import EntityIndex
            from cognifold.graph.fact_extraction import extract_facts

            fact_ids = extract_facts(graph)
            if fact_ids:
                print(f"    Fact extraction: {len(fact_ids)} fact nodes created")

            entity_idx = EntityIndex()
            entity_idx.build(graph)
            graph.entity_index = entity_idx
            print(f"    Entity index: {entity_idx.entity_count} entities indexed")
        except Exception as e:
            print(f"    Warning: fact extraction / entity indexing failed: {e}")

        if graph_logger and log_path_str:
            graph_logger.log_run_end(
                total_steps=step - 1,
                node_count=graph.node_count,
                edge_count=graph.edge_count,
            )
            graph_logger.close()

            # Generate visualization
            print(f"  Generating replay visualization for {sample_id}...")
            player = ReplayPlayer.from_log(Path(log_path_str))
            renderer = ReplayRenderer()
            html_output_path = os.path.join(OUTPUT_DIR, f"{sample_id}_replay.html")
            renderer.render(
                player=player,
                output_path=Path(html_output_path),
                title=f"Cognifold Replay: {sample_id}",
            )
            print(f"  Replay saved to {html_output_path}")

        # Run QA Evaluation
        print("  Running QA Evaluation...")
        qa_list = conversation_sample.get("qa", [])

        metrics = {"correct": 0, "partial": 0, "incorrect": 0, "error": 0}
        j_metrics = {"correct": 0, "total": 0}  # Standard J-score (binary, cats 1-4)
        j_by_cat: dict[int, dict[str, int]] = {
            1: {"correct": 0, "total": 0}, 2: {"correct": 0, "total": 0},
            3: {"correct": 0, "total": 0}, 4: {"correct": 0, "total": 0},
        }
        sample_qa_results = []

        for qa_item in qa_list:
            question = qa_item.get("question")
            answer = qa_item.get("answer")
            category = qa_item.get("category", "unknown")

            if not question:
                continue

            try:
                # Use domain-specific query
                _query_start = time.time()

                # Iter3: Mem0/EverMemOS-style question decomposition.
                # For chained-fact questions ("When is Melanie's daughter's
                # birthday?"), the bridge fact (daughter's name) doesn't
                # match the question text directly via BM25/dense. Decompose
                # into atomic sub-questions and retrieve for each, merge
                # unique nodes. Single-fact questions return [question],
                # so the overhead is one extra LLM call.
                sub_qs = _decompose_locomo_question(question, llm_model)
                # The final entry is always the original question; use its
                # result as the base.
                base_q = sub_qs[-1]
                result = query_agent.query_for_qa(
                    question=base_q,
                    domain="locomo",
                    query_mode=query_mode,
                )
                seen_node_ids = {n.node_id for n in result.nodes}
                extra_context_parts: list[str] = []
                for sq in sub_qs[:-1]:
                    try:
                        sq_result = query_agent.query_for_qa(
                            question=sq,
                            domain="locomo",
                            query_mode=query_mode,
                        )
                    except Exception:
                        continue
                    fresh_summaries = []
                    for node in sq_result.nodes:
                        if node.node_id in seen_node_ids:
                            continue
                        seen_node_ids.add(node.node_id)
                        title = node.title if hasattr(node, "title") else ""
                        desc = ""
                        if hasattr(node, "data") and isinstance(node.data, dict):
                            desc = node.data.get("description", "") or ""
                        elif hasattr(node, "description"):
                            desc = node.description or ""
                        if title or desc:
                            fresh_summaries.append(
                                f"- **{title}**: {desc[:600]}" if title else f"- {desc[:600]}"
                            )
                    if fresh_summaries:
                        extra_context_parts.append(
                            f"\n[Sub-query: {sq}]\n" + "\n".join(fresh_summaries[:8])
                        )
                context = result.context
                if extra_context_parts:
                    context = (
                        context
                        + "\n\n--- Decomposed sub-query evidence ---"
                        + "".join(extra_context_parts)
                    )

                # Generate answer from context before evaluation
                generated_answer = ""
                if use_llm_eval:
                    generated_answer = _generate_answer(
                        question=question,
                        context=context,
                        profile_templates=profile_templates,
                        model=llm_model,
                        max_tokens=100,
                    )

                # String-match pre-check: if gold answer appears in generated
                # answer (case-insensitive), it's correct regardless of LLM eval.
                # Includes date normalization to handle format differences
                # (e.g., "10 July 2023" vs "July 10, 2023").
                answer_str = str(answer).lower().strip() if answer is not None else ""
                gen_lower = generated_answer.lower()
                ctx_lower = context.lower()

                def _normalize_for_match(text: str) -> str:
                    """Normalize text for lenient matching: dates, punctuation, numbers."""
                    import re as _re
                    t = text.lower().strip()
                    # Remove punctuation except hyphens
                    t = _re.sub(r"[,\.!?;:\"'()']", " ", t)
                    # Normalize date formats: "July 10 2023" / "10 July 2023" → canonical
                    months = {
                        "january": "01", "february": "02", "march": "03", "april": "04",
                        "may": "05", "june": "06", "july": "07", "august": "08",
                        "september": "09", "october": "10", "november": "11", "december": "12",
                    }
                    for mname, mnum in months.items():
                        # "July 10 2023" → "2023-07-10"
                        t = _re.sub(
                            rf"{mname}\s+(\d{{1,2}})\s+(\d{{4}})",
                            lambda m: f"{m.group(2)}-{mnum}-{m.group(1).zfill(2)}",
                            t,
                        )
                        # "10 July 2023" → "2023-07-10"
                        t = _re.sub(
                            rf"(\d{{1,2}})\s+{mname}\s+(\d{{4}})",
                            lambda m: f"{m.group(2)}-{mnum}-{m.group(1).zfill(2)}",
                            t,
                        )
                        # "13 August" (no year) → "XX-08-13"
                        t = _re.sub(
                            rf"(\d{{1,2}})\s+{mname}\b",
                            lambda m: f"XX-{mnum}-{m.group(1).zfill(2)}",
                            t,
                        )
                        # "August 13" (no year) → "XX-08-13"
                        t = _re.sub(
                            rf"{mname}\s+(\d{{1,2}})\b",
                            lambda m: f"XX-{mnum}-{m.group(1).zfill(2)}",
                            t,
                        )
                    # Number words
                    for word, digit in [("ten", "10"), ("five", "5"), ("four", "4"), ("three", "3"), ("two", "2")]:
                        t = _re.sub(rf"\b{word}\b", digit, t)
                    t = _re.sub(r"\s+", " ", t).strip()
                    return t

                norm_answer = _normalize_for_match(answer_str)
                norm_gen = _normalize_for_match(gen_lower)
                norm_ctx = _normalize_for_match(ctx_lower)

                string_match = (
                    norm_answer
                    and len(norm_answer) > 1
                    and (norm_answer in norm_gen or norm_answer in norm_ctx)
                )

                if string_match:
                    eval_result = "CORRECT"
                    explanation = f"String match: '{answer_str}' found in response"
                elif answer is None:
                    # Adversarial question — delegate to LLM eval
                    if use_llm_eval:
                        eval_result, explanation = evaluate_with_llm(
                            question=question,
                            expected=answer,
                            context=context,
                            generated=generated_answer,
                            model=judge_model_name,
                        )
                    else:
                        eval_result = "INCORRECT"
                        explanation = "No LLM eval for adversarial"
                elif use_llm_eval:
                    # LLM-based evaluation for non-trivial cases
                    eval_result, explanation = evaluate_with_llm(
                        question=question,
                        expected=answer,
                        context=context,
                        generated=generated_answer,
                        model=judge_model_name,
                    )
                else:
                    eval_result = "INCORRECT"
                    explanation = "No string match"

                metrics[eval_result.lower()] = metrics.get(eval_result.lower(), 0) + 1

                # Standard J-score evaluation (binary, generous judge) — Mem0 protocol.
                j_correct = False
                j_reasoning = ""
                cat_int = int(category) if str(category).isdigit() else 0
                if cat_int in (1, 2, 3, 4) and use_llm_eval and generated_answer:
                    j_correct, j_reasoning = evaluate_with_jscore(
                        question=question,
                        expected=answer,
                        generated=generated_answer,
                        model=judge_model_name,
                    )
                    j_metrics["total"] += 1
                    if j_correct:
                        j_metrics["correct"] += 1
                    if cat_int in j_by_cat:
                        j_by_cat[cat_int]["total"] += 1
                        if j_correct:
                            j_by_cat[cat_int]["correct"] += 1

                qa_entry = {
                    "question": question,
                    "expected": answer,
                    "generated": generated_answer,
                    "category": category,
                    "result": eval_result,
                    "verdict": eval_result,
                    "explanation": explanation,
                    "context_length": len(context),
                    "j_correct": j_correct,
                    "j_reasoning": j_reasoning[:200],
                }
                if enrich_eval_result is not None:
                    enrich_eval_result(
                        qa_entry,
                        graph=graph,
                        query_result=result,
                        retrieval_mode=query_mode,
                        query_start_time=_query_start,
                    )
                sample_qa_results.append(qa_entry)

                # Log first few results
                if len(sample_qa_results) <= 3:
                    print(f"    Q: {question}")
                    print(f"    A: {answer}")
                    print(f"    Result: {eval_result} - {explanation}")

            except Exception as e:
                metrics["error"] += 1
                print(f"    Error querying '{question}': {e}")

        total_eval = metrics["correct"] + metrics["partial"] + metrics["incorrect"]
        score = (metrics["correct"] / total_eval * 100) if total_eval > 0 else 0
        partial_score = (
            ((metrics["correct"] + 0.5 * metrics["partial"]) / total_eval * 100)
            if total_eval > 0
            else 0
        )

        j_score_pct = (j_metrics["correct"] / j_metrics["total"] * 100) if j_metrics["total"] > 0 else 0

        print(f"  Score: {score:.1f}% strict, {partial_score:.1f}% partial")
        print(f"  J-Score (standard, cats 1-4): {j_score_pct:.1f}% ({j_metrics['correct']}/{j_metrics['total']})")
        for cat_id in sorted(j_by_cat.keys()):
            cat_data = j_by_cat[cat_id]
            cat_names = {1: "Multi-hop", 2: "Temporal", 3: "Open-domain", 4: "Single-hop"}
            if cat_data["total"] > 0:
                cat_pct = cat_data["correct"] / cat_data["total"] * 100
                print(f"    Cat {cat_id} ({cat_names.get(cat_id, '?')}): {cat_pct:.1f}% ({cat_data['correct']}/{cat_data['total']})")
        print(f"  Breakdown: {metrics}")

        results.append(
            {
                "sample_id": sample_id,
                "score_strict": score,
                "score_partial": partial_score,
                "j_score": j_score_pct,
                "j_by_category": {k: v["correct"] / v["total"] * 100 if v["total"] else 0 for k, v in j_by_cat.items()},
                "metrics": metrics,
                "total_turns": total_turns,
                "graph_nodes": graph.node_count,
            }
        )

        all_qa_results.extend(sample_qa_results)

    # Save detailed results
    results_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "summary": results,
                "qa_details": all_qa_results,
                "config": {
                    "query_mode": query_mode,
                    "use_llm_eval": use_llm_eval,
                    "disable_concepts": disable_concepts,
                },
            },
            f,
            indent=2,
        )
    print(f"\nDetailed results saved to {results_path}")

    if save_wrong_cases is not None:
        save_wrong_cases(all_qa_results, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    for res in results:
        print(
            f"  {res['sample_id']}: {res['score_strict']:.1f}% (partial: {res['score_partial']:.1f}%)"
        )

    if results:
        avg_strict = sum(r["score_strict"] for r in results) / len(results)
        avg_partial = sum(r["score_partial"] for r in results) / len(results)
        print(f"\n  Average: {avg_strict:.1f}% strict, {avg_partial:.1f}% partial")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LoCoMo Benchmark on Cognifold")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of conversations (default: None = all 10). "
        "Pass --limit 1 for quick smoke; omit for full benchmark.",
    )
    parser.add_argument(
        "--sessions", type=int, default=None, help="Limit sessions per conversation"
    )
    parser.add_argument("--visualize", action="store_true", help="Generate replay visualization")
    parser.add_argument(
        "--disable-concepts",
        action="store_true",
        help="Disable concept formation (Episodic mode)",
    )
    parser.add_argument(
        "--query-mode",
        type=str,
        default="mergefold",
        help="Query mode (base, rag, episodic, mergefold)",
    )
    parser.add_argument(
        "--no-llm-eval",
        action="store_true",
        help="Use simple string matching instead of LLM evaluation",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Don't use the locomo prompt profile",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default=None,
        help="Embedding model (e.g. openai:text-embedding-3-small, gemini:text-embedding-004, or none). Overrides profile config.",
    )
    parser.add_argument(
        "--event-stream",
        action="store_true",
        help="Event-stream protocol: run inter-session consolidation (merge, decay) between sessions to simulate real-time passage.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help='Agent backbone override, e.g. "openai:gpt-4o-mini" or '
        '"openai:gpt-4.1-mini". Takes precedence over the locomo profile. '
        "Mem0/MemOS baselines use gpt-4o-mini; EverMemOS uses gpt-4.1-mini.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help='LLM-as-judge override, e.g. "gpt-4o" or "gpt-4o-mini". '
        "Decoupled from --model so you can run a stronger judge against "
        "a smaller agent. When omitted, uses --model's value.",
    )
    args = parser.parse_args()

    run_benchmark(
        limit_conversations=args.limit,
        limit_sessions=args.sessions,
        visualize=args.visualize,
        disable_concepts=args.disable_concepts,
        query_mode=args.query_mode,
        use_llm_eval=not args.no_llm_eval,
        use_profile=not args.no_profile,
        embedding=args.embedding,
        event_stream=args.event_stream,
        model=args.model,
        judge_model=args.judge_model,
    )
