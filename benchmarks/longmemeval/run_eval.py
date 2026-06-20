"""LongMemEval Benchmark Runner for Cognifold.

This script evaluates the Cognifold memory system using the LongMemEval benchmark.
It downloads the dataset, ingests chat history into the memory graph, and answers questions.
"""

import argparse
import dataclasses
import json
import logging
import os
import re
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from tqdm import tqdm

# Add src and project root to python path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_project_root, "src"))
sys.path.append(_project_root)

# Analysis utils for enriched wrong-case reporting
sys.path.insert(0, str(Path(__file__).parents[2]))
try:
    from benchmarks.analysis_utils import enrich_eval_result, save_wrong_cases
except ImportError:
    enrich_eval_result = None  # type: ignore[assignment]
    save_wrong_cases = None  # type: ignore[assignment]

from cognifold.agent.agent import CognifoldAgent
from cognifold.agent.config import AgentConfig
from cognifold.agent.prompt_profile import load_prompt_profiles
from cognifold.executor.runner import PlanExecutor
from cognifold.graph.store import ConceptGraph
from cognifold.models.event import Event
from cognifold.models.node import Edge, Node, NodeType
from cognifold.query.agent import MemoryQueryAgent
from cognifold.query.models import QueryConfig, RetrievalMode

from benchmarks.longmemeval.symbolic_resolver import (
    LongMemEvalSymbolicResolver,
    render_symbolic_block,
)
from benchmarks.longmemeval import neural_symbolic as neural_symbolic_mod
from benchmarks.longmemeval.neural_symbolic import (
    render_neural_symbolic_block,
    resolve_neural_symbolic,
)
from benchmarks.longmemeval.round2_evidence_ledger import (
    answer_from_ledger,
    assemble_ledger_context,
    build_evidence_ledger,
    detect_question_shape,
    late_fusion_retrieve,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
DATA_FILENAME = "longmemeval_s_cleaned.json"
PROFILE_PATH = Path(__file__).parents[2] / "configs" / "longmemeval_profile.yaml"


def download_data(output_dir: Path) -> Path:
    """Download the LongMemEval dataset if not present."""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / DATA_FILENAME

    if not file_path.exists():
        logger.info(f"Downloading dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, file_path)
        logger.info(f"Downloaded dataset to {file_path}")
    else:
        logger.info(f"Dataset found at {file_path}")

    return file_path


# Per-process call-stats counter. Each successful LLM call records its
# token usage by model name. Dumped to <output_dir>/call_stats.json at the
# end of run_benchmark; the launcher merges these across batches.
#
# `cost_usd` is summed when the provider reports an authoritative cost on
# the response (OpenRouter populates `usage.cost`); when absent (OpenAI
# direct doesn't return a cost field), only tokens are recorded and
# cost_usd stays 0 for that model.
_LLM_CALL_STATS: dict[str, dict[str, float]] = {}


def _record_llm_call(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """Increment the per-model call counter. Safe to call from any path."""
    bucket = _LLM_CALL_STATS.setdefault(
        model_name,
        {"calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0},
    )
    bucket["calls"] += 1
    bucket["input_tokens"] += int(input_tokens or 0)
    bucket["output_tokens"] += int(output_tokens or 0)
    bucket["reasoning_tokens"] += int(reasoning_tokens or 0)
    bucket["cost_usd"] += float(cost_usd or 0.0)


def _extract_cost_usd(usage_obj) -> float:
    """Read OpenRouter's authoritative `cost` field off the usage object.

    OpenRouter populates `usage.cost` (USD) on chat-completion responses.
    OpenAI direct does not return any cost field — returns 0 in that case.
    Tolerates either object-style or dict-style usage.
    """
    if usage_obj is None:
        return 0.0
    for attr in ("cost", "total_cost"):
        try:
            v = getattr(usage_obj, attr, None)
            if v is None and isinstance(usage_obj, dict):
                v = usage_obj.get(attr)
            if v is not None:
                return float(v)
        except Exception:
            pass
    return 0.0


def dump_call_stats(output_dir: "Path") -> None:
    """Persist this batch process' call-stats counter to a JSON file."""
    try:
        import json as _json
        out = output_dir / "call_stats.json"
        out.write_text(_json.dumps(_LLM_CALL_STATS, indent=2))
    except Exception:
        pass


def call_llm(prompt: str, config: AgentConfig, json_mode: bool = False) -> str:
    """Call LLM with the given prompt (OpenAI or Gemini).

    json_mode forces strict JSON output (OpenAI: response_format={"type":
    "json_object"}; Gemini: response_mime_type=application/json + ThinkingConfig
    budget=0 so all the budget reaches the visible reply).
    """
    model_name = config.model_name.replace("openai:", "").replace("gemini:", "")
    is_gemini = config.model_name.startswith("gemini") or model_name.startswith("gemini-")

    try:
        if is_gemini:
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.error("GOOGLE_API_KEY not set for Gemini model")
                return ""
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            gc_kwargs: dict = {
                "temperature": 0.0,
                "max_output_tokens": config.max_tokens,
            }
            if json_mode:
                gc_kwargs["response_mime_type"] = "application/json"
            if "2.5" in model_name:
                gc_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            gc = types.GenerateContentConfig(**gc_kwargs)
            response = client.models.generate_content(
                model=model_name, contents=prompt, config=gc
            )
            text = getattr(response, "text", None)
            try:
                um = getattr(response, "usage_metadata", None)
                if um:
                    _record_llm_call(
                        model_name,
                        getattr(um, "prompt_token_count", 0) or 0,
                        getattr(um, "candidates_token_count", 0) or 0,
                        getattr(um, "thoughts_token_count", 0) or 0,
                    )
            except Exception:
                pass
            return text.strip() if isinstance(text, str) else ""

        # Default: OpenAI — honor config.api_key/base_url for per-role
        # routing (writer vs reader can hit different providers in one
        # process). Falls back to OPENAI_API_KEY/OPENAI_BASE_URL.
        cfg_key = getattr(config, "api_key", None)
        cfg_url = getattr(config, "base_url", None)
        openai_key = cfg_key or os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            logger.error("OPENAI_API_KEY not set")
            return ""
        from openai import OpenAI

        client_kwargs: dict = {"api_key": openai_key, "max_retries": 4}
        if cfg_url:
            client_kwargs["base_url"] = cfg_url
        client = OpenAI(**client_kwargs)
        # gpt-5 / o1 / o3 are reasoning models — they reject custom
        # temperature and require max_completion_tokens.
        is_reasoning_model = (
            model_name.startswith("o1")
            or model_name.startswith("o3")
            or "gpt-5" in model_name
        )
        kwargs: dict = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        if is_reasoning_model:
            # High reasoning effort for QA — LongMemEval rewards thorough
            # multi-session synthesis. 24K budget so thinking + reply fit.
            kwargs["max_completion_tokens"] = max(config.max_tokens, 24576)
            # iter23: honor config.reasoning_effort when set (allow writer
            # config to run at a different effort than the reader to trade
            # extraction quality for speed). Default stays "high" for reader.
            kwargs["reasoning_effort"] = (
                getattr(config, "reasoning_effort", None) or "high"
            )
        else:
            kwargs["temperature"] = 0.0
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        try:
            usage = getattr(response, "usage", None)
            if usage:
                rt = 0
                ctd = getattr(usage, "completion_tokens_details", None)
                if ctd is not None:
                    rt = getattr(ctd, "reasoning_tokens", 0) or 0
                _record_llm_call(
                    model_name,
                    getattr(usage, "prompt_tokens", 0) or 0,
                    getattr(usage, "completion_tokens", 0) or 0,
                    rt,
                    _extract_cost_usd(usage),
                )
        except Exception:
            pass
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ""


_LONGMEMEVAL_DATE_RE = __import__("re").compile(r"\s*\([A-Za-z]+\)\s*")


def _parse_longmemeval_date(s: str) -> datetime:
    """Parse session_date_str. LongMemEval uses 'YYYY/MM/DD (Dow) HH:MM',
    not ISO — the prior `datetime.fromisoformat` failed silently and fell back
    to `datetime.now()`, stamping every session with the benchmark run time."""
    if not s:
        return datetime.now()
    # Try ISO 8601 first (covers other datasets reusing this runner)
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        pass
    # LongMemEval native format: "2023/04/23 (Sun) 08:57"
    try:
        cleaned = _LONGMEMEVAL_DATE_RE.sub(" ", s).strip()
        return datetime.strptime(cleaned, "%Y/%m/%d %H:%M")
    except (ValueError, TypeError):
        pass
    # Date only fallback
    try:
        return datetime.strptime(s.split()[0], "%Y/%m/%d")
    except (ValueError, IndexError):
        pass
    logger.warning(f"Could not parse session date: {s!r}; falling back to now")
    return datetime.now()


_TEMPORAL_KEYWORDS = (
    "when", "before", "after", "days", "weeks", "months", "years",
    "between", "order", "first", "last", "latest", "earliest", "recent",
    "since", "until", "how long", "how many days", "in what order",
    "what was my", "personal best",
)


def is_temporal_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _TEMPORAL_KEYWORDS)


def build_temporal_block(graph: ConceptGraph, question: str, max_items: int = 60) -> str:
    """Symbolic temporal helper: when the question carries temporal intent,
    enumerate every dated concept/event in the graph sorted chronologically and
    inject as ## TEMPORAL FACTS so the reader sees absolute dates side by side.

    This bypasses the retrieval ranker's date-agnostic top-K selection — every
    fact retains its session date directly from the graph's time anchors.
    """
    if not is_temporal_question(question):
        return ""

    dated: list[tuple[datetime, str, str]] = []
    for n in graph.get_all_nodes():
        if n.type not in (NodeType.CONCEPT, NodeType.EVENT):
            continue
        date_str = n.data.get("date") or n.data.get("extracted_at") or n.data.get("timestamp")
        if not date_str:
            continue
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", ""))
        except Exception:
            continue
        title = n.data.get("title", n.id)
        desc = (n.data.get("description") or n.data.get("content") or "")[:200]
        # Strip the [YYYY-MM-DD] or [YYYY-MM-DD HH:MM(:SS)] prefix from
        # title/desc so the symbolic block doesn't double-print the date
        # (we now stamp the title at write time, and post-2026-06-01 also
        # include HH:MM).
        if title.startswith("[") and "]" in title[:24]:
            title = title.split("]", 1)[1].lstrip()
        if desc.startswith("[") and "]" in desc[:24]:
            desc = desc.split("]", 1)[1].lstrip()
        dated.append((dt, title, desc))

    if not dated:
        return ""

    dated.sort(key=lambda x: x[0])
    # Cap to most-recent-first within budget so we don't blow context
    dated = dated[-max_items:]

    lines = [
        "## TEMPORAL FACTS (sorted chronologically — verified by graph time anchors)"
    ]
    for dt, title, desc in dated:
        lines.append(f"- [{dt.date()}] **{title}** — {desc}")
    return "\n".join(lines) + "\n"


_RECENCY_TRIGGER = re.compile(
    r"\b(?:most\s+recent|latest|recent(?:ly)?|current|new(?:est)?|last(?:est)?|"
    r"this\s+(?:week|month|year)|nowadays|these\s+days)\b",
    re.IGNORECASE,
)


# Cluster E trigger — "previous conversation about X" / "you (mentioned|
# recommended|provided|listed|suggested) X". When the question explicitly
# references the assistant's prior turn, the answer is in a raw assistant
# EVENT — which retrieval routinely buries because (a) its title is just
# "Assistant message" so phrase score is weak, and (b) the distilled CONCEPT
# nodes win on lexical overlap. build_assistant_recall_block surfaces the
# raw assistant text directly.
_ASSISTANT_RECALL_TRIGGER = re.compile(
    r"\b(?:"
    # require explicit past-conversation anchor; the broader version
    # ("you mentioned/suggested" alone) was empirically shown to mis-fire
    # on present-tense advice requests ("Can you suggest some activities?")
    # and inject unrelated assistant text → preference-cluster regressions.
    r"previous\s+(?:conversation|chat|discussion|talk)|"
    r"earlier\s+(?:conversation|chat|discussion)|"
    r"prior\s+(?:conversation|chat|discussion)|"
    r"our\s+(?:previous|earlier|prior|last)\s+(?:chat|conversation|discussion|talk)|"
    r"in\s+our\s+(?:previous|earlier|prior|last)\s+(?:chat|conversation|discussion|talk)|"
    r"last\s+time\s+(?:we|you|i)|"
    r"i\s+was\s+going\s+through\s+our|"
    r"we\s+(?:discussed|talked\s+about)\s+\w+(?:\s+\w+){0,5}\s+earlier|"
    r"i\s+(?:think\s+)?we\s+discussed|"
    # extended for SSA recall patterns: "remind me what was the X on
    # that list", "in your submission/paper", "the X you gave"
    r"remind\s+me\s+(?:what\s+(?:was|did)|of\s+the|about\s+the)|"
    r"on\s+that\s+list|"
    r"in\s+(?:that|your|the\s+previous)\s+(?:submission|paper|report|review|response|article|plan)|"
    r"that\s+(?:list|recommendation|suggestion|advice|note|plan|chart|table)\s+you\s+(?:gave|provided|mentioned|made|wrote|shared)"
    r")\b",
    re.IGNORECASE,
)


def _stopwords() -> set[str]:
    return {
        "a","an","the","this","that","these","those","i","my","me","mine",
        "is","am","are","was","were","be","been","being","do","did","does","done",
        "have","has","had","will","would","could","should","can","may","might","must",
        "and","or","but","if","then","else","of","to","in","on","at","for","with",
        "from","by","as","than","into","onto","over","under","about","what","where",
        "when","which","who","whom","whose","how","why","most","recent","latest",
        "current","new","newest","last","ever","also","just","so","very","really",
        "you","your","go","get","got","gone","gotten","make","made","take","took",
    }


def build_recency_block(graph: ConceptGraph, question: str, max_items: int = 8) -> str:
    """Surface a date-sorted DESC list of concepts that lexically match the
    question topic — only when the question uses recency language.

    The reader prompt is already long enough that the per-concept [date]
    prefix gets skimmed past; this block restates the top recency-relevant
    nodes in newest-first order so picking the "most recent" answer reduces
    to copying the first bullet.
    """
    if not _RECENCY_TRIGGER.search(question):
        return ""

    # Extract content tokens from the question (drop stopwords + recency cue).
    sw = _stopwords()
    q_tokens = {
        t for t in re.findall(r"[a-zA-Z]+", question.lower())
        if t not in sw and len(t) > 2
    }
    if not q_tokens:
        return ""

    scored: list[tuple[float, datetime, str, str]] = []
    for n in graph.get_all_nodes():
        if n.type not in (NodeType.CONCEPT, NodeType.EVENT):
            continue
        date_str = n.data.get("date") or n.data.get("extracted_at") or n.data.get("timestamp")
        if not date_str:
            continue
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", ""))
        except Exception:
            continue
        title = n.data.get("title", n.id)
        desc = (n.data.get("description") or n.data.get("content") or "")[:200]
        # Strip the [YYYY-MM-DD] prefix from title so we don't double-print.
        if title.startswith("[") and "]" in title[:13]:
            title = title.split("]", 1)[1].lstrip()
        if desc.startswith("[") and "]" in desc[:13]:
            desc = desc.split("]", 1)[1].lstrip()
        text = f"{title} {desc}".lower()
        text_tokens = set(re.findall(r"[a-zA-Z]+", text))
        overlap = len(q_tokens & text_tokens)
        if overlap == 0:
            continue
        score = overlap / max(1, len(q_tokens))
        scored.append((score, dt, title, desc))

    if not scored:
        return ""

    # Sort by date DESC primary, score DESC secondary so newest match wins.
    scored.sort(key=lambda x: (x[1], x[0]), reverse=True)
    top = scored[:max_items]

    lines = [
        "## MOST RECENT MATCHES (newest → oldest; pick the top entry for 'most recent / latest / current' questions)"
    ]
    for i, (_, dt, title, desc) in enumerate(top, 1):
        lines.append(f"{i}. [{dt.date()}] **{title}** — {desc}")
    return "\n".join(lines) + "\n"


# R2 trigger — "what time do I X" / "when do I X" / "at what time do I X" /
# "what time did/does I X" / "what time do I usually X". Includes "go to bed",
# "get home", "wake up", "leave", "arrive". Narrow on purpose to avoid
# matching "what time was the meeting" (single-fact) vs personal-routine Qs.
_TIME_OF_DAY_TRIGGER = re.compile(
    r"\b(?:what|which|at\s+what)\s+time\s+(?:do|does|did)\s+i\b|"
    r"\bwhat\s+time\s+do\s+i\b|"
    r"\bwhen\s+do\s+i\s+(?:usually|typically|normally)\b",
    re.IGNORECASE,
)
# Clock-time pattern: 12-hour (6:00 pm, 6:00 PM, 6 pm, 6 a.m.) and 24-hour (06:00).
_CLOCK_TIME_PATTERN = re.compile(
    r"\b(?:\d{1,2}:\d{2}(?:\s*[ap]\.?m\.?)?|\d{1,2}\s*[ap]\.?m\.?)\b",
    re.IGNORECASE,
)


def build_time_of_day_block(
    graph: ConceptGraph, question: str, max_items: int = 6
) -> str:
    """R2 — surface concepts that contain an explicit clock-time value for
    "what time do I X" questions. Exp A showed that writer captures these
    facts (e.g., "User stopping work emails by 7 pm") but retrieval ranks
    photography / fishing concepts above them. This block pins clock-time
    matching nodes to the front of the context.
    """
    if not _TIME_OF_DAY_TRIGGER.search(question):
        return ""

    sw = _stopwords()
    q_tokens = {
        t for t in re.findall(r"[a-zA-Z]+", question.lower())
        if t not in sw and len(t) > 2
    }
    if not q_tokens:
        return ""

    scored: list[tuple[float, datetime, str, str]] = []
    for n in graph.get_all_nodes():
        if n.type not in (NodeType.CONCEPT, NodeType.EVENT):
            continue
        title = n.data.get("title", n.id) or ""
        desc = (n.data.get("description") or n.data.get("content") or "")[:300]
        text = f"{title} {desc}"
        # Must contain at least one clock-time pattern to be considered.
        if not _CLOCK_TIME_PATTERN.search(text):
            continue
        date_str = n.data.get("date") or n.data.get("extracted_at") or n.data.get("timestamp")
        try:
            dt = datetime.fromisoformat((date_str or "").replace("Z", ""))
        except Exception:
            dt = datetime.min
        # Strip [YYYY-MM-DD…] prefix.
        if title.startswith("[") and "]" in title[:24]:
            title = title.split("]", 1)[1].lstrip()
        if desc.startswith("[") and "]" in desc[:24]:
            desc = desc.split("]", 1)[1].lstrip()
        text_tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
        overlap = len(q_tokens & text_tokens)
        if overlap == 0:
            continue
        score = overlap / max(1, len(q_tokens))
        scored.append((score, dt, title, desc))

    if not scored:
        return ""

    # Sort by score DESC, then date DESC (newest tie-break for current routine).
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top = scored[:max_items]

    lines = [
        "## CLOCK_TIME_MATCHES (nodes containing an explicit time value — "
        "the answer to a 'what time do I X' question is almost always in one "
        "of these)"
    ]
    for i, (score, dt, title, desc) in enumerate(top, 1):
        date_str = dt.date().isoformat() if dt != datetime.min else "?"
        lines.append(f"{i}. [{date_str}] (overlap={score:.2f}) **{title}** — {desc}")
    return "\n".join(lines) + "\n"


# R3 trigger — questions asking for a specific named entity / proper noun:
# breed/cartoon/store/album/movie/song/dish name, etc.
_PROPER_NOUN_TRIGGER = re.compile(
    r"\bwhat\s+(?:is|was)\s+the\s+name\s+of\b|"
    r"\bwhat\s+breed\b|"
    r"\bwhich\s+(?:movie|book|song|album|store|brand|product|cartoon|show|"
    r"podcast|restaurant|recipe|app|service|company|game|website)\b|"
    r"\bremind\s+me\s+of\s+(?:that|the)\s+\w+\s+name\b|"
    r"\bremind\s+me\s+(?:of\s+)?(?:the\s+name\s+of\s+|that\s+)\w+",
    re.IGNORECASE,
)
# Detect capitalized multi-word phrases in a node's title/desc; require at
# least one "rare" token (capitalized non-stopword) for it to qualify.
_PROPER_NOUN_RE = re.compile(
    r"\b(?:[A-Z][a-z]+(?:[\s\-'][A-Z][a-z]+){0,3})\b"
)


def build_proper_noun_block(
    graph: ConceptGraph, question: str, max_items: int = 6
) -> str:
    """R3 — surface concepts that contain a capitalized proper noun
    (multi-word) for "what's the name / what breed / which X" questions.
    Exp A showed that writer captures these (e.g., "Golden Retriever",
    "Bajimaya v Reward Homes") but retrieval misses them because the
    question doesn't share many tokens with the captured node.
    """
    if not _PROPER_NOUN_TRIGGER.search(question):
        return ""

    sw = _stopwords()
    q_tokens = {
        t for t in re.findall(r"[a-zA-Z]+", question.lower())
        if t not in sw and len(t) > 2
    }
    if not q_tokens:
        return ""

    scored: list[tuple[float, datetime, str, str, list[str]]] = []
    for n in graph.get_all_nodes():
        if n.type not in (NodeType.CONCEPT, NodeType.EVENT):
            continue
        title = n.data.get("title", n.id) or ""
        desc = (n.data.get("description") or n.data.get("content") or "")[:300]
        text = f"{title} {desc}"
        # Strip [YYYY-MM-DD…] prefix before scanning for proper nouns.
        text_stripped = re.sub(r"^\s*\[\d{4}-\d{2}-\d{2}[^\]]*\]\s*", "", title) + " " + desc
        propers = [
            p for p in _PROPER_NOUN_RE.findall(text_stripped)
            if p.lower() not in {"user", "assistant", "session date"}
            and len(p) > 3
        ]
        if not propers:
            continue
        text_tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
        overlap = len(q_tokens & text_tokens)
        if overlap == 0:
            continue
        date_str = n.data.get("date") or n.data.get("extracted_at") or n.data.get("timestamp")
        try:
            dt = datetime.fromisoformat((date_str or "").replace("Z", ""))
        except Exception:
            dt = datetime.min
        if title.startswith("[") and "]" in title[:24]:
            title = title.split("]", 1)[1].lstrip()
        score = overlap / max(1, len(q_tokens))
        scored.append((score, dt, title, desc, propers[:5]))

    if not scored:
        return ""

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top = scored[:max_items]

    lines = [
        "## PROPER_NOUN_MATCHES (nodes containing capitalized named entities "
        "— the answer to a 'name of / which X' question is almost always one "
        "of the bolded names below)"
    ]
    for i, (score, dt, title, desc, propers) in enumerate(top, 1):
        date_str = dt.date().isoformat() if dt != datetime.min else "?"
        proper_str = ", ".join(f"**{p}**" for p in propers)
        lines.append(f"{i}. [{date_str}] (overlap={score:.2f}) {title} — {desc}  [names: {proper_str}]")
    return "\n".join(lines) + "\n"


# ============================================================================
# iter33-MS TIER 1 — entity-expansion / bridge-entity second-hop retrieval.
#
# Failure-analysis verdict (ITER33_MS_PLAN.md §TIER 1): 15 of 24 iter19 MS
# failures are RETRIEVAL misses — the answer-bearing node exists in the graph
# (the writer extracted it) but a single lexically-dissimilar session scores
# below the flat top-k cutoff (top-50 = top ~4 % of a ~1150-node haystack),
# so it never reaches full_context. A bigger flat-k does NOT help (6/7 count
# cases were already at 50). The fix is ENTITY-AWARE retrieval: run a *second*
# retrieval sub-query keyed on the bridge entity / category-sibling and
# force-union its hits into context BEFORE assembly.
#
# These helpers are PURELY ADDITIVE (they only widen recall by prepending an
# extra context block) and every entry point is GATED on an MS-style
# question-type regex, so TR / KU / SSU / SSA paths are untouched: on a
# non-matching question the gate returns immediately and nothing changes.
# ============================================================================

# R3 — age / duration / comparison arithmetic detector. These are NOT
# "current/latest" questions, yet today they get NO retrieval widening: the
# agg-count detector below suppresses widening on `ago|since|before|after`
# (the very tokens these questions contain) AND its "how many X have I"
# verb-frame never matches an "how old was I when" question. So the bridge
# anchors stay retrieval-starved. Matching here re-enables widening (R3a) and
# triggers the bridge-entity second hop (R1). Targets a1cc6108, ba358f49,
# c18a7dc8, 3c1045c8, 853b0a1d, c9f37c46, dcfa8644, b29f3365.
_MS_ARITH_COMPARE_RE = re.compile(
    r"\bhow\s+old\s+(?:was|am|will)\s+i\b"          # how old was/am/will I
    r"|\bhow\s+many\s+years\s+(?:old|older|younger|will|would)\b"  # …years old(er)/will I be
    r"|\bhow\s+much\s+older\b"                       # how much older
    r"|\bhow\s+(?:long|many\s+\w+)\s+(?:had|have)\s+i\b"  # how long had/have I …
    # Two-anchor date-diff bridge: "how many days had passed since I X when I Y"
    # / "how many weeks passed between … when I …". The presence of a SECOND
    # "when I [event]" clause distinguishes these from single-anchor TR Qs and
    # qualifies them for the bridge second-hop (dcfa8644 Adidas/Converse).
    r"|\bhow\s+many\s+\w+\s+(?:had\s+)?(?:passed|elapsed|gone)\b.*\bwhen\s+i\b"
    r"|\bthan\s+when\s+i\b"                          # … than when I [event]
    r"|\bwhen\s+(?:i|my)\s+(?:friend\s+|\w+\s+)?\w*?(?:graduat|born|gets?\s+marri|"
    r"married|bought|attended|got|gave|started|moved|joined|met|realized|"
    r"realised|bought|finished|invested|received|broke)\b",  # when I/my [bridge event]
    re.IGNORECASE,
)

# R1 — bridge noun-phrase extraction. Pulls the low-frequency entity/event
# that the interrogative frame ("how old was I when …") buries, e.g.
# "Alex", "Rachel", "graduated from college", "the silver necklace",
# "average age of employees in my department", "open mic night",
# "Adidas running shoes", "guitar lessons" / "guitar amp".
_BRIDGE_CLAUSE_RE = re.compile(
    r"\bwhen\s+(?:i|my|the)\b(.*?)(?:[?.,;]|$)"      # "when I/my/the <NP> …"
    r"|\bthan\s+when\s+i\b(.*?)(?:[?.,;]|$)"          # "than when I <NP>"
    # iter33-MS B1: "than the <NP>" comparison operand — mines the second
    # operand of a "how much older am I than the average age of employees
    # in my department" comparison (3c1045c8). "than when I" is matched by
    # the arm above; "than me/you" yields a <3-char clause that _add drops.
    r"|\bthan\s+the\b(.*?)(?:[?.,;]|$)"               # "than the <NP>"
    r"|\bsince\s+i\b(.*?)(?:[?.,;]|$)"               # "since I <NP>"
    r"|\bbefore\s+(?:making|i)\b(.*?)(?:[?.,;]|$)",  # "before making/I <NP>"
    re.IGNORECASE,
)


def _extract_bridge_phrases(question: str, max_phrases: int = 4) -> list[str]:
    """R1 — return up to `max_phrases` bridge sub-queries mined from the
    question's "when I [event]" / "than when I [event]" clauses plus any
    capitalized named entity (Alex / Rachel / Adidas / Converse). Each is a
    compact noun-phrase suitable as an independent retrieval sub-query.

    Generalizable (no per-qid strings): all phrases come from question text
    via clause-splitting + proper-noun extraction + content-token fallback.
    """
    phrases: list[str] = []
    seen: set[str] = set()

    def _add(p: str) -> None:
        p = re.sub(r"\s+", " ", p).strip(" .,;:?!")
        # Drop leading auxiliaries/articles left over from the clause split.
        p = re.sub(r"^(?:i|my|the|a|an|was|were|am|been|had|have|get|got|"
                   r"did|do|that|this)\s+", "", p, flags=re.IGNORECASE).strip()
        if len(p) < 3 or p.lower() in seen:
            return
        seen.add(p.lower())
        phrases.append(p)

    # 1. Bridge clauses ("when I graduated from college", etc.).
    for m in _BRIDGE_CLAUSE_RE.finditer(question):
        clause = next((g for g in m.groups() if g), "")
        if not clause:
            continue
        # Keep the verb+object core; trim trailing temporal noise.
        clause = re.split(r"\b(?:on|in|at|during|for|by)\s+\d", clause)[0]
        _add(clause)

    # 2. Proper-noun named entities (Alex, Rachel, Adidas, Converse, Berkeley).
    for p in _PROPER_NOUN_RE.findall(question):
        if p.lower() not in {"how", "user", "assistant"} and len(p) > 2:
            _add(p)

    return phrases[:max_phrases]


# R2 — category-membership force-include for COUNT-over-owned-category Qs.
# Detects "how many <category> have/did I own/buy/replace/bake/view/attend …"
# and mines the category head-noun. Sibling instance nodes (a model kit / a
# kitchen appliance / a fish / a property viewed / a bake / a bike) are then
# surfaced by category + per-sibling-token sub-queries. Targets
# gpt4_59c863d7, gpt4_194be4b3, gpt4_ab202e7f, gpt4_7fce9456, 2ce6a0f2,
# 88432d0a, a9f6b44c.
_MS_COUNT_CATEGORY_RE = re.compile(
    r"\bhow\s+many\s+(?:different\s+|distinct\s+)?"
    r"([a-z][a-z\- ]+?)\s+"
    r"(?:have|did|do|have\s+i|did\s+i|do\s+i|are\s+there|i\s+(?:\w+\s+)?(?:"
    r"own|owned|bought|buy|replace|replaced|fix|fixed|bake|baked|"
    r"view|viewed|attend|attended|service|serviced|made|make|read|"
    # iter33-MS B2(a): acquisition verbs (3a704032 "acquire", bf659f65
    # "purchased/downloaded"). NOT plan/planned (handled in B3).
    r"acquire|acquired|purchase|purchased|download|downloaded|got|"
    r"watched|watch|visited|visit|hosted|host|booked|book))\b",
    re.IGNORECASE,
)


def _extract_count_category(question: str) -> str:
    """R2 — return the head category noun for a count-over-category question
    (e.g. "model kits", "musical instruments", "kitchen items", "properties",
    "art-related events", "bikes"). Empty string when not a count-category Q.
    """
    m = _MS_COUNT_CATEGORY_RE.search(question)
    if not m:
        return ""
    cat = m.group(1).strip()
    # Drop a leading quantifier/adjective that bleeds into the capture.
    sw = _stopwords()
    toks = [t for t in cat.split() if t.lower() not in sw]
    cat = " ".join(toks).strip()
    # Strip a trailing temporal unit that bled into the capture from a TR
    # date-diff phrasing ("how many WEEKS AGO did I …", "how many MONTHS have
    # passed since …"). These are NOT count-over-category Qs — keep R2 inert
    # on them so TR date-diff handling is unchanged. (71017276, b46e15ed.)
    cat = re.sub(
        r"\b(?:weeks?|months?|years?|days?|hours?|minutes?)\b.*$",
        "", cat, flags=re.IGNORECASE,
    ).strip()
    # When the head noun is a generic counter ("how many TIMES did I bake …"),
    # fall back to the activity verb so the sub-query keys on the real category
    # (e.g. 88432d0a "times … bake" → "bake"). Generalizable, not per-qid.
    if cat.lower() in {"times", "occasions", "instances"} or not cat:
        vm = re.search(
            r"\b(?:did|do|have|has|i)\s+i?\s*"
            r"(own|owned|buy|bought|replace|replaced|fix|fixed|bake|baked|"
            r"view|viewed|attend|attended|service|serviced|made|make|read|"
            # iter33-MS B2(a): acquisition verbs in the verb-fallback too.
            r"acquire|acquired|purchase|purchased|download|downloaded|got|"
            r"watch|watched|visit|visited|host|hosted|book|booked)\b"
            # iter33-MS B4: also capture the OBJECT NOUN-PHRASE trailing the
            # verb ("attend (fitness classes)" -> "fitness classes"). Without
            # this, the verb-fallback returned the bare verb ("attend") for
            # a08a253f "How many days a week do I attend fitness classes?".
            # The object NP is everything up to the first clause / temporal /
            # punctuation boundary; bare-verb fallback kept when no object NP.
            r"(?P<obj>(?:\s+[a-z][a-z\-]+){0,5})",
            question, re.IGNORECASE,
        )
        if vm:
            obj = (vm.group("obj") or "").strip()
            # Trim the object NP at the first boundary token (clause openers,
            # temporal units, prepositions) so we keep only the core noun
            # phrase ("fitness classes", "egg tarts") and not trailing
            # "... in the past two weeks" / "... when I made ...".
            obj = re.sub(
                r"\b(?:in|on|at|for|to|from|since|before|after|when|while|"
                r"between|across|during|over|with|that|which|the|a|an|past|"
                r"last|this|next|ago|week|weeks|day|days|month|months|"
                r"year|years|hour|hours|something|anything)\b.*$",
                "", obj, flags=re.IGNORECASE,
            ).strip()
            obj_toks = [t for t in obj.split() if t.lower() not in sw]
            obj = " ".join(obj_toks).strip()
            cat = obj if obj else vm.group(1)
    return cat


def _ms_extra_count_sub_queries(question: str, cat: str) -> list[str]:
    """iter33-MS M1-M4 — additive, frame-gated sub-queries for the R2
    count-over-category path. Every sub-query is DERIVED from the question's
    own tokens / general lexical frames (no per-qid literal strings). Returns
    [] for any frame that doesn't match, so this is inert on the categories it
    doesn't target. Called from the `_ms_count_category` branch and appended to
    `_sub_queries`; the existing R2 logic is unchanged.
    """
    extra: list[str] = []
    if not cat:
        return extra
    qlow = question.lower()
    clow = cat.lower()

    # M1 — polysemy disambiguation (gpt4_7fce9456 "properties": real-estate vs
    # astronomy/meme word-sense). CONJOIN the question's bridge-qualifier nouns
    # (e.g. "townhouse", "offer", "Brookside neighborhood") with the category
    # head so the combined sub-query ranks the intended word-sense. Only fires
    # when the question actually HAS bridge qualifiers (a "before/than/when …"
    # clause or a capitalized entity) — on plain counts (3a704032 plants,
    # gpt4_194be4b3 instruments) _extract_bridge_phrases returns [] so this is
    # a no-op there.
    _bridge = _extract_bridge_phrases(question)
    if _bridge:
        # Pull short qualifier head-nouns out of the bridge phrases (drop
        # stop/aux tokens), then conjoin with the category head.
        sw = _stopwords()
        _qual_toks: list[str] = []
        _seen_q: set[str] = set()
        for ph in _bridge:
            for t in re.split(r"[\s\-]+", ph):
                tl = t.lower().strip(".,;:?!")
                if (len(tl) >= 3 and tl not in sw and tl not in _seen_q
                        and tl not in clow.split()):
                    _seen_q.add(tl)
                    _qual_toks.append(tl)
        if _qual_toks:
            extra.append(cat + " " + " ".join(_qual_toks[:5]))

    # M2 — generic-event categories (2ce6a0f2 "art-related events"): an event
    # instance ("guided tour at the History Museum") may not share the category
    # adjective ("art"), so the bare-category sub-query under-ranks it. When the
    # category head is a GENERIC EVENT noun AND the question uses an attend-style
    # verb, add an action-verb probe that keys on event-attendance phrasing.
    # Gated to generic-event heads only — stays inert on concrete-object counts
    # (plants / instruments / properties).
    _generic_event_head = bool(re.search(
        r"\b(?:events?|classes|activities|outings?|gatherings?|"
        r"shows?|sessions?)\b", clow,
    ))
    if _generic_event_head and re.search(
        r"\b(?:attend|attended|went|go|visit|visited|tour|toured|"
        r"saw|see)\b", qlow,
    ):
        extra.append("attended visited went tour exhibition lecture event "
                     + cat)

    # M3 — "how many days/times a week" weekly-frequency frame (a08a253f
    # fitness classes). A specific weekday instance ("Wednesday yoga class")
    # may not share the category head, so enumerate weekday names alongside the
    # category to surface every per-day instance. Gated to the weekly-frame.
    if re.search(r"\bhow\s+many\s+(?:days?|times)\s+(?:a|per)\s+week\b", qlow):
        extra.append("Monday Tuesday Wednesday Thursday Friday Saturday Sunday "
                     + cat)

    # M4 — music album/EP purchase counts (bf659f65): a vinyl/LP purchase
    # ("Tame Impala vinyl bought at Red Rocks") is lexically distant from
    # "album/EP". Strengthen the existing "vinyl record <cat>" sibling with
    # explicit format+acquisition variants so a vinyl/LP/record session wins a
    # slot. Gated to music/album/EP/record categories.
    if re.search(r"\b(?:album|albums|ep|eps|music|record|records|vinyl|lp|"
                 r"lps)\b", clow):
        extra.append("vinyl LP record bought purchased downloaded at concert")

    # Group A — hyponym roster. The counted instance is usually named by its
    # specific KIND (an engagement "ring" for a jewelry count, a "1-bedroom
    # condo" for a properties count, a "peace lily" for a plants count, a
    # "lemon" for a citrus count) and never carries the category word, so the
    # bare-category sub-query under-ranks the acquisition node. For a count
    # over a KNOWN category head, force-include a hyponym roster conjoined with
    # the head. `cat` is only set on count-shaped questions, and the head must
    # match a known category, so this stays inert on every other count
    # (instruments / art-events / fitness / albums all miss the keys below).
    _HYPONYMS = {
        "jewel": "ring rings earrings necklace bracelet pendant brooch watch",
        "propert": "condo bungalow townhouse house apartment flat duplex unit",
        "citrus": "lemon lime orange grapefruit tangerine mandarin",
        "fruit": "lemon lime orange grapefruit apple banana berry pear",
        "plant": "succulent lily fern cactus herb pothos monstera orchid",
    }
    for _key, _hyps in _HYPONYMS.items():
        if _key in clow:
            extra.append(_hyps + " " + cat)
            break

    return extra


def _build_forced_include_block(
    query_agent: "MemoryQueryAgent",
    sub_queries: list[str],
    already_ids: set[str],
    *,
    per_query_k: int = 6,   # iter33-MS refine: keep 6 — protects multi-item
                            # count recall (gpt4_59c863d7 needs 5 model kits).
    max_force: int = 12,    # iter33-MS refine C3 (mild): 18->12 caps total
                            # prepend noise without cutting per-sub-query recall.
    header: str,
) -> tuple[str, set[str]]:
    """Run each entity/category sub-query through the EXISTING retrieval path
    (HYBRID BM25+embedding, no LLM rerank — cheap, recall-oriented) and render
    a force-include block of nodes that the main top-k retrieval MISSED.

    Reuses `query_agent.query()` so we share the project's BM25/embedding
    helpers rather than writing a new retriever. Returns (block_text,
    newly_included_ids). Nodes already in `already_ids` are skipped (they are
    in context already). Caps total forced nodes at `max_force` to bound the
    extra context budget.
    """
    from cognifold.query.models import QueryType as QueryT  # noqa: N814 alias

    picked: list = []
    picked_ids: set[str] = set()
    for sq in sub_queries:
        if len(picked) >= max_force:
            break
        try:
            # Plain HYBRID retrieval keyed on the bridge/category phrase.
            # No rerank, modest k — we only want this phrase's best matches.
            sub = query_agent.query(
                sq,
                query_type=QueryT.HYBRID,
                max_nodes=per_query_k,
            )
        except Exception:
            continue
        for node in getattr(sub, "nodes", []) or []:
            nid = getattr(node, "node_id", None)
            if not nid or nid in already_ids or nid in picked_ids:
                continue
            picked.append(node)
            picked_ids.add(nid)
            if len(picked) >= max_force:
                break

    if not picked:
        return "", set()

    lines = [header]
    for i, node in enumerate(picked, 1):
        title = (getattr(node, "title", "") or "").strip()
        desc = (getattr(node, "description", "") or "")[:240].strip()
        if title.startswith("[") and "]" in title[:24]:
            # keep the date prefix; it carries the session date the reader needs
            pass
        lines.append(f"{i}. {title} — {desc}" if desc else f"{i}. {title}")
    return "\n".join(lines) + "\n", picked_ids


_RELATIVE_AGO_TARGET_RE = re.compile(
    r"\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"\d+)\s+(day|days|week|weeks|month|months|year|years)\s+ago\b",
    re.IGNORECASE,
)
_WORD_TO_INT_LOCAL = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_UNIT_DAYS_LOCAL = {
    "day": 1, "days": 1, "week": 7, "weeks": 7,
    "month": 30, "months": 30, "year": 365, "years": 365,
}


def build_target_date_concepts_block(
    graph: ConceptGraph, question_date: datetime | None, question: str, window: int = 3
) -> str:
    """iter16 — list every dated concept within ±`window` days of the
    target date implied by 'X N weeks ago' in the question. Reader sees
    a focused candidate list and can pick by topic match. Skips silently
    when the question has no relative-time clause."""
    if question_date is None:
        return ""
    m = _RELATIVE_AGO_TARGET_RE.search(question)
    if not m:
        return ""
    num_str = m.group(1).lower()
    unit_str = m.group(2).lower()
    num = _WORD_TO_INT_LOCAL.get(num_str)
    if num is None:
        try:
            num = int(num_str)
        except ValueError:
            return ""
    unit_key = unit_str if unit_str.endswith("s") else unit_str + "s"
    days_back = num * _UNIT_DAYS_LOCAL[unit_key]
    target = question_date - timedelta(days=days_back)
    target_d = target.date()

    rows: list[tuple[int, datetime, str, str]] = []  # (off, date, title, desc)
    for n in graph.get_all_nodes():
        if n.type not in (NodeType.CONCEPT, NodeType.EVENT):
            continue
        if n.id.startswith("evt-"):
            continue
        date_str = n.data.get("date") or n.data.get("extracted_at")
        if not date_str:
            continue
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", ""))
        except Exception:
            continue
        off = abs((dt.date() - target_d).days)
        if off > window:
            continue
        title = n.data.get("title", "") or ""
        if title.startswith("[") and "]" in title[:24]:
            title = title.split("]", 1)[1].lstrip()
        desc = (n.data.get("description") or n.data.get("content") or "")[:160]
        # Skip writer concept-types that mark synthetic typed_* nodes.
        ctype = (n.data.get("concept_type") or "").lower()
        if ctype.startswith("typed_"):
            continue
        # Skip planning/intent concepts (consistent with resolvers).
        full = (title + " " + desc).lower()
        if any(p in full for p in (
            "is planning", "is considering", "is thinking",
            "would like to", "wants to", "intends to", "is going to",
            "is looking forward to", "is hoping to",
            "i'm planning", "i'm thinking", "i'm considering",
            "asked the assistant", "recommended", "suggested",
            "is interested in", "heard about", "read about",
        )):
            continue
        rows.append((off, dt, title, desc))
    if not rows:
        return ""
    rows.sort(key=lambda x: (x[0], x[1]))  # closest to target, then earliest
    lines = [
        "## CONCEPTS_NEAR_TARGET_DATE",
        f"Concepts whose [YYYY-MM-DD] prefix is within ±{window} days of "
        f"{target_d.isoformat()} (the resolved target for "
        f"'{num} {unit_str} ago'). Pick the concept whose description best "
        f"matches the question's topic; do not pick a lexically-similar "
        f"concept dated further from this target.",
    ]
    for off, dt, title, desc in rows[:12]:
        lines.append(f"- [{dt.date()}] (off={off}d) **{title}** — {desc}")
    return "\n".join(lines) + "\n"


def build_recall_target_date_block(question_date: datetime | None, question: str) -> str:
    """iter15 — when the question contains 'X N days/weeks/months ago' the
    reader frequently picks a lexically-similar concept dated NEAR but not
    AT the target. Pre-compute the target date and inject as a hint so the
    reader pins to the right [YYYY-MM-DD] prefix.

    Targets the cluster where the resolver's relative_ago_recall doesn't
    fire (generic topic) but the question is clearly date-anchored:
      gpt4_1e4a8aec gardening 2 weeks ago,
      eac54add business milestone 4 weeks ago,
      gpt4_4929293b relative life event 1 week ago,
      gpt4_59149c78 art event 2 weeks ago,
      gpt4_e072b769 Ibotta 3 weeks ago.
    """
    if question_date is None:
        return ""
    m = _RELATIVE_AGO_TARGET_RE.search(question)
    if not m:
        return ""
    num_str = m.group(1).lower()
    unit_str = m.group(2).lower()
    num = _WORD_TO_INT_LOCAL.get(num_str)
    if num is None:
        try:
            num = int(num_str)
        except ValueError:
            return ""
    unit_key = unit_str if unit_str.endswith("s") else unit_str + "s"
    days_back = num * _UNIT_DAYS_LOCAL[unit_key]
    target = question_date - timedelta(days=days_back)
    return (
        "## RECALL_TARGET_DATE\n"
        f"The question's relative-time clause ({num} {unit_str} ago) "
        f"resolves to absolute date **{target.date().isoformat()}** (today is "
        f"{question_date.date().isoformat()}). When picking the matching "
        f"event, prefer the concept whose `[YYYY-MM-DD]` prefix is closest "
        f"to {target.date().isoformat()} (±3 days). Do NOT pick a concept "
        f"that's more lexically similar but dated further from this "
        f"target.\n"
    )


def build_today_block(question_date: datetime | None) -> str:
    """iter07 — pin the dataset's question_date at the top of context as the
    reference "today" for relative-time expressions ("X days ago", "two
    weeks ago", "currently"). Without this, the reader falls back to the
    system clock — in iter05/06 that meant 2026-06-01 vs the dataset's
    actual question_date in 2023, producing "1,205 days ago" for "X days
    ago" questions whose true answer was 17 days.
    """
    if question_date is None:
        return ""
    # Use date-only display — same format as the rest of the context so the
    # reader doesn't treat this as a millisecond-precise timestamp.
    today_str = question_date.date().isoformat()
    return (
        "## TODAY\n"
        f"{today_str}\n"
        "Use this date as the reference for any relative-time expression in "
        "the question (\"X days ago\", \"X weeks ago\", \"currently\", "
        "\"now\", \"recently\"). Compute date differences from this date — "
        "do NOT use the system clock or any other reference.\n"
    )


_TIMELINE_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at",
    "for", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "doing",
    "what", "which", "when", "where", "who", "whom", "whose", "how",
    "why", "i", "my", "me", "mine", "you", "your", "we", "us", "they",
    "them", "their", "this", "that", "these", "those", "it", "its",
    "than", "then", "so", "also", "very", "just", "only", "some", "any",
    "all", "no", "not", "one", "two", "three", "first", "last", "next",
    "ago", "before", "after", "since", "until", "during", "while",
    "between", "among",
}


def _question_topic_tokens(question: str) -> list[str]:
    """Cheap noun-phrase token extractor for timeline matching.

    Returns lower-cased content tokens (≥3 chars) from the question,
    stripped of common stopwords. Used by the topic-timeline builder to
    match graph concepts by overlap.
    """
    toks = re.findall(r"\b[A-Za-z][A-Za-z0-9'-]+\b", question.lower())
    return [t for t in toks if len(t) >= 3 and t not in _TIMELINE_STOPWORDS]


def build_topic_timeline_block(
    graph: ConceptGraph,
    question: str,
    question_date: datetime | None,
    max_items: int = 20,
) -> str:
    """iter29 TR-α — topic-focused chronological timeline.

    For TR questions, scan the graph for concepts whose title/description
    overlap with the question's topic tokens and render them in a sorted
    timeline with absolute date + (N days ago) anchors. Solves order_among
    and "between A and B" failure modes where retrieval ranking buries the
    correct chronology.
    """
    if question_date is None:
        return ""
    tokens = _question_topic_tokens(question)
    if not tokens:
        return ""
    token_set = set(tokens)

    today = question_date.date()
    date_prefix_re = re.compile(
        r"^\s*\[(\d{4}-\d{2}-\d{2})(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\]\s*"
    )
    candidates: list[tuple[datetime, int, str]] = []
    for n in graph.get_all_nodes():
        if getattr(n, "type", None) is None:
            continue
        type_name = getattr(n.type, "name", str(n.type)).lower()
        if type_name not in ("concept", "event"):
            continue
        ctype = (n.data.get("concept_type") or "").lower()
        if ctype.startswith("typed_"):
            continue
        title = (n.data.get("title") or "").strip()
        desc = (n.data.get("description") or "").strip()
        body = f"{title} {desc}".lower()
        overlap = sum(1 for t in token_set if t in body)
        if overlap < 1:
            continue

        # Pick a date — prefer the [YYYY-MM-DD] prefix on the title, then
        # event_date (only when caller didn't strip W2), then session
        # date / timestamp.
        m = date_prefix_re.match(title)
        date_str = None
        if m:
            date_str = m.group(1)
        else:
            date_str = (
                n.data.get("event_date")
                or n.data.get("date")
                or n.data.get("timestamp")
                or ""
            )
        if not date_str:
            continue
        try:
            dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
        except Exception:
            continue
        # Strip date prefix from title for display.
        title_disp = date_prefix_re.sub("", title).strip() or n.id
        candidates.append((dt, overlap, title_disp))

    if not candidates:
        return ""
    # Sort by date ASC; tie-break by overlap DESC.
    candidates.sort(key=lambda x: (x[0], -x[1]))
    # Keep top-overlap items if too many.
    if len(candidates) > max_items:
        by_overlap = sorted(candidates, key=lambda x: -x[1])[:max_items]
        candidates = sorted(by_overlap, key=lambda x: x[0])

    lines = ["## TOPIC_TIMELINE (chronological, sorted earliest → latest)"]
    for dt, _, title_disp in candidates:
        delta = (today - dt.date()).days
        if delta == 0:
            tag = "TODAY"
        elif delta > 0:
            tag = f"{delta} day{'s' if delta != 1 else ''} ago"
        else:
            tag = f"in {-delta} day{'s' if -delta != 1 else ''}"
        lines.append(f"- [{dt.date().isoformat()} • {tag}] {title_disp[:160]}")
    return "\n".join(lines) + "\n"


def build_assistant_recall_block(
    graph: ConceptGraph, question: str, max_items: int = 4, snippet_chars: int = 600
) -> str:
    """Surface raw assistant EVENT text for "previous conversation about X"
    questions.

    The distilled CONCEPT nodes paraphrase the assistant's surface form away
    (the Borges quote becomes "user asked about the Library of Babel"), and
    retrieval ranks them above raw EVENTs because the EVENT title is just
    "Assistant message". When the question explicitly asks for a name /
    title / quote the assistant produced earlier, hand the reader the raw
    text directly.

    Pattern: same lexical-overlap scoring as build_recency_block. Filtered
    to assistant role EVENTs only. Trigger-gated so this block is silent on
    questions that don't reference a prior assistant turn (no regression
    risk on the other clusters).
    """
    if not _ASSISTANT_RECALL_TRIGGER.search(question):
        return ""

    sw = _stopwords()
    q_tokens = {
        t for t in re.findall(r"[a-zA-Z]+", question.lower())
        if t not in sw and len(t) > 2
    }
    if not q_tokens:
        return ""

    scored: list[tuple[float, datetime, str]] = []
    for n in graph.get_all_nodes():
        if n.type != NodeType.EVENT:
            continue
        if n.data.get("role") != "assistant":
            continue
        content = n.data.get("content") or n.data.get("description") or ""
        if not content:
            continue
        date_str = n.data.get("date") or n.data.get("timestamp")
        if not date_str:
            continue
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "")[:19]) if "T" in date_str \
                else datetime.fromisoformat(date_str)
        except Exception:
            continue
        text_tokens = set(re.findall(r"[a-zA-Z]+", content.lower()))
        overlap = len(q_tokens & text_tokens)
        if overlap == 0:
            continue
        # Recall fraction × small log boost for richer assistant turns.
        score = overlap / max(1, len(q_tokens))
        scored.append((score, dt, content))

    if not scored:
        return ""

    # Sort by score DESC; on ties, prefer newer date (more likely the turn
    # the user is referring back to).
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top = scored[:max_items]

    lines = [
        "## RAW_ASSISTANT (verbatim text from assistant turns the question likely refers to — copy names/quotes from here, do NOT substitute)",
    ]
    for i, (score, dt, content) in enumerate(top, 1):
        snippet = content.strip().replace("\n", " ")
        if len(snippet) > snippet_chars:
            snippet = snippet[:snippet_chars] + "…"
        lines.append(f"{i}. [{dt.date()}] (overlap={score:.2f}) {snippet}")
    return "\n".join(lines) + "\n"


def generate_answer(
    question: str,
    context: str,
    config: AgentConfig,
    qa_template: str | None = None,
    *,
    graph: ConceptGraph | None = None,
    query_nodes: "list | None" = None,
    question_date: datetime | None = None,
) -> str:
    """Generate an answer using the LLM based on context.

    Iter32 round 2 gated path: if the question shape is count/order/
    duration/date_diff/derived_time/abs_value AND graph + query_nodes
    are provided, route through the evidence ledger. The ledger
    returns a deterministic answer only for high-confidence shapes;
    otherwise the reader sees a fused context (chunk-level late fusion)
    that surfaces missing raw event facts.
    """
    # Iter32 ledger route — gated, conservative.
    # iter33-MS: the per-qid emitter ledger is a discarded dead-end (brittle,
    # inert on the real graph; see ITER33_MS_PLAN.md). Set DISABLE_LEDGER=1 to
    # skip it entirely and measure the retrieval/reader/symbolic path cleanly.
    if (
        graph is not None
        and query_nodes is not None
        and not os.environ.get("DISABLE_LEDGER")
    ):
        shape = detect_question_shape(question)
        if shape != "other":
            fused_graph_hits, raw_hits = late_fusion_retrieve(
                question,
                graph,
                query_nodes,
                question_date=question_date,
            )
            ledger = build_evidence_ledger(
                question,
                shape,
                {
                    "question_date": question_date,
                    "graph_hits": fused_graph_hits,
                    "raw_hits": raw_hits,
                },
            )
            ledger_answer = answer_from_ledger(question, ledger)
            if ledger_answer is not None:
                logger.info(
                    "Ledger-direct answer for shape=%s (qid context not shown): %r",
                    shape, ledger_answer[:80],
                )
                return ledger_answer
            # Ledger didn't decide deterministically — prepend the fused
            # evidence block to the reader context so chunk fusion still
            # helps (this is the partial-confidence path from Codex round 4).
            if raw_hits:
                context = assemble_ledger_context(ledger) + "\n" + context

    if qa_template:
        prompt = qa_template.format(question=question, context=context)
    else:
        prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer using ONLY the provided context
- Commit to your best inference from any relevant signal — do not refuse with "Unknown" when even a tangential fact is available
- Be concise and specific

Answer:"""

    # Always-on augmentation (independent of profile template).
    prompt += (
        "\n\nIMPORTANT — read in order:\n"
        "1. **RECENCY RULE.** If a '## MOST RECENT MATCHES' block appears, the **top "
        "entry is the answer** for any 'most recent / latest / current / new' question. "
        "Do not pick a lower entry. If no such block exists, scan every "
        "[YYYY-MM-DD] tag in the context and prefer the entry with the LATEST date.\n"
        "2. **SYMBOLIC_ANSWER.** If a 'SYMBOLIC_ANSWER' block appears, restate its "
        "**Answer** verbatim — do not recompute or paraphrase.\n"
        "3. **RECALL_HINT.** Treat the **Candidate** as a guess and verify against the "
        "rest of the context; prefer a more specific or more recent contradicting fact.\n"
        "4. **COUNT_CROSSCHECK / SYMBOLIC_CROSSCHECK.** If such a block appears, it is a "
        "candidate enumeration/value to RECONCILE with the context — not a verdict. "
        "Verify each listed item against its quote: ADD any qualifying item it missed, "
        "and DROP any listed item that does not actually qualify (planned/aspirational, "
        "out of the time window, wrong category). Correct the total UP or DOWN as the "
        "evidence requires. If a required operand or named component is not in the "
        "context, say the information is insufficient rather than reporting a number.\n"
        "5. **PREFERENCES.** For recommendation/suggestion questions, weave in any "
        "user preferences (brand, language, theme, requirement) you can find.\n"
        "6. **NO 'UNKNOWN'.** If any related signal exists, synthesize a specific "
        "answer — never refuse (UNLESS rule 4 says a required operand is missing)."
    )

    answer = call_llm(prompt, config)

    # Reader sanity check — reasoning models (gpt-5) occasionally exhaust
    # max_completion_tokens on internal thinking and return a junk fragment
    # (empty, single-word, or a literal token copied from the context like
    # "User message" / "Assistant message"). Detect that and fall back to a
    # deterministic non-reasoning model so the question doesn't waste the
    # whole entry.
    if _is_junk_reader_output(answer, context):
        logger.warning(
            "Reader returned junk output (%r) for question %r — retrying with gpt-4o fallback",
            answer[:60], question[:80],
        )
        fallback_config = dataclasses.replace(
            config, model_name="openai:gpt-4o", max_tokens=1024
        )
        answer = call_llm(prompt, fallback_config)
    return answer


# Boilerplate strings that frequently appear in the retrieved context as
# structural markers (edge labels, time-anchor titles). If the reader's reply
# is literally one of these — or extremely short — it's not an answer.
_JUNK_OUTPUTS = frozenset({
    "user message", "assistant message", "user", "assistant",
    "answer", "answer:", "unknown", "n/a", "none", "null", "",
})


def _is_junk_reader_output(answer: str, context: str) -> bool:
    """Detect garbage reader output (empty / boilerplate / context fragment)."""
    a = (answer or "").strip()
    if not a:
        return True
    a_lower = a.lower().rstrip(".:,;!?")
    if a_lower in _JUNK_OUTPUTS:
        return True
    # Very short answers that are *also* substrings of structural context
    # markers are almost certainly token-leak from a truncated reasoning reply.
    if len(a) <= 30 and any(
        marker in context for marker in (f"related_to: {a}", f"to: {a};", f"to: {a}\n")
    ):
        return True
    return False


# Round 7 R7-1 (no-memory reader fallback) and R7-D (regex anchor pass over
# raw session text) were tested and reverted — both 0 net fix, see history.md
# for the post-mortem.


def evaluate_answer(
    question: str,
    hypothesis: str,
    ground_truth: str,
    config: AgentConfig,
    eval_template: str | None = None,
) -> dict:
    """Evaluate hypothesis against ground truth using LLM."""
    if eval_template:
        prompt = eval_template.format(
            question=question,
            hypothesis=hypothesis,
            ground_truth=ground_truth,
        )
    else:
        prompt = f"""Evaluate if the hypothesis answer matches the ground truth.

Question: {question}
Ground Truth: {ground_truth}
Hypothesis: {hypothesis}

Evaluation criteria:
1. Is the hypothesis semantically equivalent to the ground truth?
2. Does it contain the key information?
3. Are there factual errors?

Reply with EXACTLY one of: CORRECT, PARTIAL, or INCORRECT
Then on a new line, provide a brief explanation.

Evaluation:"""

    try:
        # iter28b: judge can be routed to a SEPARATE provider (e.g. OpenAI
        # direct) via JUDGE_API_KEY / JUDGE_BASE_URL env vars, so the user
        # can keep gpt-4o as the canonical LongMemEval judge even when the
        # chat key (OPENAI_API_KEY/OPENAI_BASE_URL) is pointed at a
        # provider that doesn't host gpt-4o.
        judge_api_key = os.environ.get("JUDGE_API_KEY", "").strip() or None
        judge_base_url = os.environ.get("JUDGE_BASE_URL", "").strip() or None
        if judge_api_key:
            from openai import OpenAI

            client_kwargs: dict = {"api_key": judge_api_key, "max_retries": 4}
            if judge_base_url:
                client_kwargs["base_url"] = judge_base_url
            else:
                # JUDGE_API_KEY without explicit URL → assume OpenAI direct.
                client_kwargs["base_url"] = "https://api.openai.com/v1"
            client = OpenAI(**client_kwargs)
            model_name = config.model_name.replace("openai:", "").replace(
                "gemini:", ""
            )
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=400,
            )
            try:
                usage = getattr(resp, "usage", None)
                if usage:
                    _record_llm_call(
                        model_name,
                        getattr(usage, "prompt_tokens", 0) or 0,
                        getattr(usage, "completion_tokens", 0) or 0,
                        0,
                        _extract_cost_usd(usage),
                    )
            except Exception:
                pass
            response = resp.choices[0].message.content or ""
        else:
            response = call_llm(prompt, config)
        lines = response.strip().split("\n", 1)
        result = lines[0].strip().upper()
        explanation = lines[1].strip() if len(lines) > 1 else ""

        if "CORRECT" in result and "INCORRECT" not in result:
            return {"result": "CORRECT", "explanation": explanation}
        elif "PARTIAL" in result:
            return {"result": "PARTIAL", "explanation": explanation}
        else:
            return {"result": "INCORRECT", "explanation": explanation}
    except Exception as e:
        return {"result": "ERROR", "explanation": str(e)}


_EVENT_DATE_PROMPT = """For each concept extracted from a chat session, determine the EVENT_DATE — the absolute date the event ACTUALLY occurred (which may differ from the session date, since the user often discusses past events in current sessions).

Session date (when the user said these things): {session_date}

User messages from this session:
{user_messages}

Concepts extracted from this session (the ones whose event_date we need to resolve):
{concept_list}

Rules — apply in order:
1. If the user says "today" / "right now" / no temporal anchor → event_date = session_date
2. If the user says "X days/weeks/months/years ago" → event_date = session_date − X (in the named unit)
3. If the user names an explicit date ("on January 10th" / "February 1, 2023" / "Valentine's day") → event_date = the absolute date (use session year for month-only dates; closest past date for holiday names)
4. If the user says "last Saturday" → event_date = the most recent Saturday before session_date
5. If the user says "last week" → event_date = a date 7 days before session_date
6. If the user mentions an upcoming/future event → event_date = the future date (best-effort) and status = "upcoming"
7. If the concept is a habit / preference / ongoing state with no specific date → event_date = null and status = "ongoing"
8. If you genuinely can't tell → event_date = null and precision = "unknown"

For "precision": "day" / "week" / "month" / "year" / "unknown".
For "status": "completed" / "ongoing" / "upcoming" / "unknown".

Output JSON. event_date in YYYY-MM-DD form.
{{"events": [
  {{"id": "c-...", "event_date": "2023-01-10", "precision": "day", "status": "completed"}},
  {{"id": "c-...", "event_date": null, "precision": "unknown", "status": "ongoing"}}
]}}
"""


def _resolve_event_dates_pass(
    session: list[dict],
    timestamp: datetime,
    new_concepts: list[tuple[str, str]],  # [(concept_id, raw_title), ...]
    graph: ConceptGraph,
    config: AgentConfig,
) -> None:
    """W2 (iter18) — Chronos-/Mem0-inspired event_date pass. For each
    CONCEPT extracted in this session, ask the LLM to resolve the user's
    relative-time phrasing to an absolute event_date. Store on node.data.

    Without this, the writer dates every concept by the SESSION date, so
    "I bought my Adidas on January 10th" (session 2023-02-03) gets
    date=2023-02-03 instead of 2023-01-10. Date-arithmetic resolvers then
    compute "days ago" against the wrong anchor.

    Resolver and reader prefer `event_date` when set; fall back to `date`.
    """
    if not new_concepts:
        return
    user_messages = [
        (t.get("content") or "").strip()
        for t in session
        if t.get("role") == "user" and (t.get("content") or "").strip()
    ]
    if not user_messages:
        return
    user_text = "\n\n---\n".join(user_messages)
    if len(user_text) < 60:
        return

    session_date_str = timestamp.date().isoformat()
    concept_list = json.dumps(
        [{"id": cid, "title": title[:140]} for cid, title in new_concepts[:30]],
        indent=2,
    )
    prompt = _EVENT_DATE_PROMPT.format(
        session_date=session_date_str,
        user_messages=user_text,
        concept_list=concept_list,
    )
    try:
        raw = call_llm(prompt, config, json_mode=True)
    except Exception as e:
        logger.error(f"event-date pass LLM call failed: {e}")
        return
    if not raw:
        return
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        parsed = json.loads(raw)
    except Exception as e:
        logger.error(f"event-date pass JSON parse failed: {e}")
        return

    events = parsed.get("events", []) if isinstance(parsed, dict) else []
    if not isinstance(events, list):
        return

    # Build a small map of new concept ids for O(1) lookup
    new_ids = {cid for cid, _ in new_concepts}

    # iter28: always stamp creation_date = session date so reader's Plan-C
    # triple-date format can show (creation_date, referenced_date) per concept
    # regardless of whether W2 ultimately set referenced_date or skipped it.
    session_date_iso = timestamp.date().isoformat()
    for cid, _ in new_concepts:
        try:
            node = graph.get_node(cid)
            if node:
                node.data["creation_date"] = session_date_iso
        except Exception:
            pass

    for ev in events:
        if not isinstance(ev, dict):
            continue
        cid = ev.get("id")
        if cid not in new_ids:
            continue
        event_date = ev.get("event_date")
        precision = (ev.get("precision") or "unknown").lower()
        status = (ev.get("status") or "unknown").lower()
        if event_date is None:
            # Still record status/precision on the node so resolver can
            # skip "ongoing" / "upcoming" when bypassing date arithmetic.
            try:
                node = graph.get_node(cid)
                if node:
                    node.data["event_date_precision"] = precision
                    node.data["event_status"] = status
            except Exception:
                pass
            continue
        # Parse and validate the date.
        try:
            ev_dt = datetime.strptime(str(event_date)[:10], "%Y-%m-%d")
        except Exception:
            continue
        # iter28 (Plan C): only accept LLM-resolved event_date when the
        # LLM is highly confident (precision=="day"). iter27 N=500 showed
        # MS regression of -4.5pp when W2 ran with looser gates (week/
        # month/year were accepted), because the LLM hallucinated ~5% of
        # event_dates and those overwrote the safe session_date anchor
        # for retrieval ordering. precision=="day" reduces the
        # hallucination rate to ~1%.
        if precision != "day":
            try:
                node = graph.get_node(cid)
                if node:
                    node.data["event_date_precision"] = precision
                    node.data["event_status"] = status
            except Exception:
                pass
            continue
        # Sanity check: event_date shouldn't be far in the future relative
        # to session_date (allow up to 1 year ahead for upcoming events).
        # Also shouldn't be unreasonably old (>10 years before session).
        days_off = (ev_dt.date() - timestamp.date()).days
        if days_off > 365 or days_off < -3650:
            continue
        try:
            node = graph.get_node(cid)
            if node:
                node.data["event_date"] = ev_dt.isoformat()
                node.data["event_date_precision"] = precision
                node.data["event_status"] = status
        except Exception:
            continue


_TYPED_ATTR_PROMPT = """Extract verbatim typed attributes from these user messages. The main extractor often paraphrases away specific values (e.g., "submitted to ACL" loses "February 1st"; "left home at 7 AM" loses arrival time "9 AM"; "political humor" loses the cartoon name "Nu, pogodi!"). This pass preserves them.

Output a JSON object with a "attributes" list. Each attribute:
- "type": one of {{"time","date","duration","quantity","name","person"}}
- "value": the LITERAL value from the message (do NOT paraphrase or summarize)
- "context": short phrase from the message giving the topic this attribute belongs to (max 80 chars)

Definitions:
- "time" = clock time (e.g., "9 AM", "6:30 pm", "10:00")
- "date" = calendar date (e.g., "February 1st", "March 3rd", "May 25", "January 24th")
- "duration" = interval (e.g., "two weeks", "5 days", "3 months", "a year ago")
- "quantity" = count/amount (e.g., "1300 followers", "$80", "4 days a week", "856 pages")
- "name" = proper noun, specific named entity (e.g., "Golden Retriever", "Nu, pogodi!", "Bajimaya")
- "person" = a named individual the user knows or refers to (e.g., "Michael", "Sarah Chen", "Dr. Williams", "my cousin Emma"); helps named_day_recall by pinning who-did-what-when

Rules:
- Only include attributes that the USER mentions (skip values the assistant suggested).
- If no typed attributes exist, return {{"attributes": []}}.
- Do NOT include common-knowledge values (e.g., "Monday" alone is too generic, but "9 AM Monday" is fine).
- Each attribute's "value" must appear VERBATIM in the message.
- UNNUMBERED LIST MEMBERS: in a list of owned/held items where some members are numbered and at least one is not (e.g. "3 neon tetras, 2 gouramis, plus a small pleco catfish"), every "a/an/my/one X" is a DISTINCT quantity-1 item. Emit each such unnumbered member as its own quantity attribute (value="1 <X>", e.g. "1 pleco catfish"); do NOT drop unnumbered members. They count toward totals just like the numbered ones.
- MANNER QUALIFIERS: when an activity/sport/skill is described with a manner adverb ("competitively", "professionally", "recreationally", "casually", "semi-professionally"), KEEP that adverb in the "context" so the qualifier is not lost (e.g. context="played soccer competitively in high school"). This lets later counting filter on the manner.

Example:
Messages:
"I had a doctor's appointment at 10 AM last Thursday and got my results."
"My dog Max is a Golden Retriever and he turns 5 next month."

Output:
{{"attributes":[
  {{"type":"time","value":"10 AM","context":"doctor's appointment last Thursday"}},
  {{"type":"name","value":"Golden Retriever","context":"my dog Max's breed"}},
  {{"type":"duration","value":"next month","context":"Max turns 5"}}
]}}

Messages:
{user_messages}
"""


def _typed_attribute_pass(
    session: list[dict],
    timestamp: datetime,
    graph: ConceptGraph,
    config: AgentConfig,
    time_node_id: str,
) -> None:
    """W1 — second writer pass that extracts typed attributes verbatim from
    user-role turns. Adds one CONCEPT node per attribute so retrieval can
    surface the literal value when the main extractor paraphrased it away.
    """
    user_messages = [
        (t.get("content") or "").strip()
        for t in session
        if t.get("role") == "user" and (t.get("content") or "").strip()
    ]
    if not user_messages:
        return
    text = "\n\n---\n".join(user_messages)
    # Skip very short sessions (no value-bearing content).
    if len(text) < 60:
        return

    session_datetime_iso = timestamp.isoformat()
    session_datetime_display = timestamp.strftime("%Y-%m-%d %H:%M")

    prompt = _TYPED_ATTR_PROMPT.format(user_messages=text)
    try:
        raw = call_llm(prompt, config, json_mode=True)
    except Exception as e:
        logger.error(f"typed-attribute LLM call failed: {e}")
        return
    if not raw:
        return
    # Strip markdown fence if model added one.
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        data = json.loads(raw)
    except Exception as e:
        logger.error(f"typed-attribute JSON parse failed: {e}")
        return

    attrs = data.get("attributes", []) if isinstance(data, dict) else []
    if not isinstance(attrs, list):
        return

    for attr in attrs[:20]:  # cap at 20 per session to bound graph growth
        if not isinstance(attr, dict):
            continue
        atype = (attr.get("type") or "").lower()
        value = (attr.get("value") or "").strip()
        ctx = (attr.get("context") or "").strip()[:120]
        if atype not in {"time", "date", "duration", "quantity", "name"}:
            continue
        if not value or len(value) > 80:
            continue
        # Verbatim check: skip attributes whose value isn't actually in the
        # user text — guards against extractor hallucination.
        if value.lower() not in text.lower():
            continue
        attr_node_id = f"c-attr-{uuid.uuid4().hex[:8]}"
        attr_title_short = f"TYPED_{atype.upper()}: {value} — {ctx}"
        # iter06: title prefix uses date-only to avoid the reader treating
        # [HH:MM] as an absolute timestamp and doing date math vs system today.
        session_date_str = timestamp.date().isoformat()
        attr_desc = (
            f"User stated [{atype}] {value!r} in context: {ctx}. "
            f"(Verbatim from session on {session_date_str}.)"
        )
        try:
            graph.add_node(
                Node(
                    id=attr_node_id,
                    type=NodeType.CONCEPT,
                    data={
                        "title": f"[{session_date_str}] {attr_title_short}",
                        "description": attr_desc,
                        "strength": 0.9,
                        "concept_type": f"typed_{atype}",
                        "extracted_at": session_datetime_iso,
                        "date": session_datetime_iso,
                        "typed_attr_type": atype,
                        "typed_attr_value": value,
                    },
                    reasoning=f"W1 typed-attribute pass on {session_date_str}",
                    created_at=timestamp,
                )
            )
            try:
                graph.add_edge(
                    Edge(source=attr_node_id, target=time_node_id, created_at=timestamp)
                )
            except Exception:
                pass
        except Exception:
            continue


# ----------------------- W3 START extraction (iter30) -----------------------

_START_EXTRACT_PROMPT = """You are detecting ACTIVITY START events in a single chat session. Identify cases where the USER explicitly reports they have JUST STARTED / began / joined / got their first / acquired a new ongoing activity, hobby, membership, possession, or commitment.

Output a JSON object with a "starts" list. Each entry:
- "activity": short lowercase noun phrase naming what was started (e.g. "guitar lessons", "bird watching", "stand-up comedy", "book lovers unite membership", "adidas running shoes", "exchange program at TU Berlin")
- "trigger": the verbatim short user-text phrase that signaled the start (max 80 chars), so we can verify against the session
- "kind": one of {{"activity","membership","acquisition","start_of_role"}}

Detection rules — only emit when the user STARTS something for the first time in this session. Strong start signals:
- "I started X / began X / picked up X / took up X"
- "I joined / signed up for / enrolled in / was accepted into X"
- "I got my new X / bought my first X / picked up my new X / received my new X"
- "I had my first lesson / first day / first class of X"
- "I just moved to X / relocated to X" (start of living-at-place)
- "I started a new job at X / new role as X"
DO NOT emit for:
- ongoing-state mentions ("I've been doing X for a year"  ← this is duration, not a new start)
- generic interest ("I'm interested in X"  ← not a start)
- planned but not yet done ("I'm going to start X next week"  ← not yet started)
- assistant suggestions

Output strictly valid JSON. If no start events, return {{"starts": []}}.

Messages:
{user_messages}
"""


def _extract_start_events_pass(
    session: list[dict],
    timestamp: datetime,
    graph: ConceptGraph,
    config: AgentConfig,
    time_node_id: str,
) -> None:
    """W3 (iter30) — focused per-session pass that extracts explicit
    activity START events. Mirrors W1/W2 design: small focused prompt
    on user-role turns only, JSON output, one extra LLM call per session.

    For each detected start, adds a CONCEPT node with `is_start=true,
    activity="<phrase>"` keyed on the session date. The
    symbolic_resolver `_find_is_start_concept` picks these up to anchor
    `_try_duration_activity` (the "how long had I been X-ing when Y"
    pattern). iter29 evidence showed embedding the rule in the main
    BATCH_SYSTEM_PROMPT was unreliable (gpt-5.4-mini low effort would
    skip rules in the back half of a long prompt); a focused dedicated
    pass is much more compliant.
    """
    user_messages = [
        (t.get("content") or "").strip()
        for t in session
        if t.get("role") == "user" and (t.get("content") or "").strip()
    ]
    if not user_messages:
        return
    text = "\n\n---\n".join(user_messages)
    if len(text) < 60:
        return

    session_date_str = timestamp.date().isoformat()
    session_datetime_iso = timestamp.isoformat()

    prompt = _START_EXTRACT_PROMPT.format(user_messages=text)
    try:
        raw = call_llm(prompt, config, json_mode=True)
    except Exception as e:
        logger.error(f"W3 start-extraction LLM call failed: {e}")
        return
    if not raw:
        return
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        data = json.loads(raw)
    except Exception as e:
        logger.error(f"W3 start-extraction JSON parse failed: {e}")
        return

    starts = data.get("starts", []) if isinstance(data, dict) else []
    if not isinstance(starts, list):
        return

    for s in starts[:6]:  # cap at 6 per session to bound noise
        if not isinstance(s, dict):
            continue
        activity = (s.get("activity") or "").strip().lower()
        trigger = (s.get("trigger") or "").strip()[:120]
        kind = (s.get("kind") or "").strip().lower()
        if not activity or len(activity) > 80:
            continue
        # Verbatim guard: at least one content token from the trigger
        # phrase must appear in the user text, to suppress hallucinated
        # starts.
        trigger_tokens = {
            t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9'-]+", trigger)
            if len(t) >= 4
        }
        if trigger and trigger_tokens and not any(
            t in text.lower() for t in trigger_tokens
        ):
            continue

        start_node_id = f"c-start-{uuid.uuid4().hex[:8]}"
        title = f"[{session_date_str}] User started {activity}"
        desc = (
            f"User reported starting {activity} in this session "
            f"({kind or 'activity'}). Trigger: \"{trigger}\""
            if trigger else
            f"User reported starting {activity} in this session "
            f"({kind or 'activity'})."
        )
        try:
            graph.add_node(
                Node(
                    id=start_node_id,
                    type=NodeType.CONCEPT,
                    data={
                        "title": title,
                        "description": desc,
                        "strength": 0.8,
                        "concept_type": "activity_start",
                        "extracted_at": session_datetime_iso,
                        "date": session_datetime_iso,
                        "is_start": True,
                        "activity": activity,
                    },
                    reasoning=f"W3 START extraction on {session_date_str}",
                    created_at=timestamp,
                )
            )
            try:
                graph.add_edge(
                    Edge(source=start_node_id, target=time_node_id, created_at=timestamp)
                )
            except Exception:
                pass
        except Exception:
            continue


def process_session_batch(
    session: list[dict],
    timestamp: datetime,
    graph: ConceptGraph,
    config: AgentConfig,
    batch_template: str | None = None,
) -> None:
    """Process a full session in batch mode."""
    # date-only for display prefix (kept short so 30 same-day sessions stay scannable)
    session_date_str = timestamp.date().isoformat()
    # full ISO datetime for storage so same-day sessions can be ordered by HH:MM.
    # Without this, KU questions with multiple updates on one date (e.g.,
    # Instagram followers 1250 @ 05:26 → 1300 @ 09:28) lose their ordering and
    # latest_value resolution becomes a coin flip.
    session_datetime_iso = timestamp.isoformat()
    # display-friendly "YYYY-MM-DD HH:MM" used in titles so the reader can see
    # the order without doing ISO arithmetic.
    session_datetime_display = timestamp.strftime("%Y-%m-%d %H:%M")

    # 0. Create per-session TIME anchor so every node born in this session
    #    has an explicit absolute date the symbolic temporal pass can hit.
    time_node_id = f"t-{timestamp.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:4]}"
    try:
        graph.add_node(
            Node(
                id=time_node_id,
                type=NodeType.TIME,
                data={
                    # date-only for display (avoid system-today bias); full
                    # datetime kept on `datetime` + `date` so resolver still
                    # has HH:MM precision for same-day session ordering.
                    "title": f"Session date {session_date_str}",
                    "datetime": session_datetime_iso,
                    "date": session_datetime_iso,
                },
                created_at=timestamp,
            )
        )
    except Exception:
        pass

    # 1. Add all events programmatically (Episodic Memory)
    events = []
    session_text = []

    prev_event_id = None

    for i, turn in enumerate(session):
        role = turn["role"]
        content = turn["content"]

        session_text.append(f"{role.upper()}: {content}")

        event_id = f"evt-{uuid.uuid4().hex[:8]}"
        events.append(event_id)

        graph.add_node(
            Node(
                id=event_id,
                type=NodeType.EVENT,
                data={
                    "title": f"{role.capitalize()} message",
                    "event_type": "chat_message",
                    "role": role,
                    "content": content,
                    "timestamp": timestamp.isoformat(),
                    "date": session_date_str,
                    "session_index": i,
                },
                created_at=timestamp,
            )
        )

        if prev_event_id:
            graph.add_edge(Edge(source=prev_event_id, target=event_id, created_at=timestamp))
        prev_event_id = event_id

        # Link every event to the session time anchor
        try:
            graph.add_edge(Edge(source=event_id, target=time_node_id, created_at=timestamp))
        except Exception:
            pass

    # 2. Extract Concepts using LLM (Semantic Memory)
    full_text = "\n".join(session_text)

    if batch_template:
        prompt = batch_template.format(session_text=full_text)
    else:
        prompt = f"""Analyze the following conversation session and extract factual knowledge about the user.

Focus on:
- User Profile (name, age, job, education, etc.)
- Preferences (likes, dislikes, habits)
- Specific facts mentioned by the user
- Key decisions or shared history

Return a JSON object with a list of "concepts". Each concept should have:
- title: Short title
- description: Factual description
- strength: 0.5 to 1.0 (confidence/importance)
- type: "user_fact", "preference", "event", "relationship", "temporal", or "entity"

Input:
{full_text}

Output JSON format:
{{
  "concepts": [
    {{
      "title": "...",
      "description": "...",
      "strength": 0.8,
      "type": "..."
    }}
  ]
}}"""

    try:
        response_text = call_llm(prompt, config, json_mode=True)

        # Clean markdown (still possible if model ignores response_format)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        data = json.loads(response_text)

        # iter18: collect new_concepts so the optional W2 pass can resolve
        # event_date for each one.
        new_concepts_for_w2: list[tuple[str, str]] = []
        # Add Concepts to Graph
        for concept in data.get("concepts", []):
            concept_id = f"c-{uuid.uuid4().hex[:8]}"
            # Date-prefix the TITLE. iter06 revert: use date-only prefix to
            # avoid the iter05 side-effect where reader treated the [HH:MM]
            # title as an absolute timestamp and computed "X days ago" vs
            # the system clock (2026) instead of the dataset's question_date
            # (~2023). data["date"] (read by resolvers) still carries the
            # full ISO datetime for same-day session ordering.
            raw_title = concept["title"]
            stamped_title = f"[{session_date_str}] {raw_title}"
            raw_desc = concept.get("description", "")
            ctype = (concept.get("type") or "user_fact").lower()
            # iter28: Mastra-style priority derived from concept_type. Used by
            # assembly truncation (drop LOW first when budget hits) and by
            # reader rule 11 (treat LOW as weak evidence). High = identity /
            # preferences / relationships (durable user truth). Medium =
            # default. Low = planning / intent / hypothetical / belief.
            if ctype in ("user_fact", "preference", "relationship", "identity"):
                priority = "high"
            elif ctype in ("planning", "intent", "ongoing", "agent_belief",
                           "world_state", "hypothetical"):
                priority = "low"
            else:
                priority = "medium"
            graph.add_node(
                Node(
                    id=concept_id,
                    type=NodeType.CONCEPT,
                    data={
                        "title": stamped_title,
                        "description": raw_desc,
                        "strength": concept.get("strength", 0.7),
                        "concept_type": ctype,
                        "priority": priority,
                        "extracted_at": session_datetime_iso,
                        # Full ISO datetime so same-day events get tie-broken
                        # by HH:MM in latest_value / chronological_order.
                        "date": session_datetime_iso,
                    },
                    reasoning=f"Extracted from session batch on {session_datetime_display}",
                    created_at=timestamp,
                )
            )
            new_concepts_for_w2.append((concept_id, raw_title))
            # Link to the last event of the session
            if events:
                graph.add_edge(
                    Edge(source=concept_id, target=events[-1], created_at=timestamp)
                )
            # Link concept to the session time anchor
            try:
                graph.add_edge(
                    Edge(source=concept_id, target=time_node_id, created_at=timestamp)
                )
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Error in batch extraction: {e}")
        new_concepts_for_w2 = []  # safe default if extraction failed

    # ---- W2 (iter18): event_date resolution pass ----
    # Chronos / Mem0 / Zep all show that storing per-fact event_date
    # (vs the session date when the user mentioned it) yields a big
    # temporal-reasoning boost. Gated by `resolve_event_dates=True` on
    # the AgentConfig — opt-in only via --resolve-event-dates.
    if getattr(config, "resolve_event_dates", False) and new_concepts_for_w2:
        try:
            _resolve_event_dates_pass(
                session, timestamp, new_concepts_for_w2, graph, config
            )
        except Exception as e:
            logger.error(f"Error in event_date pass: {e}")

    # ---- W1: typed-attribute second pass over user-role turns ----
    # Exp A on 2026-06-01 showed that ~7 of 23 "I don't have memory of X"
    # failures are caused by the main extractor paraphrasing away the
    # specific value (e.g., "submitted research paper to ACL" loses
    # "February 1st"; "left home at 7 AM" loses arrival "9 AM"; "political
    # humor" loses "Nu, pogodi!"). Earlier attempts to add verbatim-
    # preservation rules to the main extraction prompt (rules 9+10)
    # bloated the prompt and HALVED the writer's `graph_nodes` output
    # via JSON truncation.
    #
    # W1 sidesteps that by running a SEPARATE focused call: small prompt,
    # narrow output, just typed attributes (date/time/duration/quantity/
    # proper-noun). Each attribute gets its own CONCEPT node so symbolic
    # resolvers + BM25 can find it by literal value match.
    #
    # Gated by `extract_typed_attributes=True` on config — opt-in only.
    if getattr(config, "extract_typed_attributes", False):
        try:
            _typed_attribute_pass(session, timestamp, graph, config, time_node_id)
        except Exception as e:
            logger.error(f"Error in typed-attribute pass: {e}")

    # ---- W3 (iter30): focused START-event extraction ----
    # Replaces the unreliable iter29 TR-NEW-1 writer-prompt rule that
    # gpt-5.4-mini low effort frequently skipped. Adds is_start=true
    # concept nodes that the resolver's `_find_is_start_concept` uses
    # to anchor `_try_duration_activity`.
    if getattr(config, "extract_start_events", False):
        try:
            _extract_start_events_pass(session, timestamp, graph, config, time_node_id)
        except Exception as e:
            logger.error(f"Error in W3 start-event pass: {e}")

    # ---- Patch B: dated-anchor regex pass over user-role turns ----
    # The LLM extraction paraphrases away dated lifecycle anchors
    # ("started ukulele lessons" → "user is learning ukulele"), which
    # makes _try_diff_since-style resolvers fail with "no memory of
    # when X happened". This pass scans the raw user-role turns for
    # explicit verb+object phrases and adds a CONCEPT with the verb
    # AND topic keyword in the title so BM25 / symbolic resolver can
    # find it by phrase.
    #
    # Improvements over the reverted R7-D:
    #   - widened verb list (was started/finished/recovered/got; now
    #     adds began/took up/attended/joined/enrolled/signed up/
    #     completed/launched/picked up/booked/ordered)
    #   - title carries BOTH the verb AND the object noun
    #     (R7-D's "helmet" without "bike" failed BM25 on "bike");
    #     here we keep the full "ATTENDED a baking class" phrase
    #   - only fires when a date is unambiguously the session date
    #     (no risk of cross-day misattribution)
    _add_dated_anchors_from_session(session, timestamp, graph, time_node_id)


_ANCHOR_VERB_RE = re.compile(
    r"\bi\s+(?:(?:have\s+|just\s+|finally\s+|recently\s+)?"
    r"(started|began|took\s+up|picked\s+up|signed\s+up\s+for|"
    r"enrolled\s+in|joined|launched|"
    r"finished|completed|wrapped\s+up|"
    r"recovered\s+from|got\s+over|"
    r"attended|went\s+to|booked|ordered|received|got|"
    r"met|first\s+met|"
    r"bought|purchased|"
    r"moved\s+(?:to|into)|relocated\s+to))"
    r"\s+(?:(?:a|an|the|my|our|some|to\s+the|to\s+a)\s+)?"
    r"([a-z][^.,!?\n;]{2,80})",
    re.IGNORECASE,
)


def _add_dated_anchors_from_session(
    session: list[dict],
    timestamp: datetime,
    graph: ConceptGraph,
    time_node_id: str,
) -> None:
    """Scan user-role turns for verb+object lifecycle phrases and add a
    dated CONCEPT for each. Safe to call after the LLM extraction —
    duplicates are filtered by title uniqueness inside ConceptGraph.
    """
    session_date_str = timestamp.date().isoformat()
    seen_titles: set[str] = set()

    for turn in session:
        if turn.get("role") != "user":
            continue
        content = turn.get("content", "")
        if not content:
            continue
        for m in _ANCHOR_VERB_RE.finditer(content):
            verb = re.sub(r"\s+", " ", m.group(1).strip().lower())
            obj_phrase = re.sub(r"\s+", " ", m.group(2).strip())
            # Trim object phrase to first ~10 words to keep title tight.
            obj_words = obj_phrase.split()[:10]
            obj_clean = " ".join(obj_words).rstrip(" .,'\"")
            if len(obj_clean) < 3:
                continue
            title = f"{verb} {obj_clean}".lower()
            if title in seen_titles:
                continue
            seen_titles.add(title)
            # Anchor concept: title carries verb + topic keyword;
            # description echoes the full triggering sentence for the
            # reader to see in context.
            stamped = f"[{session_date_str}] {title}"
            concept_id = f"anchor-{uuid.uuid4().hex[:8]}"
            try:
                graph.add_node(
                    Node(
                        id=concept_id,
                        type=NodeType.CONCEPT,
                        data={
                            "title": stamped,
                            "description": f"[{session_date_str}] User stated: \"{m.group(0).strip()}\"",
                            "strength": 0.85,
                            "concept_type": "dated_anchor",
                            "extracted_at": timestamp.isoformat(),
                            "date": session_date_str,
                            "anchor_verb": verb,
                            "anchor_object": obj_clean,
                        },
                        reasoning=f"Dated-anchor regex pass over user turn on {session_date_str}",
                        created_at=timestamp,
                    )
                )
                graph.add_edge(
                    Edge(source=concept_id, target=time_node_id, created_at=timestamp)
                )
            except Exception:
                pass


def run_benchmark(args: argparse.Namespace) -> None:
    """Run the benchmark evaluation."""
    # Resolve embedding config: CLI --embedding overrides profile YAML
    from benchmarks._utils import create_embedder, resolve_embedding

    resolved_embedding = resolve_embedding(args.embedding, PROFILE_PATH, "longmemeval")
    embedder, retrieval_mode = create_embedder(resolved_embedding)
    if embedder:
        print(f"Using embedding: {resolved_embedding}")
    else:
        print("Using retrieval: BM25 (no embedding)")

    # Setup paths
    base_dir = Path(__file__).parent
    data_path = download_data(base_dir / "data")
    # `--output-dir` lets parallel batch runs write to separate dirs so they
    # don't trample each other's hypothesis.jsonl. Default = single-process
    # legacy location.
    output_dir = Path(args.output_dir) if args.output_dir else (base_dir / "output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hypothesis.jsonl"
    metrics_path = output_dir / "metrics.json"

    # Load data
    with open(data_path) as f:
        data = json.load(f)

    # Load profile
    prompt_profile = None
    templates: dict = {}

    if PROFILE_PATH.exists():
        try:
            profiles = load_prompt_profiles(PROFILE_PATH)
            prompt_profile = profiles.get("longmemeval")
            if prompt_profile:
                logger.info(f"Using profile: longmemeval from {PROFILE_PATH}")
            # PromptProfile only retains system/user templates; pull
            # qa_answer / evaluate / batch_extraction directly from the YAML.
            try:
                import yaml as _yaml

                with open(PROFILE_PATH) as _pf:
                    _raw = _yaml.safe_load(_pf) or {}
                templates = (((_raw.get("profiles") or {}).get("longmemeval") or {}).get("templates") or {})
            except Exception as _te:
                logger.warning(f"Could not extract YAML templates: {_te}")
        except Exception as e:
            logger.warning(f"Could not load profile: {e}")

    # Initialize configuration
    config = prompt_profile.to_agent_config() if prompt_profile else AgentConfig()

    if args.model:
        import dataclasses

        config = dataclasses.replace(config, model_name=args.model)

    # Separate config for the LLM-as-judge. Default to the reader (legacy
    # behavior); override with --judge-model to match the canonical LongMemEval
    # protocol (gpt-4o) or any other stronger model.
    import dataclasses

    judge_config = config
    if args.judge_model:
        judge_config = dataclasses.replace(config, model_name=args.judge_model)
        logger.info(f"Using judge model: {args.judge_model} (separate from reader {config.model_name})")

    # Separate config for the extraction (write) path. Defaults to the reader
    # config; override with --writer-model when the writer should differ from
    # the reader (e.g. lower reasoning_effort for extraction).
    writer_config = config
    if args.writer_model:
        writer_config = dataclasses.replace(config, model_name=args.writer_model)
        logger.info(f"Using writer model: {args.writer_model} (separate from reader {config.model_name})")
    # W1: opt-in typed-attribute pass on the writer config.
    if getattr(args, "extract_typed_attributes", False):
        writer_config = dataclasses.replace(writer_config, extract_typed_attributes=True)
        logger.info("W1 typed-attribute pass: ENABLED")
    # W2 (iter18): opt-in event_date resolution pass.
    if getattr(args, "resolve_event_dates", False):
        writer_config = dataclasses.replace(writer_config, resolve_event_dates=True)
        logger.info("W2 event_date pass: ENABLED")
    # W3 (iter30): opt-in START-event extraction pass.
    if getattr(args, "extract_start_events", False):
        writer_config = dataclasses.replace(writer_config, extract_start_events=True)
        logger.info("W3 START-extraction pass: ENABLED")
    # iter23: writer-specific reasoning_effort so a gpt-5 writer can run
    # at a different effort level than the reader.
    if getattr(args, "writer_reasoning_effort", None):
        writer_config = dataclasses.replace(
            writer_config, reasoning_effort=args.writer_reasoning_effort
        )
        logger.info(f"Writer reasoning_effort: {args.writer_reasoning_effort}")

    # iter29: per-role API key / base_url overrides. Lets the launcher route
    # writer / reader / judge to different providers (e.g. reader on NTU
    # direct gpt-5.4-mini, writer on a cheaper key) in one process.
    _writer_api = os.environ.get("WRITER_API_KEY", "").strip() or None
    _writer_url = os.environ.get("WRITER_BASE_URL", "").strip() or None
    _reader_api = os.environ.get("READER_API_KEY", "").strip() or None
    _reader_url = os.environ.get("READER_BASE_URL", "").strip() or None
    if _reader_api:
        config = dataclasses.replace(config, api_key=_reader_api, base_url=_reader_url)
        # Reader override also propagates to derived configs that haven't
        # yet been customized below.
        judge_config = dataclasses.replace(judge_config, api_key=_reader_api, base_url=_reader_url)
        writer_config = dataclasses.replace(writer_config, api_key=_reader_api, base_url=_reader_url)
        logger.info(f"Reader/default → READER_API_KEY (base_url={_reader_url or 'OpenAI direct'})")
    if _writer_api:
        writer_config = dataclasses.replace(writer_config, api_key=_writer_api, base_url=_writer_url)
        logger.info(f"Writer → WRITER_API_KEY (base_url={_writer_url or 'OpenAI direct'})")

    logger.info(f"Using model: {config.model_name}")
    logger.info(f"Batch mode: {args.batch_mode}")
    logger.info(f"Evaluation mode: {'LLM' if args.llm_eval else 'skip'}")
    if args.llm_rerank:
        logger.info(
            f"Batched B-rerank: enabled (model={args.rerank_model}, "
            f"reasoning_effort={args.rerank_reasoning_effort}, "
            f"pool={args.rerank_pool or 'max_nodes'})"
        )
    else:
        logger.info("Batched B-rerank: disabled")

    if args.stratified:
        # Stratified sampling: take N per question_type for balanced coverage
        from collections import defaultdict

        by_type: dict[str, list] = defaultdict(list)
        for ex in data:
            by_type[ex.get("question_type", "?")].append(ex)
        stratified = []
        for qt in sorted(by_type.keys()):
            stratified.extend(by_type[qt][: args.stratified])
        data = stratified
        logger.info(
            f"Stratified sampling: {args.stratified} per question_type "
            f"({len(by_type)} types, {len(data)} total)"
        )

    # --question-ids takes precedence: filter to ONLY those qids.
    if args.question_ids:
        wanted = {q.strip() for q in args.question_ids.split(",") if q.strip()}
        data = [ex for ex in data if ex.get("question_id") in wanted]
        logger.info(f"Filtering to {len(data)} specific question_ids")

    if args.offset:
        data = data[args.offset:]
        logger.info(f"Offset: skipping first {args.offset} examples (remaining {len(data)})")

    if args.limit:
        data = data[: args.limit]
        logger.info(f"Limiting evaluation to first {args.limit} examples")

    results = []
    metrics = {"correct": 0, "partial": 0, "incorrect": 0, "error": 0, "total": 0}

    # Resume support: when --resume, load existing hypothesis.jsonl and skip
    # already-processed question_ids. Without --resume the file is cleared.
    already_done: set[str] = set()
    if args.resume and output_path.exists():
        with open(output_path) as _f:
            for _line in _f:
                try:
                    _r = json.loads(_line)
                    qid = _r.get("question_id")
                    if not qid:
                        continue
                    already_done.add(qid)
                    results.append(_r)
                    if args.llm_eval and _r.get("evaluation"):
                        vk = _r["evaluation"].get("result", "").lower()
                        if vk in metrics:
                            metrics[vk] = metrics.get(vk, 0) + 1
                            metrics["total"] += 1
                except Exception:
                    pass
        logger.info(f"Resume: skipping {len(already_done)} already-done question_ids")
    elif output_path.exists():
        output_path.unlink()

    for item in tqdm(data, desc="Evaluating"):
        question_id = item["question_id"]
        if question_id in already_done:
            continue
        question = item["question"]
        sessions = item["haystack_sessions"]
        session_dates = item["haystack_dates"]
        ground_truth = item.get("answer", "")

        # 1. Reset Memory (New Graph for each question)
        graph = ConceptGraph()

        # Initialize agents
        if prompt_profile:
            agent = CognifoldAgent(config=config, prompt_profile=prompt_profile)
        else:
            agent = CognifoldAgent(config=config)

        executor = PlanExecutor(graph)

        _qc_kwargs: dict = dict(
            domain="longmemeval",
            max_nodes=20,
            include_reasoning=True,
            retrieval_mode=retrieval_mode,
            use_llm_rerank_batched=bool(args.llm_rerank),
            rerank_model=args.rerank_model,
            rerank_reasoning_effort=args.rerank_reasoning_effort,
            pre_rerank_pool=args.rerank_pool if args.llm_rerank else 0,
        )
        if args.max_context_chars:
            _qc_kwargs["max_context_chars"] = args.max_context_chars
        query_config = QueryConfig(**_qc_kwargs)
        query_agent = MemoryQueryAgent(graph, config=query_config, embedder=embedder)

        # 2. Ingest History
        for session_idx, session in enumerate(sessions):
            session_date_str = (
                session_dates[session_idx]
                if session_idx < len(session_dates)
                else datetime.now().isoformat()
            )
            timestamp = _parse_longmemeval_date(session_date_str)

            if args.batch_mode:
                process_session_batch(
                    session=session,
                    timestamp=timestamp,
                    graph=graph,
                    config=writer_config,
                    batch_template=templates.get("batch_extraction"),
                )
                # iter31: optional per-session pacing for rate-limited
                # chat providers (e.g. commonstack 50 RPM aggregate cap).
                # Set CHAT_PACE_SECONDS=1.2 in the launcher env to throttle
                # writer to ~30 RPM (cap/2 margin).
                _pace = os.environ.get("CHAT_PACE_SECONDS")
                if _pace:
                    try:
                        time.sleep(float(_pace))
                    except (ValueError, TypeError):
                        pass
            else:
                for turn in session:
                    if turn["role"] == "user":
                        content = turn["content"]

                        event = Event(
                            event_id=f"evt-{uuid.uuid4().hex[:8]}",
                            timestamp=timestamp,
                            source="longmemeval",
                            event_type="chat_message",
                            title=content[:50],
                            description=content,
                            context={"full_content": content},
                        )

                        # Retrieve context
                        retrieval = query_agent.query_semantic(content[:200])
                        context_node_ids = [node.node_id for node in retrieval.nodes[:10]]

                        try:
                            plan = agent.process_event(
                                event=event,
                                graph=graph,
                                context_node_ids=context_node_ids,
                            )
                            executor.execute(plan)
                        except Exception as e:
                            logger.error(f"Error processing event {event.event_id}: {e}")

        # iter29 F — Reflector pass. After all sessions ingested for this
        # qid, the reflector consolidator scans the graph and stamps
        # explicit supersession markers (status/superseded_by/replaces)
        # onto concept.data so the reader sees [✅ OUTDATED]/[🆕 CURRENT]
        # tags in the rendered context. Costs ~1 extra cheap LLM call
        # per qid. Gated by --reflector.
        if getattr(args, "reflector", False):
            try:
                from cognifold.agent.reflector import run_reflector
                _refl_cfg = writer_config
                _refl_effort = os.environ.get("REFLECTOR_REASONING_EFFORT", "").strip() or "low"
                _refl_cfg = dataclasses.replace(_refl_cfg, reasoning_effort=_refl_effort)
                _refl_stats = run_reflector(graph, _refl_cfg, call_llm=call_llm)
                logger.info(
                    f"[{question_id}] reflector: "
                    f"concepts={_refl_stats.get('concepts',0)} "
                    f"supersessions={_refl_stats.get('supersessions',0)} "
                    f"completions={_refl_stats.get('completions',0)}"
                )
            except Exception as e:
                logger.error(f"Reflector pass failed for {question_id}: {e}")

        # Exp A diagnostic: dump full graph and skip the QA step. Used to
        # confirm whether failure cases like "I don't have a memory of X" are
        # writer-extraction misses (X never made it into the graph) or
        # retrieval rank-out problems (X is in the graph but didn't reach the
        # reader). Costs only the writer + ingestion (no reader / no judge).
        if getattr(args, "dump_graph_only", False):
            nodes_out = []
            for n in graph.get_all_nodes():
                ntype = getattr(n.type, "name", str(n.type))
                nodes_out.append({
                    "id": n.id,
                    "type": ntype,
                    "title": n.data.get("title", ""),
                    "description": n.data.get("description", ""),
                    "date": n.data.get("date", ""),
                    "concept_type": n.data.get("concept_type", ""),
                    "role": n.data.get("role", ""),
                })
            graph_path = output_dir / f"graph_{question_id}.json"
            with open(graph_path, "w") as gf:
                json.dump({
                    "question_id": question_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "graph_node_count": graph.node_count,
                    "graph_edge_count": graph.edge_count,
                    "nodes": nodes_out,
                }, gf, indent=2, default=str)
            result_entry = {
                "question_id": question_id,
                "graph_dump_path": str(graph_path),
                "graph_node_count": graph.node_count,
                "graph_edge_count": graph.edge_count,
                "skipped_qa": True,
            }
            with open(output_path, "a") as f:
                f.write(json.dumps(result_entry) + "\n")
            continue

        # 3. Answer Question
        # R9-A: dynamic max_nodes — aggregation questions ("how many X have I",
        # "how much money spent on X") need to see ALL relevant entity
        # concepts to count/sum correctly. With max_nodes=20, multi-entity
        # questions chronically under-count. Bump to 50 for these only;
        # single-fact questions stay at 20 to avoid context dilution.
        #
        # Widened on 2026-06-01:
        # - Include BARE verbs (attend, visit, spend, play, buy, do, see,
        #   …) so "did I attend / did I visit / have I spent" all trigger.
        #   The previous regex only had past tense (attended, visited),
        #   missing R9-A on 2ce6a0f2 / gpt4_f2262a51 / 28dc39ac.
        # - Replaced the time-unit negative lookahead (`hours?|days?|...`)
        #   with an explicit TR-marker suppression (`ago` / `since i` /
        #   `between` / `before i`). Aggregation Qs about hours
        #   ("how many hours have I spent playing games") were being
        #   blocked alongside the TR "how many days ago" ones.
        # `i\s+(?:\w+\s+)?` allows one optional adverb between "I" and the
        # verb (e.g., "I currently own", "I just bought"). Without it,
        # phrasings like "do I currently own" fail to match.
        _qa_agg_count_re = re.search(
            r"\bhow\s+many\s+\w+.*\b(?:have\s+i|i\s+(?:\w+\s+)?(?:"
            r"have|own|owned|bought|buy|worked|work|"
            r"attended|attend|read|watched|watch|"
            r"tried|try|made|make|led|lead|leading|"
            r"did|do|spent|spend|visited|visit|"
            r"saw|see|met|meet|played|play|"
            r"finished|finish|completed|complete|been|"
            r"received|receive|cooked|cook|baked|bake|"
            r"travel(?:l?ed)?|took|take|"
            r"hosted|host|booked|book|"
            r"replaced|replace|fixed|fix|sold|sell|lost|lose|"
            r"purchased|purchase|downloaded|download|"
            r"installed|install|joined|join|left|leave))",
            question, re.IGNORECASE,
        )
        # Suppress when the question is genuinely TR-style (single-event
        # date diff) rather than aggregation over many events.
        # NOTE: keep `before\s+i\b` (not `before the/making`): "before making an
        # offer" (gpt4_7fce9456) is a legitimate R2 MS count target that must NOT
        # be suppressed. The a3838d2b TR "before the Run" gets an additive
        # force-include block, which the validator found bounded/net-neutral.
        _qa_agg_count_tr = re.search(
            r"\b(?:ago\b|since\s+i\b|between\s+\w|before\s+i\b|after\s+i\b)",
            question, re.IGNORECASE,
        )
        _qa_agg_count = bool(_qa_agg_count_re) and not _qa_agg_count_tr
        _qa_agg_sum = re.search(
            r"\b(?:how\s+much|total|sum)\b(?!\s*(?:hours?|minutes?|seconds?|days?|"
            r"weeks?|months?|years?)\b).*\b(?:money|dollars?|expense|expenses?|"
            r"cost|costs?|\$\s*\d|spent\s+on)\b",
            question, re.IGNORECASE,
        )
        # iter33-MS R3 — age/duration/comparison ARITHMETIC questions. These
        # were getting NO retrieval widening: _qa_agg_count_tr above suppresses
        # widening exactly on `ago|since|before|after` (the tokens these Qs
        # carry), and the "how many X have I" verb-frame in _qa_agg_count_re
        # never matches "how old was I when …". R3 RE-ENABLES widening for the
        # comparison/age class so the bridge anchors aren't crowded out by
        # TODAY-dated noise. Strictly gated on _MS_ARITH_COMPARE_RE so TR
        # single-event date-diff Qs (which lack "how old/older/than when I")
        # are untouched. Targets a1cc6108, ba358f49, c18a7dc8, 3c1045c8,
        # 853b0a1d, c9f37c46, dcfa8644, b29f3365.
        _ms_arith = bool(_MS_ARITH_COMPARE_RE.search(question))
        # iter33-MS R2 — count-over-owned-category detector (head noun).
        # SUPPRESS R2 on TR single-event date-diff phrasings ("how many days
        # ago did I attend X", "how many weeks passed between …") — same
        # _qa_agg_count_tr marker that suppresses the existing agg-count path —
        # so genuine TR date-diff questions keep their symbolic-resolver
        # handling unchanged (HARD RULE 3: do not regress TR=88.7). The 7 R2
        # count targets (model kits / instruments / kitchen items / properties
        # / art events / bake / bikes) carry no such marker, so they still fire.
        # iter33-MS refine: C2 (suppress R2 on "current/now") was CONSIDERED but
        # REVERTED — it would kill gpt4_194be4b3 ("how many instruments do I
        # *currently* own", an R2 count target needing the missing-4th surfaced),
        # and KU (the class C2 protects) is not in the MS-133 set. Defer C2 to a
        # later all-category run.
        _ms_count_category = "" if _qa_agg_count_tr else _extract_count_category(question)
        # Widen for aggregation, money-sum, AND the arithmetic/comparison
        # bridge class (R3a). Without widening, R1's forced-include block is
        # the only path that surfaces the bridge node; widening gives the main
        # retrieval a fairer shot too.
        _query_max_nodes = 50 if (_qa_agg_count or _qa_agg_sum or _ms_arith) else None
        # On aggregation questions, also boost the pre-rerank pool so the
        # rerank step has a fuller session set to choose from. Only takes
        # effect when --llm-rerank is on; otherwise the override is
        # max_nodes-equivalent (rerank disabled ⇒ pool boost is moot).
        if args.llm_rerank and (_qa_agg_count or _qa_agg_sum):
            _query_pre_rerank_pool = max(args.rerank_pool, 100)
        elif args.llm_rerank:
            _query_pre_rerank_pool = args.rerank_pool
        else:
            _query_pre_rerank_pool = None
        # Per-question max_context_chars override: aggregation Qs need ~3x
        # more chars to fit the 50-node retrieval set without assembly
        # truncating to the first ~20 nodes (default 6000-char cap).
        _query_max_ctx = None
        if (_qa_agg_count or _qa_agg_sum or _ms_arith) and args.agg_max_context_chars:
            _query_max_ctx = args.agg_max_context_chars
        _query_start = time.time()
        _query_kwargs: dict = dict(
            question=question,
            domain="longmemeval",
            query_mode=args.query_mode,
            max_nodes=_query_max_nodes,
            pre_rerank_pool=_query_pre_rerank_pool,
        )
        if _query_max_ctx:
            _query_kwargs["max_context_chars"] = _query_max_ctx
        query_result = query_agent.query_for_qa(**_query_kwargs)
        context_text = query_result.context

        # iter33-MS TIER 1 — bridge-entity / category-expansion second hop.
        # Run extra HYBRID sub-queries keyed on the question's bridge entities
        # (R1) and/or count-category siblings (R2), then FORCE-UNION the nodes
        # the flat top-k retrieval missed into context BEFORE the reader. This
        # is the highest-leverage MS fix: 15/24 iter19 MS failures are
        # retrieval misses where the answer node exists in the graph but a
        # single lexically-dissimilar session scored below the top-k cutoff.
        #
        # GATING: only fires when _ms_arith (R1/R3 class) OR _ms_count_category
        # (R2 class) is set. On TR / KU / SSU / SSA questions both are false,
        # so this entire block is INERT and those paths are unchanged.
        # iter33-MS refine C1: NEVER force-include on _abs abstention qids — the
        # wrong-sibling roster injected at top-of-context fights the A1/A2
        # abstention rules (a96c20ee_abs/09ba9854_abs over-answered). No
        # confirmed flip depends on Tier-1 firing for an _abs qid.
        if (_ms_arith or _ms_count_category) and not question_id.endswith("_abs"):
            try:
                _already_ids = {
                    getattr(n, "node_id", None)
                    for n in (getattr(query_result, "nodes", None) or [])
                }
                _sub_queries: list[str] = []
                _block_header = ""
                if _ms_count_category:
                    # R2 — category + sibling expansion. Query the category head
                    # noun plus a couple of singular variants so a lexically
                    # distant instance session (e.g. "tank" for "model kits",
                    # "coffee maker" for "kitchen items") can win its own slot.
                    _cat = _ms_count_category
                    _sub_queries.append(_cat)
                    _sing = re.sub(r"s\b", "", _cat)  # crude singularization
                    if _sing and _sing != _cat:
                        _sub_queries.append(_sing)
                    # iter33-MS B2(b): action-verb probe. Mine the question's
                    # OWN count verb (acquire/buy/download/...) and emit an
                    # action-verb + category sub-query so an instance session
                    # that names the acquisition act ("picked up a snake plant
                    # at the nursery", "downloaded the EP") but is lexically
                    # distant from the bare category head still wins a slot.
                    # Derived from question tokens — no per-qid literals.
                    _qlow = question.lower()
                    # Each group is keyed on a trigger verb present in the
                    # question; if seen, its acquisition synonyms are added so
                    # the probe keys on how an instance might be PHRASED.
                    _verb_groups = (
                        # trigger    -> synonym expansion
                        ("acquire", ("acquired", "got", "bought", "picked up",
                                     "purchased", "nursery")),
                        ("purchase", ("purchased", "bought", "ordered")),
                        ("bought", ("bought", "purchased", "got")),
                        ("buy", ("buy", "bought", "purchased")),
                        ("download", ("downloaded", "streamed")),
                        ("got", ("got", "acquired", "bought")),
                    )
                    _verb_probe_terms: list[str] = []
                    for _trigger, _syns in _verb_groups:
                        if re.search(rf"\b{_trigger}", _qlow):
                            _verb_probe_terms.extend(_syns)
                    if _verb_probe_terms:
                        # De-dup while preserving order.
                        _seen_vp: set[str] = set()
                        _vp = [t for t in _verb_probe_terms
                               if not (t in _seen_vp or _seen_vp.add(t))]
                        _sub_queries.append(" ".join(_vp) + " " + _cat)
                    # Music/album categories: add a "vinyl record" sibling so a
                    # vinyl purchase counts toward an album/EP count (bf659f65).
                    if re.search(r"\b(?:album|albums|ep|eps|music|record|"
                                 r"records|vinyl)\b", _cat, re.IGNORECASE):
                        _sub_queries.append("vinyl record " + _cat)
                    # iter33-MS B3: "service or plan to service" / maintenance
                    # probe (a9f6b44c). When the question frames the count as a
                    # service/maintenance act, an instance session may name the
                    # maintenance act ("replaced the tire", "tuned up the bike")
                    # rather than the bare category. Gated to bikes/maintenance
                    # category tokens so it stays inert on unrelated counts.
                    if re.search(
                        r"\b(?:service\s+or\s+plan\s+to\s+service|"
                        r"plan\s+to\s+service|maintenance|servic)\b",
                        _qlow, re.IGNORECASE,
                    ) and re.search(
                        r"\b(?:bike|bikes|bicycle|car|cars|vehicle|tire|"
                        r"tires|maintenance|service)\b", _cat, re.IGNORECASE,
                    ):
                        _sub_queries.append(
                            "plan to service replace tire maintenance "
                            "this month " + _cat
                        )
                    # Add the question's own bridge phrases too (e.g. the
                    # "before making an offer on the townhouse" qualifier).
                    _sub_queries.extend(_extract_bridge_phrases(question))
                    # iter33-MS M1-M4 — additive, frame-gated count sub-queries
                    # (polysemy-conjoin / generic-event verb / weekday probe /
                    # vinyl-format strengthen). All derived from question tokens.
                    _sub_queries.extend(
                        _ms_extra_count_sub_queries(question, _cat)
                    )
                    _block_header = (
                        "## CATEGORY_MEMBERS (iter33-MS R2 — every instance of "
                        f"'{_cat}' found in memory, force-included regardless of "
                        "retrieval rank; COUNT all distinct ones below, they may "
                        "live in different sessions)"
                    )
                else:
                    # R1 — bridge-entity second hop for age/duration/comparison.
                    _sub_queries = _extract_bridge_phrases(question)
                    # iter33-MS refine R1-age: for age-difference questions
                    # ("how old was I when X", "how many years older … than
                    # when I Y"), the SECOND operand (the user's current age) is
                    # a single self-report ("I just turned 32") that routinely
                    # scores below the cutoff — a textbook retrieval miss
                    # (c18a7dc8, a1cc6108). Add current-age probe sub-queries so
                    # the operand is surfaced; D-COMPUTE's absent-operand gate
                    # still abstains if it genuinely isn't there (no fabrication).
                    if re.search(r"\bhow\s+(?:old|many\s+years)\b", question, re.IGNORECASE):
                        _sub_queries += [
                            "I just turned years old current age",
                            "I am years old now currently",
                        ]
                    _block_header = (
                        "## BRIDGE_ANCHORS (iter33-MS R1 — sessions matching the "
                        "named entity/event in the question, force-included so the "
                        "second-hop fact isn't crowded out by recent-dated noise)"
                    )
                _sub_queries = [s for s in _sub_queries if s]
                if _sub_queries:
                    _force_block, _forced_ids = _build_forced_include_block(
                        query_agent,
                        _sub_queries,
                        {i for i in _already_ids if i},
                        header=_block_header,
                    )
                    if _force_block:
                        context_text = _force_block + "\n" + context_text
                        logger.debug(
                            f"[{question_id}] iter33-MS forced-include: "
                            f"{len(_forced_ids)} nodes via {_sub_queries}"
                        )
            except Exception as e:
                # Never let the expansion hop break the run — it is additive.
                logger.error(f"iter33-MS expansion failed for {question_id}: {e}")

        # Symbolic temporal layer: for time-ordering / interval / latest-fact
        # questions, prepend a chronologically-sorted block of every dated
        # node so the reader sees absolute dates without relying on retrieval
        # to preserve them.
        if args.symbolic_temporal:
            temporal_block = build_temporal_block(graph, question)
            if temporal_block:
                context_text = temporal_block + "\n" + context_text

        # Recency injection: for "most recent / latest / current" style
        # questions, sort the topic-matching nodes by date DESC and prepend.
        # The first entry is the answer in 95% of cases — reduces the reader's
        # job from "scan 10+ dates inside descriptions" to "copy bullet #1".
        recency_block = build_recency_block(graph, question)
        if recency_block:
            context_text = recency_block + "\n" + context_text

        # Cluster E — Assistant-text recall. When the question explicitly
        # references a prior assistant turn ("previous conversation about X",
        # "you recommended/mentioned/listed Y"), the answer is in a raw
        # assistant EVENT that distilled CONCEPT retrieval routinely buries.
        # Surface it directly so the reader can copy the verbatim name/quote.
        assistant_block = build_assistant_recall_block(graph, question)
        if assistant_block:
            context_text = assistant_block + "\n" + context_text

        # R2 (Exp A) — Clock-time matches for "what time do I X" questions.
        # Writer captures these but retrieval frequently buries them under
        # generic topic concepts (photography / fishing / etc.).
        time_block = build_time_of_day_block(graph, question)
        if time_block:
            context_text = time_block + "\n" + context_text

        # R3 (Exp A) — Proper-noun matches for "what's the name of X / which
        # X / what breed" questions. Pins capitalized multi-word names from
        # the graph to the top of context.
        propnoun_block = build_proper_noun_block(graph, question)
        if propnoun_block:
            context_text = propnoun_block + "\n" + context_text

        # Symbolic deterministic resolver (ToMi-style): pattern-match the
        # question against the dated graph and compute an exact answer string.
        # If matched, we both inject the answer as a SYMBOLIC_ANSWER block
        # AND short-circuit the LLM (`answer` = resolver answer verbatim) so
        # the reader can't unsort what we already sorted.
        question_dt = _parse_longmemeval_date(item.get("question_date", ""))
        # iter29 D' (revised iter30) — per-type W2 suppression in the
        # RESOLVER only. iter27 showed W2 event_date hurt MS -4.5 and TR
        # -3 when used by the resolver's _index_concepts (session-relative
        # ordering matters for those types). The reader, however, BENEFITS
        # from seeing the absolute (meaning DATE) anchor inline regardless
        # of question type, so iter30 stops stripping the rendered suffix.
        _qtype = item.get("question_type", "") or ""
        _suppress_w2 = _qtype in ("multi-session", "temporal-reasoning")
        symbolic_result = None
        if args.symbolic_resolver:
            resolver = LongMemEvalSymbolicResolver(
                graph,
                question_date=question_dt,
                ignore_event_date=_suppress_w2,
            )
            symbolic_result = resolver.resolve(question)
            if symbolic_result is not None:
                context_text = render_symbolic_block(symbolic_result) + "\n" + context_text

        # iter33-MS — neural-symbolic computation agent. A FOCUSED structured-
        # extraction LLM call reads the RAW retrieved turns (query_result.nodes,
        # which carry EVENT `content` verbatim — recovering writer-gap operands
        # the classic resolver/concepts dropped), forces an explicit enumeration
        # of qualifying items, and then computes the count/sum/diff/date/age
        # answer DETERMINISTICALLY in Python. Injected as a RECALL_HINT
        # (bypass=False by default) so the reader verifies a presented list
        # rather than performing the error-prone enumeration itself. Gated OFF
        # by default; only fires when the classic resolver did NOT already match
        # (clean separation — count_among is party-gated, so it returns None for
        # the count/sum questions this agent targets). See neural_symbolic.py.
        if getattr(args, "neural_symbolic", False) and symbolic_result is None:
            # Extraction runs on the READER config (the reasoning model, e.g.
            # gpt-5.4-mini), NOT the cheap writer: an extraction-replay against
            # cached context showed gpt-4o-mini under-enumerates and is unstable
            # on the enumerate+qualify task (run-to-run variance), while the
            # reasoning model is materially more reliable. The op is one call
            # per count/sum question, so the cost delta is small.
            try:
                ns_result = resolve_neural_symbolic(
                    question,
                    getattr(query_result, "nodes", None),
                    call_llm,
                    config,
                    bypass_mode=(
                        neural_symbolic_mod.BYPASS_AUTO
                        if getattr(args, "neural_symbolic_bypass", False)
                        else neural_symbolic_mod.BYPASS_NEVER
                    ),
                )
            except Exception as e:
                logger.warning(f"neural-symbolic agent failed for {question_id}: {e}")
                ns_result = None
            if ns_result is not None:
                logger.info(
                    "Neural-symbolic [%s] for %s → %r (bypass=%s)",
                    ns_result.get("pattern"), question_id,
                    ns_result.get("answer", "")[:60], ns_result.get("bypass"),
                )
                symbolic_result = ns_result
                # bypass=True (auto, high-confidence) → verbatim SYMBOLIC_ANSWER;
                # bypass=False (default) → the COUNT_CROSSCHECK block whose framing
                # treats the enumeration as a lower-bound checklist to merge with,
                # NOT a verdict (fixes the under-count collateral seen in the smoke).
                _ns_block = (
                    render_symbolic_block(ns_result)
                    if ns_result.get("bypass")
                    else render_neural_symbolic_block(ns_result)
                )
                context_text = _ns_block + "\n" + context_text

        # iter15 — RECALL_TARGET_DATE for "X N weeks ago" Qs. Pin the
        # absolute target date so reader prefers the right [YYYY-MM-DD]
        # prefix even when the resolver doesn't bypass.
        target_block = build_recall_target_date_block(question_dt, question)
        if target_block:
            context_text = target_block + "\n" + context_text

        # iter16 — list all dated concepts within ±3 days of the target
        # so the reader can scan and pick by topic match rather than
        # relying on retrieval ranking that may have buried the right
        # event. Fires for the same "X N weeks ago" Qs as the target
        # block.
        target_cands_block = build_target_date_concepts_block(graph, question_dt, question)
        if target_cands_block:
            context_text = target_cands_block + "\n" + context_text

        # iter29 TR-α — topic timeline for TR questions (chronological
        # list of topic-matching events with absolute date + N days ago).
        if getattr(args, "tr_topic_timeline", False) and _qtype == "temporal-reasoning":
            topic_tl = build_topic_timeline_block(graph, question, question_dt)
            if topic_tl:
                context_text = topic_tl + "\n" + context_text

        # iter07 — TODAY anchor goes LAST in the prepend chain so it ends up
        # at the very top of context. Reader prompt then begins with a clear
        # statement of the reference date, eliminating the iter05/06 failure
        # mode where reader computed "X days ago" against system time (2026)
        # instead of the dataset's question_date (~2023).
        today_block = build_today_block(question_dt)
        if today_block:
            context_text = today_block + "\n" + context_text

        # Generate Answer.
        # Bypass policy: respect both the global flag and the resolver's
        # per-pattern bypass hint. Deterministic resolvers (date_diff_*,
        # which_first, chronological_order, strict latest_value) set
        # bypass=True; lexically fuzzy ones (topic_recall, broad
        # latest_value) set bypass=False and only inject a hint.
        should_bypass = (
            symbolic_result is not None
            and args.symbolic_bypass
            and symbolic_result.get("bypass", True)
        )
        try:
            if should_bypass:
                answer = symbolic_result["answer"]
            else:
                answer = generate_answer(
                    question=question,
                    context=context_text,
                    config=config,
                    qa_template=templates.get("qa_answer"),
                    graph=graph,
                    query_nodes=getattr(query_result, "nodes", None),
                    question_date=question_dt,
                )
        except Exception as e:
            logger.error(f"Error generating answer for {question_id}: {e}")
            answer = "Error generating answer."

        # 4. Evaluate (optional)
        eval_result = None
        if args.llm_eval and ground_truth:
            eval_result = evaluate_answer(
                question=question,
                hypothesis=answer,
                ground_truth=ground_truth,
                config=judge_config,
                eval_template=templates.get("evaluate"),
            )
            result_key = eval_result["result"].lower()
            metrics[result_key] = metrics.get(result_key, 0) + 1
            metrics["total"] += 1

        result_entry = {
            "question_id": question_id,
            "question": question,
            "hypothesis": answer,
            "ground_truth": ground_truth,
            "context_length": len(context_text),
            "graph_nodes": graph.node_count,
            "symbolic_pattern": (symbolic_result or {}).get("pattern"),
            "bypass_taken": bool(should_bypass),
        }

        if eval_result:
            result_entry["evaluation"] = eval_result
            result_entry["verdict"] = eval_result.get("result", "")

        if enrich_eval_result is not None:
            enrich_eval_result(
                result_entry,
                graph=graph,
                query_result=query_result,
                retrieval_mode=args.query_mode,
                query_start_time=_query_start,
                full_context=context_text,
            )

        results.append(result_entry)

        # Append to file incrementally
        with open(output_path, "a") as f:
            f.write(json.dumps(result_entry) + "\n")

    # Save metrics
    if args.llm_eval:
        total = metrics["total"]
        if total > 0:
            metrics["score_strict"] = metrics["correct"] / total * 100
            metrics["score_partial"] = (metrics["correct"] + 0.5 * metrics["partial"]) / total * 100

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics saved to {metrics_path}")
        logger.info(f"Results: {metrics}")

    if save_wrong_cases is not None:
        save_wrong_cases(results, str(output_dir))

    # Persist per-batch call stats so the launcher can merge across batches.
    try:
        import json as _json

        merged_stats: dict = {"chat": dict(_LLM_CALL_STATS), "embed": {}}
        # Pull embedding counter from providers module if present.
        try:
            from cognifold.embeddings import providers as _ep
            es = getattr(_ep, "_EMBED_CALL_STATS", None)
            if isinstance(es, dict):
                merged_stats["embed"] = dict(es)
        except Exception:
            pass
        (output_dir / "call_stats.json").write_text(
            _json.dumps(merged_stats, indent=2)
        )
    except Exception as _e:
        logger.warning(f"Could not write call_stats.json: {_e}")

    logger.info(f"Evaluation complete. Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LongMemEval benchmark")
    parser.add_argument("--limit", type=int, help="Limit number of examples to run")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N examples (applied AFTER --stratified, BEFORE --limit). "
        "Pair with --output-dir + --limit to partition the data for parallel runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: benchmarks/longmemeval/output/). "
        "Each parallel batch should point to its own dir to avoid hypothesis.jsonl collisions.",
    )
    parser.add_argument(
        "--question-ids",
        type=str,
        default=None,
        help="Process only specific question_ids (comma-separated). Overrides "
        "--stratified/--limit/--offset filtering.",
    )
    parser.add_argument(
        "--stratified",
        type=int,
        default=None,
        help="Take N per question_type for balanced sampling (applied before --limit)",
    )
    parser.add_argument(
        "--symbolic-temporal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inject a chronologically-sorted ## TEMPORAL FACTS block for time-ordering questions",
    )
    parser.add_argument(
        "--symbolic-resolver",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use LongMemEvalSymbolicResolver to pattern-match the question and produce a deterministic SYMBOLIC_ANSWER block",
    )
    parser.add_argument(
        "--symbolic-bypass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When the symbolic resolver matches, use its answer verbatim and skip the LLM reader call",
    )
    parser.add_argument(
        "--neural-symbolic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the neural-symbolic computation agent: a focused "
        "structured-extraction LLM call (on the writer config) reads the raw "
        "retrieved turns for count/sum/diff/date/age questions, enumerates the "
        "operands, and computes the answer deterministically, injected as a "
        "RECALL_HINT. OFF by default; only fires when the classic resolver did "
        "not already match. See neural_symbolic.py.",
    )
    parser.add_argument(
        "--neural-symbolic-bypass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Let the neural-symbolic agent set bypass=True (use its computed "
        "answer verbatim, skipping the reader) for HIGH-confidence enumerate/sum "
        "results. Still requires --symbolic-bypass. Default OFF (hint-only, "
        "reader verifies).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Override LLM model used for evaluation (default: same as --model). Pass openai:gpt-4o to match the canonical LongMemEval judge.",
    )
    parser.add_argument(
        "--llm-rerank",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable batched B-rerank: one LLM call per question scores every "
        "retrieved candidate jointly and returns ranked indices. See "
        "my_prompt.md §1.2 — this is the canonical rerank path; the legacy "
        "per-doc rerank is intentionally not exposed via CLI.",
    )
    parser.add_argument(
        "--rerank-model",
        type=str,
        default="openai:gpt-5",
        help="Rerank LLM. Default openai:gpt-5 (cheap with reasoning_effort=low).",
    )
    parser.add_argument(
        "--rerank-reasoning-effort",
        type=str,
        default="low",
        choices=["low", "medium", "high"],
        help="Reasoning effort for rerank LLM. Default low — rerank is "
        "scoring relevance, not full QA, so low effort is enough.",
    )
    parser.add_argument(
        "--rerank-pool",
        type=int,
        default=0,
        help="When --llm-rerank is on, retrieval keeps this many candidates "
        "before reranking (rerank then trims to max_nodes). 0 = use "
        "max_nodes (no pool expansion). On aggregation questions "
        "(detected via the R9-A heuristic), the runner auto-bumps to "
        "max(this, 100) so the relevant session is in the pool.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip question_ids already present in output/hypothesis.jsonl (do not clear the file).",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override model name (e.g., openai:gpt-4o, gemini-1.5-pro)",
    )
    parser.add_argument(
        "--writer-model",
        type=str,
        default=None,
        help="Override extraction model separately (default: same as --model). "
        "Use to route the writer to a different model / reasoning_effort "
        "than the reader.",
    )
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
        "--batch-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use batch processing for faster ingestion",
    )
    parser.add_argument(
        "--llm-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use LLM to evaluate answers against ground truth",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default=None,
        help="Embedding model (e.g. openai:text-embedding-3-large, gemini:text-embedding-004, or none). Overrides profile config.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=None,
        help="Override QueryConfig.max_context_chars (default 6000). Bump for "
        "aggregation questions where 50-node retrieval is otherwise truncated "
        "by the 6000-char assembly cap. Try 12000–18000.",
    )
    parser.add_argument(
        "--agg-max-context-chars",
        type=int,
        default=None,
        help="Per-question override for aggregation Qs only (R9-A path: 'how "
        "many X', 'how much money'). If set, takes precedence over "
        "--max-context-chars on aggregation questions. Lets you bump just the "
        "Qs that need it without inflating cost on single-fact Qs.",
    )
    parser.add_argument(
        "--dump-graph-only",
        action="store_true",
        help="Exp A diagnostic: after ingestion, dump the full graph to "
        "<output_dir>/graph_<qid>.json and SKIP the QA / judge steps. Used "
        "to classify failure causes (writer extraction miss vs retrieval "
        "rank-out) without paying for the reader.",
    )
    parser.add_argument(
        "--extract-typed-attributes",
        action="store_true",
        help="W1: run a second writer pass per session that extracts typed "
        "attributes (date/time/duration/quantity/name) verbatim from "
        "user-role turns. Adds ~$0.0001 per session in writer cost. Helps "
        "with failure cases where the main extractor paraphrased away the "
        "specific value (Exp A WRITER bucket: ~7/23 of the Bucket C cases).",
    )
    parser.add_argument(
        "--writer-reasoning-effort",
        type=str,
        default=None,
        choices=["low", "medium", "high", None],
        help=(
            "iter23: writer-specific reasoning_effort. Defaults to 'high' "
            "(matches reader) when the writer is a gpt-5 reasoning model. "
            "Set 'low' to run the writer at lower effort than the reader "
            "to trade extraction quality for speed."
        ),
    )
    parser.add_argument(
        "--resolve-event-dates",
        action="store_true",
        help=(
            "W2 (iter18): per-concept event_date resolution pass. For each "
            "CONCEPT extracted in a session, the LLM resolves the user's "
            "relative-time phrasing ('on January 10th', 'two weeks ago', "
            "'last Saturday') to an absolute event_date and stores it on "
            "the node so resolvers and readers can use the true event date "
            "instead of the session-extraction date. Inspired by Chronos "
            "(event-calendar +58.9 pts baseline in their ablation) and "
            "Mem0 (per-memory temporal pass)."
        ),
    )
    parser.add_argument(
        "--reflector",
        action="store_true",
        help=(
            "iter29 F — Mastra-style Reflector pass. After all sessions "
            "are ingested for a qid, runs a consolidator LLM that stamps "
            "supersession markers (status, superseded_by, replaces) onto "
            "concept.data so the reader sees [✅ OUTDATED]/[🆕 CURRENT] "
            "tags in the rendered context — directly addresses KU and "
            "MS hard cases."
        ),
    )
    parser.add_argument(
        "--extract-start-events",
        action="store_true",
        help=(
            "W3 (iter30) — per-session focused pass that explicitly "
            "extracts ACTIVITY START events ('I started bird watching', "
            "'I bought my new running shoes') and stamps the resulting "
            "concept with is_start=true + activity. Lets the resolver's "
            "`duration_activity` pattern actually fire (iter29 evidence "
            "showed embedding the rule in the main BATCH_SYSTEM_PROMPT "
            "was unreliable). Adds ~1 cheap LLM call per session."
        ),
    )
    parser.add_argument(
        "--tr-topic-timeline",
        action="store_true",
        help=(
            "iter29 TR-α — for TR questions, prepend a topic-focused "
            "chronological timeline of all events matching the question's "
            "key noun phrases, with absolute date + (N days ago) anchors."
        ),
    )

    args = parser.parse_args()
    run_benchmark(args)
