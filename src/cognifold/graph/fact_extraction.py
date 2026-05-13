"""Post-ingestion fact extraction from graph nodes.

Scans concept and event nodes for structured fact patterns
(person+attribute+value) and creates dedicated high-priority concept
nodes with clear, searchable titles like "Alice birthday: March 5".

This module is LLM-free -- uses regex/heuristic patterns to extract
facts without API costs.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from cognifold.models.node import BaseEdgeType, Edge, Node, NodeType

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fact extraction patterns
# ---------------------------------------------------------------------------

# Each pattern is (compiled_regex, template_fn) where template_fn takes
# the match object and returns (fact_title, fact_description).
# Patterns are designed to capture person+attribute+value triples.

_POSSESSIVE_ATTR = re.compile(
    r"(?:^|[.!?]\s+)"
    r"(\w[\w\s]{0,30}?)"  # subject
    r"'s\s+"
    r"(name|birthday|birth\s*day|age|job|occupation|profession|"
    r"hobby|hobbies|favorite|favourite|wife|husband|spouse|partner|"
    r"brother|sister|son|daughter|mother|father|parent|pet|dog|cat|"
    r"car|home|house|apartment|school|university|college|"
    r"hometown|home\s*town|nationality|religion|major|degree|"
    r"phone|email|address|company|employer|workplace|"
    r"dream|goal|plan|allergy|allergies|"
    r"child|children|kid|kids|friend)\w*"
    r"\s+(?:is|are|was|were)\s+"
    r"(.+?)(?:[.!?]|$)",
    re.IGNORECASE | re.MULTILINE,
)

_SUBJECT_IS_VALUE = re.compile(
    r"(?:^|[.!?]\s+)"
    r"(\w[\w\s]{0,30}?)"  # subject
    r"\s+(?:is|was|has been|used to be)\s+"
    r"(?:a\s+|an\s+|the\s+)?"
    r"(\w[\w\s,]{2,60}?)"  # value
    r"(?:[.!?]|$)",
    re.IGNORECASE | re.MULTILINE,
)

_SUBJECT_LIVES_IN = re.compile(
    r"(?:^|[.!?]\s+)"
    r"(\w[\w\s]{0,30}?)"  # subject
    r"\s+(?:lives?|lived|residing|resides?|moved|is from|comes? from|grew up|born)\s+"
    r"(?:in|at|to)\s+"
    r"(\w[\w\s,]{2,60}?)"  # location
    r"(?:[.!?]|$)",
    re.IGNORECASE | re.MULTILINE,
)

_SUBJECT_WORKS_AT = re.compile(
    r"(?:^|[.!?]\s+)"
    r"(\w[\w\s]{0,30}?)"  # subject
    r"\s+(?:works?|worked|working|employed|is employed)\s+"
    r"(?:at|for|with)\s+"
    r"(\w[\w\s,]{2,60}?)"  # employer
    r"(?:[.!?]|$)",
    re.IGNORECASE | re.MULTILINE,
)

_SUBJECT_LIKES = re.compile(
    r"(?:^|[.!?]\s+)"
    r"(\w[\w\s]{0,30}?)"  # subject
    r"\s+(?:likes?|loves?|enjoys?|prefers?|is into|is fond of|"
    r"is a fan of|is passionate about|is interested in)\s+"
    r"(\w[\w\s,]{2,60}?)"  # thing
    r"(?:[.!?]|$)",
    re.IGNORECASE | re.MULTILINE,
)

_SUBJECT_HAS = re.compile(
    r"(?:^|[.!?]\s+)"
    r"(\w[\w\s]{0,30}?)"  # subject
    r"\s+(?:has|had|have|got|owns?|owned)\s+"
    r"(?:a\s+|an\s+|the\s+)?"
    r"(\w[\w\s,]{2,60}?)"  # object
    r"(?:[.!?]|$)",
    re.IGNORECASE | re.MULTILINE,
)

_SUBJECT_WANTS = re.compile(
    r"(?:^|[.!?]\s+)"
    r"(\w[\w\s]{0,30}?)"  # subject
    r"\s+(?:wants?|wanted|planning|plans?|hopes?|hoping|intends?|"
    r"is going|going to|will|is planning)\s+"
    r"(?:to\s+)?"
    r"(\w[\w\s,]{2,60}?)"  # activity/thing
    r"(?:[.!?]|$)",
    re.IGNORECASE | re.MULTILINE,
)

# Conversational patterns: "Speaker: I went/did/visited/attended ..."
_SPEAKER_ACTION = re.compile(
    r"^(\w[\w\s]{0,20}?):\s+"  # "Caroline: "
    r"(?:I\s+)?"
    r"(went to|visited|attended|researched|started|joined|signed up for|"
    r"adopted|bought|moved to|graduated from|enrolled in|applied to|"
    r"quit|left|got|received|found|discovered|tried|began|"
    r"am single|am married|am divorced|am engaged|"
    r"am (?:a |an )?[\w\s]+?(?:er|ist|or|ian|ant))\s*"
    r"(.+?)(?:[.!?\n]|$)",
    re.IGNORECASE | re.MULTILINE,
)

# Speaker state: "Caroline: I'm single/I'm a counselor/I have 2 kids"
_SPEAKER_STATE = re.compile(
    r"^(\w[\w\s]{0,20}?):\s+"  # "Caroline: "
    r"(?:I'm|I am|I've been|I have|I got)\s+"
    r"(.+?)(?:[.!?\n]|$)",
    re.IGNORECASE | re.MULTILINE,
)

# Simple pronoun subjects to skip (these create noisy facts)
_PRONOUN_SUBJECTS = frozenset(
    {
        "i",
        "he",
        "she",
        "it",
        "we",
        "they",
        "you",
        "that",
        "this",
        "there",
        "here",
        "one",
        "what",
        "who",
        "which",
        "someone",
        "something",
        "everyone",
        "everything",
        "anyone",
        "anything",
        "nobody",
        "nothing",
    }
)


def _clean_subject(subject: str) -> str:
    """Clean up a subject string."""
    return subject.strip().rstrip(",.:;")


def _clean_value(value: str) -> str:
    """Clean up a value string."""
    return value.strip().rstrip(",.:;!?")


def _is_valid_subject(subject: str) -> bool:
    """Check if a subject is a valid entity (not a pronoun, etc.)."""
    clean = subject.strip().lower()
    if clean in _PRONOUN_SUBJECTS:
        return False
    return len(clean) >= 2


def extract_facts_from_text(text: str) -> list[tuple[str, str]]:
    """Extract structured facts from text.

    Returns a list of (fact_title, fact_description) tuples.
    """
    if not text:
        return []

    facts: list[tuple[str, str]] = []
    seen_titles: set[str] = set()

    # Pattern 1: X's Y is Z
    for match in _POSSESSIVE_ATTR.finditer(text):
        subject = _clean_subject(match.group(1))
        attr = match.group(2).strip().lower()
        value = _clean_value(match.group(3))
        if _is_valid_subject(subject) and value:
            title = f"{subject} {attr}: {value}"
            title_key = title.lower()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                facts.append((title, f"{subject}'s {attr} is {value}"))

    # Pattern 2: X lives/lived/is from Y
    for match in _SUBJECT_LIVES_IN.finditer(text):
        subject = _clean_subject(match.group(1))
        location = _clean_value(match.group(2))
        if _is_valid_subject(subject) and location:
            title = f"{subject} location: {location}"
            title_key = title.lower()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                facts.append((title, f"{subject} lives in {location}"))

    # Pattern 3: X works at Y
    for match in _SUBJECT_WORKS_AT.finditer(text):
        subject = _clean_subject(match.group(1))
        employer = _clean_value(match.group(2))
        if _is_valid_subject(subject) and employer:
            title = f"{subject} workplace: {employer}"
            title_key = title.lower()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                facts.append((title, f"{subject} works at {employer}"))

    # Pattern 4: X likes/loves/enjoys Y
    for match in _SUBJECT_LIKES.finditer(text):
        subject = _clean_subject(match.group(1))
        thing = _clean_value(match.group(2))
        if _is_valid_subject(subject) and thing:
            title = f"{subject} likes: {thing}"
            title_key = title.lower()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                facts.append((title, f"{subject} likes {thing}"))

    # Pattern 5: Speaker action (from dialogue) — "Caroline: I went to LGBTQ group"
    for match in _SPEAKER_ACTION.finditer(text):
        speaker = _clean_subject(match.group(1))
        action = match.group(2).strip().lower()
        obj = _clean_value(match.group(3))
        if _is_valid_subject(speaker) and obj:
            title = f"{speaker}: {action} {obj}"
            title_key = title.lower()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                facts.append((title, f"{speaker} {action} {obj}"))

    # Pattern 6: Speaker state — "Caroline: I'm single / I'm a counselor"
    for match in _SPEAKER_STATE.finditer(text):
        speaker = _clean_subject(match.group(1))
        state = _clean_value(match.group(2))
        if _is_valid_subject(speaker) and state and len(state) > 2:
            title = f"{speaker}: {state}"
            title_key = title.lower()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                facts.append((title, f"{speaker} is/has {state}"))

    return facts


def extract_facts(graph: ConceptGraph) -> list[str]:
    """Scan graph nodes and create fact concept nodes.

    For each extracted fact, creates a new concept node with a clear
    title and links it to the source node with a DERIVED_FROM edge.

    Args:
        graph: The concept graph to enrich with facts.

    Returns:
        List of created fact node IDs.
    """
    created_ids: list[str] = []
    seen_fact_titles: set[str] = set()

    # Snapshot node data before iteration (we'll add nodes during the loop)
    node_snapshots: list[tuple[str, dict[str, object]]] = [
        (nid, dict(attrs)) for nid, attrs in graph.internal_graph.nodes(data=True)
    ]

    # Collect existing titles to avoid duplicates
    for _node_id, attrs in node_snapshots:
        data = attrs.get("data", {})
        if isinstance(data, dict):
            title = str(data.get("title", "")).lower()
            if title:
                seen_fact_titles.add(title)

    for node_id, attrs in node_snapshots:
        raw_data = attrs.get("data", {})
        data = raw_data if isinstance(raw_data, dict) else {}
        title = str(data.get("title", ""))
        description = str(data.get("description", ""))
        text = f"{title}. {description}"

        facts = extract_facts_from_text(text)

        for fact_title, fact_description in facts:
            fact_key = fact_title.lower()
            if fact_key in seen_fact_titles:
                continue
            seen_fact_titles.add(fact_key)

            fact_id = f"fact-{uuid.uuid4().hex[:8]}"
            now = datetime.now()

            fact_node = Node(
                id=fact_id,
                type=NodeType.CONCEPT,
                data={
                    "title": fact_title,
                    "description": fact_description,
                },
                created_at=now,
                last_accessed=now,
                access_count=0,
                reasoning=f"Extracted structured fact from node '{node_id}'",
                grounded_in=[node_id],
            )

            try:
                graph.add_node(fact_node)
                # Link fact to source node
                edge = Edge.create(
                    source=fact_id,
                    target=node_id,
                    edge_type=BaseEdgeType.DERIVED_FROM.value,
                )
                graph.add_edge(edge)
                created_ids.append(fact_id)
            except (ValueError, KeyError) as e:
                logger.debug("Could not add fact node %s: %s", fact_id, e)

    if created_ids:
        logger.info("Fact extraction created %d fact nodes", len(created_ids))

    return created_ids
