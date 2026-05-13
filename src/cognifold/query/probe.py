"""Symbolic Probe Agent for evaluating memory structure."""

from __future__ import annotations

import json
import logging
from typing import Any

from cognifold.graph.store import ConceptGraph
from cognifold.models.node import NodeType

logger = logging.getLogger(__name__)

PROMPT_SYMBOLIC_EMERGENCE = """
SYSTEM: You are evaluating a symbolic memory graph.
INPUT:
- Concepts: {concepts}
- Random sample events: {events}

TASK:
A) For each concept, write 2 canonical queries that should be answered by this concept.
B) For each query, identify the minimal supporting events.
C) Mark if concept is too broad / too narrow / redundant with another concept.

OUTPUT JSON ONLY.
Structure:
[
  {{
    "concept_id": "...",
    "canonical_queries": ["Q1", "Q2"],
    "supporting_events": ["evt-1", "evt-2"],
    "assessment": "optimal" | "too_broad" | "too_narrow" | "redundant"
  }}
]
"""


class SymbolicProbeAgent:
    """Agent for probing the symbolic structure of the memory graph."""

    def __init__(self, graph: ConceptGraph, model_name: str = "gpt-4o"):
        self.graph = graph
        self.model_name = model_name

    def probe_emergence(self, sample_size: int = 20) -> list[dict[str, Any]]:
        """Run the symbolic emergence probe."""

        # Gather concepts
        concepts = self.graph.get_nodes_by_type(NodeType.CONCEPT)
        if not concepts:
            return []

        # Gather random events (or recent ones)
        events = self.graph.get_nodes_by_type(NodeType.EVENT)
        # Simple sampling: take last N
        sample_events = events[-sample_size:] if len(events) > sample_size else events

        # Format input
        concepts_str = json.dumps(
            [
                {"id": c.id, "title": c.data.get("title"), "desc": c.data.get("description")}
                for c in concepts
            ],
            indent=1,
        )

        events_str = json.dumps(
            [{"id": e.id, "desc": e.data.get("description")} for e in sample_events], indent=1
        )

        prompt = PROMPT_SYMBOLIC_EMERGENCE.format(concepts=concepts_str, events=events_str)

        # Call LLM
        response_text = self._call_llm(prompt)

        try:
            # Clean markdown
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[0].strip()

            return json.loads(response_text)
        except Exception as e:
            logger.error(f"Error parsing probe response: {e}")
            return []

    def _call_llm(self, prompt: str) -> str:
        # Use OpenAI directly for simplicity in this probe
        from openai import OpenAI

        from cognifold.service.llm_keys import get_api_key

        client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": prompt}], temperature=0.0
        )
        return response.choices[0].message.content or ""
