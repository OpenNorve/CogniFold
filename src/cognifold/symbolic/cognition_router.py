"""CognitionRouter: unified dispatch layer for Cognifold's dual-memory system.

Coordinates between symbolic memory (deterministic facts) and graph memory
(semantic associations) to answer queries. Implements a three-phase protocol:

1. Recognition: Can symbolic memory answer directly? (System 1 - fast, certain)
2. Reconstruction: Search graph memory, constrained by symbolic state (System 2 - slow, rich)
3. Validation: Check graph results against symbolic facts (consistency guarantee)

Principle: deterministic first, fuzzy second, validate always.

Usage:
    router = CognitionRouter(symbolic_tracker, query_agent)
    result = router.answer(question, domain="tomi", query_mode="mergefold")
    if result.direct_answer:
        use(result.direct_answer)  # symbolic resolved it, no LLM needed
    else:
        use(result.context)  # combined context for LLM
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cognifold.query.agent import MemoryQueryAgent
    from cognifold.query.models import QueryResult
    from cognifold.symbolic.state_tracker import SymbolicStateTracker

logger = logging.getLogger(__name__)


@dataclass
class SymbolicConstraints:
    """Constraints extracted from symbolic memory for a specific query."""

    relevant_entities: set[str] = field(default_factory=set)
    relevant_agents: set[str] = field(default_factory=set)
    verified_facts: dict[str, dict[str, str]] = field(default_factory=dict)
    agent_beliefs: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)
    initial_state: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def has_constraints(self) -> bool:
        return bool(self.verified_facts or self.agent_beliefs or self.initial_state)

    @property
    def entity_coverage(self) -> int:
        """How many query-relevant entities have symbolic facts."""
        return len(self.relevant_entities & set(self.verified_facts.keys()))


@dataclass
class CognitionResult:
    """Result from the cognition router."""

    context: str
    """Final context for LLM (graph + symbolic, or symbolic only)."""

    direct_answer: str | None = None
    """If symbolic can answer directly, the answer. None otherwise."""

    source: str = "graph"
    """Where the answer came from: 'symbolic', 'hybrid', or 'graph'."""

    constraints: SymbolicConstraints | None = None
    """Symbolic constraints that were applied (for debugging)."""

    query_result: Any = None
    """The underlying QueryResult from graph search."""


class CognitionRouter:
    """Unified dispatch layer for dual-memory cognition.

    Sits above both symbolic and graph memory, coordinating queries
    through the recognition → reconstruction → validation pipeline.
    """

    def __init__(
        self,
        symbolic: SymbolicStateTracker | None,
        query_agent: MemoryQueryAgent,
    ) -> None:
        self.symbolic = symbolic
        self.query_agent = query_agent

    def answer(
        self,
        question: str,
        domain: str | None = None,
        query_mode: str = "mergefold",
        **kwargs: Any,
    ) -> CognitionResult:
        """Answer a question using the dual-memory system.

        Two phases:
        1. Reconstruction: graph search with symbolic constraints
        2. Validation: inject verified facts (including symbolic answer) into context

        Symbolic memory NEVER bypasses the LLM — it augments the graph context
        with verified facts so the LLM can reason with both memory systems.
        """
        # Phase 1: Extract symbolic constraints + recognition hint
        constraints = self._extract_constraints(question)

        # If symbolic can answer directly, record it as a high-confidence
        # hint — but still go through graph retrieval + LLM
        symbolic_hint = self._try_recognition(question)
        if symbolic_hint is not None:
            logger.info("Symbolic hint: %s (will inject into context)", symbolic_hint)

        # Phase 2: Reconstruction (constrained graph search)
        query_result = self._constrained_search(
            question, constraints, domain=domain, query_mode=query_mode, **kwargs
        )

        # Phase 3: Validation (inject verified facts + symbolic hint)
        context = self._validate_and_fuse(
            question, query_result, constraints, symbolic_hint=symbolic_hint
        )

        source = "hybrid" if constraints.has_constraints or symbolic_hint else "graph"

        return CognitionResult(
            context=context,
            direct_answer=None,
            source=source,
            constraints=constraints,
            query_result=query_result,
        )

    # ---- Phase 1: Recognition ----

    def _try_recognition(self, question: str) -> str | None:
        """Try to answer directly from symbolic memory."""
        if self.symbolic is None or not self.symbolic.has_state:
            return None
        return self.symbolic.answer_query(question)

    # ---- Phase 2: Reconstruction ----

    def _extract_constraints(self, question: str) -> SymbolicConstraints:
        """Extract symbolic constraints relevant to this query."""
        constraints = SymbolicConstraints()

        if self.symbolic is None or not self.symbolic.has_state:
            return constraints

        q = question.lower()

        # Find entities and agents mentioned in the query
        for entity in self.symbolic.known_entities:
            if entity in q:
                constraints.relevant_entities.add(entity)
        for agent in self.symbolic.known_agents:
            if agent in q:
                constraints.relevant_agents.add(agent)
                # Agents reference entities, so include all entities
                # when agent beliefs are relevant
                constraints.relevant_entities.update(self.symbolic.known_entities)

        if not constraints.relevant_entities and not constraints.relevant_agents:
            return constraints

        # Collect verified facts for relevant entities
        for entity in constraints.relevant_entities:
            attrs = self.symbolic.state.entity_attributes.get(entity, {})
            if attrs:
                constraints.verified_facts[entity] = dict(attrs)

            init = self.symbolic.state.initial_attributes.get(entity, {})
            current = self.symbolic.state.entity_attributes.get(entity, {})
            changed = {k: v for k, v in init.items() if current.get(k) != v}
            if changed:
                constraints.initial_state[entity] = changed

        # Collect agent beliefs
        agents_to_check = constraints.relevant_agents or set(
            self.symbolic.state.agent_beliefs.keys()
        )
        for agent in agents_to_check:
            beliefs = self.symbolic.state.agent_beliefs.get(agent, {})
            if not beliefs:
                continue
            relevant = {
                e: dict(a) for e, a in beliefs.items() if e in constraints.relevant_entities
            }
            if relevant:
                constraints.agent_beliefs[agent] = relevant

        return constraints

    def _constrained_search(
        self,
        question: str,
        constraints: SymbolicConstraints,
        domain: str | None = None,
        query_mode: str = "mergefold",
        **kwargs: Any,
    ) -> QueryResult:
        """Search graph memory, using symbolic constraints to guide retrieval.

        When symbolic state knows about entities in the query, we enhance
        the search query with those entity names to improve BM25/semantic
        matching on relevant nodes.
        """
        if constraints.relevant_entities:
            # Append entity names to improve retrieval relevance
            entity_terms = " ".join(sorted(constraints.relevant_entities))
            enhanced_query = f"{question} {entity_terms}"
        else:
            enhanced_query = question

        return self.query_agent.query_for_qa(
            question=enhanced_query,
            domain=domain,
            query_mode=query_mode,
            **kwargs,
        )

    # ---- Phase 3: Validation ----

    def _validate_and_fuse(
        self,
        question: str,
        query_result: QueryResult,
        constraints: SymbolicConstraints,
        symbolic_hint: str | None = None,
    ) -> str:
        """Fuse graph context with verified symbolic facts.

        Symbolic memory augments the LLM's graph context — it never replaces it.
        When symbolic can answer directly, the answer is injected as a
        high-confidence hint alongside other verified facts, so the LLM
        can reason with both the graph memory and symbolic verification.
        """
        graph_context = query_result.context

        if not constraints.has_constraints and not symbolic_hint:
            return graph_context

        # Build concise verified facts section
        fact_lines: list[str] = []

        # Symbolic recognition hint (highest confidence)
        if symbolic_hint:
            fact_lines.append(f"  ANSWER HINT: {symbolic_hint}")

        # Check query type to decide what facts to inject
        q = question.lower()
        is_belief_query = any(kw in q for kw in ("think", "believe", "look for", "search for"))
        is_initial_query = any(
            kw in q for kw in ("beginning", "initially", "originally", "at first", "start")
        )

        # Initial state facts (only for "beginning/originally" queries)
        if is_initial_query and constraints.initial_state:
            for entity, attrs in sorted(constraints.initial_state.items()):
                for attr, val in sorted(attrs.items()):
                    fact_lines.append(f"  INITIAL: {entity} {attr} was {val}")

        # Agent beliefs (for belief queries, or when beliefs differ from truth)
        if constraints.agent_beliefs:
            for agent, entity_beliefs in sorted(constraints.agent_beliefs.items()):
                for entity, attrs in sorted(entity_beliefs.items()):
                    for attr, believed_val in sorted(attrs.items()):
                        true_val = constraints.verified_facts.get(entity, {}).get(attr)
                        is_false = true_val is not None and believed_val != true_val
                        if is_belief_query or is_false:
                            line = f"  {agent.title()} believes {entity} {attr}: {believed_val}"
                            if is_false:
                                line += f" [FALSE BELIEF — actual: {true_val}]"
                            fact_lines.append(line)

        # World state (for direct state queries)
        if not is_belief_query:
            for entity, attrs in sorted(constraints.verified_facts.items()):
                for attr, val in sorted(attrs.items()):
                    fact_lines.append(f"  VERIFIED: {entity} {attr} = {val}")

        if not fact_lines:
            return graph_context

        verified_section = "=== VERIFIED FACTS (deterministic, use as ground truth) ===\n"
        verified_section += "\n".join(fact_lines)

        return verified_section + "\n\n" + graph_context
