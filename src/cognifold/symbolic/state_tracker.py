"""General-purpose symbolic state tracker for Cognifold.

Maintains a deterministic state model that tracks:
- Entity attributes (location, status, ownership, etc.)
- Agent beliefs (what each agent believes about entity attributes)
- Observer model (who can see what — presence-based belief updates)

Unlike the ToMi-specific BeliefTracker, this module works with
LLM-extracted structured actions rather than hardcoded regex patterns.
The LLM does the language understanding; this module does the logic.

Principle: "LLM for language, code for logic."

Usage:
    tracker = SymbolicStateTracker()

    # During ingestion, LLM outputs symbolic_actions alongside operations
    for action in plan.symbolic_actions:
        tracker.process_action(action)

    # After ingestion, inject state into graph and validate LLM concepts
    nodes = tracker.generate_state_nodes()
    corrections = tracker.validate_graph(graph)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph

logger = logging.getLogger(__name__)


@dataclass
class SymbolicAction:
    """A structured action extracted by the LLM during ingestion.

    Action types:
    - STATE_CHANGE: An entity's attribute changed
        (subject, attribute, value, old_value?, actor?)
    - PRESENCE_CHANGE: An agent entered/exited a location
        (agent, location, direction: "enter"|"exit")
    - FACT_ASSERTION: A fact is stated
        (subject, predicate, value)
    """

    type: str  # STATE_CHANGE, PRESENCE_CHANGE, FACT_ASSERTION
    subject: str = ""
    attribute: str = ""
    value: str = ""
    old_value: str = ""
    actor: str = ""
    agent: str = ""
    location: str = ""
    direction: str = ""  # enter / exit
    predicate: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SymbolicAction:
        return cls(
            type=d.get("type", ""),
            subject=d.get("subject", ""),
            attribute=d.get("attribute", ""),
            value=d.get("value", ""),
            old_value=d.get("old_value", ""),
            actor=d.get("actor", ""),
            agent=d.get("agent", ""),
            location=d.get("location", ""),
            direction=d.get("direction", ""),
            predicate=d.get("predicate", ""),
        )


@dataclass
class GeneralState:
    """General-purpose state model."""

    entity_attributes: dict[str, dict[str, str]] = field(default_factory=dict)
    """Current true state: {entity: {attribute: value}}
    e.g. {"ball": {"location": "garden"}, "alice": {"job": "teacher"}}"""

    initial_attributes: dict[str, dict[str, str]] = field(default_factory=dict)
    """First observed values: {entity: {attribute: initial_value}}"""

    agent_beliefs: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)
    """Per-agent beliefs: {agent: {entity: {attribute: believed_value}}}"""

    agent_locations: dict[str, str | None] = field(default_factory=dict)
    """Where each agent currently is: {agent: location_or_None}"""

    observers: dict[str, set[str]] = field(default_factory=dict)
    """Who is present at each location: {location: {agents_present}}"""

    facts: dict[str, dict[str, str]] = field(default_factory=dict)
    """Asserted facts: {subject: {predicate: value}}"""


class SymbolicStateTracker:
    """General-purpose deterministic state tracking.

    Processes LLM-extracted SymbolicActions and maintains a state model.
    Key rules (deterministic, not learned):
    1. When an entity attribute changes, only present agents update beliefs
    2. Absent agents retain their old beliefs (frozen)
    3. Initial values are recorded for "beginning/originally" queries
    4. Facts are accumulated and never overwritten unless explicitly changed
    """

    def __init__(self) -> None:
        self.state = GeneralState()
        self._known_agents: set[str] = set()
        self._known_entities: set[str] = set()
        self._action_count = 0

    @property
    def has_state(self) -> bool:
        """Whether any state has been tracked."""
        return bool(self.state.entity_attributes or self.state.agent_beliefs or self.state.facts)

    @property
    def known_agents(self) -> set[str]:
        """Public accessor for known agent names."""
        return self._known_agents

    @property
    def known_entities(self) -> set[str]:
        """Public accessor for known entity names."""
        return self._known_entities

    def process_action(self, action: SymbolicAction | dict[str, Any]) -> None:
        """Process a single symbolic action and update state."""
        if isinstance(action, dict):
            action = SymbolicAction.from_dict(action)

        action_type = action.type.upper()
        if action_type == "STATE_CHANGE":
            self._handle_state_change(action)
        elif action_type == "PRESENCE_CHANGE":
            self._handle_presence_change(action)
        elif action_type == "FACT_ASSERTION":
            self._handle_fact_assertion(action)
        else:
            logger.debug("Unknown symbolic action type: %s", action_type)

        self._action_count += 1

    def _handle_state_change(self, action: SymbolicAction) -> None:
        """Handle an entity attribute change."""
        subject = action.subject.lower().strip()
        attribute = action.attribute.lower().strip()
        value = action.value.lower().strip()
        actor = action.actor.lower().strip() if action.actor else ""

        if not subject or not attribute or not value:
            return

        if actor and actor not in self._known_agents:
            logger.warning("State change by unknown actor: %s", actor)

        self._known_entities.add(subject)

        # Record initial value
        if subject not in self.state.initial_attributes:
            self.state.initial_attributes[subject] = {}
        if attribute not in self.state.initial_attributes[subject]:
            self.state.initial_attributes[subject][attribute] = value

        # Update ground truth
        if subject not in self.state.entity_attributes:
            self.state.entity_attributes[subject] = {}
        self.state.entity_attributes[subject][attribute] = value

        # Update beliefs of present agents only
        # Find witnesses: agents in the same location as the actor
        witnesses = self._get_witnesses(actor)
        for agent in self._known_agents:
            if agent not in self.state.agent_beliefs:
                self.state.agent_beliefs[agent] = {}
            if subject not in self.state.agent_beliefs[agent]:
                self.state.agent_beliefs[agent][subject] = {}

            if agent in witnesses:
                self.state.agent_beliefs[agent][subject][attribute] = value
            # Absent agents: beliefs FROZEN (key insight)

    def _handle_presence_change(self, action: SymbolicAction) -> None:
        """Handle an agent entering or exiting a location."""
        agent = action.agent.lower().strip()
        location = action.location.lower().strip()
        direction = action.direction.lower().strip()

        if not agent or not location:
            return

        if direction == "exit" and agent not in self._known_agents:
            logger.warning("Exit for unknown agent (never entered): %s", agent)

        self._known_agents.add(agent)

        if direction == "enter":
            self.state.agent_locations[agent] = location

            if location not in self.state.observers:
                self.state.observers[location] = set()
            self.state.observers[location].add(agent)

            # When entering, agent sees current state of everything
            if agent not in self.state.agent_beliefs:
                self.state.agent_beliefs[agent] = {}
            for entity, attrs in self.state.entity_attributes.items():
                if entity not in self.state.agent_beliefs[agent]:
                    self.state.agent_beliefs[agent][entity] = {}
                for attr, val in attrs.items():
                    self.state.agent_beliefs[agent][entity][attr] = val

        elif direction == "exit":
            self.state.agent_locations[agent] = None
            if location in self.state.observers:
                self.state.observers[location].discard(agent)
            # Beliefs FROZEN on exit

    def _handle_fact_assertion(self, action: SymbolicAction) -> None:
        """Handle a factual statement."""
        subject = action.subject.lower().strip()
        predicate = action.predicate.lower().strip() or action.attribute.lower().strip()
        value = action.value.lower().strip()

        if not subject or not value:
            return

        self._known_entities.add(subject)

        if not predicate:
            predicate = "fact"

        if subject not in self.state.facts:
            self.state.facts[subject] = {}
        self.state.facts[subject][predicate] = value

        # Facts are also entity attributes (common knowledge)
        if subject not in self.state.entity_attributes:
            self.state.entity_attributes[subject] = {}
        self.state.entity_attributes[subject][predicate] = value

        # Record initial
        if subject not in self.state.initial_attributes:
            self.state.initial_attributes[subject] = {}
        if predicate not in self.state.initial_attributes[subject]:
            self.state.initial_attributes[subject][predicate] = value

        # All known agents learn asserted facts (common knowledge)
        for agent in self._known_agents:
            if agent not in self.state.agent_beliefs:
                self.state.agent_beliefs[agent] = {}
            if subject not in self.state.agent_beliefs[agent]:
                self.state.agent_beliefs[agent][subject] = {}
            self.state.agent_beliefs[agent][subject][predicate] = value

    def _get_witnesses(self, actor: str) -> set[str]:
        """Get all agents who can witness an action by the actor."""
        if not actor:
            # No actor specified — all present agents witness
            witnesses: set[str] = set()
            for agents in self.state.observers.values():
                witnesses.update(agents)
            return witnesses

        # Find actor's location, then all agents at that location
        actor_loc = self.state.agent_locations.get(actor)
        if actor_loc:
            witnesses = self.state.observers.get(actor_loc, set()).copy()
            witnesses.add(actor)
            return witnesses

        # Actor location unknown — only actor witnesses
        return {actor} if actor in self._known_agents else set()

    def generate_state_nodes(self) -> list[dict[str, Any]]:
        """Generate graph node descriptors from current state.

        Returns list of dicts with: title, description, symbolic_type,
        entity, attribute, value, and optionally agent/is_false_belief.
        """
        nodes: list[dict[str, Any]] = []

        # Entity attribute nodes (ground truth)
        for entity, attrs in self.state.entity_attributes.items():
            for attr, value in attrs.items():
                nodes.append(
                    {
                        "title": f"[WORLD STATE] {entity} {attr}: {value}",
                        "description": f"Verified current state: {entity}'s {attr} is {value}.",
                        "symbolic_type": "world_state",
                        "entity": entity,
                        "attribute": attr,
                        "value": value,
                    }
                )

        # Initial state nodes (when different from current)
        for entity, attrs in self.state.initial_attributes.items():
            for attr, init_val in attrs.items():
                current_val = self.state.entity_attributes.get(entity, {}).get(attr)
                if init_val != current_val:
                    nodes.append(
                        {
                            "title": f"[INITIAL STATE] {entity} {attr} was originally: {init_val}",
                            "description": (
                                f"The {entity}'s {attr} was initially {init_val}"
                                f" before any changes occurred."
                            ),
                            "symbolic_type": "initial_state",
                            "entity": entity,
                            "attribute": attr,
                            "value": init_val,
                        }
                    )

        # Per-agent belief nodes
        for agent, entity_beliefs in self.state.agent_beliefs.items():
            for entity, attrs in entity_beliefs.items():
                for attr, believed_val in attrs.items():
                    true_val = self.state.entity_attributes.get(entity, {}).get(attr, "?")
                    is_false = believed_val != true_val

                    desc = f"{agent.title()} believes {entity}'s {attr} is {believed_val}."
                    if is_false:
                        desc += (
                            f" This is a FALSE BELIEF — the {entity}'s {attr}"
                            f" is actually {true_val}."
                            f" {agent.title()} was not present when the change occurred."
                        )

                    nodes.append(
                        {
                            "title": f"[BELIEF] {agent.title()} thinks {entity} {attr}: {believed_val}",
                            "description": desc,
                            "symbolic_type": "agent_belief",
                            "agent": agent,
                            "entity": entity,
                            "attribute": attr,
                            "value": believed_val,
                            "is_false_belief": is_false,
                        }
                    )

        # Fact nodes
        for subject, preds in self.state.facts.items():
            for pred, value in preds.items():
                # Skip if already covered by entity_attributes
                if self.state.entity_attributes.get(subject, {}).get(pred) == value:
                    continue
                nodes.append(
                    {
                        "title": f"[FACT] {subject} {pred}: {value}",
                        "description": f"Established fact: {subject}'s {pred} is {value}.",
                        "symbolic_type": "fact",
                        "entity": subject,
                        "attribute": pred,
                        "value": value,
                    }
                )

        return nodes

    def answer_query(self, query: str) -> str | None:
        """Try to answer a query directly from symbolic state.

        Handles patterns like:
        - "Where is X?" / "Where is X located?" -> entity location
        - "Where does Y think X is?" -> agent belief about entity location
        - "Where was X at the beginning?" -> initial location
        - "Who moved X?" / "What happened to X?" -> state change history
        - Factual lookups: "What is X's Y?" -> entity attribute

        Returns the answer string, or None if not answerable symbolically.
        """
        q = query.lower().strip()

        # Extract agent (whose belief is being asked about)
        agent = self._extract_query_agent(q)
        # Extract entity being asked about
        entity = self._extract_query_entity(q)

        if not entity:
            return None

        # "beginning" / "initially" / "originally" queries -> initial state
        if any(kw in q for kw in ("beginning", "initially", "at first", "original", "start")):
            init_attrs = self.state.initial_attributes.get(entity, {})
            if "location" in init_attrs:
                return init_attrs["location"]
            if init_attrs:
                # Return first initial attribute
                _attr, val = next(iter(init_attrs.items()))
                return val

        # Agent belief query: "Where does Y think X is?"
        if agent:
            beliefs = self.state.agent_beliefs.get(agent, {})
            entity_beliefs = beliefs.get(entity, {})
            if "location" in entity_beliefs:
                return entity_beliefs["location"]
            if entity_beliefs:
                # Return first belief attribute
                _attr, val = next(iter(entity_beliefs.items()))
                return val

        # Direct entity state query: "Where is X?" / "What is X's location?"
        attrs = self.state.entity_attributes.get(entity, {})
        if attrs:
            # Check for location-specific queries
            if (
                any(kw in q for kw in ("where", "location", "located", "place"))
                and "location" in attrs
            ):
                return attrs["location"]
            # Check for specific attribute queries
            for attr_name, attr_val in attrs.items():
                if attr_name in q:
                    return attr_val
            # If asking "where" and we have any attribute, try location
            if "where" in q and "location" in attrs:
                return attrs["location"]

        # Fact lookup
        facts = self.state.facts.get(entity, {})
        if facts:
            for pred, val in facts.items():
                if pred in q:
                    return val

        return None

    def get_relevant_state_context(self, query: str) -> str:
        """Get only the symbolic state relevant to a query.

        Unlike get_state_context() which returns everything, this filters
        to entities/agents mentioned in the query.
        """
        q = query.lower().strip()
        lines: list[str] = []

        # Find relevant entities and agents
        relevant_entities: set[str] = set()
        relevant_agents: set[str] = set()

        for entity in self._known_entities:
            if entity in q:
                relevant_entities.add(entity)
        for agent in self._known_agents:
            if agent in q:
                relevant_agents.add(agent)

        # If no specific entities/agents found, return full state
        if not relevant_entities and not relevant_agents:
            return self.get_state_context()

        # Include all entities if agents are mentioned (agent beliefs reference entities)
        if relevant_agents:
            relevant_entities = self._known_entities.copy()

        # World state for relevant entities
        if relevant_entities:
            ws_lines = []
            for entity in sorted(relevant_entities):
                attrs = self.state.entity_attributes.get(entity, {})
                for attr, val in sorted(attrs.items()):
                    ws_lines.append(f"  {entity} {attr}: {val}")
            if ws_lines:
                lines.append("=== World State (Ground Truth) ===")
                lines.extend(ws_lines)

        # Agent beliefs for relevant agents (or all agents if entities are queried)
        agents_to_show = relevant_agents or set(self.state.agent_beliefs.keys())
        for agent in sorted(agents_to_show):
            beliefs = self.state.agent_beliefs.get(agent, {})
            if not beliefs:
                continue
            # Filter to relevant entities
            filtered = (
                {e: a for e, a in beliefs.items() if e in relevant_entities}
                if relevant_entities
                else beliefs
            )
            if not filtered:
                continue
            present = self.state.agent_locations.get(agent)
            status = f"(currently in {present})" if present else "(not present)"
            lines.append(f"\n=== {agent.title()}'s Beliefs {status} ===")
            for entity, attrs in sorted(filtered.items()):
                for attr, val in sorted(attrs.items()):
                    true_val = self.state.entity_attributes.get(entity, {}).get(attr, "?")
                    marker = " [FALSE BELIEF]" if val != true_val else ""
                    lines.append(f"  {agent.title()} believes {entity} {attr}: {val}{marker}")

        return "\n".join(lines)

    def _extract_query_agent(self, query: str) -> str | None:
        """Extract the agent whose belief is being queried."""
        import re

        # "Where does Sally think..." / "Where will Sally look..."
        m = re.search(r"(?:does|will|would)\s+(\w+)\s+(?:think|believe|look|search)", query)
        if m:
            agent = m.group(1).lower()
            if agent in self._known_agents:
                return agent

        # "What does Sally believe..."
        m = re.search(r"(?:does|did)\s+(\w+)\s+(?:believe|think|know)", query)
        if m:
            agent = m.group(1).lower()
            if agent in self._known_agents:
                return agent

        # Check if any known agent appears in belief context
        for agent in self._known_agents:
            if agent in query and ("think" in query or "believe" in query or "look for" in query):
                return agent

        return None

    def _extract_query_entity(self, query: str) -> str | None:
        """Extract the entity being asked about."""
        import re

        # "where is the ball" / "where does X think the ball is"
        m = re.search(r"the\s+(\w+)", query)
        if m:
            entity = m.group(1).lower()
            if entity in self._known_entities:
                return entity

        # Try matching any known entity mentioned in query
        for entity in self._known_entities:
            if entity in query:
                return entity

        return None

    def get_state_context(self) -> str:
        """Generate text summary of all tracked state for QA context injection."""
        lines: list[str] = []

        if self.state.entity_attributes:
            lines.append("=== World State (Ground Truth) ===")
            for entity, attrs in sorted(self.state.entity_attributes.items()):
                for attr, val in sorted(attrs.items()):
                    lines.append(f"  {entity} {attr}: {val}")

        for agent in sorted(self.state.agent_beliefs.keys()):
            beliefs = self.state.agent_beliefs[agent]
            if not beliefs:
                continue
            present = self.state.agent_locations.get(agent)
            status = f"(currently in {present})" if present else "(not present)"
            lines.append(f"\n=== {agent.title()}'s Beliefs {status} ===")
            for entity, attrs in sorted(beliefs.items()):
                for attr, val in sorted(attrs.items()):
                    true_val = self.state.entity_attributes.get(entity, {}).get(attr, "?")
                    marker = " [FALSE BELIEF]" if val != true_val else ""
                    lines.append(f"  {agent.title()} believes {entity} {attr}: {val}{marker}")

        if self.state.facts:
            lines.append("\n=== Established Facts ===")
            for subject, preds in sorted(self.state.facts.items()):
                for pred, val in sorted(preds.items()):
                    lines.append(f"  {subject} {pred}: {val}")

        return "\n".join(lines)

    def inject_into_graph(self, graph: ConceptGraph) -> tuple[int, int]:
        """Inject symbolic state nodes into graph and validate LLM concepts.

        Returns (nodes_injected, corrections_made).
        """
        from cognifold.models.node import Edge, Node, NodeType

        state_nodes = self.generate_state_nodes()
        if not state_nodes:
            return 0, 0

        node_ids: list[tuple[str, str, str]] = []  # (node_id, entity, attribute)

        for sdata in state_nodes:
            node_id = f"sym-{uuid.uuid4().hex[:8]}"
            node = Node(
                id=node_id,
                type=NodeType.CONCEPT,
                data={
                    "title": str(sdata["title"]),
                    "description": str(sdata["description"]),
                    "symbolic_type": str(sdata.get("symbolic_type", "")),
                },
                created_at=datetime.now(),
            )
            graph.add_node(node)
            node_ids.append(
                (
                    node_id,
                    str(sdata.get("entity", "")),
                    str(sdata.get("attribute", "")),
                )
            )

        # Connect symbolic nodes to related event nodes
        event_nodes = [n for n in graph.get_all_nodes() if n.type == "event"]
        for sym_id, entity, _attr in node_ids:
            if not entity:
                continue
            for ev in event_nodes:
                desc = (ev.data.get("description", "") or "").lower()
                title = (ev.data.get("title", "") or "").lower()
                if entity in desc or entity in title:
                    graph.add_edge(
                        Edge(source=ev.id, target=sym_id, edge_type="GROUNDS", weight=0.9)
                    )

        # Validate and correct LLM-generated concepts
        corrections = self._validate_llm_concepts(graph)

        return len(state_nodes), corrections

    def _validate_llm_concepts(self, graph: ConceptGraph) -> int:
        """Correct LLM-generated concepts that contradict symbolic state."""
        corrections = 0

        for node in graph.get_all_nodes():
            if node.type != "concept":
                continue
            # Skip our own symbolic nodes
            if node.data.get("symbolic_type"):
                continue

            title = (node.data.get("title", "") or "").lower()
            desc = (node.data.get("description", "") or "").lower()

            # Check agent beliefs
            for agent, entity_beliefs in self.state.agent_beliefs.items():
                for entity, attrs in entity_beliefs.items():
                    for attr, correct_val in attrs.items():
                        # Detect LLM belief concepts
                        if (
                            agent in title
                            and entity in title
                            and ("believ" in title or "think" in title)
                            and correct_val not in desc
                        ):
                            graph.update_node(
                                node.id,
                                {
                                    "title": (
                                        f"{agent.title()} believes {entity} {attr}: {correct_val}"
                                    ),
                                    "description": (
                                        f"{agent.title()} believes the {entity}'s"
                                        f" {attr} is {correct_val}."
                                        f" [Corrected by symbolic tracker]"
                                    ),
                                },
                            )
                            corrections += 1

            # Check entity state concepts
            for entity, attrs in self.state.entity_attributes.items():
                for attr, true_val in attrs.items():
                    if (
                        "true" in title
                        and entity in title
                        and attr in title
                        and true_val not in desc
                    ):
                        graph.update_node(
                            node.id,
                            {
                                "title": f"True {entity} {attr}: {true_val}",
                                "description": (
                                    f"The {entity}'s {attr} is currently {true_val}."
                                    f" [Corrected by symbolic tracker]"
                                ),
                            },
                        )
                        corrections += 1

        return corrections
