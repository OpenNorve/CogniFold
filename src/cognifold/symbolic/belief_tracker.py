"""Symbolic belief tracker for Theory of Mind reasoning.

SymbolicToM-inspired: maintains per-agent belief states as a deterministic
state machine. The LLM cannot reliably track "who knows what" because it
reasons about world state, not mental models. This module provides what
the LLM cannot: deterministic, provably correct belief tracking.

Key insight: absent agents' beliefs are FROZEN — they retain whatever
they last observed, not the current world state.

Usage:
    tracker = SymbolicBeliefTracker()
    for event in events:
        tracker.process_event(event)
    # At QA time:
    answer = tracker.answer_belief_query("Where does Sally think the ball is?")
    if answer:
        return answer  # Skip LLM
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cognifold.models.event import Event


@dataclass
class Action:
    """A parsed action from an event description."""

    type: str  # ENTER, EXIT, MOVE_OBJECT, FACT_ASSERTION
    agent: str = ""
    entity: str = ""
    location: str = ""
    from_loc: str = ""
    to_loc: str = ""
    subject: str = ""
    predicate: str = ""
    value: str = ""


@dataclass
class BeliefState:
    """Complete belief state for a story/scenario."""

    world_state: dict[str, str] = field(default_factory=dict)
    """Current true state: {entity: current_location}"""

    initial_locations: dict[str, str] = field(default_factory=dict)
    """First observed location: {entity: initial_location} (for 'beginning' queries)"""

    agent_beliefs: dict[str, dict[str, str]] = field(default_factory=dict)
    """Per-agent beliefs: {agent: {entity: believed_location}}"""

    agent_locations: dict[str, str | None] = field(default_factory=dict)
    """Where each agent currently is: {agent: location_or_None}"""

    observers: dict[str, set[str]] = field(default_factory=dict)
    """Who is present at each location: {location: {agents_present}}"""


# Regex patterns for ToMi-style action parsing
_ENTER_PATTERN = re.compile(r"(\w+)\s+enter(?:s|ed)?\s+the\s+(\w+)", re.IGNORECASE)
_EXIT_PATTERN = re.compile(
    r"(\w+)\s+(?:exit(?:s|ed)?|left|leaves|leave(?:s)?)\s+the\s+(\w+)", re.IGNORECASE
)
_MOVE_PATTERN = re.compile(r"(\w+)\s+moved\s+the\s+(\w+)\s+to\s+the\s+(\w+)", re.IGNORECASE)
_LOCATION_STATE = re.compile(r"the\s+(\w+)\s+is\s+in\s+the\s+(\w+)", re.IGNORECASE)


class SymbolicBeliefTracker:
    """Deterministic belief tracking per agent.

    Processes events and maintains a symbolic state machine that tracks:
    - World state (ground truth entity locations)
    - Agent beliefs (what each agent believes, based on what they observed)
    - Observer sets (who is present at each location)
    """

    def __init__(self) -> None:
        self.state = BeliefState()
        self._known_agents: set[str] = set()
        self._known_entities: set[str] = set()

    def process_event(self, event: Event) -> None:
        """Parse event description and update belief state.

        Args:
            event: A Cognifold Event to process.
        """
        desc = event.description
        if not desc:
            return

        actions = self._parse_actions(desc)
        for action in actions:
            if action.type == "ENTER":
                self._agent_enters(action.agent, action.location)
            elif action.type == "EXIT":
                self._agent_exits(action.agent, action.location)
            elif action.type == "MOVE_OBJECT":
                self._move_object(action.agent, action.entity, action.to_loc)
            elif action.type == "LOCATION_STATE":
                self._set_initial_location(action.entity, action.location)

    def _parse_actions(self, text: str) -> list[Action]:
        """Parse event text into structured actions.

        Handles ToMi-style sentences. Multiple actions can occur in one text.
        """
        actions: list[Action] = []

        # Check each pattern
        for m in _ENTER_PATTERN.finditer(text):
            agent = m.group(1).strip()
            location = m.group(2).strip()
            actions.append(Action(type="ENTER", agent=agent, location=location))

        for m in _EXIT_PATTERN.finditer(text):
            agent = m.group(1).strip()
            location = m.group(2).strip()
            actions.append(Action(type="EXIT", agent=agent, location=location))

        for m in _MOVE_PATTERN.finditer(text):
            agent = m.group(1).strip()
            entity = m.group(2).strip()
            to_loc = m.group(3).strip()
            actions.append(Action(type="MOVE_OBJECT", agent=agent, entity=entity, to_loc=to_loc))

        for m in _LOCATION_STATE.finditer(text):
            entity = m.group(1).strip()
            location = m.group(2).strip()
            # Only treat as initial state if no MOVE in this text
            if not any(a.type == "MOVE_OBJECT" for a in actions):
                actions.append(Action(type="LOCATION_STATE", entity=entity, location=location))

        return actions

    def _agent_enters(self, agent: str, location: str) -> None:
        """Agent enters a location: add to observers, update their beliefs."""
        agent_lower = agent.lower()
        loc_lower = location.lower()
        self._known_agents.add(agent_lower)

        self.state.agent_locations[agent_lower] = loc_lower

        if loc_lower not in self.state.observers:
            self.state.observers[loc_lower] = set()
        self.state.observers[loc_lower].add(agent_lower)

        # Ensure agent has a beliefs dict
        if agent_lower not in self.state.agent_beliefs:
            self.state.agent_beliefs[agent_lower] = {}

        # When entering, agent can now see the current state of the room.
        # In ToMi scenarios, containers (box, basket) are within rooms,
        # so entering a room means seeing all objects and their containers.
        # Update beliefs for ALL entities to current world state.
        for entity, true_loc in self.state.world_state.items():
            self.state.agent_beliefs[agent_lower][entity] = true_loc

    def _agent_exits(self, agent: str, location: str) -> None:
        """Agent exits a location: remove from observers, beliefs FROZEN."""
        agent_lower = agent.lower()
        loc_lower = location.lower()

        self.state.agent_locations[agent_lower] = None

        if loc_lower in self.state.observers:
            self.state.observers[loc_lower].discard(agent_lower)

        # Key insight: beliefs are NOT updated when leaving.
        # Agent retains their last observation.

    def _move_object(self, mover: str, entity: str, to_loc: str) -> None:
        """Move object: update world state, only update present agents' beliefs.

        Key fix: In ToMi, objects are in containers (green_bucket) but agents
        are in rooms (front_yard). We find witnesses by looking at agents in
        the MOVER's room, not at the container locations.
        """
        mover_lower = mover.lower()
        entity_lower = entity.lower()
        to_loc_lower = to_loc.lower()

        self._known_entities.add(entity_lower)

        # Update world state (ground truth)
        self.state.world_state[entity_lower] = to_loc_lower

        # Find witnesses: all agents in the same ROOM as the mover.
        # In ToMi, containers (box, basket) are inside rooms (kitchen, garden).
        # Agents track room-level presence, so we check the mover's room.
        mover_room = self.state.agent_locations.get(mover_lower)
        all_witnesses = self.state.observers.get(mover_room, set()).copy() if mover_room else set()
        # The mover always witnesses their own action
        all_witnesses.add(mover_lower)

        # Only update beliefs of agents who can see the move
        for agent in self._known_agents:
            if agent in all_witnesses:
                if agent not in self.state.agent_beliefs:
                    self.state.agent_beliefs[agent] = {}
                self.state.agent_beliefs[agent][entity_lower] = to_loc_lower
            # Absent agents: beliefs FROZEN (the key insight from SymbolicToM)

    def _set_initial_location(self, entity: str, location: str) -> None:
        """Set initial entity location (from "the X is in the Y" statements)."""
        entity_lower = entity.lower()
        loc_lower = location.lower()
        self._known_entities.add(entity_lower)
        self.state.world_state[entity_lower] = loc_lower

        # Track the first-ever location for "beginning" queries
        if entity_lower not in self.state.initial_locations:
            self.state.initial_locations[entity_lower] = loc_lower

        # All known agents who are present get this belief
        observers = self.state.observers.get(loc_lower, set())
        for agent in observers:
            if agent not in self.state.agent_beliefs:
                self.state.agent_beliefs[agent] = {}
            self.state.agent_beliefs[agent][entity_lower] = loc_lower

        # Also set for ALL known agents (initial state is common knowledge)
        for agent in self._known_agents:
            if agent not in self.state.agent_beliefs:
                self.state.agent_beliefs[agent] = {}
            self.state.agent_beliefs[agent][entity_lower] = loc_lower

    def answer_belief_query(self, query: str) -> str | None:
        """Try to answer a belief/location query from symbolic state.

        Handles patterns like:
        - "Where does Sally think the ball is?" → Sally's belief about ball
        - "Where is the ball really?" → world state
        - "Where will Sally look for the ball?" → Sally's belief
        - "Where was the ball at the beginning?" → initial location

        Args:
            query: The query string.

        Returns:
            The answer string, or None if the query can't be resolved symbolically.
        """
        q = query.lower()

        agent = self._extract_query_agent(q)
        entity = self._extract_query_entity(q)

        if not entity:
            return None

        # Check for "beginning" / "initially" / "at first" queries → initial location
        if any(kw in q for kw in ("beginning", "initially", "at first", "original", "start")):
            answer = self.state.initial_locations.get(entity)
            if answer:
                return answer

        # If query asks about an agent's belief
        if agent:
            beliefs = self.state.agent_beliefs.get(agent, {})
            answer = beliefs.get(entity)
            if answer:
                return answer

        # If query asks about true location (no agent specified, or "really")
        if not agent or "really" in q or "true" in q or "actual" in q:
            answer = self.state.world_state.get(entity)
            if answer:
                return answer

        return None

    def _extract_query_agent(self, query: str) -> str | None:
        """Extract the agent whose belief is being queried."""
        # "Where does Sally think..."
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

        # Check if any known agent name appears in a belief context
        for agent in self._known_agents:
            if agent in query and ("think" in query or "believe" in query or "look for" in query):
                return agent

        return None

    def _extract_query_entity(self, query: str) -> str | None:
        """Extract the entity being asked about."""
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

    def generate_belief_nodes(self) -> list[dict[str, Any]]:
        """Generate graph node descriptors from current belief state.

        Returns a list of dicts, each describing a concept node to add to the
        graph.  Keys: title, description, symbolic_type, entity, value, and
        optionally agent / is_false_belief.
        """
        nodes: list[dict[str, Any]] = []

        # World-state nodes (ground truth)
        for entity, location in self.state.world_state.items():
            nodes.append(
                {
                    "title": f"[WORLD STATE] {entity} is in the {location}",
                    "description": (
                        f"Verified current location: the {entity} is in the {location}."
                    ),
                    "symbolic_type": "world_state",
                    "entity": entity,
                    "value": location,
                }
            )

        # Initial-location nodes (only when different from current)
        for entity, location in self.state.initial_locations.items():
            if location != self.state.world_state.get(entity):
                nodes.append(
                    {
                        "title": f"[INITIAL STATE] {entity} was originally in the {location}",
                        "description": (
                            f"The {entity} started in the {location} before any moves occurred."
                        ),
                        "symbolic_type": "initial_state",
                        "entity": entity,
                        "value": location,
                    }
                )

        # Per-agent belief nodes
        for agent, beliefs in self.state.agent_beliefs.items():
            for entity, believed_loc in beliefs.items():
                true_loc = self.state.world_state.get(entity, "?")
                is_false = believed_loc != true_loc
                present = self.state.agent_locations.get(agent)

                desc = f"{agent.title()} believes the {entity} is in the {believed_loc}."
                if is_false:
                    desc += (
                        f" This is a FALSE BELIEF — the {entity} is actually in the"
                        f" {true_loc}. {agent.title()} was not present when the"
                        f" {entity} was moved."
                    )
                else:
                    status = f"in the {present}" if present else "no longer present"
                    desc += f" {agent.title()} is {status}."

                nodes.append(
                    {
                        "title": (f"[BELIEF] {agent.title()} thinks {entity} is in {believed_loc}"),
                        "description": desc,
                        "symbolic_type": "agent_belief",
                        "agent": agent,
                        "entity": entity,
                        "value": believed_loc,
                        "is_false_belief": is_false,
                    }
                )

        return nodes

    def get_belief_context(self) -> str:
        """Generate a text summary of all beliefs for graph context injection.

        Returns:
            A formatted string describing world state and agent beliefs.
        """
        lines: list[str] = []

        if self.state.world_state:
            lines.append("=== World State (Ground Truth) ===")
            for entity, loc in sorted(self.state.world_state.items()):
                lines.append(f"  {entity} is in the {loc}")

        for agent in sorted(self.state.agent_beliefs.keys()):
            beliefs = self.state.agent_beliefs[agent]
            if beliefs:
                present = self.state.agent_locations.get(agent)
                status = f"(currently in {present})" if present else "(not present)"
                lines.append(f"\n=== {agent.title()}'s Beliefs {status} ===")
                for entity, loc in sorted(beliefs.items()):
                    true_loc = self.state.world_state.get(entity, "?")
                    marker = " [FALSE BELIEF]" if loc != true_loc else ""
                    lines.append(f"  {agent.title()} believes {entity} is in the {loc}{marker}")

        return "\n".join(lines)
