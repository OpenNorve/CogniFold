"""Query intent classification and edge-type weight routing.

MAGMA-inspired: routes queries to relevant graph views by assigning
edge-type weight multipliers based on detected query intent.

Different query intents benefit from different graph traversal strategies:
- "when did X happen?" → prioritize CAUSES / temporal edges
- "why did X happen?" → prioritize CAUSES / TRIGGERS
- "where does Sally think the ball is?" → prioritize GROUNDS (belief nodes)
- "who is X?" → prioritize GROUNDS / REINFORCES (entity nodes)
"""

from __future__ import annotations


class QueryIntent:
    """Known query intent types."""

    TEMPORAL = "TEMPORAL"
    CAUSAL = "CAUSAL"
    ENTITY = "ENTITY"
    SEMANTIC = "SEMANTIC"
    BELIEF = "BELIEF"


# Edge-type weight multipliers per query intent.
# Values > 1.0 boost that edge type; < 1.0 suppress it.
INTENT_EDGE_WEIGHTS: dict[str, dict[str, float]] = {
    QueryIntent.TEMPORAL: {
        "causes": 1.5,
        "deadline_for": 2.0,
        "triggers": 0.5,
        "related_to": 0.3,
        "grounds": 0.8,
        "reinforces": 0.5,
        "part_of": 0.5,
        "derived_from": 0.5,
    },
    QueryIntent.CAUSAL: {
        "causes": 2.0,
        "triggers": 1.5,
        "grounds": 1.0,
        "related_to": 0.5,
        "reinforces": 0.8,
        "part_of": 0.8,
        "derived_from": 0.8,
        "deadline_for": 0.3,
    },
    QueryIntent.ENTITY: {
        "grounds": 1.5,
        "reinforces": 1.5,
        "part_of": 1.0,
        "causes": 0.5,
        "related_to": 1.0,
        "derived_from": 1.0,
        "triggers": 0.5,
        "deadline_for": 0.3,
    },
    QueryIntent.BELIEF: {
        "grounds": 2.0,
        "user_feedback": 1.5,
        "related_to": 0.3,
        "causes": 0.5,
        "triggers": 0.5,
        "reinforces": 1.0,
        "part_of": 0.5,
        "derived_from": 0.5,
    },
    QueryIntent.SEMANTIC: {
        "related_to": 1.2,
        "derived_from": 1.2,
        "part_of": 1.0,
        "grounds": 1.0,
        "reinforces": 1.0,
        "causes": 0.8,
        "triggers": 0.8,
        "deadline_for": 0.5,
    },
}

# Keywords for intent classification
_TEMPORAL_KEYWORDS = frozenset(
    {"when", "what time", "what date", "how long", "before", "after", "during", "until", "since"}
)
_CAUSAL_KEYWORDS = frozenset(
    {"why", "because", "cause", "reason", "lead to", "result in", "due to", "consequence"}
)
_BELIEF_KEYWORDS = frozenset(
    {"think", "believe", "know", "expect", "assume", "suppose", "opinion", "thought", "belief"}
)
_ENTITY_KEYWORDS = frozenset(
    {"who", "where", "which person", "what place", "what is", "what are", "what was"}
)


class QueryIntentRouter:
    """Classifies query intent and returns edge-type weight multipliers."""

    def classify_intent(self, query: str) -> str:
        """Classify query intent using keyword matching.

        Args:
            query: The query string.

        Returns:
            One of the QueryIntent constants.
        """
        q = query.lower()

        # Check each intent's keywords
        for kw in _BELIEF_KEYWORDS:
            if kw in q:
                return QueryIntent.BELIEF

        for kw in _TEMPORAL_KEYWORDS:
            if kw in q:
                return QueryIntent.TEMPORAL

        for kw in _CAUSAL_KEYWORDS:
            if kw in q:
                return QueryIntent.CAUSAL

        for kw in _ENTITY_KEYWORDS:
            if kw in q:
                return QueryIntent.ENTITY

        return QueryIntent.SEMANTIC

    def get_edge_weights(self, query: str) -> dict[str, float]:
        """Return edge-type weight multipliers for the given query.

        Args:
            query: The query string.

        Returns:
            Dict mapping edge_type -> weight multiplier.
        """
        intent = self.classify_intent(query)
        return INTENT_EDGE_WEIGHTS.get(intent, INTENT_EDGE_WEIGHTS[QueryIntent.SEMANTIC])
