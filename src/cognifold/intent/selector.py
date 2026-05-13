"""Intent selector for choosing actionable intents.

This module provides the IntentSelector class which evaluates intents
and determines which ones should have actions generated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph
    from cognifold.intent.calibrator import IntentCalibrator
    from cognifold.models.node import Node

logger = logging.getLogger(__name__)


@dataclass
class IntentScore:
    """Score for an intent's actionability.

    Attributes:
        intent_id: ID of the intent.
        urgency: Urgency score (0.0-1.0).
        importance: Importance score (0.0-1.0).
        combined: Combined actionability score.
    """

    intent_id: str
    urgency: float
    importance: float
    combined: float


class IntentSelector:
    """Selects intents that should have actions generated.

    Not all intents need immediate action generation. The IntentSelector
    evaluates intents based on:
    - Urgency: How time-sensitive is the intent?
    - Importance: How connected/important is the intent?
    - Status: Only 'pending' intents are considered.

    Example:
        >>> from cognifold.intent import IntentSelector
        >>> selector = IntentSelector(graph)
        >>> actionable = selector.select_actionable_intents(
        ...     current_time=datetime.now(),
        ...     min_urgency=0.3,
        ... )
        >>> for intent in actionable:
        ...     print(f"Actionable: {intent.id}")
    """

    def __init__(
        self,
        graph: ConceptGraph,
        scoring_config: dict[str, Any] | None = None,
        calibrator: IntentCalibrator | None = None,
    ) -> None:
        """Initialize the selector.

        Args:
            graph: The concept graph containing intents.
            scoring_config: Optional scoring configuration.
            calibrator: Optional IntentCalibrator for personalized scoring.
        """
        self.graph = graph
        self.scoring_config = scoring_config or {}
        self.calibrator = calibrator

        # Default weights
        self.urgency_weight = self.scoring_config.get("urgency_weight", 0.6)
        self.importance_weight = self.scoring_config.get("importance_weight", 0.4)

    def select_actionable_intents(
        self,
        current_time: datetime,
        min_urgency: float = 0.3,
        max_intents: int = 5,
    ) -> list[Node]:
        """Select intents that should have actions generated.

        Criteria:
        - Status is 'pending'
        - Combined score above threshold
        - Not already processed in this session

        Args:
            current_time: Current time for urgency calculation.
            min_urgency: Minimum urgency threshold.
            max_intents: Maximum intents to return.

        Returns:
            List of intent nodes ready for action generation.
        """
        from cognifold.models.node import NodeType

        # Get all intent nodes
        intents = list(self.graph.get_nodes_by_type(NodeType.INTENT))

        if not intents:
            return []

        # Score and filter intents
        scored_intents: list[tuple[Node, IntentScore]] = []

        for intent in intents:
            # Check status
            status = intent.data.get("status", "pending")
            if status != "pending":
                continue

            # Calculate scores
            score = self._score_intent(intent, current_time)

            # Check threshold
            if score.urgency >= min_urgency or score.combined >= 0.5:
                scored_intents.append((intent, score))

        # Sort by combined score descending
        scored_intents.sort(key=lambda x: x[1].combined, reverse=True)

        # Return top intents
        selected = [intent for intent, _ in scored_intents[:max_intents]]

        logger.debug(f"Selected {len(selected)} actionable intents from {len(intents)} total")

        return selected

    def _score_intent(self, intent: Node, current_time: datetime) -> IntentScore:
        """Calculate actionability score for an intent.

        If a calibrator is configured, the combined score is multiplied
        by a personalization factor in [0.1, 2.0].

        Args:
            intent: The intent node to score.
            current_time: Current time.

        Returns:
            IntentScore with calculated scores.
        """
        # Calculate urgency from priority and time connections
        urgency = self._calculate_urgency(intent, current_time)

        # Calculate importance from graph connectivity
        importance = self._calculate_importance(intent)

        # Combined score
        combined = self.urgency_weight * urgency + self.importance_weight * importance

        # Apply calibration multiplier if available
        if self.calibrator is not None:
            multiplier = self.calibrator.get_score_multiplier(intent)
            combined = min(1.0, combined * multiplier)

        return IntentScore(
            intent_id=intent.id,
            urgency=urgency,
            importance=importance,
            combined=combined,
        )

    def _calculate_urgency(self, intent: Node, current_time: datetime) -> float:
        """Calculate urgency score for an intent.

        Urgency is based on:
        - Explicit priority field
        - Connection to time nodes with approaching deadlines
        - Recency of creation

        Args:
            intent: The intent node.
            current_time: Current time.

        Returns:
            Urgency score (0.0-1.0).
        """
        priority = intent.data.get("priority", "medium")
        priority_scores = {
            "urgent": 1.0,
            "high": 0.75,
            "medium": 0.5,
            "low": 0.25,
        }
        base_urgency = priority_scores.get(priority, 0.5)

        # Check for time node connections
        time_boost = 0.0
        neighbors = self.graph.get_neighbors(intent.id)
        predecessors = self.graph.get_predecessors(intent.id)

        for node_id in neighbors + predecessors:
            if not self.graph.has_node(node_id):
                continue
            node = self.graph.get_node(node_id)
            if node.type.value == "time":
                # Check if deadline is approaching
                scheduled_str = node.data.get("scheduled_time", "")
                if scheduled_str:
                    try:
                        scheduled = datetime.fromisoformat(scheduled_str.replace("Z", "+00:00"))
                        # Calculate hours until deadline
                        hours_until = (scheduled - current_time).total_seconds() / 3600
                        if hours_until < 24:
                            time_boost = max(time_boost, 0.3)
                        elif hours_until < 48:
                            time_boost = max(time_boost, 0.2)
                        elif hours_until < 168:  # 1 week
                            time_boost = max(time_boost, 0.1)
                    except (ValueError, TypeError):
                        pass

        return min(1.0, base_urgency + time_boost)

    def _calculate_importance(self, intent: Node) -> float:
        """Calculate importance score for an intent.

        Importance is based on graph connectivity:
        - Number of connections to other nodes
        - Types of connected nodes (concepts > events)

        Args:
            intent: The intent node.

        Returns:
            Importance score (0.0-1.0).
        """
        neighbors = self.graph.get_neighbors(intent.id)
        predecessors = self.graph.get_predecessors(intent.id)
        total_connections = len(set(neighbors + predecessors))

        # Base importance from connection count
        # Scale: 0 connections = 0.1, 5+ connections = 0.9
        connection_score = min(0.9, 0.1 + (total_connections * 0.16))

        # Boost for concept connections
        concept_count = 0
        for node_id in neighbors + predecessors:
            if not self.graph.has_node(node_id):
                continue
            node = self.graph.get_node(node_id)
            if node.type.value == "concept":
                concept_count += 1

        concept_boost = min(0.2, concept_count * 0.1)

        return min(1.0, connection_score + concept_boost)

    def get_intent_scores(self, current_time: datetime) -> list[IntentScore]:
        """Get scores for all pending intents.

        Useful for debugging and analysis.

        Args:
            current_time: Current time.

        Returns:
            List of IntentScore objects for all pending intents.
        """
        from cognifold.models.node import NodeType

        intents = list(self.graph.get_nodes_by_type(NodeType.INTENT))
        scores: list[IntentScore] = []

        for intent in intents:
            status = intent.data.get("status", "pending")
            if status == "pending":
                score = self._score_intent(intent, current_time)
                scores.append(score)

        # Sort by combined score
        scores.sort(key=lambda s: s.combined, reverse=True)
        return scores
