"""Feedback storage backed by the concept graph.

Each feedback is stored as an event node with a USER_FEEDBACK edge
to the target intent, keeping the full audit trail in the graph.
"""

from __future__ import annotations

import logging
import uuid
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING

from cognifold.intent.personalization import (
    FeedbackStats,
    FeedbackType,
    IntentFeedback,
)
from cognifold.models.node import BaseEdgeType, Edge, IntentStatus, Node, NodeType

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph

logger = logging.getLogger(__name__)

_FEEDBACK_EVENT_TYPE = "intent_feedback"


class FeedbackStore:
    """Persists intent feedback as graph nodes + edges.

    Each call to :meth:`add_feedback` creates an event node
    (type ``intent_feedback``) and a ``USER_FEEDBACK`` edge
    pointing at the target intent.
    """

    def __init__(self, graph: ConceptGraph) -> None:
        self.graph = graph

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_feedback(self, fb: IntentFeedback) -> Node:
        """Store feedback as an event node with a USER_FEEDBACK edge.

        Also updates the intent node's status based on the feedback type:
        - REJECT → REJECTED
        - DEFER  → DEFERRED
        - ACCEPT / MODIFY → status unchanged (remains pending/action_scheduled)

        Returns the created feedback event node.
        """
        # Build event node
        node_id = f"e-fb-{fb.feedback_id}"
        data = {
            "event_id": node_id,
            "event_type": _FEEDBACK_EVENT_TYPE,
            "title": f"Feedback: {fb.feedback_type.value} intent {fb.intent_id}",
            "feedback_id": fb.feedback_id,
            "intent_id": fb.intent_id,
            "feedback_type": fb.feedback_type.value,
            "category_tags": fb.category_tags,
        }
        if fb.user_comment:
            data["user_comment"] = fb.user_comment
        if fb.modified_priority:
            data["modified_priority"] = fb.modified_priority
        if fb.modified_description:
            data["modified_description"] = fb.modified_description

        node = Node(
            id=node_id,
            type=NodeType.EVENT,
            data=data,
            created_at=fb.timestamp,
        )
        self.graph.add_node(node)

        # Edge: feedback event → intent
        edge = Edge.create(
            source=node_id,
            target=fb.intent_id,
            edge_type=BaseEdgeType.USER_FEEDBACK.value,
        )
        self.graph.add_edge(edge)

        # Update intent status for reject / defer
        if self.graph.has_node(fb.intent_id):
            new_status: str | None = None
            if fb.feedback_type == FeedbackType.REJECT:
                new_status = IntentStatus.REJECTED.value
            elif fb.feedback_type == FeedbackType.DEFER:
                new_status = IntentStatus.DEFERRED.value

            if new_status:
                self.graph.update_node(fb.intent_id, {"status": new_status})

            # For MODIFY, apply adjustments
            if fb.feedback_type == FeedbackType.MODIFY:
                modify_data: dict[str, object] = {}
                if fb.modified_priority:
                    modify_data["priority"] = fb.modified_priority
                if fb.modified_description:
                    modify_data["description"] = fb.modified_description
                if modify_data:
                    self.graph.update_node(fb.intent_id, modify_data)

        logger.info(
            "Stored feedback %s for intent %s (%s)",
            fb.feedback_id,
            fb.intent_id,
            fb.feedback_type.value,
        )
        return node

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_feedback_for_intent(self, intent_id: str) -> list[IntentFeedback]:
        """Return all feedback entries targeting a specific intent."""
        results: list[IntentFeedback] = []
        for node in self.graph.get_all_nodes():
            if node.data.get("event_type") != _FEEDBACK_EVENT_TYPE:
                continue
            if node.data.get("intent_id") != intent_id:
                continue
            results.append(self._node_to_feedback(node))
        results.sort(key=lambda f: f.timestamp)
        return results

    def get_all_feedback(self) -> list[IntentFeedback]:
        """Return all feedback entries in the graph."""
        results: list[IntentFeedback] = []
        for node in self.graph.get_all_nodes():
            if node.data.get("event_type") == _FEEDBACK_EVENT_TYPE:
                results.append(self._node_to_feedback(node))
        results.sort(key=lambda f: f.timestamp)
        return results

    def get_stats(self) -> FeedbackStats:
        """Compute aggregate feedback statistics."""
        all_fb = self.get_all_feedback()
        if not all_fb:
            return FeedbackStats()

        type_counts = Counter(fb.feedback_type for fb in all_fb)
        accept = type_counts.get(FeedbackType.ACCEPT, 0)
        reject = type_counts.get(FeedbackType.REJECT, 0)
        defer = type_counts.get(FeedbackType.DEFER, 0)
        modify = type_counts.get(FeedbackType.MODIFY, 0)
        total = len(all_fb)

        # By-category breakdown
        by_category: dict[str, dict[str, int]] = {}
        for fb in all_fb:
            for tag in fb.category_tags:
                cat = by_category.setdefault(
                    tag, {"accept": 0, "reject": 0, "defer": 0, "modify": 0}
                )
                cat[fb.feedback_type.value] = cat.get(fb.feedback_type.value, 0) + 1

        return FeedbackStats(
            total_count=total,
            accept_count=accept,
            reject_count=reject,
            defer_count=defer,
            modify_count=modify,
            acceptance_rate=accept / total if total > 0 else 0.0,
            by_category=by_category,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _node_to_feedback(node: Node) -> IntentFeedback:
        """Convert a feedback event node back to an IntentFeedback model."""
        d = node.data
        return IntentFeedback(
            feedback_id=d.get("feedback_id", node.id),
            intent_id=d.get("intent_id", ""),
            feedback_type=FeedbackType(d.get("feedback_type", "accept")),
            user_comment=d.get("user_comment"),
            modified_priority=d.get("modified_priority"),
            modified_description=d.get("modified_description"),
            timestamp=node.created_at,
            category_tags=d.get("category_tags", []),
        )

    @staticmethod
    def make_feedback(
        intent_id: str,
        feedback_type: FeedbackType | str,
        *,
        user_comment: str | None = None,
        modified_priority: str | None = None,
        modified_description: str | None = None,
        category_tags: list[str] | None = None,
    ) -> IntentFeedback:
        """Convenience factory for creating IntentFeedback instances."""
        if not isinstance(feedback_type, FeedbackType):
            feedback_type = FeedbackType(feedback_type)
        return IntentFeedback(
            feedback_id=uuid.uuid4().hex[:12],
            intent_id=intent_id,
            feedback_type=feedback_type,
            user_comment=user_comment,
            modified_priority=modified_priority,
            modified_description=modified_description,
            timestamp=datetime.now(),
            category_tags=category_tags or [],
        )
