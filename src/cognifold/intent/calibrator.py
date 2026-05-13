"""Intent calibrator — learns user preferences from feedback history.

Uses Exponential Moving Average (EMA) to weight recent feedback more
heavily, allowing preferences to drift over time.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING

from cognifold.intent.feedback_store import FeedbackStore
from cognifold.intent.personalization import (
    CalibrationProfile,
    FeedbackType,
    IntentFeedback,
)

if TYPE_CHECKING:
    from cognifold.models.node import Node

logger = logging.getLogger(__name__)

# Clamp range for score multiplier
_MIN_MULTIPLIER = 0.1
_MAX_MULTIPLIER = 2.0

# Max patterns to track
_MAX_PATTERNS = 10


class IntentCalibrator:
    """Builds a calibration profile from feedback and provides scoring adjustments.

    The calibrator computes:
    - Per-category weights via EMA (>1 means preferred, <1 means disliked)
    - A score multiplier for each intent [0.1, 2.0]
    - A prompt context string that can be injected into the agent prompt
    """

    def __init__(self, store: FeedbackStore, ema_alpha: float = 0.3) -> None:
        """Initialize the calibrator.

        Args:
            store: FeedbackStore to read feedback history from.
            ema_alpha: EMA smoothing factor (0 < alpha ≤ 1).
                       Higher = more weight on recent feedback.
        """
        self.store = store
        self.ema_alpha = ema_alpha
        self._cached_profile: CalibrationProfile | None = None

    def compute_profile(self) -> CalibrationProfile:
        """Aggregate all historical feedback into a CalibrationProfile via EMA."""
        all_fb = self.store.get_all_feedback()
        if not all_fb:
            self._cached_profile = CalibrationProfile()
            return self._cached_profile

        # --- Category weights via EMA ---
        category_weights: dict[str, float] = {}
        for fb in all_fb:
            signal = self._feedback_signal(fb.feedback_type)
            for tag in fb.category_tags:
                prev = category_weights.get(tag, 1.0)
                category_weights[tag] = (1 - self.ema_alpha) * prev + self.ema_alpha * signal

        # --- Priority bias ---
        priority_bias = self._compute_priority_bias(all_fb)

        # --- Acceptance rate ---
        stats = self.store.get_stats()
        acceptance_rate = stats.acceptance_rate

        # --- Keyword patterns ---
        rejection_patterns = self._extract_patterns(all_fb, {FeedbackType.REJECT})
        preferred_patterns = self._extract_patterns(all_fb, {FeedbackType.ACCEPT})

        profile = CalibrationProfile(
            category_weights=category_weights,
            priority_bias=priority_bias,
            acceptance_rate=acceptance_rate,
            rejection_patterns=rejection_patterns,
            preferred_patterns=preferred_patterns,
            last_updated=datetime.now(),
        )
        self._cached_profile = profile
        return profile

    def get_score_multiplier(self, intent: Node) -> float:
        """Return a calibrated scoring multiplier for an intent.

        The multiplier is clamped to [0.1, 2.0] so no intent is completely
        zeroed out or over-amplified.
        """
        profile = self._cached_profile or self.compute_profile()

        # Start at neutral
        multiplier = 1.0

        # Category adjustment
        tags = self._extract_intent_tags(intent)
        if tags:
            cat_weights = [profile.category_weights.get(t, 1.0) for t in tags]
            avg_weight = sum(cat_weights) / len(cat_weights)
            multiplier *= avg_weight

        # Rejection pattern penalty
        title = intent.data.get("title", "").lower()
        desc = intent.data.get("description", "").lower()
        text = f"{title} {desc}"
        for pattern in profile.rejection_patterns:
            if pattern.lower() in text:
                multiplier *= 0.7
                break  # Only penalize once

        # Preferred pattern boost
        for pattern in profile.preferred_patterns:
            if pattern.lower() in text:
                multiplier *= 1.3
                break

        return max(_MIN_MULTIPLIER, min(_MAX_MULTIPLIER, multiplier))

    def get_prompt_context(self) -> str:
        """Generate a prompt section describing user intent preferences.

        Returns an empty string if there's no feedback yet.
        """
        profile = self._cached_profile or self.compute_profile()
        stats = self.store.get_stats()

        if stats.total_count == 0:
            return ""

        lines = ["## User Intent Preferences"]

        # Preferred categories
        preferred = [
            tag
            for tag, w in sorted(profile.category_weights.items(), key=lambda x: -x[1])
            if w > 1.1
        ]
        if preferred:
            lines.append(f"- Preferred: {', '.join(preferred[:5])}")

        # Disliked categories
        disliked = [
            tag
            for tag, w in sorted(profile.category_weights.items(), key=lambda x: x[1])
            if w < 0.9
        ]
        if disliked:
            lines.append(f"- Disliked: {', '.join(disliked[:5])}")

        # Rejection/preferred keywords
        if profile.rejection_patterns:
            lines.append(f"- Rejected keywords: {', '.join(profile.rejection_patterns[:5])}")
        if profile.preferred_patterns:
            lines.append(f"- Preferred keywords: {', '.join(profile.preferred_patterns[:5])}")

        # Acceptance rate guidance
        rate = profile.acceptance_rate
        if rate < 0.3:
            lines.append(
                f"- Acceptance rate: {rate:.0%} → be very selective, only suggest high-confidence intents"
            )
        elif rate < 0.6:
            lines.append(f"- Acceptance rate: {rate:.0%} → be moderately selective")
        else:
            lines.append(f"- Acceptance rate: {rate:.0%} → user is receptive to suggestions")

        lines.append(f"- Total feedback received: {stats.total_count}")

        return "\n".join(lines)

    def get_adjusted_min_urgency(self, base_min_urgency: float = 0.3) -> float:
        """Return an adjusted min_urgency threshold based on acceptance rate.

        If acceptance rate is very low (<30%), raise the bar to only surface
        high-confidence intents. If very high (>80%), lower the bar.

        Args:
            base_min_urgency: The default min_urgency threshold.

        Returns:
            Adjusted threshold in [0.1, 0.8].
        """
        profile = self._cached_profile or self.compute_profile()
        rate = profile.acceptance_rate

        if rate < 0.3:
            # User is rejecting most intents → raise threshold
            adjusted = base_min_urgency + 0.2
        elif rate > 0.8:
            # User accepts most → lower threshold to show more
            adjusted = base_min_urgency - 0.1
        else:
            adjusted = base_min_urgency

        return max(0.1, min(0.8, adjusted))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _feedback_signal(ft: FeedbackType) -> float:
        """Convert a feedback type to a numeric signal for EMA.

        ACCEPT → 1.5 (boost), MODIFY → 1.1 (slight boost),
        DEFER → 0.8, REJECT → 0.3.
        """
        return {
            FeedbackType.ACCEPT: 1.5,
            FeedbackType.MODIFY: 1.1,
            FeedbackType.DEFER: 0.8,
            FeedbackType.REJECT: 0.3,
        }[ft]

    @staticmethod
    def _compute_priority_bias(feedback: list[IntentFeedback]) -> float:
        """Estimate whether the system over/under-estimates intent priority.

        Positive bias = system priorities are too high (user rejects high-priority).
        Negative bias = system priorities are too low.
        """
        priority_map = {"urgent": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25}
        adjustments: list[float] = []

        for fb in feedback:
            if fb.feedback_type == FeedbackType.REJECT:
                # Rejection of high priority = system over-estimates
                adjustments.append(0.1)
            elif fb.feedback_type == FeedbackType.ACCEPT:
                adjustments.append(-0.05)
            elif fb.feedback_type == FeedbackType.MODIFY and fb.modified_priority:
                # User changed priority — compute direction
                orig = priority_map.get(fb.modified_priority, 0.5)
                adjustments.append(0.5 - orig)  # Positive if lowered

        if not adjustments:
            return 0.0
        return max(-1.0, min(1.0, sum(adjustments) / len(adjustments)))

    def _extract_patterns(
        self, feedback: list[IntentFeedback], types: set[FeedbackType]
    ) -> list[str]:
        """Extract common title keywords from feedback of given types."""
        word_counts: Counter[str] = Counter()

        for fb in feedback:
            if fb.feedback_type not in types:
                continue
            # Get the intent title if available
            if self.store.graph.has_node(fb.intent_id):
                intent = self.store.graph.get_node(fb.intent_id)
                title = intent.data.get("title", "")
                words = [w.lower().strip() for w in title.split() if len(w) > 3]
                word_counts.update(words)

        # Return most common non-trivial words
        stop_words = {
            "this",
            "that",
            "with",
            "from",
            "have",
            "been",
            "will",
            "should",
            "could",
            "would",
        }
        return [
            word
            for word, _ in word_counts.most_common(_MAX_PATTERNS + len(stop_words))
            if word not in stop_words
        ][:_MAX_PATTERNS]

    @staticmethod
    def _extract_intent_tags(intent: Node) -> list[str]:
        """Extract category tags from an intent node's data."""
        tags: list[str] = []
        # Explicit tags
        if intent.data.get("category_tags"):
            tags.extend(intent.data["category_tags"])
        # Event type as category
        if intent.data.get("event_type"):
            tags.append(intent.data["event_type"])
        # Extract from title keywords as fallback
        if not tags:
            title = intent.data.get("title", "")
            words = [w.lower().strip() for w in title.split() if len(w) > 3]
            tags.extend(words[:3])
        return tags
