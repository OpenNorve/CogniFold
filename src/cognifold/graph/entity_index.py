"""Entity index for fast entity-to-node lookups.

Extracts named entities from node titles and descriptions, builds an
inverted index mapping normalized entity names to node IDs. Used during
query to supplement BM25/semantic entry points with entity-exact matches.

This module is LLM-free -- entity extraction uses heuristic patterns
(capitalized multi-word phrases, quoted strings, known attribute patterns).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph

logger = logging.getLogger(__name__)

# Words that should not be treated as entities on their own
_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "because",
        "about",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "it",
        "its",
        "they",
        "them",
        "their",
        # Common generic words that appear capitalized at sentence start
        "said",
        "also",
        "however",
        "then",
        "when",
        "where",
        "how",
        "why",
        "there",
        "here",
        "now",
        "still",
        "already",
        "never",
        "always",
        "sometimes",
        "often",
        "usually",
        "one",
        "two",
        "three",
        "four",
        "five",
        "first",
        "second",
        "third",
        "new",
        "old",
        "big",
        "small",
        "long",
        "short",
        "good",
        "bad",
        "great",
        "little",
        "much",
        "many",
        "well",
        "back",
        "even",
        "like",
        "really",
        "went",
        "going",
        "get",
        "got",
        "make",
        "made",
        "know",
        "think",
        "want",
        "see",
        "come",
        "take",
        "find",
        "give",
        "tell",
        "say",
        "ask",
        "use",
        "try",
        "keep",
        "let",
        "begin",
        "seem",
        "help",
        "show",
        "hear",
        "play",
        "run",
        "move",
        "live",
        "believe",
        "user1",
        "user2",
        "speaker1",
        "speaker2",
    }
)

# Pattern for "UserN" or "SpeakerN" references (keep as entities)
_SPEAKER_PATTERN = re.compile(r"\b(user\s*\d+|speaker\s*\d+|person\s*\d+)\b", re.IGNORECASE)

# Capitalized multi-word phrases (e.g. "New York City", "John Smith")
_CAPITALIZED_PHRASE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

# Single capitalized word that's likely a proper noun (>= 2 chars, not sentence-start)
_SINGLE_PROPER_NOUN = re.compile(r"(?<=[.!?]\s|[:,]\s)[A-Z][a-z]{2,}")

# Quoted strings
_QUOTED = re.compile(r'"([^"]{2,50})"')


def normalize_entity(name: str) -> str:
    """Normalize an entity name for index lookups.

    Lowercases, strips articles and extra whitespace.
    """
    name = name.lower().strip()
    # Strip leading articles
    name = re.sub(r"^(the|a|an)\s+", "", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name)
    return name


def extract_entities_from_text(text: str) -> set[str]:
    """Extract entity mentions from text using heuristic NER.

    Returns a set of normalized entity names.
    """
    if not text:
        return set()

    entities: set[str] = set()

    # 1. Speaker/user references
    for match in _SPEAKER_PATTERN.finditer(text):
        raw = match.group(1).lower().replace(" ", "")
        entities.add(raw)

    # 2. Capitalized multi-word phrases (high confidence proper nouns)
    for match in _CAPITALIZED_PHRASE.finditer(text):
        phrase = match.group(1)
        normalized = normalize_entity(phrase)
        # Filter out stop-word-only phrases
        words = normalized.split()
        if any(w not in _STOP_WORDS for w in words):
            entities.add(normalized)

    # 3. Quoted strings (often entity names, titles, etc.)
    for match in _QUOTED.finditer(text):
        quoted = match.group(1).strip()
        normalized = normalize_entity(quoted)
        if len(normalized) >= 2 and normalized not in _STOP_WORDS:
            entities.add(normalized)

    # 4. Single capitalized words mid-sentence (proper nouns)
    # Split into sentences and check non-first words
    sentences = re.split(r"[.!?]\s+", text)
    for sentence in sentences:
        words = sentence.split()
        for word in words[1:]:  # Skip first word (sentence start)
            # Check if it's a capitalized word (not ALL CAPS)
            clean = re.sub(r"[^\w]", "", word)
            if (
                clean
                and clean[0].isupper()
                and not clean.isupper()
                and len(clean) >= 2
                and clean.lower() not in _STOP_WORDS
            ):
                entities.add(clean.lower())

    return entities


class EntityIndex:
    """Inverted index mapping entity names to graph node IDs.

    Build once after ingestion, then query during retrieval to find
    nodes mentioning specific entities.
    """

    def __init__(self) -> None:
        self._index: dict[str, set[str]] = defaultdict(set)
        self._node_entities: dict[str, set[str]] = {}

    @property
    def entity_count(self) -> int:
        """Number of unique entities in the index."""
        return len(self._index)

    def build(self, graph: ConceptGraph) -> None:
        """Scan all nodes and build the entity index.

        Args:
            graph: The concept graph to index.
        """
        self._index.clear()
        self._node_entities.clear()

        for node_id, attrs in graph.internal_graph.nodes(data=True):
            data = attrs.get("data", {})
            title = str(data.get("title", ""))
            description = str(data.get("description", ""))
            text = f"{title} {description}"

            entities = extract_entities_from_text(text)
            if entities:
                self._node_entities[node_id] = entities
                for entity in entities:
                    self._index[entity].add(node_id)

        logger.info(
            "EntityIndex built: %d entities across %d nodes",
            len(self._index),
            len(self._node_entities),
        )

    def query(self, entity_name: str) -> list[str]:
        """Find node IDs that mention an entity.

        Args:
            entity_name: Entity name to look up (will be normalized).

        Returns:
            List of matching node IDs (may be empty).
        """
        normalized = normalize_entity(entity_name)
        return list(self._index.get(normalized, set()))

    def query_text(self, text: str) -> dict[str, list[str]]:
        """Extract entities from query text and return matching nodes.

        Args:
            text: Query text to extract entities from.

        Returns:
            Dict mapping entity names to lists of matching node IDs.
        """
        entities = extract_entities_from_text(text)
        results: dict[str, list[str]] = {}
        for entity in entities:
            node_ids = self.query(entity)
            if node_ids:
                results[entity] = node_ids
        return results

    def query_all_matches(self, text: str) -> list[str]:
        """Extract entities from query text and return all unique matching node IDs.

        Args:
            text: Query text to extract entities from.

        Returns:
            De-duplicated list of node IDs matching any entity in the text.
        """
        entity_matches = self.query_text(text)
        all_ids: set[str] = set()
        for node_ids in entity_matches.values():
            all_ids.update(node_ids)
        return list(all_ids)
