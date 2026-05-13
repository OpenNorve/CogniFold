"""BM25 index for keyword-based retrieval."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cognifold.graph.store import ConceptGraph
    from cognifold.models.node import Node

from cognifold.retrieval.result import RetrievalResult


@dataclass
class BM25Config:
    """Configuration for BM25 index.

    Attributes:
        k1: Term frequency saturation parameter (typically 1.2-2.0).
        b: Length normalization parameter (typically 0.75).
        min_term_length: Minimum token length to index.
        stopwords: Set of stopwords to exclude.
    """

    k1: float = 1.5
    b: float = 0.75
    min_term_length: int = 2
    stopwords: set[str] = field(
        default_factory=lambda: {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "but",
            "by",
            "do",
            "for",
            "from",
            "had",
            "has",
            "have",
            "he",
            "her",
            "him",
            "his",
            "if",
            "in",
            "into",
            "is",
            "it",
            "its",
            "my",
            "no",
            "not",
            "of",
            "on",
            "or",
            "our",
            "she",
            "so",
            "than",
            "that",
            "the",
            "their",
            "them",
            "then",
            "there",
            "these",
            "they",
            "this",
            "to",
            "was",
            "we",
            "were",
            "what",
            "when",
            "which",
            "who",
            "will",
            "with",
            "you",
            "your",
        }
    )


class BM25Index:
    """BM25 inverted index for keyword retrieval.

    Implements the Okapi BM25 ranking function:
    score(D, Q) = sum(IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl)))

    Where:
    - IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5))
    - f(qi, D) = term frequency of qi in document D
    - |D| = document length
    - avgdl = average document length
    - k1, b = tuning parameters

    Example:
        >>> index = BM25Index()
        >>> index.build(graph)
        >>> results = index.search("exercise fitness", top_k=10)
    """

    def __init__(self, config: BM25Config | None = None) -> None:
        """Initialize BM25 index.

        Args:
            config: BM25 configuration.
        """
        self.config = config or BM25Config()

        # Inverted index: term -> {doc_id: term_frequency}
        self._inverted_index: dict[str, dict[str, int]] = defaultdict(dict)

        # Document lengths: doc_id -> length
        self._doc_lengths: dict[str, int] = {}

        # Document count
        self._doc_count: int = 0

        # Average document length
        self._avg_doc_length: float = 0.0

        # Document frequency: term -> number of documents containing term
        self._doc_freq: dict[str, int] = defaultdict(int)

        # Node reference for result creation
        self._indexed_graph: ConceptGraph | None = None

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into terms.

        Args:
            text: The text to tokenize.

        Returns:
            List of normalized tokens.
        """
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r"\b\w+\b", text.lower())

        # Filter by length and stopwords
        return [
            t
            for t in tokens
            if len(t) >= self.config.min_term_length and t not in self.config.stopwords
        ]

    def _get_node_text(self, node: Node) -> str:
        """Extract searchable text from a node.

        Args:
            node: The node to extract text from.

        Returns:
            Combined text for indexing.
        """
        parts: list[str] = []

        # Title/name
        title = node.data.get("title") or node.data.get("name") or ""
        if title:
            parts.append(title)

        # Description
        description = node.data.get("description") or ""
        if description:
            parts.append(description)

        # Reasoning
        if node.reasoning:
            parts.append(node.reasoning)

        # Event-specific fields
        event_type = node.data.get("event_type")
        if event_type:
            parts.append(event_type)

        context = node.data.get("context")
        if context:
            if isinstance(context, dict):
                # Index as key:value pairs to preserve entity relationships
                # e.g. {"location": "bathroom", "agent": "Sandra"} becomes
                # "location bathroom agent Sandra location:bathroom agent:Sandra"
                for k, v in context.items():
                    if v is not None:
                        v_str = str(v)
                        parts.append(v_str)
                        parts.append(f"{k}:{v_str}")
            elif isinstance(context, str):
                parts.append(context)
            else:
                parts.append(str(context))

        # Index additional data fields that may contain answer-bearing info
        for key in ("speaker", "entity", "subject", "object", "state", "value"):
            val = node.data.get(key)
            if val and isinstance(val, str):
                parts.append(val)

        # Index raw content field (chat messages, document text, etc.)
        # Critical for LongMemEval/MSC where event nodes store chat in data["content"]
        content = node.data.get("content")
        if content and isinstance(content, str):
            parts.append(content)

        # Index role field for speaker-aware retrieval
        role = node.data.get("role")
        if role and isinstance(role, str):
            parts.append(role)

        # Boost entity names from concept node IDs for better entity-specific retrieval.
        # E.g. "c-mary-location" → boost "mary" so "Where is Mary?" matches correctly.
        if node.id and node.id.startswith("c-"):
            id_parts = node.id.split("-")
            if len(id_parts) >= 2:
                entity_name = id_parts[1]
                if entity_name and len(entity_name) >= 2:
                    for _ in range(3):
                        parts.append(entity_name)

        return " ".join(parts)

    def build(self, graph: ConceptGraph) -> None:
        """Build the BM25 index from graph nodes.

        Args:
            graph: The concept graph to index.
        """
        # Reset index
        self._inverted_index.clear()
        self._doc_lengths.clear()
        self._doc_freq.clear()
        self._doc_count = 0
        self._indexed_graph = graph

        total_length = 0

        for node in graph.get_all_nodes():
            text = self._get_node_text(node)
            tokens = self._tokenize(text)

            # Store document length
            self._doc_lengths[node.id] = len(tokens)
            total_length += len(tokens)
            self._doc_count += 1

            # Build term frequencies
            term_freq: dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1

            # Update inverted index and document frequencies
            for term, freq in term_freq.items():
                self._inverted_index[term][node.id] = freq
                self._doc_freq[term] += 1

        # Calculate average document length
        if self._doc_count > 0:
            self._avg_doc_length = total_length / self._doc_count

    def _calculate_idf(self, term: str) -> float:
        """Calculate IDF (Inverse Document Frequency) for a term.

        Uses the Robertson-Sparck Jones formula:
        IDF = log((N - n + 0.5) / (n + 0.5))

        Args:
            term: The term to calculate IDF for.

        Returns:
            IDF value.
        """
        n = self._doc_freq.get(term, 0)
        if n == 0:
            return 0.0

        # Robertson-Sparck Jones IDF
        return math.log((self._doc_count - n + 0.5) / (n + 0.5) + 1.0)

    def _score_document(self, doc_id: str, query_terms: list[str]) -> float:
        """Calculate BM25 score for a document.

        Args:
            doc_id: The document ID.
            query_terms: List of query terms.

        Returns:
            BM25 score.
        """
        score = 0.0
        doc_length = self._doc_lengths.get(doc_id, 0)

        if doc_length == 0 or self._avg_doc_length == 0:
            return 0.0

        k1 = self.config.k1
        b = self.config.b

        for term in query_terms:
            if term not in self._inverted_index:
                continue

            tf = self._inverted_index[term].get(doc_id, 0)
            if tf == 0:
                continue

            idf = self._calculate_idf(term)

            # BM25 term score
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_length / self._avg_doc_length)
            score += idf * (numerator / denominator)

        return score

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        include_node_types: list[str] | None = None,
        exclude_node_types: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Search the index with a query.

        Args:
            query: The search query.
            top_k: Maximum number of results.
            min_score: Minimum BM25 score threshold.
            include_node_types: Only include these node types.
            exclude_node_types: Exclude these node types.

        Returns:
            List of RetrievalResult sorted by score.
        """
        if self._doc_count == 0:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        # Find candidate documents (documents containing at least one query term)
        candidates: set[str] = set()
        for term in query_terms:
            if term in self._inverted_index:
                candidates.update(self._inverted_index[term].keys())

        # Score candidates
        results: list[RetrievalResult] = []

        for doc_id in candidates:
            # Apply type filters
            node = None
            if self._indexed_graph is not None:
                try:
                    node = self._indexed_graph.get_node(doc_id)
                except KeyError:
                    # Document was added directly (not from the graph)
                    node = None

                if node is not None:
                    if include_node_types is not None and node.type.value not in include_node_types:
                        continue

                    if exclude_node_types is not None and node.type.value in exclude_node_types:
                        continue

            score = self._score_document(doc_id, query_terms)

            if score >= min_score:
                results.append(
                    RetrievalResult(
                        node_id=doc_id,
                        final_score=score,
                        bm25_score=score,
                        node=node,
                    )
                )

        # Sort by score and assign ranks
        results.sort(key=lambda r: r.final_score, reverse=True)

        for rank, result in enumerate(results):
            result.bm25_rank = rank + 1

        return results[:top_k]

    def get_term_frequency(self, term: str, doc_id: str) -> int:
        """Get term frequency in a document.

        Args:
            term: The term.
            doc_id: The document ID.

        Returns:
            Term frequency (0 if not found).
        """
        term = term.lower()
        return self._inverted_index.get(term, {}).get(doc_id, 0)

    def get_document_frequency(self, term: str) -> int:
        """Get document frequency for a term.

        Args:
            term: The term.

        Returns:
            Number of documents containing the term.
        """
        term = term.lower()
        return self._doc_freq.get(term, 0)

    def get_vocabulary_size(self) -> int:
        """Get the number of unique terms in the index.

        Returns:
            Vocabulary size.
        """
        return len(self._inverted_index)

    def get_document_count(self) -> int:
        """Get the number of indexed documents.

        Returns:
            Document count.
        """
        return self._doc_count

    def add_document(self, node: Node) -> None:
        """Add a single document to the index.

        Args:
            node: The node to add.
        """
        text = self._get_node_text(node)
        tokens = self._tokenize(text)

        # Update document length
        old_length = self._doc_lengths.get(node.id, 0)
        self._doc_lengths[node.id] = len(tokens)

        if old_length == 0:
            self._doc_count += 1

        # Update term frequencies
        term_freq: dict[str, int] = defaultdict(int)
        for token in tokens:
            term_freq[token] += 1

        # Update inverted index
        for term, freq in term_freq.items():
            old_freq = self._inverted_index[term].get(node.id, 0)
            self._inverted_index[term][node.id] = freq

            if old_freq == 0:
                self._doc_freq[term] += 1

        # Update average document length
        total_length = sum(self._doc_lengths.values())
        self._avg_doc_length = total_length / self._doc_count if self._doc_count > 0 else 0.0

    def remove_document(self, node_id: str) -> None:
        """Remove a document from the index.

        Args:
            node_id: The node ID to remove.
        """
        if node_id not in self._doc_lengths:
            return

        # Remove from inverted index
        terms_to_remove: list[str] = []
        for term, postings in self._inverted_index.items():
            if node_id in postings:
                del postings[node_id]
                self._doc_freq[term] -= 1

                if self._doc_freq[term] == 0:
                    terms_to_remove.append(term)

        # Clean up empty terms
        for term in terms_to_remove:
            del self._inverted_index[term]
            del self._doc_freq[term]

        # Update counts
        del self._doc_lengths[node_id]
        self._doc_count -= 1

        # Update average document length
        total_length = sum(self._doc_lengths.values())
        self._avg_doc_length = total_length / self._doc_count if self._doc_count > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize index state.

        Returns:
            Dictionary with index data.
        """
        return {
            "inverted_index": {
                term: dict(postings) for term, postings in self._inverted_index.items()
            },
            "doc_lengths": dict(self._doc_lengths),
            "doc_freq": dict(self._doc_freq),
            "doc_count": self._doc_count,
            "avg_doc_length": self._avg_doc_length,
            "config": {
                "k1": self.config.k1,
                "b": self.config.b,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BM25Index:
        """Deserialize index from dictionary.

        Args:
            data: Serialized index data.

        Returns:
            BM25Index instance.
        """
        config_data = data.get("config", {})
        config = BM25Config(
            k1=config_data.get("k1", 1.5),
            b=config_data.get("b", 0.75),
        )

        index = cls(config)
        index._inverted_index = defaultdict(
            dict,
            {term: dict(postings) for term, postings in data.get("inverted_index", {}).items()},
        )
        index._doc_lengths = dict(data.get("doc_lengths", {}))
        index._doc_freq = defaultdict(int, data.get("doc_freq", {}))
        index._doc_count = data.get("doc_count", 0)
        index._avg_doc_length = data.get("avg_doc_length", 0.0)

        return index
