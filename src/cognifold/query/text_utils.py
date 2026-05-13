"""Shared text utilities for query processing.

This module provides common text operations used across the query subsystem:
- STOP_WORDS: Comprehensive set of English stop words to filter during keyword extraction
- extract_keywords(): Extract meaningful keywords from text
- compute_text_similarity(): Keyword-based similarity between query and text
"""

from __future__ import annotations

import re

# Comprehensive English stop words for keyword filtering.
# Union of sets from both scoring.py and strategies.py originals.
STOP_WORDS: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "the",
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
        "must",
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
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "and",
        "but",
        "if",
        "or",
        "because",
        "until",
        "while",
        "about",
        "against",
        "up",
        "down",
        "out",
        "off",
        "over",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
    ]
)


def extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from text.

    Tokenizes on non-alpha characters, lowercases, removes stop words
    and words shorter than 3 characters.

    Args:
        text: Input text.

    Returns:
        Set of lowercase keywords.
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in STOP_WORDS and len(w) > 2}


def compute_text_similarity(query: str, text: str) -> float:
    """Compute keyword-based similarity between query and text.

    Uses Jaccard-like scoring weighted by query keywords: the fraction
    of query keywords that appear in the text.

    Args:
        query: The query string.
        text: The text to match against.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    query_keywords = extract_keywords(query)
    text_keywords = extract_keywords(text)

    if not query_keywords or not text_keywords:
        return 0.0

    matches = query_keywords & text_keywords
    if not matches:
        return 0.0

    return len(matches) / len(query_keywords)
