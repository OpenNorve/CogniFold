"""Cross-encoder reranker for retrieval results.

Wraps a HuggingFace CrossEncoder model (default: BAAI/bge-reranker-v2-m3)
to re-score (query, node) pairs and return the top-k most relevant nodes.
The cross-encoder concatenates query and document into a single transformer
forward pass — slower than dual-encoder cosine but markedly more accurate
on the relevance ranking task. Used after dense retrieval to lift the most
query-relevant raw-turn EVENT nodes ahead of concept summaries.

Iter6 LoCoMo addition. MIT license, fully local (no API cost).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

# Lazy-loaded CrossEncoder cache so each (model_name) is loaded once per process.
_MODEL_CACHE: dict[str, Any] = {}


def _build_text(node: Any, max_chars: int = 800) -> str:
    """Concatenate node title + description for reranking (cap to 800 chars)."""
    title = ""
    desc = ""
    data = getattr(node, "data", None)
    if isinstance(data, dict):
        title = (data.get("title") or "").strip()
        desc = (data.get("description") or "").strip()
    else:
        title = getattr(node, "title", "") or ""
        desc = getattr(node, "description", "") or ""
    text = f"{title}. {desc}".strip()
    if not text or text == ".":
        text = title or desc or ""
    return text[:max_chars]


def get_cross_encoder(model_name: str = "BAAI/bge-reranker-v2-m3") -> Any | None:
    """Load (and cache) a CrossEncoder model. Returns None if import fails."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        logger.warning(
            "sentence_transformers not installed; cross-encoder reranking disabled"
        )
        _MODEL_CACHE[model_name] = None
        return None
    try:
        # max_length 512 is the BGE default; covers most node descriptions.
        model = CrossEncoder(model_name, max_length=512)
    except Exception as e:
        logger.warning(f"Failed to load CrossEncoder {model_name}: {e}")
        _MODEL_CACHE[model_name] = None
        return None
    _MODEL_CACHE[model_name] = model
    logger.info(f"Loaded CrossEncoder reranker: {model_name}")
    return model


def rerank_nodes(
    query: str,
    nodes: Sequence[Any],
    top_k: int | None = None,
    model_name: str = "BAAI/bge-reranker-v2-m3",
) -> list[Any]:
    """Re-rank `nodes` by cross-encoder relevance to `query`.

    Args:
        query: question text.
        nodes: iterable of node-like objects (must have `.data['description']`
            or `.data['title']`).
        top_k: keep only top-k after reranking. None = return all reordered.
        model_name: HF reranker repo. Override via env CF_RERANK_MODEL.

    Returns:
        List of nodes ordered by descending relevance, truncated to top_k.
        On failure (model not available), returns the input list unchanged.
    """
    nodes = list(nodes)
    if not nodes:
        return nodes
    if len(nodes) == 1:
        return nodes
    model_name = os.environ.get("CF_RERANK_MODEL", model_name)
    model = get_cross_encoder(model_name)
    if model is None:
        return nodes
    try:
        pairs = [[query, _build_text(n)] for n in nodes]
        scores = model.predict(pairs, show_progress_bar=False)
    except Exception as e:
        logger.warning(f"CrossEncoder rerank failed, returning original order: {e}")
        return nodes
    ranked = sorted(zip(scores, nodes), key=lambda x: -float(x[0]))
    out = [n for _, n in ranked]
    if top_k is not None and top_k > 0:
        out = out[:top_k]
    return out
