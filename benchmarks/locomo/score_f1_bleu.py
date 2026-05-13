"""Post-process LoCoMo benchmark_results.json to add F1 and BLEU-1 metrics.

Reads `qa_details[]`, computes per-question F1 and BLEU-1 against `expected`,
aggregates per `category_key` + overall, and writes back into the same JSON
under `secondary_metrics`. Idempotent.

Usage:
    python -m benchmarks.locomo.score_f1_bleu <results.json> [<results.json> ...]
"""

from __future__ import annotations

import json
import math
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")


def _normalize(text: object) -> str:
    if text is None or text == "":
        return ""
    text = str(text).lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def _tokens(text: str) -> list[str]:
    return _normalize(text).split()


def squad_f1(prediction: str, gold: str) -> float:
    pred = _tokens(prediction)
    g = _tokens(gold)
    if not pred and not g:
        return 1.0
    if not pred or not g:
        return 0.0
    common = Counter(pred) & Counter(g)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    p = n_common / len(pred)
    r = n_common / len(g)
    return 2 * p * r / (p + r)


def bleu1(prediction: str, gold: str) -> float:
    """Sentence-level BLEU-1 with brevity penalty (matches NLTK BLEU-1, smoothing-free)."""
    pred = _tokens(prediction)
    g = _tokens(gold)
    if not pred or not g:
        return 0.0
    pred_counts = Counter(pred)
    gold_counts = Counter(g)
    overlap = sum(min(pred_counts[t], gold_counts[t]) for t in pred_counts)
    precision = overlap / len(pred) if pred else 0.0
    if precision == 0.0:
        return 0.0
    bp = 1.0 if len(pred) >= len(g) else math.exp(1 - len(g) / len(pred))
    return bp * precision


def _aggregate(values: Iterable[float]) -> float:
    vs = list(values)
    return sum(vs) / len(vs) if vs else 0.0


def annotate(path: Path) -> dict:
    data = json.loads(path.read_text())
    qa = data.get("qa_details", [])
    if not qa:
        return {"path": str(path), "n": 0, "skipped": True}

    per_cat: dict[str, list[tuple[float, float]]] = {}
    scored: list[tuple[float, float]] = []
    for entry in qa:
        pred = entry.get("generated", "") or ""
        gold = entry.get("expected", "") or ""
        f1 = squad_f1(pred, gold)
        b1 = bleu1(pred, gold)
        entry["f1"] = round(f1, 4)
        entry["bleu1"] = round(b1, 4)
        cat = entry.get("category_key", "unknown")
        per_cat.setdefault(cat, []).append((f1, b1))
        # Mem0 protocol: cat 5 (Adversarial, key=unknown, gold=None) excluded from aggregates.
        if cat != "unknown" and gold not in (None, ""):
            scored.append((f1, b1))

    cat_summary = {
        cat: {
            "n": len(items),
            "f1_mean": round(_aggregate(f for f, _ in items), 4),
            "bleu1_mean": round(_aggregate(b for _, b in items), 4),
        }
        for cat, items in sorted(per_cat.items())
    }
    overall = {
        "n": len(scored),
        "n_excluded_adversarial": len(qa) - len(scored),
        "f1_mean": round(_aggregate(f for f, _ in scored), 4),
        "bleu1_mean": round(_aggregate(b for _, b in scored), 4),
    }
    data["secondary_metrics"] = {"per_category": cat_summary, "overall": overall}

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return {"path": str(path), "n": len(qa), "overall": overall, "per_category": cat_summary}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    for raw in argv:
        p = Path(raw)
        if not p.exists():
            print(f"skip (missing): {p}", file=sys.stderr)
            continue
        result = annotate(p)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
