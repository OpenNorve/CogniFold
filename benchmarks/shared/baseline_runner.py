#!/usr/bin/env python3
"""Baseline runner for CogniFold benchmarks.

Provides two baseline modes for comparison against CogniFold's graph-based approach:

1. **Direct LLM** (zero-shot): Send full context + question to LLM, no retrieval
2. **Standard RAG**: Chunk context into paragraphs, embed with Gemini, retrieve
   top-k by cosine similarity, send retrieved chunks + question to LLM

Supports all five standard benchmarks: mutual, socialiqa, tomi, babilong, rgb.
Outputs results in the same JSON format as existing benchmark runners.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_project_root, "src"))
sys.path.append(_project_root)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark registry: how to load data and extract fields per benchmark
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY: dict[str, dict[str, Any]] = {
    "mutual": {
        "data_file": "mutual/data/mutual_dev.json",
        "task_type": "mc",  # multiple-choice
    },
    "socialiqa": {
        "data_file": "socialiqa/data/socialiqa_validation.json",
        "task_type": "mc",
    },
    "tomi": {
        "data_file": "tomi/data/tomi_test.json",
        "task_type": "freeform",
    },
    "babilong": {
        "data_file": "babilong/data/babilong_0k_qa1.json",
        "task_type": "freeform",
    },
    "rgb": {
        "data_file": "rgb/data/rgb_test.json",
        "task_type": "freeform",
    },
    "musique": {
        "data_file": "musique/data/musique_validation.json",
        "task_type": "freeform",
    },
    "narrativeqa": {
        "data_file": "narrativeqa/data/narrativeqa_test.json",
        "task_type": "freeform",
    },
    "streamingqa": {
        "data_file": "streamingqa/data/streamingqa_eval.json",
        # Closed-book: dataset has no usable passage text; question only.
        "task_type": "freeform",
    },
    "qmsum": {
        "data_file": "qmsum/data/qmsum_test.json",
        "task_type": "freeform",
    },
    "safetybench": {
        "data_file": "safetybench/data/safetybench_en_test.json",
        "task_type": "mc",
    },
}


# ---------------------------------------------------------------------------
# Benchmark-specific data extraction
# ---------------------------------------------------------------------------


def extract_mutual(example: dict[str, Any]) -> dict[str, Any]:
    """Extract context, question, options, and answer from a MuTual example."""
    article = example.get("article", [])
    dialogue_text = "\n".join(article)
    option_list = example.get("options", [])
    correct_letter = example.get("answers", "A")
    options: dict[str, str] = {}
    for j, opt in enumerate(option_list):
        options[chr(ord("A") + j)] = opt
    question = "Given the following dialogue, what is the best next response?"
    return {
        "context": dialogue_text,
        "question": question,
        "options": options,
        "correct": correct_letter,
        "example_id": example.get("id", ""),
    }


def extract_socialiqa(example: dict[str, Any]) -> dict[str, Any]:
    """Extract context, question, options, and answer from a SocialIQA example."""
    context_text = example.get("context", "")
    question = example.get("question", "")
    options = {
        "A": example.get("answerA", ""),
        "B": example.get("answerB", ""),
        "C": example.get("answerC", ""),
    }
    label_idx = int(example.get("label", "1")) - 1
    correct_letter = chr(ord("A") + label_idx)
    return {
        "context": context_text,
        "question": question,
        "options": options,
        "correct": correct_letter,
    }


def extract_tomi(example: dict[str, Any]) -> dict[str, Any]:
    """Extract context, question, and answer from a ToMi example."""
    story = example.get("story", [])
    context = "\n".join(story)
    question = example.get("question", "")
    answer = example.get("answer", "")
    return {
        "context": context,
        "question": question,
        "correct": answer,
        "question_type": example.get("question_type", "unknown"),
        "example_id": example.get("id", ""),
    }


def extract_babilong(example: dict[str, Any]) -> dict[str, Any]:
    """Extract context, question, and answer from a BABILong example."""
    input_text = example.get("input", "")
    question = example.get("question", "").strip()
    target = example.get("target", "").strip()
    return {
        "context": input_text,
        "question": question,
        "correct": target,
        "task": example.get("task", "qa1"),
        "example_id": example.get("id", ""),
    }


def extract_rgb(example: dict[str, Any]) -> dict[str, Any]:
    """Extract context, question, and answer from an RGB example."""
    # Gather passages
    passages: list[str] = []
    positive = example.get("positive", [])
    negative = example.get("negative", [])
    if positive or negative:
        if isinstance(positive, list):
            passages.extend(str(p) for p in positive)
        if isinstance(negative, list):
            passages.extend(str(p) for p in negative)
    if not passages:
        raw_passages = example.get("passages")
        if raw_passages and isinstance(raw_passages, list):
            for p in raw_passages:
                if isinstance(p, str):
                    passages.append(p)
                elif isinstance(p, dict):
                    passages.append(p.get("text", p.get("content", str(p))))
                else:
                    passages.append(str(p))
    if not passages:
        ctx = example.get("context")
        if ctx:
            if isinstance(ctx, list):
                passages = [str(c) for c in ctx]
            elif isinstance(ctx, str):
                paragraphs = [p.strip() for p in ctx.split("\n\n") if p.strip()]
                passages = paragraphs if len(paragraphs) > 1 else [ctx]

    context = "\n\n".join(passages)
    question = example.get("query", example.get("question", ""))

    raw_answer = example.get("answer", "")
    if isinstance(raw_answer, list):
        gold_answers: list[str] = []
        for item in raw_answer:
            if isinstance(item, list):
                gold_answers.extend(str(a) for a in item)
            else:
                gold_answers.append(str(item))
        gold_answer = gold_answers[0] if gold_answers else ""
    else:
        gold_answer = str(raw_answer)
        gold_answers = [gold_answer]

    return {
        "context": context,
        "question": question,
        "correct": gold_answer,
        "gold_answers": gold_answers,
        "type": example.get("type", "unknown"),
    }


def extract_musique(example: dict[str, Any]) -> dict[str, Any]:
    paragraphs = example.get("paragraphs", [])
    parts: list[str] = []
    for p in paragraphs:
        title = p.get("title", "")
        text = p.get("paragraph_text", "") or p.get("text", "")
        parts.append(f"{title}\n{text}".strip() if title else str(text))
    context = "\n\n".join(parts)
    raw = example.get("answer", "")
    if isinstance(raw, list):
        gold = [str(a) for a in raw]
    else:
        gold = [str(raw)]
    return {
        "context": context,
        "question": example.get("question", ""),
        "correct": gold[0] if gold else "",
        "gold_answers": gold,
        "example_id": example.get("id", ""),
    }


def extract_narrativeqa(example: dict[str, Any]) -> dict[str, Any]:
    doc = example.get("document", {})
    if isinstance(doc, dict):
        context = doc.get("summary", "") or doc.get("text", "")
    else:
        context = str(doc)
    raw = example.get("answers", [])
    if isinstance(raw, list):
        gold = [str(a) for a in raw]
    else:
        gold = [str(raw)]
    return {
        "context": str(context),
        "question": example.get("question", ""),
        "correct": gold[0] if gold else "",
        "gold_answers": gold,
        "example_id": example.get("id", ""),
    }


def extract_streamingqa(example: dict[str, Any]) -> dict[str, Any]:
    """StreamingQA closed-book: dataset has no usable passage text, so we
    issue the question with no context — measures the LLM's parametric
    knowledge of the news fact."""
    raw = example.get("answers", [])
    additional = example.get("answers_additional", []) or []
    gold = [str(a) for a in (raw or [])] + [str(a) for a in additional]
    return {
        "context": "",
        "question": example.get("question", ""),
        "correct": gold[0] if gold else "",
        "gold_answers": gold,
        "example_id": example.get("qa_id", ""),
    }


def extract_qmsum(example: dict[str, Any]) -> dict[str, Any]:
    """QMSum extractor with two-mode behaviour:
    - When called on a raw record (with `meeting_transcripts`): returns the
      transcript + a `_qmsum_subqueries` list, used by load_data to expand
      into many (transcript, query) pairs.
    - When called on an already-flattened record (with `_qmsum_context`):
      returns the standard {context, question, correct, gold_answers} form.
    """
    if "_qmsum_context" in example:
        return {
            "context": example["_qmsum_context"],
            "question": example["_qmsum_query"],
            "correct": example["_qmsum_answer"],
            "gold_answers": [example["_qmsum_answer"]],
        }
    transcripts = example.get("meeting_transcripts", []) or []
    parts: list[str] = []
    for t in transcripts:
        sp = t.get("speaker", "")
        ct = t.get("content", "")
        parts.append(f"{sp}: {ct}".strip(": "))
    context = "\n".join(parts)
    queries: list[dict[str, Any]] = []
    for q in (example.get("general_query_list") or []) + (example.get("specific_query_list") or []):
        queries.append({"query": q.get("query", ""), "answer": q.get("answer", "")})
    return {
        "context": context,
        "_qmsum_subqueries": queries,
    }


def extract_safetybench(example: dict[str, Any]) -> dict[str, Any]:
    """SafetyBench multiple-choice; closed-book (no context)."""
    options = {}
    for letter in ("A", "B", "C", "D"):
        if letter in example and example.get(letter):
            options[letter] = str(example.get(letter))
    correct = example.get("answer", "")
    if isinstance(correct, int):
        correct = chr(ord("A") + correct)
    return {
        "context": "",
        "question": example.get("question", ""),
        "options": options,
        "correct": str(correct).upper()[:1] if correct else "",
        "example_id": example.get("id", ""),
        "category": example.get("category", "unknown"),
    }


EXTRACTORS: dict[str, Any] = {
    "mutual": extract_mutual,
    "socialiqa": extract_socialiqa,
    "tomi": extract_tomi,
    "babilong": extract_babilong,
    "rgb": extract_rgb,
    "musique": extract_musique,
    "narrativeqa": extract_narrativeqa,
    "streamingqa": extract_streamingqa,
    "qmsum": extract_qmsum,
    "safetybench": extract_safetybench,
}


# ---------------------------------------------------------------------------
# LLM helpers (OpenAI + Gemini fallback)
# ---------------------------------------------------------------------------

_LLM_PROVIDER: str | None = None  # "openai" or "gemini", auto-detected


def _detect_provider() -> str:
    """Auto-detect LLM provider from available API keys."""
    global _LLM_PROVIDER
    if _LLM_PROVIDER:
        return _LLM_PROVIDER
    if os.environ.get("OPENAI_API_KEY"):
        _LLM_PROVIDER = "openai"
    elif os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        _LLM_PROVIDER = "gemini"
    else:
        raise RuntimeError("OPENAI_API_KEY or GOOGLE_API_KEY must be set")
    return _LLM_PROVIDER


def call_llm(prompt: str, model: str | None = None, max_retries: int = 3) -> str:
    """Call LLM with retry logic. Auto-detects OpenAI or Gemini."""
    provider = _detect_provider()
    if provider == "openai":
        return _call_openai(prompt, model or "gpt-4o-mini", max_retries)
    else:
        return _call_gemini(prompt, model or "gemini-2.0-flash", max_retries)


def _call_openai(prompt: str, model: str, max_retries: int) -> str:
    from openai import OpenAI
    client = OpenAI()
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.0,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(f"Rate limit, retrying in {wait}s...")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                logger.warning(f"OpenAI call failed (attempt {attempt + 1}): {e}")
                time.sleep(1)
            else:
                logger.error(f"OpenAI call failed after {max_retries} attempts: {e}")
                return ""
    return ""


def _call_gemini(prompt: str, model: str, max_retries: int) -> str:
    google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not google_key:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY must be set")
    from google import genai
    client = genai.Client(api_key=google_key)
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return (response.text or "").strip() if response.text else ""
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(f"Rate limit, retrying in {wait}s...")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                logger.warning(f"Gemini call failed (attempt {attempt + 1}): {e}")
                time.sleep(1)
            else:
                logger.error(f"Gemini call failed after {max_retries} attempts: {e}")
                return ""
    return ""


def answer_mc_direct(
    context: str,
    question: str,
    options: dict[str, str],
    model: str = "gpt-4o-mini",
) -> str:
    """Answer a multiple-choice question. Returns the selected option letter."""
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    prompt = (
        f"Answer the following multiple-choice question based on the context.\n"
        f"Reply with ONLY the letter of the correct answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Answer (letter only):"
    )
    raw = call_llm(prompt, model=model).upper()
    for ch in raw:
        if ch in options:
            return ch
    return raw[:1] if raw else ""


def answer_freeform_direct(
    context: str,
    question: str,
    model: str = "gpt-4o-mini",
) -> str:
    """Answer a free-form question. Returns the answer text."""
    prompt = (
        f"Answer the following question based on the context.\n"
        f"Give a concise answer (a single word or short phrase when possible).\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return call_llm(prompt, model=model)


# ---------------------------------------------------------------------------
# RAG helpers: chunking, embedding, retrieval
# ---------------------------------------------------------------------------


def chunk_text(text: str, max_chunk_chars: int = 500) -> list[str]:
    """Split text into chunks by paragraph boundaries, respecting max size.

    Strategy:
    1. Split on double newlines (paragraphs)
    2. If a paragraph is too long, split on single newlines
    3. If still too long, split on sentence boundaries
    4. If still too long, split by character limit
    """
    if not text.strip():
        return []

    # First split by double newlines
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chunk_chars:
            chunks.append(para)
        else:
            # Split long paragraphs by single newlines
            lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
            current = ""
            for line in lines:
                if current and len(current) + len(line) + 1 > max_chunk_chars:
                    chunks.append(current)
                    current = line
                else:
                    current = f"{current}\n{line}" if current else line
            if current:
                # If still too long, split by sentences
                if len(current) > max_chunk_chars:
                    sentences = re.split(r"(?<=[.!?])\s+", current)
                    buf = ""
                    for sent in sentences:
                        if buf and len(buf) + len(sent) + 1 > max_chunk_chars:
                            chunks.append(buf)
                            buf = sent
                        else:
                            buf = f"{buf} {sent}" if buf else sent
                    if buf:
                        chunks.append(buf)
                else:
                    chunks.append(current)

    return chunks


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Embed texts using OpenAI or Gemini (auto-detected)."""
    if not texts:
        return []
    provider = _detect_provider()
    all_embeddings: list[list[float]] = []
    batch_size = 2048 if provider == "openai" else 100
    for i in range(0, len(texts), batch_size):
        batch = [t[:8000] for t in texts[i : i + batch_size]]
        try:
            if provider == "openai":
                from openai import OpenAI
                client = OpenAI()
                result = client.embeddings.create(
                    model=model or "text-embedding-3-small", input=batch,
                )
                for item in result.data:
                    all_embeddings.append(item.embedding)
            else:
                google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
                from google import genai
                client = genai.Client(api_key=google_key)
                result = client.models.embed_content(
                    model=model or "text-embedding-004", contents=batch,
                )
                for emb in result.embeddings:
                    all_embeddings.append(emb.values)
        except Exception as e:
            logger.error(f"Embedding failed for batch {i}: {e}")
            dim = len(all_embeddings[0]) if all_embeddings else 768
            all_embeddings.extend([[0.0] * dim] * len(batch))
    return all_embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_top_k(
    query_embedding: list[float],
    chunk_embeddings: list[list[float]],
    chunks: list[str],
    k: int = 5,
) -> list[str]:
    """Retrieve top-k chunks by cosine similarity to the query embedding."""
    scored = []
    for i, emb in enumerate(chunk_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        scored.append((sim, i))
    scored.sort(reverse=True)
    return [chunks[idx] for _, idx in scored[:k]]


# ---------------------------------------------------------------------------
# Evaluation helpers (matching existing runner output formats)
# ---------------------------------------------------------------------------


def normalize_answer(text: str) -> str:
    """Normalize answer text for evaluation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\$\.]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_f1(predicted: str, gold: str) -> float:
    """Token-level F1 score."""
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(predicted: str, gold: str) -> bool:
    """Exact match with containment fallback."""
    norm_pred = normalize_answer(predicted)
    norm_gold = normalize_answer(gold)
    if norm_pred == norm_gold:
        return True
    return bool(norm_gold and norm_gold in norm_pred)


# ---------------------------------------------------------------------------
# Wilson score confidence interval
# ---------------------------------------------------------------------------


def wilson_ci_95(n_success: int, n_total: int) -> tuple[float, float]:
    """Compute 95% confidence interval using Wilson score interval.

    Returns (lower, upper) bounds as fractions in [0, 1].
    """
    if n_total == 0:
        return 0.0, 0.0

    z = 1.96  # 95% CI
    p_hat = n_success / n_total
    denominator = 1 + z * z / n_total
    centre = (p_hat + z * z / (2 * n_total)) / denominator
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n_total)) / n_total) / denominator

    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    return lower, upper


# ---------------------------------------------------------------------------
# Main baseline runner
# ---------------------------------------------------------------------------


class BaselineRunner:
    """Runs direct LLM or standard RAG baselines across benchmarks."""

    def __init__(
        self,
        benchmark: str,
        mode: str,
        model: str = "gpt-4o-mini",
        limit: Optional[int] = None,
        top_k: int = 5,
        data_path: Optional[str] = None,
        babilong_config: str = "0k",
        babilong_tasks: str = "qa1",
    ) -> None:
        if benchmark not in BENCHMARK_REGISTRY:
            raise ValueError(
                f"Unknown benchmark: {benchmark}. "
                f"Choose from: {', '.join(BENCHMARK_REGISTRY.keys())}"
            )
        if mode not in ("direct", "rag"):
            raise ValueError(f"Unknown mode: {mode}. Choose from: direct, rag")

        self.benchmark = benchmark
        self.mode = mode
        self.model = model
        self.limit = limit
        self.top_k = top_k
        self.babilong_config = babilong_config
        self.babilong_tasks = babilong_tasks

        bench_dir = Path(__file__).parent.parent
        reg = BENCHMARK_REGISTRY[benchmark]

        if data_path:
            self.data_path = Path(data_path)
        elif benchmark == "babilong":
            self.data_path = bench_dir / "babilong" / "data" / f"babilong_{babilong_config}_{babilong_tasks.split(',')[0]}.json"
        else:
            self.data_path = bench_dir / reg["data_file"]

        self.task_type = reg["task_type"]
        self.extractor = EXTRACTORS[benchmark]

        self.output_dir = bench_dir / "shared" / "output" / f"baseline_{benchmark}_{mode}"

    def load_data(self) -> list[dict[str, Any]]:
        """Load and optionally filter/limit benchmark data."""
        if not self.data_path.exists():
            print(f"ERROR: Dataset not found at {self.data_path}")
            print("Please run the benchmark's download_data.py first.")
            sys.exit(1)

        with open(self.data_path) as f:
            data = json.load(f)

        # BABILong task filtering
        if self.benchmark == "babilong":
            task_list = [t.strip() for t in self.babilong_tasks.split(",")]
            data = [ex for ex in data if ex.get("task") in task_list]
            print(f"Filtered to tasks: {', '.join(task_list)}")

        # QMSum: flatten one example into many (transcript, query) pairs
        # so each query is evaluated independently with shared context.
        if self.benchmark == "qmsum":
            flat: list[dict[str, Any]] = []
            for ex in data:
                ext = self.extractor(ex)
                ctx = ext.get("context", "")
                for sub in ext.get("_qmsum_subqueries", []):
                    flat.append(
                        {
                            "_qmsum_context": ctx,
                            "_qmsum_query": sub["query"],
                            "_qmsum_answer": sub["answer"],
                        }
                    )
            data = flat
            print(f"Flattened to {len(data)} (transcript, query) pairs")

        if self.limit:
            data = data[: self.limit]

        print(f"Loaded {len(data)} examples from {self.data_path}")
        return data

    def run_direct(self, extracted: dict[str, Any]) -> str:
        """Direct LLM mode: send full context + question."""
        context = extracted["context"]
        question = extracted["question"]

        if self.task_type == "mc":
            return answer_mc_direct(context, question, extracted["options"], model=self.model)
        else:
            return answer_freeform_direct(context, question, model=self.model)

    def run_rag(self, extracted: dict[str, Any]) -> str:
        """Standard RAG mode: chunk, embed, retrieve top-k, answer."""
        context = extracted["context"]
        question = extracted["question"]

        # Chunk the context
        chunks = chunk_text(context)
        if not chunks:
            # Fallback to direct if no chunks
            return self.run_direct(extracted)

        # Embed chunks and query
        try:
            all_texts = chunks + [question]
            embeddings = embed_texts(all_texts)
            chunk_embeddings = embeddings[:-1]
            query_embedding = embeddings[-1]
        except Exception as e:
            logger.error(f"Embedding failed, falling back to direct: {e}")
            return self.run_direct(extracted)

        # Retrieve top-k
        retrieved = retrieve_top_k(query_embedding, chunk_embeddings, chunks, k=self.top_k)
        rag_context = "\n\n".join(retrieved)

        if self.task_type == "mc":
            return answer_mc_direct(rag_context, question, extracted["options"], model=self.model)
        else:
            return answer_freeform_direct(rag_context, question, model=self.model)

    def evaluate_one(self, extracted: dict[str, Any], predicted: str) -> dict[str, Any]:
        """Evaluate a single prediction. Returns result dict matching existing formats."""
        correct = extracted["correct"]

        if self.task_type == "mc":
            is_correct = predicted == correct
            result: dict[str, Any] = {
                "question": extracted["question"],
                "correct": correct,
                "predicted": predicted,
                "is_correct": is_correct,
            }
            if "example_id" in extracted:
                result["example_id"] = extracted["example_id"]
            if self.benchmark == "socialiqa":
                result["context"] = extracted.get("context", "")
            return result

        else:
            # Free-form evaluation
            if self.benchmark in ("rgb", "musique", "narrativeqa", "streamingqa", "qmsum"):
                gold_answers = extracted.get("gold_answers", [correct])
                exact_match = any(compute_exact_match(predicted, ga) for ga in gold_answers)
                f1 = max((compute_f1(predicted, ga) for ga in gold_answers), default=0.0)
                result = {
                    "question": extracted.get("question", ""),
                    "gold_answer": correct,
                    "predicted": predicted,
                    "exact_match": exact_match,
                    "f1": f1,
                }
                if self.benchmark == "rgb":
                    result["type"] = extracted.get("type", "unknown")
                return result
            else:
                # tomi / babilong
                target_norm = correct.lower().strip()
                answer_norm = predicted.lower().strip()
                exact_match = target_norm == answer_norm
                contains_match = target_norm in answer_norm if target_norm else False

                # Use LLM verdict for non-exact matches
                if not exact_match:
                    verdict_prompt = (
                        f"Evaluate if the generated answer matches the expected answer.\n\n"
                        f"Question: {extracted['question']}\n"
                        f"Expected Answer: {correct}\n"
                        f"Generated Answer: {predicted}\n\n"
                        f"Reply with ONLY one word: CORRECT, PARTIAL, or INCORRECT"
                    )
                    verdict_raw = call_llm(verdict_prompt, model=self.model).upper()
                    if "CORRECT" in verdict_raw and "INCORRECT" not in verdict_raw:
                        verdict = "CORRECT"
                    elif "PARTIAL" in verdict_raw:
                        verdict = "PARTIAL"
                    else:
                        verdict = "INCORRECT"
                else:
                    verdict = "CORRECT"

                result = {
                    "question": extracted["question"],
                    "target": correct,
                    "generated_answer": predicted,
                    "verdict": verdict,
                    "exact_match": exact_match,
                    "contains_match": contains_match,
                }
                if "example_id" in extracted:
                    result["example_id"] = extracted["example_id"]
                if self.benchmark == "tomi":
                    result["question_type"] = extracted.get("question_type", "unknown")
                if self.benchmark == "babilong":
                    result["task"] = extracted.get("task", "qa1")
                return result

    def run(self) -> None:
        """Run the full baseline evaluation."""
        try:
            provider = _detect_provider()
            print(f"  Provider: {provider}")
        except RuntimeError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        data = self.load_data()
        os.makedirs(self.output_dir, exist_ok=True)

        all_results: list[dict[str, Any]] = []
        total = len(data)

        print(f"\nRunning {self.mode.upper()} baseline on {self.benchmark.upper()}")
        print(f"  Model: {self.model}")
        print(f"  Examples: {total}")
        if self.mode == "rag":
            print(f"  Top-k: {self.top_k}")
        print()

        for i, example in enumerate(data):
            extracted = self.extractor(example)

            print(f"[{i + 1}/{total}] ", end="", flush=True)

            try:
                if self.mode == "direct":
                    predicted = self.run_direct(extracted)
                else:
                    predicted = self.run_rag(extracted)
            except Exception as e:
                logger.error(f"Error on example {i}: {e}")
                predicted = ""

            result = self.evaluate_one(extracted, predicted)
            all_results.append(result)

            # Print inline status
            self._print_inline_status(result)

            # Rate limit pacing
            time.sleep(0.3)

        self._save_results(all_results)
        self._print_summary(all_results)

    def _print_inline_status(self, result: dict[str, Any]) -> None:
        """Print a one-line status for the current example."""
        if self.task_type == "mc":
            status = "CORRECT" if result.get("is_correct") else "INCORRECT"
            print(f"{status} (predicted={result.get('predicted')}, correct={result.get('correct')})")
        elif self.benchmark in ("rgb", "musique", "narrativeqa", "streamingqa", "qmsum"):
            em_str = "EM" if result.get("exact_match") else "NO-EM"
            print(f"{em_str} F1={result.get('f1', 0):.3f} pred='{str(result.get('predicted', ''))[:60]}'")
        else:
            print(f"{result.get('verdict')} (target={result.get('target')}, got={result.get('generated_answer', '')[:60]})")

    def _save_results(self, all_results: list[dict[str, Any]]) -> None:
        """Save results in JSON format matching existing benchmark runners."""
        total = len(all_results)
        if total == 0:
            print("No results to save.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.output_dir / f"baseline_results_{timestamp}.json"

        # Build summary based on benchmark type
        summary = self._build_summary(all_results)

        output = {
            "metadata": {
                "benchmark": self.benchmark,
                "mode": self.mode,
                "model": self.model,
                "timestamp": timestamp,
                "total_examples": total,
                "data_path": str(self.data_path),
            },
            "summary": summary,
            "results": all_results,
        }

        # Add babilong-specific metadata
        if self.benchmark == "babilong":
            output["metadata"]["context_length"] = self.babilong_config
            output["metadata"]["tasks"] = self.babilong_tasks

        with open(results_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {results_path}")

        # Also save as benchmark_results.json for easy comparison
        latest_path = self.output_dir / "benchmark_results.json"
        with open(latest_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Latest results also at {latest_path}")

    def _build_summary(self, all_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Build a summary dict matching the format of the relevant benchmark runner."""
        total = len(all_results)

        if self.task_type == "mc":
            # mutual / socialiqa
            correct_count = sum(1 for r in all_results if r.get("is_correct"))
            accuracy = (correct_count / total * 100) if total > 0 else 0.0
            ci_lower, ci_upper = wilson_ci_95(correct_count, total)
            return {
                "total": total,
                "correct": correct_count,
                "accuracy": accuracy,
                "accuracy_95ci_lower": ci_lower * 100,
                "accuracy_95ci_upper": ci_upper * 100,
            }

        elif self.benchmark in ("rgb", "musique", "narrativeqa", "streamingqa", "qmsum"):
            exact_matches = sum(1 for r in all_results if r.get("exact_match"))
            em_rate = (exact_matches / total * 100) if total > 0 else 0.0
            avg_f1 = sum(r.get("f1", 0) for r in all_results) / total if total > 0 else 0.0
            ci_lower, ci_upper = wilson_ci_95(exact_matches, total)
            if self.benchmark != "rgb":
                return {
                    "total": total,
                    "exact_matches": exact_matches,
                    "exact_match_rate": em_rate,
                    "average_f1": avg_f1,
                    "em_95ci_lower": ci_lower * 100,
                    "em_95ci_upper": ci_upper * 100,
                }

            type_stats: dict[str, dict[str, Any]] = defaultdict(
                lambda: {"em_count": 0, "f1_sum": 0.0, "count": 0}
            )
            for r in all_results:
                t = r.get("type", "unknown")
                type_stats[t]["count"] += 1
                type_stats[t]["f1_sum"] += r.get("f1", 0)
                if r.get("exact_match"):
                    type_stats[t]["em_count"] += 1

            type_summary = {}
            for t, stats in sorted(type_stats.items()):
                count = stats["count"]
                t_em = stats["em_count"]
                t_ci_lo, t_ci_hi = wilson_ci_95(t_em, count)
                type_summary[t] = {
                    "count": count,
                    "exact_match_rate": t_em / count * 100 if count > 0 else 0.0,
                    "average_f1": stats["f1_sum"] / count if count > 0 else 0.0,
                    "em_95ci_lower": t_ci_lo * 100,
                    "em_95ci_upper": t_ci_hi * 100,
                }

            return {
                "total": total,
                "exact_matches": exact_matches,
                "exact_match_rate": em_rate,
                "average_f1": avg_f1,
                "em_95ci_lower": ci_lower * 100,
                "em_95ci_upper": ci_upper * 100,
                "by_type": type_summary,
            }

        else:
            # tomi / babilong
            exact_match_count = sum(1 for r in all_results if r.get("exact_match"))
            contains_match_count = sum(1 for r in all_results if r.get("contains_match"))
            correct = sum(1 for r in all_results if r.get("verdict") == "CORRECT")
            partial = sum(1 for r in all_results if r.get("verdict") == "PARTIAL")
            ci_lower, ci_upper = wilson_ci_95(correct, total)

            summary: dict[str, Any] = {
                "total": total,
                "exact_match": exact_match_count,
                "contains_match": contains_match_count,
                "correct": correct,
                "partial": partial,
                "exact_match_rate": exact_match_count / total if total > 0 else 0.0,
                "correct_rate": correct / total if total > 0 else 0.0,
                "correct_95ci_lower": ci_lower * 100,
                "correct_95ci_upper": ci_upper * 100,
            }

            # Per-type/task breakdown
            group_key = "question_type" if self.benchmark == "tomi" else "task"
            per_group: dict[str, dict[str, int]] = defaultdict(
                lambda: {"total": 0, "correct": 0, "exact_match": 0}
            )
            for r in all_results:
                g = r.get(group_key, "unknown")
                per_group[g]["total"] += 1
                if r.get("verdict") == "CORRECT":
                    per_group[g]["correct"] += 1
                if r.get("exact_match"):
                    per_group[g]["exact_match"] += 1

            breakdown = {}
            for g, stats in sorted(per_group.items()):
                g_ci_lo, g_ci_hi = wilson_ci_95(stats["correct"], stats["total"])
                breakdown[g] = {
                    **stats,
                    "correct_95ci_lower": g_ci_lo * 100,
                    "correct_95ci_upper": g_ci_hi * 100,
                }
            summary[f"per_{group_key}"] = breakdown

            return summary

    def _print_summary(self, all_results: list[dict[str, Any]]) -> None:
        """Print summary to console."""
        total = len(all_results)
        if total == 0:
            print("No results to report.")
            return

        print("\n" + "=" * 60)
        print(f"BASELINE SUMMARY: {self.benchmark.upper()} / {self.mode.upper()}")
        print("=" * 60)
        print(f"  Model: {self.model}")
        print(f"  Total: {total}")

        if self.task_type == "mc":
            correct_count = sum(1 for r in all_results if r.get("is_correct"))
            accuracy = correct_count / total * 100
            ci_lo, ci_hi = wilson_ci_95(correct_count, total)
            print(f"  Correct: {correct_count}/{total} ({accuracy:.1f}%)")
            print(f"  95% CI: [{ci_lo * 100:.1f}%, {ci_hi * 100:.1f}%]")

        elif self.benchmark in ("rgb", "musique", "narrativeqa", "streamingqa", "qmsum"):
            exact_matches = sum(1 for r in all_results if r.get("exact_match"))
            em_rate = exact_matches / total * 100
            avg_f1 = sum(r.get("f1", 0) for r in all_results) / total
            ci_lo, ci_hi = wilson_ci_95(exact_matches, total)
            print(f"  Exact Match: {exact_matches}/{total} ({em_rate:.1f}%)")
            print(f"  Average F1: {avg_f1:.4f}")
            print(f"  EM 95% CI: [{ci_lo * 100:.1f}%, {ci_hi * 100:.1f}%]")

            type_stats: dict[str, dict[str, Any]] = defaultdict(
                lambda: {"em_count": 0, "f1_sum": 0.0, "count": 0}
            )
            for r in all_results:
                t = r.get("type", "unknown")
                type_stats[t]["count"] += 1
                type_stats[t]["f1_sum"] += r.get("f1", 0)
                if r.get("exact_match"):
                    type_stats[t]["em_count"] += 1

            if len(type_stats) > 1 or (len(type_stats) == 1 and "unknown" not in type_stats):
                print("  Breakdown by type:")
                for t, stats in sorted(type_stats.items()):
                    count = stats["count"]
                    t_em = stats["em_count"] / count * 100 if count > 0 else 0.0
                    t_f1 = stats["f1_sum"] / count if count > 0 else 0.0
                    print(f"    {t}: EM={t_em:.1f}% F1={t_f1:.4f} (n={count})")

        else:
            # tomi / babilong
            exact_match_count = sum(1 for r in all_results if r.get("exact_match"))
            correct = sum(1 for r in all_results if r.get("verdict") == "CORRECT")
            partial = sum(1 for r in all_results if r.get("verdict") == "PARTIAL")
            ci_lo, ci_hi = wilson_ci_95(correct, total)

            print(f"  Exact Match: {exact_match_count}/{total} ({exact_match_count / total * 100:.1f}%)")
            print(f"  Correct (verdict): {correct}/{total} ({correct / total * 100:.1f}%)")
            print(f"  Partial: {partial}/{total} ({partial / total * 100:.1f}%)")
            print(f"  Correct 95% CI: [{ci_lo * 100:.1f}%, {ci_hi * 100:.1f}%]")

            group_key = "question_type" if self.benchmark == "tomi" else "task"
            per_group: dict[str, dict[str, int]] = defaultdict(
                lambda: {"total": 0, "correct": 0, "exact_match": 0}
            )
            for r in all_results:
                g = r.get(group_key, "unknown")
                per_group[g]["total"] += 1
                if r.get("verdict") == "CORRECT":
                    per_group[g]["correct"] += 1
                if r.get("exact_match"):
                    per_group[g]["exact_match"] += 1

            if len(per_group) > 1:
                print(f"  Per-{group_key} breakdown:")
                for g, stats in sorted(per_group.items()):
                    pct = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
                    print(f"    {g}: {stats['correct']}/{stats['total']} correct ({pct:.1f}%)")

        print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the baseline."""
    parser = argparse.ArgumentParser(
        description="Run baseline (direct LLM or standard RAG) on CogniFold benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct LLM baseline on MuTual (first 20 examples)
  python baseline_runner.py --benchmark mutual --mode direct --limit 20

  # Standard RAG baseline on ToMi
  python baseline_runner.py --benchmark tomi --mode rag --limit 50

  # RAG on BABILong with specific config
  python baseline_runner.py --benchmark babilong --mode rag --babilong-config 4k --babilong-tasks qa1

  # Direct baseline on RGB with a specific model
  python baseline_runner.py --benchmark rgb --mode direct --model gemini-2.0-flash --limit 30
""",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=list(BENCHMARK_REGISTRY.keys()),
        help="Benchmark to evaluate: mutual, socialiqa, tomi, babilong, rgb",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["direct", "rag"],
        help="Baseline mode: 'direct' (zero-shot LLM) or 'rag' (chunk+embed+retrieve)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to evaluate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Gemini model name (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve in RAG mode (default: 5)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Override data file path",
    )
    parser.add_argument(
        "--babilong-config",
        type=str,
        default="0k",
        help="BABILong context length config (default: 0k)",
    )
    parser.add_argument(
        "--babilong-tasks",
        type=str,
        default="qa1",
        help="BABILong task names, comma-separated (default: qa1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    runner = BaselineRunner(
        benchmark=args.benchmark,
        mode=args.mode,
        model=args.model,
        limit=args.limit,
        top_k=args.top_k,
        data_path=args.data,
        babilong_config=args.babilong_config,
        babilong_tasks=args.babilong_tasks,
    )
    runner.run()


if __name__ == "__main__":
    main()
