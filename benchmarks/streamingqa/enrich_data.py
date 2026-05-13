#!/usr/bin/env python3
"""Enrich StreamingQA dataset with article passages.

The raw StreamingQA eval data contains only questions, answers, and evidence_ids
(references to WMT News Crawl articles). Without the actual article text, the
benchmark is meaningless — it only tests parametric knowledge.

This script enriches the dataset by using the Gemini API to generate
factual article passages based on each question's topic and evidence date.
The generated passages simulate the news articles the system would have
read, allowing the benchmark to properly test memory retrieval.

Usage:
    python enrich_data.py [--limit 500] [--output data/streamingqa_eval_enriched.json]
    python enrich_data.py --resume  # Resume from last checkpoint
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def generate_passages_with_gemini(
    question: str,
    answers: list[str],
    evidence_date: str,
    api_key: str,
    model: str = "gemini-2.5-flash",
) -> list[dict[str, str]]:
    """Generate 2-3 factual news article passages using Gemini.

    Returns a list of passage dicts with 'passage', 'source', 'publication_date'.
    Generates a primary passage containing the answer and a background passage
    with related context, so the graph has richer content for retrieval.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    primary_answer = answers[0] if answers else "N/A"
    additional = answers[1] if len(answers) > 1 else ""

    prompt = f"""Generate 2 short news article paragraphs (3-5 sentences each) dated around {evidence_date}. These should read like real news excerpts covering the same topic from different angles.

**PARAGRAPH 1 — Primary Report**: A news article paragraph that contains the key factual information. It must include the following fact naturally embedded in the text: "{primary_answer}". Do NOT write it as Q&A format.

**PARAGRAPH 2 — Background Context**: A related background paragraph providing additional context about the same topic. Include related entities, dates, or events.{f" Include this detail if relevant: {additional}" if additional else ""}

Topic: {question}
Date: around {evidence_date}

Format your response as:
PARAGRAPH 1:
[text]

PARAGRAPH 2:
[text]

Write ONLY the paragraphs. No headlines, labels, or meta-commentary beyond the PARAGRAPH markers."""

    config = types.GenerateContentConfig(
        temperature=0.3,
        max_output_tokens=2048,
    )

    try:
        response = client.models.generate_content(model=model, contents=prompt, config=config)
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            return []

        # Parse the two paragraphs
        passages = []
        parts = text.split("PARAGRAPH 2:")
        p1_text = parts[0].replace("PARAGRAPH 1:", "").strip()
        p2_text = parts[1].strip() if len(parts) > 1 else ""

        if p1_text:
            passages.append(
                {
                    "passage": p1_text,
                    "source": "primary_report",
                    "publication_date": evidence_date,
                }
            )
        if p2_text:
            passages.append(
                {
                    "passage": p2_text,
                    "source": "background_context",
                    "publication_date": evidence_date,
                }
            )

        return passages
    except Exception as e:
        print(f"  Warning: Gemini call failed: {e}")
        return []


def enrich_dataset(
    input_path: Path,
    output_path: Path,
    limit: int = 500,
    api_key: str | None = None,
    resume: bool = False,
) -> None:
    """Enrich StreamingQA dataset with article passages."""
    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY required for enrichment")
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    examples = data[:limit]

    # Resume from checkpoint if available
    start_idx = 0
    checkpoint_path = output_path.parent / ".enrich_checkpoint.json"
    if resume and output_path.exists():
        with open(output_path) as f:
            enriched_so_far = json.load(f)
        start_idx = len(enriched_so_far)
        examples = data[:limit]  # Full set
        print(f"Resuming from checkpoint: {start_idx}/{len(examples)} already enriched")
    else:
        enriched_so_far = []

    print(f"Enriching {len(examples)} examples from {input_path} (starting at {start_idx})")

    enriched = list(enriched_so_far)
    for i in range(start_idx, len(examples)):
        ex = examples[i]
        question = ex.get("question", "")
        answers = ex.get("answers", [])
        evidence_ts = ex.get("evidence_ts", 0)

        if evidence_ts:
            evidence_date = datetime.fromtimestamp(evidence_ts).strftime("%B %d, %Y")
        else:
            evidence_date = "unknown date"

        print(f"  [{i + 1}/{len(examples)}] {question[:80]}...")

        passages = generate_passages_with_gemini(
            question=question,
            answers=answers,
            evidence_date=evidence_date,
            api_key=api_key,
        )

        if passages:
            # Tag each passage with the qa_id
            qa_id = ex.get("qa_id", str(i))
            for _j, p in enumerate(passages):
                p["source"] = f"{p['source']}_{qa_id}"
            ex["supporting_passages"] = passages
            print(
                f"    OK ({len(passages)} passages, {sum(len(p['passage']) for p in passages)} chars)"
            )
        else:
            print("    FAILED — no passages generated")

        enriched.append(ex)

        # Checkpoint every 50 examples
        if (i + 1) % 50 == 0:
            with open(output_path, "w") as f:
                json.dump(enriched, f, indent=2)
            print(f"  Checkpoint saved ({i + 1}/{len(examples)})")

        time.sleep(0.3)  # Rate limit

    with open(output_path, "w") as f:
        json.dump(enriched, f, indent=2)

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    success = sum(1 for ex in enriched if ex.get("supporting_passages"))
    print(f"\nEnriched {success}/{len(enriched)} examples")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich StreamingQA with article passages")
    parser.add_argument("--limit", type=int, default=500, help="Number of examples to enrich")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "data" / "streamingqa_eval.json",
        help="Input data file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "streamingqa_eval_enriched.json",
        help="Output enriched file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    args = parser.parse_args()

    enrich_dataset(args.input, args.output, args.limit, resume=args.resume)
