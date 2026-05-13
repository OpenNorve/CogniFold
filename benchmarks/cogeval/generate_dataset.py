"""CogEval-Bench dataset generator.

Takes gold concept graphs and generates:
1. Event streams (LLM-generated from concept specifications)
2. QA pairs across 6 question types
3. Gold annotations for emergence evaluation

Usage:
    OPENAI_API_KEY=... python -m benchmarks.cogeval.generate_dataset \
        --scenario software_engineer --scale small
    OPENAI_API_KEY=... python -m benchmarks.cogeval.generate_dataset --all
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

GOLD_GRAPHS_DIR = Path(__file__).parent / "data" / "gold_graphs"
OUTPUT_DIR = Path(__file__).parent / "data" / "generated"

SCALES: dict[str, dict[str, Any]] = {
    "small": {"event_multiplier": 0.7, "distractor_extra": 0},
    "medium": {"event_multiplier": 1.3, "distractor_extra": 8},
    "large": {"event_multiplier": 2.0, "distractor_extra": 20},
}

QA_TYPES = [
    "factual_recall",
    "multi_hop",
    "temporal_pattern",
    "concept_emergence",
    "state_tracking",
    "adversarial",
]


def _call_openai(
    prompt: str,
    system: str = "You are a helpful assistant that generates realistic event data.",
    temperature: float = 0.8,
    max_tokens: int = 4000,
    model: str = "gpt-4o-mini",
) -> str:
    """Call OpenAI API and return the text response."""
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def _parse_json_from_response(text: str) -> Any:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        start = 0
        end = len(lines)
        for i, line in enumerate(lines):
            if line.strip().startswith("```") and i == 0:
                start = i + 1
                continue
            if line.strip() == "```":
                end = i
                break
        text = "\n".join(lines[start:end])
    return json.loads(text)


def generate_events_for_concept(
    concept: dict,
    scenario: dict,
    base_time: datetime,
    temporal_span_days: int,
    scale_multiplier: float = 1.0,
) -> list[dict]:
    """Generate events for a single concept using LLM."""
    n_events = max(1, int(concept["expected_events"] * scale_multiplier))
    if n_events == 0:
        return []

    actors = scenario.get("actors", ["the person"])
    actor_str = ", ".join(actors) if len(actors) > 1 else actors[0]

    prompt = (
        f"Generate exactly {n_events} realistic events for a person's life.\n\n"
        f"Context: {scenario['description']}\n"
        f"Concept: {concept['label']} — {concept['description']}\n"
        f"Keywords to incorporate naturally: {', '.join(concept['keywords'][:5])}\n"
        f"People involved: {actor_str}\n"
        f"Time span: {temporal_span_days} days\n\n"
        "Requirements:\n"
        "- Each event is a short, concrete description (1-2 sentences)\n"
        "- Events should be specific and grounded (include times, places, details)\n"
        "- Events should naturally relate to the concept but vary in how they "
        "express it\n"
        "- Distribute events across the time span (not all on the same day)\n"
        "- Include small sensory/emotional details that make events feel real\n\n"
        "Return ONLY a JSON array of objects, each with:\n"
        "- \"description\": the event text\n"
        f"- \"day_offset\": which day (0 to {temporal_span_days - 1})\n"
        "- \"hour\": hour of day (0-23)\n"
        "- \"title\": short 3-5 word title\n\n"
        "Example format:\n"
        "[\n"
        "  {\"description\": \"Woke up at 6:30am, made pour-over coffee while "
        "listening to a podcast about distributed systems.\", \"day_offset\": 0, "
        "\"hour\": 6, \"title\": \"Morning coffee routine\"}\n"
        "]"
    )

    try:
        resp = _call_openai(prompt, temperature=0.9)
        events_raw = _parse_json_from_response(resp)
    except Exception as e:
        print(f"  Warning: LLM generation failed for {concept['id']}: {e}")
        events_raw = []
        for i in range(n_events):
            kw = concept["keywords"][i % len(concept["keywords"])]
            events_raw.append(
                {
                    "description": f"Event related to {concept['label']}: {kw}",
                    "day_offset": (i * temporal_span_days) // max(n_events, 1),
                    "hour": 9 + i,
                    "title": f"{concept['label']} event {i + 1}",
                }
            )

    events = []
    for ev in events_raw[:n_events]:
        day = ev.get("day_offset", 0) % temporal_span_days
        hour = ev.get("hour", 12) % 24
        ts = base_time + timedelta(
            days=day, hours=hour, minutes=random.randint(0, 59)
        )
        events.append(
            {
                "event_id": str(uuid.uuid4()),
                "timestamp": ts.isoformat(),
                "source": scenario.get("domain", "personal-timeline"),
                "event_type": "life_event",
                "title": ev.get("title", f"{concept['label']} event"),
                "description": ev["description"],
                "gold_concept": concept["id"],
                "gold_concept_label": concept["label"],
            }
        )

    return events


def generate_chain_events(
    chain: dict, base_time: datetime, temporal_span_days: int
) -> list[dict]:
    """Generate events for a planted multi-hop chain."""
    events = []
    n_steps = len(chain["steps"])
    gap = max(1, temporal_span_days // (n_steps + 1))

    for i, step in enumerate(chain["steps"]):
        day = (i + 1) * gap
        hour = random.randint(8, 18)
        ts = base_time + timedelta(days=min(day, temporal_span_days - 1), hours=hour)
        events.append(
            {
                "event_id": str(uuid.uuid4()),
                "timestamp": ts.isoformat(),
                "source": "personal-timeline",
                "event_type": "life_event",
                "title": step["event"][:50].rstrip("."),
                "description": step["event"],
                "gold_chain_id": chain["id"],
                "gold_chain_step": i,
                "gold_entity": step.get("entity", ""),
                "gold_actor": step.get("actor", ""),
            }
        )

    return events


def generate_distractor_events(
    scenario: dict,
    n_distractors: int,
    base_time: datetime,
    temporal_span_days: int,
) -> list[dict]:
    """Generate distractor events unrelated to the scenario's concepts."""
    prompt = (
        f"Generate exactly {n_distractors} random, unrelated life events that "
        f"have NOTHING to do with: {scenario['description']}\n\n"
        "These events should be about completely different topics like:\n"
        "- Random news (weather, sports scores, celebrity gossip)\n"
        "- Mundane unrelated activities (found a penny, saw a funny bumper sticker)\n"
        "- Random observations (the elevator music was different today)\n\n"
        "Return ONLY a JSON array of objects with \"description\", "
        f"\"day_offset\" (0-{temporal_span_days - 1}), \"hour\" (0-23), \"title\".\n"
    )

    try:
        resp = _call_openai(prompt, temperature=1.0)
        raw = _parse_json_from_response(resp)
    except Exception:
        raw = [
            {
                "description": (
                    f"Random unrelated event #{i}: saw a cloud shaped like a cat"
                ),
                "day_offset": i % temporal_span_days,
                "hour": 12,
                "title": f"Random observation {i}",
            }
            for i in range(n_distractors)
        ]

    events = []
    for ev in raw[:n_distractors]:
        day = int(ev.get("day_offset", 0)) % temporal_span_days
        hour = int(ev.get("hour", 12)) % 24
        ts = base_time + timedelta(
            days=day, hours=hour, minutes=random.randint(0, 59)
        )
        events.append(
            {
                "event_id": str(uuid.uuid4()),
                "timestamp": ts.isoformat(),
                "source": "personal-timeline",
                "event_type": "observation",
                "title": ev.get("title", "Random event"),
                "description": ev["description"],
                "gold_concept": "__distractor__",
                "gold_concept_label": "Distractor",
            }
        )

    return events


def generate_factual_recall_questions(
    events: list[dict], n: int = 8
) -> list[dict]:
    """Generate factual recall questions from individual events."""
    candidates = [e for e in events if e.get("gold_concept") != "__distractor__"]
    random.shuffle(candidates)
    candidates = candidates[: min(n * 2, len(candidates))]

    if not candidates:
        return []

    events_text = "\n".join(
        f"- [{e['timestamp'][:10]}] {e['description']}" for e in candidates
    )

    prompt = (
        f"Given these life events, generate exactly {n} factual recall "
        "questions.\n\n"
        f"Events:\n{events_text}\n\n"
        "Requirements:\n"
        "- Each question should be answerable from a SINGLE event\n"
        "- Questions should ask about specific details (who, what, when, where)\n"
        "- Answers should be short (1-2 sentences), directly from the event text\n"
        "- Vary question formats (What did..., When did..., Where was..., Who...)\n\n"
        "Return ONLY a JSON array:\n"
        "[\n"
        "  {\"question\": \"...\", \"answer\": \"...\", "
        "\"source_event_description\": \"the event text this comes from\"}\n"
        "]"
    )

    try:
        resp = _call_openai(prompt, temperature=0.5, max_tokens=3000)
        qas = _parse_json_from_response(resp)
    except Exception as e:
        print(f"  Warning: Factual QA generation failed: {e}")
        return []

    result = []
    for qa in qas[:n]:
        result.append(
            {
                "question_id": str(uuid.uuid4()),
                "question": qa["question"],
                "answer": qa["answer"],
                "type": "factual_recall",
                "difficulty": "easy",
                "source_events": [qa.get("source_event_description", "")],
                "requires_hops": 1,
            }
        )
    return result


def generate_multi_hop_questions(chains: list[dict]) -> list[dict]:
    """Generate multi-hop questions from planted chains (gold answers)."""
    result = []
    for chain in chains:
        result.append(
            {
                "question_id": str(uuid.uuid4()),
                "question": chain["question"],
                "answer": chain["answer"],
                "type": "multi_hop",
                "difficulty": chain.get("difficulty", "medium"),
                "source_chain_id": chain["id"],
                "requires_hops": chain["hops"],
                "source_events": [step["event"] for step in chain["steps"]],
            }
        )
    return result


def generate_temporal_questions(
    events: list[dict], n: int = 6
) -> list[dict]:
    """Generate temporal pattern questions about event ordering and sequences."""
    real_events = sorted(
        [e for e in events if e.get("gold_concept") != "__distractor__"],
        key=lambda e: e["timestamp"],
    )
    if len(real_events) < 10:
        return []

    events_text = "\n".join(
        f"- [{e['timestamp'][:16]}] {e['description']}" for e in real_events[:30]
    )

    prompt = (
        f"Given these chronologically ordered life events, generate exactly {n} "
        "temporal reasoning questions.\n\n"
        f"Events (chronological):\n{events_text}\n\n"
        "Question types to include:\n"
        "1. \"Before/After\" questions: \"What happened before/after X?\"\n"
        "2. \"Sequence\" questions: \"What was the order of events related to Y?\"\n"
        "3. \"Temporal gap\" questions: \"How long between X and Y?\"\n"
        "4. \"First/Last\" questions: \"What was the first time X happened?\"\n\n"
        "Requirements:\n"
        "- Each answer must be derivable from the event timestamps and descriptions\n"
        "- Include mix of difficulty levels\n"
        "- Answers should be specific and grounded\n\n"
        "Return ONLY a JSON array:\n"
        "[\n"
        "  {\"question\": \"...\", \"answer\": \"...\", "
        "\"temporal_type\": \"before_after|sequence|gap|first_last\"}\n"
        "]"
    )

    try:
        resp = _call_openai(prompt, temperature=0.5, max_tokens=3000)
        qas = _parse_json_from_response(resp)
    except Exception as e:
        print(f"  Warning: Temporal QA generation failed: {e}")
        return []

    result = []
    for qa in qas[:n]:
        result.append(
            {
                "question_id": str(uuid.uuid4()),
                "question": qa["question"],
                "answer": qa["answer"],
                "type": "temporal_pattern",
                "difficulty": "medium",
                "temporal_subtype": qa.get("temporal_type", "sequence"),
                "requires_hops": 2,
            }
        )
    return result


def generate_concept_emergence_questions(
    concepts: list[dict],
    relationships: list[dict],
    events: list[dict],
) -> list[dict]:
    """Generate structure-dependent concept emergence questions.

    These questions REQUIRE having formed concept nodes and edges to answer.
    A flat RAG system retrieving raw events cannot answer them because:
    - Cross-concept questions require traversing edges between concepts
    - Reinforcement questions require counting event-concept links
    - Hierarchy questions require knowing parent-child relationships
    - Causal impact questions require following TRIGGERS/CAUSES edges
    """
    result: list[dict] = []

    concept_map = {c["id"]: c for c in concepts}

    concept_events: dict[str, list[str]] = {}
    for ev in events:
        cid = ev.get("gold_concept", "")
        if not cid:
            continue
        if cid == "__distractor__":
            continue
        concept_events.setdefault(cid, []).append(ev["description"])

    leaf_concepts = [
        c
        for c in concepts
        if not c.get("is_hierarchy_parent") and c["id"] in concept_events
    ]

    causal_edges = [r for r in relationships if r["type"] in ("TRIGGERS", "CAUSES")]
    for edge in causal_edges:
        src = concept_map.get(edge["source"])
        tgt = concept_map.get(edge["target"])
        if not src or not tgt:
            continue
        if src.get("is_hierarchy_parent") or tgt.get("is_hierarchy_parent"):
            continue
        result.append(
            {
                "question_id": str(uuid.uuid4()),
                "question": (
                    f"How does {src['label'].lower()} affect or lead to "
                    f"changes in {tgt['label'].lower()}?"
                ),
                "answer": (
                    f"{src['label']} {edge['type'].lower().replace('_', 's ')} "
                    f"{tgt['label']}. Events in {src['label'].lower()} "
                    f"directly influence {tgt['label'].lower()} through a "
                    f"{edge['type']} relationship."
                ),
                "type": "concept_emergence",
                "difficulty": "hard",
                "emergence_subtype": "cross_concept_causal",
                "required_structure": {
                    "edge": edge["type"],
                    "source": src["label"],
                    "target": tgt["label"],
                },
                "requires_hops": 2,
            }
        )

    if len(leaf_concepts) >= 2:
        sorted_by_count = sorted(
            leaf_concepts,
            key=lambda c: len(concept_events.get(c["id"], [])),
            reverse=True,
        )
        strongest = sorted_by_count[0]
        weakest = sorted_by_count[-1]
        strong_n = len(concept_events.get(strongest["id"], []))
        weak_n = len(concept_events.get(weakest["id"], []))

        result.append(
            {
                "question_id": str(uuid.uuid4()),
                "question": (
                    "Which recurring activity or theme has the most supporting "
                    "events? How many distinct instances does it have?"
                ),
                "answer": (
                    f"{strongest['label']} has the most supporting events with "
                    f"{strong_n} instances, making it the most reinforced concept."
                ),
                "type": "concept_emergence",
                "difficulty": "hard",
                "emergence_subtype": "reinforcement_strength",
                "required_structure": {
                    "concept": strongest["label"],
                    "event_count": strong_n,
                },
                "requires_hops": 1,
            }
        )
        result.append(
            {
                "question_id": str(uuid.uuid4()),
                "question": (
                    "Which recurring theme has the fewest supporting events "
                    "and might be an emerging pattern rather than an "
                    "established one?"
                ),
                "answer": (
                    f"{weakest['label']} has only {weak_n} supporting events, "
                    "making it the least reinforced and potentially "
                    "still-emerging pattern."
                ),
                "type": "concept_emergence",
                "difficulty": "hard",
                "emergence_subtype": "reinforcement_strength",
                "required_structure": {
                    "concept": weakest["label"],
                    "event_count": weak_n,
                },
                "requires_hops": 1,
            }
        )

    part_of_edges = [r for r in relationships if r["type"] == "PART_OF"]
    parent_children: dict[str, list[str]] = {}
    for edge in part_of_edges:
        parent_children.setdefault(edge["target"], []).append(edge["source"])

    for parent_id, children_ids in parent_children.items():
        parent = concept_map.get(parent_id)
        if not parent or len(children_ids) < 2:
            continue
        children_labels = [
            concept_map[cid]["label"] for cid in children_ids if cid in concept_map
        ]
        if len(children_labels) < 2:
            continue
        result.append(
            {
                "question_id": str(uuid.uuid4()),
                "question": (
                    f"What sub-themes or activities make up the broader "
                    f"category of {parent['label'].lower()}? List them."
                ),
                "answer": (
                    f"{parent['label']} consists of: "
                    f"{', '.join(children_labels)}. These "
                    f"{len(children_labels)} activities are all part of "
                    f"{parent['label'].lower()}."
                ),
                "type": "concept_emergence",
                "difficulty": "hard",
                "emergence_subtype": "hierarchy_discovery",
                "required_structure": {
                    "parent": parent["label"],
                    "children": children_labels,
                },
                "requires_hops": 2,
            }
        )

    reinforces_edges = [r for r in relationships if r["type"] == "REINFORCES"]
    related_edges = [r for r in relationships if r["type"] == "RELATED_TO"]
    for edge in reinforces_edges + related_edges:
        src = concept_map.get(edge["source"])
        tgt = concept_map.get(edge["target"])
        if not src or not tgt:
            continue
        if src.get("is_hierarchy_parent") or tgt.get("is_hierarchy_parent"):
            continue
        result.append(
            {
                "question_id": str(uuid.uuid4()),
                "question": (
                    f"What is the relationship between {src['label'].lower()} "
                    f"and {tgt['label'].lower()}? How do they interact?"
                ),
                "answer": (
                    f"{src['label']} {edge['type'].lower().replace('_', ' ')} "
                    f"{tgt['label']}. These two patterns are connected through "
                    "their mutual influence."
                ),
                "type": "concept_emergence",
                "difficulty": "hard",
                "emergence_subtype": "cross_concept_bridge",
                "required_structure": {
                    "edge": edge["type"],
                    "source": src["label"],
                    "target": tgt["label"],
                },
                "requires_hops": 2,
            }
        )

    return result


def generate_state_tracking_questions(scenario: dict) -> list[dict]:
    """Generate state tracking questions from gold state transitions."""
    state_tracking = scenario.get("state_tracking", [])
    belief_tracking = scenario.get("belief_tracking", [])

    result = []

    for st in state_tracking:
        result.append(
            {
                "question_id": str(uuid.uuid4()),
                "question": st["question"],
                "answer": st["answer"],
                "type": "state_tracking",
                "difficulty": "medium",
                "entity": st["entity"],
                "attribute": st["attribute"],
                "transitions": st["transitions"],
                "requires_hops": len(st["transitions"]),
            }
        )

    for bt in belief_tracking:
        final_belief = bt["transitions"][-1]["belief"]
        initial_belief = bt["transitions"][0]["belief"]
        result.append(
            {
                "question_id": str(uuid.uuid4()),
                "question": (
                    f"How did {bt['entity']}'s view on {bt['belief_about']} "
                    "change over time?"
                ),
                "answer": (
                    f"{bt['entity']}'s belief changed from '{initial_belief}' "
                    f"to '{final_belief}'."
                ),
                "type": "state_tracking",
                "difficulty": "hard",
                "entity": bt["entity"],
                "attribute": bt["belief_about"],
                "transitions": bt["transitions"],
                "requires_hops": len(bt["transitions"]),
            }
        )

    return result


def generate_adversarial_questions(
    events: list[dict], n: int = 4
) -> list[dict]:
    """Generate adversarial questions about things NOT in the events.

    Tests whether the system hallucinate when it should say 'unknown'.
    """
    topics = set()
    all_keywords = set()
    for ev in events:
        label = ev.get("gold_concept_label", "")
        if label and label != "Distractor":
            topics.add(label)
        desc = ev.get("description", "").lower()
        for word in (
            "gym",
            "run",
            "jog",
            "yoga",
            "exercise",
            "hobby",
            "hobbies",
            "goal",
            "career",
            "friend",
            "social",
            "dinner",
            "movie",
            "read",
            "learn",
            "study",
            "cook",
            "meal",
        ):
            if word in desc:
                all_keywords.add(word)

    topics_str = ", ".join(sorted(topics))
    keywords_str = ", ".join(sorted(all_keywords))

    prompt = (
        f"A person's life events cover these topics: {topics_str}\n"
        f"Keywords found in events: {keywords_str}\n\n"
        f"Generate exactly {n} questions that CANNOT be answered from these "
        "events because they are about completely different, unrelated "
        "domains.\n\n"
        "CRITICAL RULES:\n"
        "- Do NOT ask about hobbies, career goals, exercise, fitness, social "
        "life, learning, or eating habits — these ARE covered in the events\n"
        "- Do NOT ask about anything that could be reasonably inferred from "
        "the events above\n"
        "- Ask about genuinely unrelated domains: childhood memories, "
        "political views, pets, travel history, family details, financial "
        "investments, religious beliefs, medical history, educational "
        "background\n"
        "- The correct answer for ALL questions is: \"This information is not "
        "available in the events.\"\n\n"
        "Return ONLY a JSON array:\n"
        "[\n"
        "  {\"question\": \"...\", \"plausible_but_wrong_answer\": \"...\", "
        "\"why_unanswerable\": \"...\"}\n"
        "]"
    )

    try:
        resp = _call_openai(prompt, temperature=0.7, max_tokens=2000)
        qas = _parse_json_from_response(resp)
    except Exception as e:
        print(f"  Warning: Adversarial QA generation failed: {e}")
        return []

    result = []
    for qa in qas[:n]:
        result.append(
            {
                "question_id": str(uuid.uuid4()),
                "question": qa["question"],
                "answer": "This information is not available in the recorded events.",
                "type": "adversarial",
                "difficulty": "medium",
                "why_unanswerable": qa.get("why_unanswerable", "Topic not covered"),
                "plausible_wrong_answer": qa.get("plausible_but_wrong_answer", ""),
                "requires_hops": 0,
            }
        )
    return result


def quality_check_qa(qas: list[dict]) -> list[dict]:
    """Use LLM-as-judge to filter low-quality QA pairs."""
    if not qas:
        return []

    to_check = [
        q
        for q in qas
        if q["type"] in ("factual_recall", "temporal_pattern", "concept_emergence")
    ]
    pass_through = [
        q
        for q in qas
        if q["type"] not in ("factual_recall", "temporal_pattern", "concept_emergence")
    ]

    if not to_check:
        return pass_through

    checked: list[dict] = []
    for batch_start in range(0, len(to_check), 10):
        batch = to_check[batch_start : batch_start + 10]
        qa_text = "\n".join(
            f"{i + 1}. Q: {q['question']}\n   A: {q['answer']}"
            for i, q in enumerate(batch)
        )
        prompt = (
            "Rate each QA pair for quality. A good QA pair has:\n"
            "- A clear, unambiguous question\n"
            "- An answer that is correct and specific (not vague)\n"
            "- The question type matches what it claims to test\n\n"
            f"QA Pairs:\n{qa_text}\n\n"
            "Return ONLY a JSON array of booleans (true = keep, false = remove):\n"
            "[true, false, true, ...]"
        )
        try:
            resp = _call_openai(prompt, temperature=0.0, max_tokens=200)
            verdicts = _parse_json_from_response(resp)
            for q, keep in zip(batch, verdicts):
                if keep:
                    checked.append(q)
                else:
                    print(f"  QC rejected: {q['question'][:60]}...")
        except Exception:
            checked.extend(batch)

        time.sleep(0.3)

    return pass_through + checked


def generate_scenario_dataset(
    scenario_path: Path, scale: str = "small", seed: int = 42
) -> dict:
    """Generate a complete CogEval-Bench dataset for one scenario."""
    random.seed(seed)

    with open(scenario_path) as f:
        scenario = json.load(f)

    scale_config = SCALES[scale]
    multiplier = scale_config["event_multiplier"]

    print("\n" + "=" * 60)
    print(f"Generating: {scenario['name']} (scale={scale})")
    print("=" * 60)

    base_time = datetime(2024, 6, 1, 0, 0, 0)
    temporal_span = scenario.get("temporal_span_days", 7)

    all_events: list[dict] = []
    concept_map: dict[str, dict] = {}

    for concept in scenario["concepts"]:
        concept_map[concept["id"]] = concept
        if concept.get("is_hierarchy_parent") or concept["expected_events"] == 0:
            continue
        print(f"  Generating events for: {concept['label']}...")
        events = generate_events_for_concept(
            concept, scenario, base_time, temporal_span, multiplier
        )
        all_events.extend(events)
        print(f"    → {len(events)} events")
        time.sleep(0.5)

    for chain in scenario.get("planted_chains", []):
        print(f"  Generating chain: {chain['id']}...")
        chain_events = generate_chain_events(chain, base_time, temporal_span)
        all_events.extend(chain_events)

    distractor_ratio = scenario.get("distractor_ratio", 0.15)
    n_distractors_total = int(
        len(all_events) * distractor_ratio + scale_config["distractor_extra"]
    )
    if n_distractors_total > 0:
        print(f"  Generating {n_distractors_total} distractor events...")
        distractors = generate_distractor_events(
            scenario, n_distractors_total, base_time, temporal_span
        )
        all_events.extend(distractors)

    all_events.sort(key=lambda e: e["timestamp"])
    for i, ev in enumerate(all_events):
        ev["event_index"] = i

    print(f"\n  Total events: {len(all_events)}")
    concept_counts: dict[str, int] = {}
    for ev in all_events:
        c = ev.get("gold_concept", ev.get("gold_chain_id", "chain"))
        concept_counts[c] = concept_counts.get(c, 0) + 1
    for c, cnt in sorted(concept_counts.items()):
        print(f"    {c}: {cnt} events")

    print("\n  Generating QA pairs...")

    all_qas: list[dict] = []

    print("    Factual recall...")
    factual = generate_factual_recall_questions(all_events, n=8)
    all_qas.extend(factual)
    print(f"      → {len(factual)} questions")
    time.sleep(0.5)

    print("    Multi-hop (planted chains)...")
    multi_hop = generate_multi_hop_questions(scenario.get("planted_chains", []))
    all_qas.extend(multi_hop)
    print(f"      → {len(multi_hop)} questions")

    print("    Temporal patterns...")
    temporal = generate_temporal_questions(all_events, n=6)
    all_qas.extend(temporal)
    print(f"      → {len(temporal)} questions")
    time.sleep(0.5)

    print("    Concept emergence (core)...")
    emergence = generate_concept_emergence_questions(
        scenario["concepts"],
        scenario.get("relationships", []),
        all_events,
    )
    all_qas.extend(emergence)
    print(f"      → {len(emergence)} questions")
    time.sleep(0.5)

    print("    State tracking...")
    state = generate_state_tracking_questions(scenario)
    all_qas.extend(state)
    print(f"      → {len(state)} questions")

    print("    Adversarial...")
    adversarial = generate_adversarial_questions(all_events, n=4)
    all_qas.extend(adversarial)
    print(f"      → {len(adversarial)} questions")
    time.sleep(0.5)

    print(f"\n  Running quality control on {len(all_qas)} QA pairs...")
    all_qas = quality_check_qa(all_qas)
    print(f"    → {len(all_qas)} passed QC")

    dataset = {
        "scenario_id": scenario["scenario_id"],
        "name": scenario["name"],
        "description": scenario["description"],
        "scale": scale,
        "generation_timestamp": datetime.now().isoformat(),
        "statistics": {
            "total_events": len(all_events),
            "concept_events": len(
                [
                    e
                    for e in all_events
                    if e.get("gold_concept") not in (None, "__distractor__")
                ]
            ),
            "chain_events": len([e for e in all_events if e.get("gold_chain_id")]),
            "distractor_events": len(
                [e for e in all_events if e.get("gold_concept") == "__distractor__"]
            ),
            "total_questions": len(all_qas),
            "questions_by_type": {
                qt: len([q for q in all_qas if q["type"] == qt]) for qt in QA_TYPES
            },
        },
        "gold_graph": {
            "concepts": scenario["concepts"],
            "relationships": scenario["relationships"],
            "planted_chains": scenario.get("planted_chains", []),
            "expected_intents": scenario.get("expected_intents", []),
            "state_tracking": scenario.get("state_tracking", []),
            "belief_tracking": scenario.get("belief_tracking", []),
        },
        "events": all_events,
        "questions": all_qas,
    }

    return dataset


def save_dataset(dataset: dict, output_dir: Path) -> Path:
    """Save dataset to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{dataset['scenario_id']}_{dataset['scale']}.json"
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {path} ({path.stat().st_size / 1024:.1f} KB)")
    return path


def convert_to_cognifold_events(dataset: dict) -> list[dict]:
    """Convert generated events to Cognifold Event format for ingestion.

    Returns a list of dicts matching the Cognifold Event schema, ready
    to be used by run_benchmark.py.
    """
    cognifold_events = []
    for ev in dataset["events"]:
        cognifold_events.append(
            {
                "event_id": ev["event_id"],
                "timestamp": ev["timestamp"],
                "source": ev.get("source", "cogeval-bench"),
                "event_type": ev.get("event_type", "life_event"),
                "title": ev.get("title", "Event"),
                "description": ev["description"],
                "context": {
                    "benchmark": "cogeval",
                    "scenario": dataset["scenario_id"],
                    "gold_concept": ev.get("gold_concept", ""),
                    "gold_chain_id": ev.get("gold_chain_id", ""),
                },
            }
        )
    return cognifold_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CogEval-Bench dataset")
    parser.add_argument(
        "--scenario",
        choices=[
            "software_engineer",
            "health_journey",
            "team_project",
            "news_stream",
            "academic_research",
            "customer_support",
        ],
        help="Which scenario to generate",
    )
    parser.add_argument(
        "--all", action="store_true", help="Generate all scenarios"
    )
    parser.add_argument(
        "--scale",
        choices=["small", "medium", "large"],
        default="small",
        help="Dataset scale (default: small)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    output_dir = Path(args.output)

    if args.all:
        scenarios = list(GOLD_GRAPHS_DIR.glob("*.json"))
    elif args.scenario:
        scenarios = [GOLD_GRAPHS_DIR / f"{args.scenario}.json"]
    else:
        parser.error("Specify --scenario or --all")

    all_datasets = []
    for scenario_path in scenarios:
        if not scenario_path.exists():
            print(f"Warning: {scenario_path} not found, skipping")
            continue
        dataset = generate_scenario_dataset(scenario_path, args.scale, args.seed)
        save_dataset(dataset, output_dir)
        all_datasets.append(dataset)

    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    for ds in all_datasets:
        stats = ds["statistics"]
        print(f"\n{ds['name']} ({ds['scale']}):")
        print(
            f"  Events: {stats['total_events']} ({stats['concept_events']} "
            f"concept, {stats['chain_events']} chain, "
            f"{stats['distractor_events']} distractor)"
        )
        print(f"  Questions: {stats['total_questions']}")
        for qt, cnt in stats["questions_by_type"].items():
            print(f"    {qt}: {cnt}")


if __name__ == "__main__":
    main()
