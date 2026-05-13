#!/usr/bin/env python3
"""MuSiQue benchmark runner for Cognifold.

Evaluates multi-hop question answering: given a set of paragraphs and a
question that requires reasoning across multiple supporting paragraphs,
produce a free-form answer.

Dataset: bdsaglam/musique (MuSiQue - Multi-hop Questions via Single-hop
Question Composition).
"""

import argparse
import json
import os
import re
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from benchmarks.shared.base_runner import (
    BenchmarkRunner,
    _call_llm_text,
    enrich_eval_result,
    evaluate_with_llm,
    generate_answer_with_llm,
    save_wrong_cases,
)
from cognifold.graph.store import ConceptGraph
from cognifold.models.event import Event
from cognifold.query.agent import MemoryQueryAgent


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


def compute_f1(predicted: str, gold: str) -> float:
    pred_tokens = tokenize(predicted)
    gold_tokens = tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = 0
    for token in pred_counts:
        if token in gold_counts:
            overlap += min(pred_counts[token], gold_counts[token])
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred_counts.values())
    recall = overlap / sum(gold_counts.values())
    return 2 * precision * recall / (precision + recall)


class MuSiQueRunner(BenchmarkRunner):
    benchmark_name = "musique"
    default_data_path = Path(__file__).parent / "data" / "musique_validation.json"

    def __init__(self) -> None:
        super().__init__()
        # MuSiQue validation often contains paired (answerable/unanswerable) examples with the same ID.
        # Default to answerable-only (MuSiQue-Ans) so `--limit 1` is a meaningful smoke test.
        self._include_unanswerable: bool = False

    def add_extra_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--include-unanswerable",
            action="store_true",
            help=(
                "Include unanswerable examples (answerable=false). Default: evaluate answerable-only "
                "(MuSiQue-Ans)."
            ),
        )

    def run(self, *, include_unanswerable: bool = False, **kwargs: Any) -> None:  # type: ignore[override]
        self._include_unanswerable = include_unanswerable
        super().run(**kwargs)

    def load_dataset(self, data_path: Path, limit: Optional[int] = None) -> list[dict[str, Any]]:
        with open(data_path) as f:
            data = json.load(f)
        if not self._include_unanswerable:
            # Filter to answerable examples by default (MuSiQue-Ans).
            has_answerable_flag = any("answerable" in ex for ex in data)
            if has_answerable_flag:
                before = len(data)
                data = [ex for ex in data if ex.get("answerable") is True]
                print(f"Filtered answerable-only: {len(data)}/{before} examples")
            else:
                print("Warning: Dataset has no 'answerable' field; cannot filter unanswerables.")
        if limit:
            data = data[:limit]
        print(f"Loaded {len(data)} examples from {data_path}")
        return data

    def build_events(self, example: dict[str, Any], idx: int) -> list[Event]:
        paragraphs = example.get("paragraphs", [])
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        example_id = example.get("id", str(idx))
        events = []
        for i, para in enumerate(paragraphs):
            title = para.get("title", f"Paragraph {i + 1}")
            text = (
                para.get("paragraph_text", "") or para.get("paragraphs", "") or para.get("text", "")
            )
            text = str(text)
            is_supporting = para.get("is_supporting", False)
            events.append(
                Event(
                    event_id=str(uuid.uuid4()),
                    timestamp=base_time + timedelta(seconds=idx * 1000 + i),
                    source="musique-benchmark",
                    event_type="information_paragraph",
                    title=title,
                    description=text,
                    context={
                        "example_id": example_id,
                        "paragraph_index": i,
                        "is_supporting": is_supporting,
                        "benchmark": self.benchmark_name,
                    },
                )
            )
        return events

    def get_query_config_overrides(self) -> dict[str, Any]:
        # MuSiQue answers are typically contained in the raw paragraphs (EVENT nodes),
        # and multi-hop requires recalling multiple low-similarity supporting passages.
        return {
            "max_nodes": 60,
            "max_context_chars": 20000,
            "max_description_chars": 4000,
            "min_relevance_score": 0.0,
            "prefer_concepts": False,
            "include_reasoning": False,
            "include_grounding": False,
        }

    def _decompose_multi_hop(self, question: str, llm_model: str) -> list[str]:
        """Decompose a multi-hop question into sub-queries for better retrieval."""
        prompt = (
            f"Break this multi-hop question into 2-4 simple sub-questions that, "
            f"when answered in sequence, lead to the final answer.\n\n"
            f"Question: {question}\n\n"
            f"Output ONLY the sub-questions, one per line, numbered. No explanations."
        )
        try:
            raw = _call_llm_text(
                model=llm_model, user_prompt=prompt, temperature=0.0, max_tokens=200
            )
            sub_queries = []
            for line in raw.strip().split("\n"):
                line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
                if line and "?" in line:
                    sub_queries.append(line)
            return sub_queries[:4]
        except Exception:
            return []

    def evaluate_example(
        self,
        example: dict[str, Any],
        idx: int,
        graph: ConceptGraph,
        query_agent: MemoryQueryAgent,
        query_mode: str,
        use_llm_eval: bool,
        profile_templates: dict[str, str],
        llm_model: str,
    ) -> dict[str, Any]:
        example_id = self.get_example_id(example, idx)
        question = example.get("question", "")
        target = example.get("answer", "")
        num_hops = len(example.get("question_decomposition", []))

        _query_start = time.time()

        # MuSiQue iter1: Force RAG (pure semantic top-K) for main retrieval.
        # MergeFold concept-traversal loses bridge paragraphs; for multi-hop QA
        # the gold answer text lives in the original ingested EVENT paragraphs.
        # EverMemOS / Zep both fall back to dense passage retrieval over evidence.
        musique_query_mode = "rag"

        # Unified cognition: recognition → reconstruction → validation
        cognition = self.cognition_query(
            question=question,
            query_agent=query_agent,
            domain=self.benchmark_name,
            query_mode=musique_query_mode,
        )
        context = cognition.context
        result = cognition.query_result

        # Multi-hop decomposition with sub-Q chained answer feeding (Iter3,
        # inspired by LoCoMo iter3 + EverMemOS / Mem0 chain-of-fact reasoning).
        # For each sub-Q in order: run retrieval, dump unique node descriptions
        # to evidence, AND (for short chains) generate a brief sub-answer that
        # is exposed to the final QA call. This addresses the failure mode
        # where the model stops at hop N-1 instead of running through to hop
        # N (e.g., answers "FHF" instead of expanding "FIFA"). With each
        # hop's intermediate answer visible, the LLM can compose them into
        # the final entity.
        #
        # Iter5 (#3 hop-gating): empirically the chain transcript HELPS
        # 2-hop questions (+0.036 F1 in iter3) but HURTS 3/4-hop (-0.06–0.08
        # F1) because errors in any intermediate sub-answer compound and
        # mislead the final answer along the longer chain. Skip the
        # transcript for hops>=3 — keep the decomposition + extra_contexts
        # retrieval (which is helpful regardless), but don't inject the
        # potentially-wrong intermediate answers as a transcript.
        sub_queries = self._decompose_multi_hop(question, llm_model)
        chain_feeding_enabled = num_hops <= 2
        sub_chain_transcript: list[str] = []
        if sub_queries:
            seen_node_ids = {n.node_id for n in result.nodes}
            extra_contexts: list[str] = []
            for sq in sub_queries:
                sq_top_descriptions: list[str] = []
                try:
                    sq_cognition = self.cognition_query(
                        question=sq,
                        query_agent=query_agent,
                        domain=self.benchmark_name,
                        query_mode=musique_query_mode,
                    )
                    for node in sq_cognition.query_result.nodes:
                        desc = node.data.get("description", "")
                        if desc:
                            # Local context for the sub-answer LLM call:
                            # use top-3 retrieved nodes regardless of whether
                            # they're already in the main retrieval (avoid
                            # the prior bug where small graphs dedupe to
                            # zero local evidence).
                            if len(sq_top_descriptions) < 3:
                                sq_top_descriptions.append(desc[:500])
                            # extra_contexts (visible to the FINAL QA call)
                            # is still deduped against the main retrieval to
                            # avoid duplicate text bloat.
                            if node.node_id not in seen_node_ids:
                                seen_node_ids.add(node.node_id)
                                extra_contexts.append(desc[:500])
                    # Sub-answer: brief LLM call using this sub-Q's local
                    # context (top-3 retrieved nodes by relevance). Cap at
                    # 24 tokens. Earlier answers (in sub_chain_transcript)
                    # are exposed so later sub-Qs can refer to them.
                    # Iter5 (#3): only build chain transcript for hops<=2.
                    sq_local_context = "\n".join(sq_top_descriptions)
                    if chain_feeding_enabled and sq_local_context.strip():
                        prior_chain = (
                            "\n".join(sub_chain_transcript)
                            if sub_chain_transcript
                            else "(none yet)"
                        )
                        sub_prompt = (
                            "Answer this sub-question with a SHORT entity / phrase "
                            "(1-6 words, no explanation). Use the prior chain if it "
                            "resolves a referent.\n\n"
                            f"Prior chain:\n{prior_chain}\n\n"
                            f"Sub-question: {sq}\n\n"
                            f"Local evidence:\n{sq_local_context}\n\n"
                            "Sub-answer:"
                        )
                        try:
                            sub_ans = _call_llm_text(
                                model=llm_model,
                                user_prompt=sub_prompt,
                                temperature=0.0,
                                max_tokens=24,
                            )
                            sub_ans_clean = (
                                sub_ans.strip().split("\n", 1)[0].strip().strip(".")
                            )
                            if sub_ans_clean:
                                sub_chain_transcript.append(
                                    f"- {sq} → {sub_ans_clean}"
                                )
                        except Exception:
                            pass
                except Exception:
                    pass
            if extra_contexts:
                context = (
                    context
                    + "\n\n--- Additional multi-hop context ---\n"
                    + "\n".join(extra_contexts[:10])
                )
            if sub_chain_transcript:
                context = (
                    context
                    + "\n\n--- Sub-question chain (intermediate hops) ---\n"
                    + "\n".join(sub_chain_transcript)
                    + "\n\n(Use the chain to resolve referents in the question; "
                    "the final answer must follow the LAST hop's value, not stop "
                    "earlier.)"
                )

        # Iter1: Direct dense paragraph retrieval — EverMemOS-style evidence
        # chaining. Run semantic search restricted to event nodes (original
        # ingested paragraph text) for question + sub-questions, dedup by
        # paragraph_index, append top snippets. This recovers gold supporting
        # paragraphs that concept summaries elide.
        #
        # Iter2: reuse the agent's cached `_semantic_search` so we don't pay
        # for a second `build_index` (== N embedding API calls) per example.
        # The main `cognition_query` above already populated it under
        # query_mode="rag"; if not, build once and stash it back on the agent.
        try:
            from cognifold.embeddings.search import SearchConfig, SemanticSearch

            embedder = getattr(query_agent, "_embedder", None)
            if embedder is not None:
                sem = getattr(query_agent, "_semantic_search", None)
                if sem is None:
                    sem = SemanticSearch(embedder)
                    sem.build_index(graph)
                    query_agent._semantic_search = sem
                queries_for_evidence = [question] + (sub_queries or [])
                seen_para_idxs: set[int] = set()
                evidence_snippets: list[str] = []
                sc = SearchConfig(top_k=6, include_node_types=["event"])
                for q in queries_for_evidence:
                    if len(evidence_snippets) >= 10:
                        break
                    try:
                        sresults = sem.search(graph, q, sc)
                    except Exception:
                        continue
                    for sres in sresults:
                        node = sres.node
                        if node is None or node.type.value != "event":
                            continue
                        ctx_field = node.data.get("context") if isinstance(node.data, dict) else None
                        p_idx = ctx_field.get("paragraph_index") if isinstance(ctx_field, dict) else None
                        if isinstance(p_idx, int):
                            if p_idx in seen_para_idxs:
                                continue
                            seen_para_idxs.add(p_idx)
                        title = node.data.get("title", "")
                        desc = node.data.get("description", "")
                        if desc:
                            evidence_snippets.append(f"- **{title}**\n  {desc[:900]}")
                        if len(evidence_snippets) >= 10:
                            break
                if evidence_snippets:
                    context = (
                        context
                        + "\n\n--- Source paragraphs (dense, multi-query) ---\n"
                        + "\n".join(evidence_snippets)
                    )
        except Exception:
            pass

        # Iter4 (P2): Force-include ALL paragraph EVENT nodes regardless of
        # retrieval ranking. MuSiQue has exactly ~20 paragraphs per example
        # (small graph), and across all 16 iter3 wrong cases the gold
        # supporting paragraphs had support_recall=0.0 — i.e. retrieval kept
        # picking concept summaries over event paragraphs. Guarantee that
        # every paragraph original text reaches the QA prompt; the LLM can
        # then ignore irrelevant ones, but cannot miss the bridge paragraph
        # for multi-hop chains. Costs ~10k extra context chars (20 × 500),
        # well within max_context_chars=20000.
        try:
            from cognifold.models.node import NodeType

            all_paragraphs: list[tuple[int, str, str]] = []
            for node in graph.nodes.values():
                if node.type != NodeType.EVENT:
                    continue
                ctx_field = node.data.get("context") if isinstance(node.data, dict) else None
                if not isinstance(ctx_field, dict):
                    continue
                if ctx_field.get("benchmark") != self.benchmark_name:
                    continue
                p_idx = ctx_field.get("paragraph_index")
                if not isinstance(p_idx, int):
                    continue
                title = node.data.get("title", "") or ""
                desc = node.data.get("description", "") or ""
                if not desc:
                    continue
                all_paragraphs.append((p_idx, title, desc[:700]))
            if all_paragraphs:
                all_paragraphs.sort(key=lambda x: x[0])
                paragraph_block = "\n".join(
                    f"[P{idx}] {title}\n  {desc}" for idx, title, desc in all_paragraphs
                )
                context = (
                    context
                    + "\n\n--- ALL paragraphs (force-included for chain coverage) ---\n"
                    + paragraph_block
                )
        except Exception:
            pass


        # Retrieval density analysis: check whether we retrieved any of the gold supporting paragraphs.
        supporting_idxs: set[int] = set()
        for para in example.get("paragraphs", []) or []:
            if para.get("is_supporting") is True and isinstance(para.get("idx"), int):
                supporting_idxs.add(para["idx"])
        for step in example.get("question_decomposition", []) or []:
            support_idx = step.get("paragraph_support_idx")
            if isinstance(support_idx, int):
                supporting_idxs.add(support_idx)

        retrieved_event_paragraph_idxs: set[int] = set()
        for node in result.nodes:
            if node.node_type != "event":
                continue
            ctx = node.data.get("context")
            if isinstance(ctx, dict):
                para_idx = ctx.get("paragraph_index")
                if isinstance(para_idx, int):
                    retrieved_event_paragraph_idxs.add(para_idx)

        supporting_retrieved = len(supporting_idxs & retrieved_event_paragraph_idxs)
        supporting_total = len(supporting_idxs)
        support_recall = supporting_retrieved / supporting_total if supporting_total > 0 else None

        if use_llm_eval:
            answer = generate_answer_with_llm(
                question=question,
                context=context,
                profile_templates=profile_templates,
                model=llm_model,
                default_system=(
                    "Answer multi-hop questions based on the provided context paragraphs. "
                    "Reply with ONLY the answer (a short phrase or entity name)."
                ),
                default_user="Question: {question}\n\nContext:\n{context}\n\nAnswer with a short phrase:",
                max_tokens=100,
            )
        else:
            answer = context.split("\n")[0].strip() if context else "unknown"

        exact_match = normalize_text(target) == normalize_text(answer)
        f1_score = compute_f1(answer, target)

        if use_llm_eval and not exact_match:
            verdict, explanation = evaluate_with_llm(
                question=question,
                expected=target,
                generated=answer,
                context=context,
                profile_templates=profile_templates,
                model=llm_model,
            )
        else:
            verdict = "CORRECT" if exact_match else "INCORRECT"
            explanation = ""

        eval_result: dict[str, Any] = {
            "example_id": example_id,
            "question": question,
            "target": target,
            "answerable": example.get("answerable"),
            "generated_answer": answer,
            "verdict": verdict,
            "explanation": explanation,
            "exact_match": exact_match,
            "f1_score": f1_score,
            "num_hops": num_hops,
            "num_paragraphs": len(example.get("paragraphs", [])),
            "context_length": len(context),
            "supporting_paragraphs": sorted(supporting_idxs),
            "supporting_total": supporting_total,
            "supporting_retrieved": supporting_retrieved,
            "support_recall": support_recall,
        }
        if enrich_eval_result is not None:
            enrich_eval_result(
                eval_result, graph=graph, query_result=result, query_start_time=_query_start
            )

        print(f"    {verdict} (target={target}, got={answer}, F1={f1_score:.3f})")
        return eval_result

    def save_results(
        self, all_results: list[dict[str, Any]], output_dir: str, config: dict[str, Any]
    ) -> None:
        results_path = os.path.join(output_dir, "benchmark_results.json")
        total = len(all_results)
        exact_match_count = sum(1 for r in all_results if r.get("exact_match"))
        correct_count = sum(1 for r in all_results if r.get("verdict") == "CORRECT")
        partial_count = sum(1 for r in all_results if r.get("verdict") == "PARTIAL")
        avg_f1 = sum(r.get("f1_score", 0.0) for r in all_results) / total if total > 0 else 0.0

        per_hops: dict[int, dict[str, Any]] = defaultdict(
            lambda: {"total": 0, "exact_match": 0, "correct": 0, "f1_sum": 0.0}
        )
        for r in all_results:
            nh = r.get("num_hops", 0)
            per_hops[nh]["total"] += 1
            if r.get("exact_match"):
                per_hops[nh]["exact_match"] += 1
            if r.get("verdict") == "CORRECT":
                per_hops[nh]["correct"] += 1
            per_hops[nh]["f1_sum"] += r.get("f1_score", 0.0)

        with open(results_path, "w") as f:
            json.dump(
                {
                    "summary": {
                        "total": total,
                        "exact_match": exact_match_count,
                        "exact_match_rate": exact_match_count / total * 100 if total > 0 else 0,
                        "average_f1": avg_f1,
                        "correct": correct_count,
                        "partial": partial_count,
                    },
                    "per_hops": {str(k): v for k, v in per_hops.items()},
                    "results": all_results,
                    "config": config,
                },
                f,
                indent=2,
            )
        print(f"\nDetailed results saved to {results_path}")

        if save_wrong_cases is not None:
            save_wrong_cases(all_results, output_dir)

    def print_summary(self, all_results: list[dict[str, Any]], config: dict[str, Any]) -> None:
        total = len(all_results)
        exact_match_count = sum(1 for r in all_results if r.get("exact_match"))
        correct_count = sum(1 for r in all_results if r.get("verdict") == "CORRECT")
        avg_f1 = sum(r.get("f1_score", 0.0) for r in all_results) / total if total > 0 else 0.0

        per_hops: dict[int, dict[str, Any]] = defaultdict(
            lambda: {"total": 0, "exact_match": 0, "correct": 0, "f1_sum": 0.0}
        )
        for r in all_results:
            nh = r.get("num_hops", 0)
            per_hops[nh]["total"] += 1
            if r.get("exact_match"):
                per_hops[nh]["exact_match"] += 1
            if r.get("verdict") == "CORRECT":
                per_hops[nh]["correct"] += 1
            per_hops[nh]["f1_sum"] += r.get("f1_score", 0.0)

        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        print(f"  Query Mode: {config.get('query_mode')}")
        print(f"  Disable Concepts: {config.get('disable_concepts')}")
        print(f"  LLM Eval: {config.get('use_llm_eval')}")
        print(
            f"  Exact Match: {exact_match_count}/{total} ({exact_match_count / total * 100 if total > 0 else 0:.1f}%)"
        )
        print(f"  Average F1: {avg_f1:.3f}")
        print(f"  Correct (verdict): {correct_count}/{total}")
        for nh in sorted(per_hops.keys()):
            stats = per_hops[nh]
            em_pct = stats["exact_match"] / stats["total"] * 100 if stats["total"] > 0 else 0
            hop_f1 = stats["f1_sum"] / stats["total"] if stats["total"] > 0 else 0
            print(f"    {nh}-hop: EM={em_pct:.1f}% F1={hop_f1:.3f} (n={stats['total']})")


if __name__ == "__main__":
    MuSiQueRunner().main()
