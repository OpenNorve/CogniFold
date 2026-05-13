"""Base class for Cognifold benchmark runners.

Provides the common infrastructure shared by all standard benchmark runners:
- API key checking
- LLM helper functions (eval, free-form QA, multiple-choice QA, verdict evaluation)
- Embedding resolution and profile loading
- Graph + agent + executor + query agent setup
- Event ingestion loop with optional replay visualization
- Results saving with wrong-case analysis
- Standard CLI argument parsing

Subclasses implement only the dataset-specific parts:
- load_dataset() -- load and return examples
- build_events() -- convert a single example to Cognifold events
- evaluate_example() -- evaluate a single example after ingestion
- print_summary() / save_results() -- (optional) custom summary/results

Benchmarks with very different structures (locomo, msc, futurex) do NOT use
this base class and keep their own standalone runners.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup (must run before cognifold imports)
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_project_root, "src"))
sys.path.append(_project_root)

try:
    from cognifold.agent.agent import CognifoldAgent
    from cognifold.agent.config import AgentConfig
    from cognifold.agent.prompt_profile import load_prompt_profiles
    from cognifold.executor.runner import PlanExecutor
    from cognifold.graph.store import ConceptGraph
    from cognifold.models.event import Event
    from cognifold.query.agent import MemoryQueryAgent
    from cognifold.query.models import QueryConfig
    from cognifold.replay.logger import GraphLogger
    from cognifold.replay.player import ReplayPlayer
    from cognifold.replay.renderer import ReplayRenderer
except ImportError as e:
    print(f"Error importing Cognifold modules: {e}")
    print("Please run from project root or set PYTHONPATH=src")
    sys.exit(1)

# Analysis utils (optional, graceful fallback)
try:
    from benchmarks.analysis_utils import enrich_eval_result, save_wrong_cases
except ImportError:
    enrich_eval_result = None  # type: ignore[assignment]
    save_wrong_cases = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared utility functions
# ---------------------------------------------------------------------------


def check_api_keys() -> bool:
    """Check if API keys are set in the environment."""
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("\nERROR: No API keys found in environment variables.")
        print("Please set OPENAI_API_KEY or GOOGLE_API_KEY to run the benchmark.")
        return False
    return True


def _resolve_llm_model_shorthand(model: str) -> str:
    """Resolve common benchmark CLI shorthands to full model IDs.

    This keeps backwards compatibility with earlier runner scripts that used
    short names like "flash"/"light".
    """
    m = model.strip()
    if m.lower() in {"flash", "light"}:
        # "light" historically meant "cheap/fast". We map both to flash for now.
        return "gemini-2.0-flash"
    return m


def _split_llm_model(model: str) -> tuple[str, str]:
    """Split model into (provider, model_name).

    Conventions:
    - OpenAI models may be prefixed with "openai:" (preferred for agent config).
    - Gemini models are typically unprefixed (e.g. "gemini-2.0-flash"), but
      "gemini:" is also accepted for symmetry with embedding strings.
    - For backwards compatibility, bare OpenAI IDs like "gpt-4o-mini" are allowed.
    """
    model = _resolve_llm_model_shorthand(model)
    if model.startswith("openai:"):
        return "openai", model.split(":", 1)[1]
    if model.startswith("gemini:"):
        return "gemini", model.split(":", 1)[1]
    if model.startswith("gemini-"):
        return "gemini", model
    if model.startswith(("gpt-", "o1", "o3")):
        return "openai", model
    # Fall back to whichever API key is present; prefer OpenAI for compatibility.
    if os.environ.get("OPENAI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        return "openai", model
    if os.environ.get("GOOGLE_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        return "gemini", model
    return "openai", model


def _normalize_agent_model_name(model: str) -> str:
    """Normalize a CLI/profile model string into an AgentConfig-compatible name.

    AgentConfig convention:
    - OpenAI models MUST be prefixed with "openai:" so the agent dispatches correctly.
    - Gemini models should be unprefixed model IDs (e.g. "gemini-2.0-flash").
    """
    provider, model_name = _split_llm_model(model)
    if provider == "openai":
        return f"openai:{model_name}"
    return model_name


def _call_llm_text(
    *,
    model: str,
    user_prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 50,
) -> str:
    """Call either OpenAI or Gemini, depending on model/provider."""
    provider, model_name = _split_llm_model(model)

    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set but an OpenAI model was requested")
        from openai import OpenAI

        client = OpenAI()
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        # GPT-5.x models require max_completion_tokens instead of max_tokens
        token_kwargs: dict[str, int] = {}
        if model_name.startswith("gpt-5"):
            token_kwargs["max_completion_tokens"] = max_tokens
        else:
            token_kwargs["max_tokens"] = max_tokens
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            **token_kwargs,
        )
        return (response.choices[0].message.content or "").strip()

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set but a Gemini model was requested")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    prompt = user_prompt if not system_prompt else f"{system_prompt}\n\n{user_prompt}"
    # Gemini 2.5+ thinking models need higher token budgets because internal
    # reasoning consumes part of the max_output_tokens budget.
    # Gemini 2.5 thinking models use output tokens for internal reasoning.
    # Scale up budget proportionally: small requests get 2x, large ones get 3x.
    if "2.5" in model_name:
        effective_max_tokens = max(max_tokens * 3, 512) if max_tokens >= 100 else max(max_tokens, 256)
    else:
        effective_max_tokens = max_tokens
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=effective_max_tokens,
    )
    response = client.models.generate_content(model=model_name, contents=prompt, config=config)
    text = getattr(response, "text", None)
    return text.strip() if isinstance(text, str) else ""


def call_llm_for_eval(prompt: str, max_tokens: int = 50, model: str = "openai:gpt-4o-mini") -> str:
    """Call LLM for evaluation (OpenAI or Gemini)."""
    try:
        return _call_llm_text(
            model=model, user_prompt=prompt, temperature=0.0, max_tokens=max_tokens
        )
    except Exception as e:
        logger.error(f"Gemini call failed: {e}")
        return ""



def generate_answer_with_llm(
    question: str,
    context: str,
    profile_templates: dict[str, str],
    model: str = "openai:gpt-4o-mini",
    default_system: str = "Answer the question based on the retrieved context. Give a concise answer.",
    default_user: str = "Question: {question}\n\nContext:\n{context}\n\nAnswer:",
    max_tokens: int = 50,
) -> str:
    """Generate a free-form answer using LLM with profile templates. Tries OpenAI, falls back to Gemini."""
    system_prompt = profile_templates.get("qa_system", default_system)
    user_template = profile_templates.get("qa_answer", default_user)
    user_prompt = user_template.format(question=question, context=context)

    try:
        return _call_llm_text(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.error(f"LLM QA failed: {e}")
        return ""


def answer_mc_with_llm(
    question: str,
    context: str,
    options: dict[str, str],
    profile_templates: dict[str, str],
    model: str = "openai:gpt-4o-mini",
) -> str:
    """Answer a multiple-choice question using LLM. Tries OpenAI, falls back to Gemini.

    Returns the selected option letter (e.g. A, B, C, D).

    Robustness logic:
    1. Primary call uses the profile system+user templates.
    2. If the response is empty or contains no valid option letter, retry once
       with a tightened directive prompt and a slightly larger token budget.
       This recovers transient API failures and cases where the model emits
       extra prose that gets truncated before any letter (a known failure mode
       on SocialIQA where ~15% of wrong cases had empty predictions).
    3. If still no letter, fall back to a context-free pure-commonsense pass
       (without any graph context) — useful for empty-graph cases where the
       graph context string ("The memory graph is empty…") is just noise.
    """
    system_prompt = profile_templates.get(
        "qa_system",
        "Answer multiple-choice questions based on retrieved context. Reply with ONLY the letter.",
    )
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()) if v.strip())
    user_template = profile_templates.get(
        "qa_answer",
        "Question: {question}\n\nContext:\n{context}\n\nOptions:\n{options}\n\nAnswer (letter only):",
    )
    user_prompt = user_template.format(question=question, context=context, options=options_text)
    valid_letters = {k for k in options if options.get(k, "").strip()}

    def _extract_letter(raw: str) -> str:
        if not raw:
            return ""
        cleaned = raw.strip().upper()
        # Prefer letters that appear after a marker like "ANSWER:" or "FINAL:".
        for marker in ("ANSWER:", "FINAL ANSWER:", "FINAL:", "CHOICE:"):
            if marker in cleaned:
                tail = cleaned.split(marker, 1)[1]
                for ch in tail:
                    if ch in valid_letters:
                        return ch
        # Otherwise: prefer the LAST letter that matches a valid option,
        # which is robust to chain-of-thought style outputs ending in "...so B".
        last = ""
        for ch in cleaned:
            if ch in valid_letters:
                last = ch
        if last:
            return last
        return ""

    # --- Primary attempt -----------------------------------------------------
    try:
        raw = _call_llm_text(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=10,
        )
        letter = _extract_letter(raw)
        if letter:
            return letter
    except Exception as e:
        logger.error(f"LLM QA primary call failed: {e}")

    # --- Retry with strict directive + larger budget -------------------------
    strict_user = (
        f"Question: {question}\n\n"
        f"Options:\n{options_text}\n\n"
        "Pick the single best answer based on social commonsense. "
        "Respond with EXACTLY one letter on its own line (A, B, or C). "
        "Do not include any other text.\n\nAnswer:"
    )
    strict_system = (
        "You are answering a 3-choice social commonsense question. "
        "Output exactly one letter: A, B, or C. No prose, no punctuation."
    )
    try:
        raw = _call_llm_text(
            model=model,
            system_prompt=strict_system,
            user_prompt=strict_user,
            temperature=0.0,
            max_tokens=10,
        )
        letter = _extract_letter(raw)
        if letter:
            return letter
    except Exception as e:
        logger.error(f"LLM QA retry call failed: {e}")

    # --- Final fallback: context-free pure commonsense -----------------------
    # Helps when context was empty/noisy (e.g. graph ingestion failed earlier
    # in the run due to transient rate-limit). The original SocialIQA "context"
    # field is already embedded in `question` by the caller, so this is enough.
    try:
        raw = _call_llm_text(
            model=model,
            system_prompt=strict_system,
            user_prompt=strict_user,
            temperature=0.2,
            max_tokens=10,
        )
        letter = _extract_letter(raw)
        if letter:
            return letter
    except Exception as e:
        logger.error(f"LLM QA fallback call failed: {e}")

    return ""


def evaluate_with_llm(
    question: str,
    expected: str,
    generated: str,
    context: str,
    profile_templates: dict[str, str],
    model: str = "openai:gpt-4o-mini",
) -> tuple[str, str]:
    """Evaluate whether the generated answer matches expected via LLM verdict.

    Returns:
        Tuple of (CORRECT/PARTIAL/INCORRECT, explanation).
    """
    template = profile_templates.get("evaluate", "")
    if template:
        prompt = template.format(
            question=question,
            expected=expected,
            generated=generated,
            context=context[:1000],
        )
    else:
        prompt = f"""Evaluate if the generated answer matches the expected answer.

Question: {question}
Expected Answer: {expected}
Generated Answer: {generated}
Context: {context[:1000]}

Reply with one of: CORRECT, PARTIAL, or INCORRECT
Then a brief explanation on the next line."""

    try:
        response = call_llm_for_eval(prompt, max_tokens=100, model=model)
        lines = response.strip().split("\n", 1)
        result = lines[0].strip().upper()
        explanation = lines[1].strip() if len(lines) > 1 else ""

        if "CORRECT" in result and "INCORRECT" not in result:
            return "CORRECT", explanation
        elif "PARTIAL" in result:
            return "PARTIAL", explanation
        else:
            return "INCORRECT", explanation
    except Exception as e:
        return "ERROR", str(e)


# ---------------------------------------------------------------------------
# BenchmarkRunner base class
# ---------------------------------------------------------------------------


class BenchmarkRunner(ABC):
    """Base class for standard Cognifold benchmark runners.

    Subclasses must implement:
    - benchmark_name: class attribute with the benchmark name
    - default_data_path: class attribute with the default data file path
    - load_dataset(data_path, limit) -> list of examples
    - build_events(example, idx) -> list of Event objects
    - evaluate_example(example, idx, graph, query_agent, ...) -> dict with eval results

    Optional overrides:
    - add_extra_args(parser) -- add benchmark-specific CLI args
    - filter_events(events) -- filter events before ingestion (e.g. skip noise)
    - save_results(all_results, output_dir, config) -- custom result saving
    - print_summary(all_results, config) -- custom summary printing
    - get_query_config_overrides() -- override QueryConfig defaults
    """

    benchmark_name: str = ""
    default_data_path: Path = Path(".")

    def __init__(self) -> None:
        self.profile_path = (
            Path(__file__).parents[2] / "configs" / f"{self.benchmark_name}_profile.yaml"
        )
        self.output_dir = f"benchmarks/{self.benchmark_name}/output"
        self._sym_tracker: Any = None  # Set per-example during run()

    # --- Abstract methods (must be implemented by subclasses) ---

    @abstractmethod
    def load_dataset(self, data_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
        """Load the benchmark dataset.

        Args:
            data_path: Path to the data file.
            limit: Optional limit on number of examples.

        Returns:
            List of example dicts.
        """

    @abstractmethod
    def build_events(self, example: dict[str, Any], idx: int) -> list[Event]:
        """Convert a single dataset example to Cognifold events.

        Args:
            example: A single example dict from the dataset.
            idx: The index of this example in the dataset.

        Returns:
            List of Event objects for ingestion.
        """

    @abstractmethod
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
        """Evaluate a single example after event ingestion.

        Args:
            example: The example dict.
            idx: Example index.
            graph: The populated ConceptGraph.
            query_agent: The MemoryQueryAgent to use for querying.
            query_mode: Query mode string.
            use_llm_eval: Whether to use LLM for evaluation.
            profile_templates: Prompt profile templates.
            llm_model: LLM model name for evaluation.

        Returns:
            Dict with evaluation results (must include at least 'is_correct' or 'verdict').
        """

    # --- Optional overrides ---

    def add_extra_args(self, parser: argparse.ArgumentParser) -> None:
        """Add benchmark-specific CLI arguments. Override in subclass if needed."""
        return None

    def filter_events(self, events: list[Event]) -> list[Event]:
        """Filter events before ingestion. Override to skip noise, etc."""
        return events

    def post_ingest(self, graph: ConceptGraph, events: list[Event]) -> None:
        """Hook called after all events are ingested. Runs edge inference by default."""
        if graph.node_count < 5:
            return

        sample_nodes = graph.get_all_nodes()[:5]
        has_embeddings = any(n.embedding for n in sample_nodes)
        if not has_embeddings:
            return

        try:
            # Ablation hook: COGNIFOLD_ABLATE_KNN=1 skips kNN edge inference.
            # Used by ablation benchmark runs to isolate kNN's causal
            # contribution; production runs leave it unset.
            if os.environ.get("COGNIFOLD_ABLATE_KNN") == "1":
                print("    Post-ingest: kNN inference [ABLATED]")
                return
            from cognifold.graph.edge_inference import EdgeInferenceEngine

            engine = EdgeInferenceEngine(
                graph,
                similarity_threshold=0.35,
                max_edges_per_node=3,
                source_types=["concept"],
                target_types=["concept"],
            )
            new_edges = engine.infer_edges()
            if new_edges:
                print(f"    Post-ingest: inferred {len(new_edges)} edges")
        except Exception as e:
            logger.debug(f"Edge inference skipped: {e}")

    def get_query_config_overrides(self) -> dict[str, Any]:
        """Return overrides for QueryConfig. Override for custom settings."""
        return {}

    def save_results(
        self,
        all_results: list[dict[str, Any]],
        output_dir: str,
        config: dict[str, Any],
    ) -> None:
        """Save benchmark results. Override for custom format."""
        results_path = os.path.join(output_dir, "benchmark_results.json")
        with open(results_path, "w") as f:
            json.dump(
                {
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
        """Print benchmark summary. Override for custom summary."""
        total = len(all_results)
        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        print(f"  Query Mode: {config.get('query_mode', 'unknown')}")
        print(f"  Disable Concepts: {config.get('disable_concepts', False)}")
        print(f"  LLM Eval: {config.get('use_llm_eval', True)}")
        print(f"  Total: {total}")

    def get_cognition_router(
        self, query_agent: MemoryQueryAgent,
    ) -> Any:
        """Get a CognitionRouter for this benchmark run.

        The CognitionRouter coordinates between symbolic memory (deterministic
        facts) and graph memory (semantic associations) using a three-phase
        protocol: recognition → reconstruction → validation.
        """
        from cognifold.symbolic.cognition_router import CognitionRouter

        tracker = self._sym_tracker if hasattr(self, "_sym_tracker") else None
        return CognitionRouter(symbolic=tracker, query_agent=query_agent)

    def cognition_query(
        self,
        question: str,
        query_agent: MemoryQueryAgent,
        domain: str | None = None,
        query_mode: str = "mergefold",
        **kwargs: Any,
    ) -> Any:
        """Query using the unified cognition router.

        Returns a CognitionResult with:
        - .context: fused context for LLM
        - .direct_answer: symbolic answer (if available, bypasses LLM)
        - .source: 'symbolic', 'hybrid', or 'graph'
        - .query_result: underlying QueryResult
        """
        router = self.get_cognition_router(query_agent)
        return router.answer(
            question, domain=domain, query_mode=query_mode, **kwargs
        )

    def get_example_id(self, example: dict[str, Any], idx: int) -> str:
        """Extract or generate an example ID."""
        return example.get("id", f"ex_{idx}")

    def print_example_header(self, example: dict[str, Any], idx: int, total: int) -> None:
        """Print a header line for the current example."""
        example_id = self.get_example_id(example, idx)
        print(f"\nProcessing Example {idx + 1}/{total} (ID: {example_id})")

    # --- Core run method ---

    def run(
        self,
        limit: int | None = None,
        visualize: bool = False,
        disable_concepts: bool = False,
        query_mode: str = "mergefold",
        use_llm_eval: bool = True,
        use_profile: bool = True,
        data_path: Path | None = None,
        embedding: str | None = None,
        model: str | None = None,
        **extra_kwargs: Any,
    ) -> None:
        """Run the benchmark (main entry point)."""
        if not check_api_keys():
            return

        # Resolve embedding
        from benchmarks._utils import create_embedder, resolve_embedding

        resolved_embedding = resolve_embedding(embedding, self.profile_path, self.benchmark_name)
        try:
            embedder, retrieval_mode = create_embedder(resolved_embedding)
        except (RuntimeError, ValueError) as e:
            print(f"Warning: Embedding init failed ({e}), falling back to BM25")
            embedder = None
            from cognifold.query.models import RetrievalMode

            retrieval_mode = RetrievalMode.BM25
        if embedder:
            print(f"Using embedding: {resolved_embedding}")
        else:
            print("Using retrieval: BM25 (no embedding)")

        # Load prompt profile
        prompt_profile = None
        profile_templates: dict[str, str] = {}
        llm_model = "openai:gpt-4o-mini" if os.environ.get("OPENAI_API_KEY") else "gemini-2.5-flash"
        if use_profile and self.profile_path.exists():
            try:
                profiles = load_prompt_profiles(self.profile_path)
                prompt_profile = profiles.get(self.benchmark_name)
                if prompt_profile:
                    print(f"Using profile: {self.benchmark_name} from {self.profile_path}")
                import yaml

                with open(self.profile_path) as f:
                    raw = yaml.safe_load(f)
                bench_raw = raw.get("profiles", {}).get(self.benchmark_name, {})
                profile_templates = bench_raw.get("templates", {})
                raw_model = bench_raw.get("model", {}).get("name", "")
                if raw_model:
                    llm_model = raw_model
            except Exception as e:
                print(f"Warning: Could not load profile: {e}")
        if model:
            llm_model = model

        # Load data
        dp = data_path or self.default_data_path
        if not dp.exists():
            print(f"Dataset not found at {dp}. Please run download_data.py first.")
            sys.exit(1)

        data = self.load_dataset(dp, limit)
        os.makedirs(self.output_dir, exist_ok=True)

        all_results: list[dict[str, Any]] = []

        for i, example in enumerate(data):
            self.print_example_header(example, i, len(data))
            example_id = self.get_example_id(example, i)

            # Initialize fresh graph
            graph = ConceptGraph()

            # Configure agent
            if prompt_profile:
                config = prompt_profile.to_agent_config()
                if disable_concepts:
                    config = dataclasses.replace(config, disable_concepts=True)
                if model:
                    config = dataclasses.replace(
                        config, model_name=_normalize_agent_model_name(model)
                    )
                agent = CognifoldAgent(config=config, prompt_profile=prompt_profile)
            else:
                # Default to whichever key is available: OpenAI if present, else Gemini.
                default_model = (
                    "openai:gpt-4o-mini" if os.environ.get("OPENAI_API_KEY") else "gemini-2.5-flash"
                )
                config = AgentConfig(model_name=default_model, temperature=0.0)
                if disable_concepts:
                    config = dataclasses.replace(config, disable_concepts=True)
                if model:
                    config = dataclasses.replace(
                        config, model_name=_normalize_agent_model_name(model)
                    )
                agent = CognifoldAgent(config=config)

            executor = PlanExecutor(graph)

            # Build query config with optional overrides
            qc_kwargs: dict[str, Any] = {
                "domain": self.benchmark_name,
                "max_nodes": 20,
                "include_reasoning": True,
                "retrieval_mode": retrieval_mode,
            }
            qc_kwargs.update(self.get_query_config_overrides())
            query_config = QueryConfig(**qc_kwargs)
            query_agent = MemoryQueryAgent(graph, config=query_config, embedder=embedder)

            # Initialize replay logger if visualizing
            graph_logger = None
            log_path_str: str | None = None
            if visualize:
                log_path_str = os.path.join(self.output_dir, f"replay_{example_id}.jsonl")
                graph_logger = GraphLogger(log_path=Path(log_path_str))
                graph_logger.log_run_start(
                    timeline_path=f"{self.benchmark_name}_{example_id}",
                    total_events=0,
                    config={"disable_concepts": disable_concepts},
                )

            # Build and filter events
            events = self.build_events(example, i)
            events = self.filter_events(events)
            print(f"  Ingesting {len(events)} events...")

            # Initialize symbolic state tracker
            from cognifold.symbolic.state_tracker import SymbolicStateTracker

            sym_tracker = SymbolicStateTracker()
            sym_action_count = 0

            # Ingest events via agent
            step = 1
            for event in events:
                if graph_logger:
                    graph_logger.log_event_start(
                        step=step,
                        event_id=event.event_id,
                        event_type=event.event_type,
                        title=event.title,
                        timestamp=event.timestamp.isoformat(),
                        event_data=event.model_dump(mode="json"),
                    )

                try:
                    retrieval = query_agent.query_semantic(event.description[:200])
                    context_node_ids = [n.node_id for n in retrieval.nodes[:10]]

                    plan = agent.process_event(
                        event=event,
                        graph=graph,
                        context_node_ids=context_node_ids,
                        node_scores={},
                    )
                    executor.execute(plan)

                    # Feed symbolic actions to state tracker
                    if plan.symbolic_actions:
                        for sa in plan.symbolic_actions:
                            sym_tracker.process_action(sa)
                            sym_action_count += 1

                    if graph_logger and plan.operations:
                        for op in plan.operations:
                            graph_logger.log_operation(
                                step=step,
                                op_type=op.op.value,
                                op_data=op.model_dump(mode="json"),
                                success=True,
                            )

                    if graph_logger:
                        graph_logger.log_event_end(
                            step=step,
                            event_id=event.event_id,
                            operations_count=len(plan.operations),
                            reasoning=plan.reasoning,
                        )

                    time.sleep(0.1)

                except Exception as e:
                    print(f"    Error processing event: {e}")
                    if "429" in str(e):
                        print("    Rate limit hit, sleeping for 10s...")
                        time.sleep(10)

                step += 1

            print(f"    Graph: {graph.node_count} nodes, {graph.edge_count} edges")

            # Symbolic state injection: inject tracked state into graph
            if sym_tracker.has_state:
                nodes_injected, corrections = sym_tracker.inject_into_graph(graph)
                print(
                    f"    Symbolic: {sym_action_count} actions processed,"
                    f" {nodes_injected} state nodes injected,"
                    f" {corrections} LLM concepts corrected"
                )

            # Store tracker for QA-time context injection
            self._sym_tracker = sym_tracker

            # Post-ingestion consolidation: merge similar concepts & tag orphans
            try:
                from cognifold.graph.consolidation import (
                    merge_similar_concepts,
                    prune_orphan_concepts,
                )

                # Ablation hook: COGNIFOLD_ABLATE_MERGE=1 skips merges.
                # Used by ablation benchmark runs to isolate MERGE_NODES'
                # causal contribution; production runs leave it unset.
                ablate_merge = os.environ.get("COGNIFOLD_ABLATE_MERGE") == "1"
                merges = 0 if ablate_merge else merge_similar_concepts(graph)
                orphans = prune_orphan_concepts(graph)
                if merges or orphans or ablate_merge:
                    tag = " [ABLATED]" if ablate_merge else ""
                    print(f"    Consolidation{tag}: {merges} merges, {orphans} orphans tagged")
            except Exception as e:
                logger.debug("Consolidation step failed: %s", e)

            # Post-ingestion: fact extraction and entity indexing
            try:
                from cognifold.graph.entity_index import EntityIndex
                from cognifold.graph.fact_extraction import extract_facts

                fact_ids = extract_facts(graph)
                if fact_ids:
                    print(f"    Fact extraction: {len(fact_ids)} fact nodes created")

                entity_idx = EntityIndex()
                entity_idx.build(graph)
                graph.entity_index = entity_idx
                print(f"    Entity index: {entity_idx.entity_count} entities indexed")
            except Exception as e:
                logger.debug("Fact extraction / entity indexing failed: %s", e)

            # Post-ingestion hook (e.g., add temporal adjacency edges)
            self.post_ingest(graph, events)

            # Generate replay visualization
            if graph_logger and log_path_str:
                graph_logger.log_run_end(
                    total_steps=step - 1,
                    node_count=graph.node_count,
                    edge_count=graph.edge_count,
                )
                graph_logger.close()

                print("  Generating replay visualization...")
                try:
                    player = ReplayPlayer.from_log(Path(log_path_str))
                    renderer = ReplayRenderer()
                    html_path = os.path.join(self.output_dir, f"{example_id}_replay.html")
                    renderer.render(
                        player=player,
                        output_path=Path(html_path),
                        title=f"Cognifold Replay: {self.benchmark_name} {example_id}",
                    )
                    print(f"  Replay saved to {html_path}")
                except Exception as e:
                    print(f"  Replay generation failed: {e}")

            # Invalidate semantic search cache so query uses the fully
            # populated graph (index built during ingestion is stale).
            query_agent.invalidate_search_cache()

            # Make symbolic tracker accessible to evaluate_example
            # via self._sym_tracker (already set at line 691)

            # Evaluate
            print("  Running QA Evaluation...")
            try:
                eval_result = self.evaluate_example(
                    example=example,
                    idx=i,
                    graph=graph,
                    query_agent=query_agent,
                    query_mode=query_mode,
                    use_llm_eval=use_llm_eval,
                    profile_templates=profile_templates,
                    llm_model=llm_model,
                )
                # Support returning a list of results (e.g. qmsum with multiple queries)
                if isinstance(eval_result, list):
                    all_results.extend(eval_result)
                else:
                    all_results.append(eval_result)
            except Exception as e:
                print(f"    Error evaluating: {e}")

        # Save results and print summary
        run_config = {
            "query_mode": query_mode,
            "use_llm_eval": use_llm_eval,
            "disable_concepts": disable_concepts,
        }
        self.save_results(all_results, self.output_dir, run_config)
        self.print_summary(all_results, run_config)

    # --- CLI entry point ---

    def main(self) -> None:
        """Parse CLI arguments and run the benchmark."""
        parser = argparse.ArgumentParser(
            description=f"Run {self.benchmark_name.upper()} Benchmark on Cognifold"
        )
        parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
        parser.add_argument(
            "--visualize", action="store_true", help="Generate replay visualization"
        )
        parser.add_argument(
            "--disable-concepts",
            action="store_true",
            help="Disable concept formation (Episodic mode)",
        )
        parser.add_argument(
            "--query-mode",
            type=str,
            default="mergefold",
            help="Query mode (base, rag, episodic, mergefold)",
        )
        parser.add_argument(
            "--no-llm-eval",
            action="store_true",
            help="Use simple extraction instead of LLM evaluation",
        )
        parser.add_argument(
            "--no-profile",
            action="store_true",
            help="Don't use the prompt profile",
        )
        parser.add_argument(
            "--data",
            type=Path,
            default=None,
            help="Data file path",
        )
        parser.add_argument(
            "--embedding",
            type=str,
            default=None,
            help="Embedding model (e.g. openai:text-embedding-3-small, gemini:text-embedding-004, or none). Overrides profile config.",
        )
        parser.add_argument(
            "--model",
            type=str,
            default=None,
            help=(
                "Override the LLM model used for BOTH ingestion and QA/eval. "
                "Examples: openai:gpt-4o-mini-2024-07-18, gemini-2.0-flash, flash. "
                "Note: OpenAI models should be prefixed with 'openai:' for ingestion."
            ),
        )

        # Let subclass add extra args
        self.add_extra_args(parser)

        args = parser.parse_args()

        # Collect extra kwargs from extra args
        known_args = {
            "limit",
            "visualize",
            "disable_concepts",
            "query_mode",
            "no_llm_eval",
            "no_profile",
            "data",
            "embedding",
            "model",
        }
        extra_kwargs = {k: v for k, v in vars(args).items() if k not in known_args}

        self.run(
            limit=args.limit,
            visualize=args.visualize,
            disable_concepts=args.disable_concepts,
            query_mode=args.query_mode,
            use_llm_eval=not args.no_llm_eval,
            use_profile=not args.no_profile,
            data_path=args.data,
            embedding=args.embedding,
            model=args.model,
            **extra_kwargs,
        )
