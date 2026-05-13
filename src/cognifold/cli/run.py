"""Run command for Cognifold CLI."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any


def add_run_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore
    """Add the run subcommand parser."""
    run_parser = subparsers.add_parser("run", help="Run simulation on a timeline")
    run_parser.add_argument("timeline", type=str, help="Path to timeline JSON file")
    run_parser.add_argument("--agent", action="store_true", help="Use LLM agent for updates")
    run_parser.add_argument("--output", "-o", type=str, help="Output directory for visualizations")
    run_parser.add_argument("--config", "-c", type=str, help="Path to config YAML file")
    run_parser.add_argument("--steps", "-n", type=int, help="Number of steps to run (default: all)")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    run_parser.add_argument("--save-graph", type=str, help="Save final graph to JSON file")
    run_parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for log files (default: logs/)",
    )
    run_parser.add_argument(
        "--open",
        action="store_true",
        help="Open visualization in browser after run",
    )
    run_parser.add_argument(
        "--prompt-profile",
        type=str,
        help="Prompt profile ID for agent mode (e.g., wiki-v1, personal-v1)",
    )
    run_parser.add_argument(
        "--prompt-profiles",
        type=str,
        default="configs/prompt_profiles.yaml",
        help="Path to prompt profiles YAML (default: configs/prompt_profiles.yaml)",
    )
    # Action mode flags (Phase 8)
    run_parser.add_argument(
        "--action-mode",
        action="store_true",
        help="Enable action mode: generate and execute actions from intents",
    )
    run_parser.add_argument(
        "--action-llm",
        type=str,
        default="mock",
        choices=["mock", "gemini"],
        help="LLM provider for action generation (default: mock)",
    )
    run_parser.add_argument(
        "--min-urgency",
        type=float,
        default=0.3,
        help="Minimum urgency threshold for actionable intents (default: 0.3)",
    )
    run_parser.add_argument(
        "--save-actions",
        type=str,
        help="Save action queue to JSON file after simulation",
    )
    # Fast mode flags (layered pipeline)
    run_parser.add_argument(
        "--fast",
        action="store_true",
        help="Use layered pipeline for fast ingest (<30s) + progressive enrichment",
    )
    run_parser.add_argument(
        "--layer",
        type=int,
        choices=[1, 2, 3],
        help="Run only a specific layer (requires --fast)",
    )
    run_parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Events per LLM batch in Layer 2 (default: 10)",
    )
    run_parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip Layer 3 (batch embeddings)",
    )


def run_command(args: argparse.Namespace) -> int:
    """Run the simulation."""
    from cognifold.config import CognifoldConfig
    from cognifold.logging import get_logger, setup_logging
    from cognifold.replay.logger import GraphLogger
    from cognifold.scoring.ranker import ScoringConfig
    from cognifold.simulator import Simulator

    # Load configuration
    config = CognifoldConfig.load(args.config)

    # Setup logging with file output
    if args.verbose:
        config.logging.level = "DEBUG"

    # Create timestamped log file
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timeline_name = Path(args.timeline).stem
    log_file = log_dir / f"run_{timeline_name}_{timestamp}.log"
    config.logging.file = str(log_file)

    # Create replay log file (JSONL format)
    replay_log_file = log_dir / f"replay_{timeline_name}_{timestamp}.jsonl"

    setup_logging(config.logging)
    logger = get_logger("cli")

    # Log run parameters
    logger.info("=== Cognifold Run Started ===")
    logger.info(f"Timeline: {args.timeline}")
    logger.info(f"Agent mode: {args.agent}")
    logger.info(f"Steps: {args.steps or 'all'}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Replay log: {replay_log_file}")
    print(f"Logging to: {log_file}")

    # Check API key if using agent
    if args.agent and not config.api_key:
        logger.error("GOOGLE_API_KEY environment variable is required when using --agent")
        print("Error: GOOGLE_API_KEY environment variable is required when using --agent")
        print("Set it with: export GOOGLE_API_KEY='your-api-key'")
        return 1

    # Check timeline exists
    timeline_path = Path(args.timeline)
    if not timeline_path.exists():
        logger.error(f"Timeline file not found: {timeline_path}")
        print(f"Error: Timeline file not found: {timeline_path}")
        return 1

    # ── Fast mode (layered pipeline) ──────────────────────────────────
    if getattr(args, "fast", False):
        return _run_fast_mode(args, config, logger, timeline_path, timestamp, log_file)

    # Create graph logger for replay
    graph_logger = GraphLogger(log_path=replay_log_file)

    # Create simulator with config and logger
    scoring_config = ScoringConfig(
        alpha=config.scoring.alpha,
        beta=config.scoring.beta,
        gamma=config.scoring.gamma,
        decay_rate=config.scoring.decay_rate,
        context_window_size=config.context.max_nodes,
        min_score_threshold=config.context.min_score_threshold,
    )
    agent_config = None
    prompt_profile = None
    if args.agent:
        from cognifold.agent import AgentConfig
        from cognifold.agent.prompt_profile import load_prompt_profiles

        base_agent_config = AgentConfig(
            model_name=config.model.name,
            temperature=config.model.temperature,
            max_tokens=config.model.max_tokens,
            max_exploration_steps=config.model.max_exploration_steps,
        )

        if args.prompt_profile:
            profiles_path = Path(args.prompt_profiles)
            if profiles_path.exists():
                profiles = load_prompt_profiles(profiles_path)
                prompt_profile = profiles.get(args.prompt_profile)
            if not prompt_profile:
                logger.error(f"Prompt profile not found: {args.prompt_profile}")
                print(f"Error: Prompt profile not found: {args.prompt_profile}")
                return 1
            agent_config = prompt_profile.to_agent_config(base_agent_config)
        else:
            agent_config = base_agent_config

    # Action mode configuration
    action_mode = getattr(args, "action_mode", False)
    action_config = None
    if action_mode:
        action_config = {
            "llm_provider": getattr(args, "action_llm", "mock"),
            "min_urgency": getattr(args, "min_urgency", 0.3),
            "max_intents_per_step": 3,
            "max_actions_per_intent": 3,
            "time_compression": 1.0,
        }
        logger.info(f"Action mode enabled with config: {action_config}")

    simulator = Simulator(
        scoring_config=scoring_config,
        graph_logger=graph_logger,
        agent_config=agent_config,
        prompt_profile=prompt_profile,
        action_mode=action_mode,
        action_config=action_config,
    )
    simulator.run_config = {
        "model": {
            "name": config.model.name,
            "temperature": config.model.temperature,
            "max_tokens": config.model.max_tokens,
            "max_exploration_steps": config.model.max_exploration_steps,
        },
        "scoring": {
            "alpha": config.scoring.alpha,
            "beta": config.scoring.beta,
            "gamma": config.scoring.gamma,
            "decay_rate": config.scoring.decay_rate,
        },
        "context": {
            "max_nodes": config.context.max_nodes,
            "min_score_threshold": config.context.min_score_threshold,
        },
        "prompt_profile": {
            "id": prompt_profile.profile_id if prompt_profile else None,
            "domain": prompt_profile.domain if prompt_profile else None,
            "mode": prompt_profile.mode.value if prompt_profile and prompt_profile.mode else None,
            "features": prompt_profile.features if prompt_profile else {},
        },
    }

    # Load timeline
    logger.info(f"Loading timeline from {timeline_path}")
    simulator.load_timeline(timeline_path)
    total_events = len(simulator.state.timeline) if simulator.state.timeline else 0
    logger.info(f"Loaded {total_events} events")
    print(f"Loaded {total_events} events from {timeline_path}")

    # Determine steps to run
    max_steps = args.steps if args.steps else total_events

    # Run simulation
    step = 0
    while not simulator.state.is_complete and step < max_steps:
        event = simulator.state.current_event
        if event is None:
            break

        logger.info(f"Processing event {step + 1}/{max_steps}: {event.title}")
        print(f"[{step + 1}/{max_steps}] {event.title}")

        try:
            if args.agent and action_mode:
                # Use agent for plan generation + action mode for intent execution
                # First, step with agent to generate intents
                simulator.step_with_agent()
                # Then process any actionable intents that were created
                if simulator._action_queue is None:  # type: ignore[reportPrivateUsage]
                    simulator._init_action_mode()  # type: ignore[reportPrivateUsage]
                simulator._process_actionable_intents(event.timestamp)  # type: ignore[reportPrivateUsage]
                # Execute any actions due before the next event
                next_event = simulator.state.current_event
                if next_event:
                    simulator._execute_due_actions(event.timestamp, next_event.timestamp)  # type: ignore[reportPrivateUsage]
            elif args.agent:
                # Use agent for plan generation only
                simulator.step_with_agent()
            elif action_mode:
                # Use action mode (with default plans)
                simulator.step_with_actions()
            else:
                # Simple default plan
                simulator.step()
        except Exception as e:
            logger.error(f"Step error: {e}")
            print(f"  Warning: Step error, using fallback: {e}")
            try:
                simulator.step()
            except Exception as e2:
                logger.warning(f"Fallback also failed: {e2}")
                # Advance to next event even if both plans fail
                simulator.state.current_step += 1

        step += 1

    # Show final stats
    status = simulator.get_status()

    # Log run end for replay
    graph_logger.log_run_end(
        total_steps=status["current_step"],
        node_count=status["graph"]["node_count"],
        edge_count=status["graph"]["edge_count"],
    )
    graph_logger.close()

    logger.info("=== Simulation Complete ===")
    logger.info(f"Events processed: {status['current_step']}")
    logger.info(f"Nodes in graph: {status['graph']['node_count']}")
    logger.info(f"Edges in graph: {status['graph']['edge_count']}")
    logger.info(f"Plans applied: {status['plans_applied']}")
    logger.info(f"Replay log saved: {replay_log_file}")

    print("\nSimulation complete:")
    print(f"  Events processed: {status['current_step']}")
    print(f"  Nodes in graph: {status['graph']['node_count']}")
    print(f"  Edges in graph: {status['graph']['edge_count']}")
    print(f"  Plans applied: {status['plans_applied']}")
    print(f"  Replay log: {replay_log_file}")

    # Action mode summary
    if action_mode:
        action_summary = simulator.get_action_summary()
        if action_summary.get("action_mode"):
            logger.info(f"Action mode summary: {action_summary}")
            print("\nAction mode results:")
            print(f"  Total actions: {action_summary['total_actions']}")
            print(f"  Action results processed: {action_summary['action_results_processed']}")
            print(f"  Intents processed: {action_summary['intents_processed']}")
            for status_name, count in action_summary.get("status_counts", {}).items():
                print(f"    - {status_name}: {count}")

    # Generate visualizations if output directory specified
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use timestamped filename for visualization
        viz_file = f"graph_{timestamp}.html"
        logger.info(f"Generating visualization to {output_dir / viz_file}")
        output_path = simulator.visualize(
            output_dir / viz_file,
            title=f"Graph State - {timeline_name} ({timestamp})",
        )
        logger.info(f"Visualization saved to: {output_path}")
        print(f"\nVisualization saved to: {output_path}")

        # Open in browser if requested
        if args.open:
            import webbrowser

            webbrowser.open(f"file://{output_path.absolute()}")
            print("Opened visualization in browser")

    # Save graph if requested
    if args.save_graph:
        from cognifold.graph.persistence import save_graph

        save_path = Path(args.save_graph)
        save_graph(simulator.state.graph, save_path)
        logger.info(f"Graph saved to {save_path}")
        print(f"Graph saved to: {save_path}")

    # Save action queue if requested
    if getattr(args, "save_actions", None) and action_mode:
        save_path = Path(args.save_actions)
        if simulator._action_queue:  # type: ignore[reportPrivateUsage]
            simulator._action_queue.save(save_path)  # type: ignore[reportPrivateUsage]
            logger.info(f"Action queue saved to {save_path}")
            print(f"Action queue saved to: {save_path}")
        else:
            # Create empty queue file to indicate action mode ran
            import json

            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(
                    {"version": "1.0", "actions": [], "note": "No actions generated"}, f, indent=2
                )
            logger.info(f"Empty action queue saved to {save_path} (no intents found)")
            print(f"Action queue saved to: {save_path} (no actions)")

    logger.info(f"Log file saved: {log_file}")
    print(f"\nFull log saved to: {log_file}")

    return 0


def _run_fast_mode(
    args: argparse.Namespace,
    config: Any,
    logger: Any,
    timeline_path: Path,
    timestamp: str,
    log_file: Path,
) -> int:
    """Run the fast (layered) pipeline.

    Behavior matrix:
    | Command                                      | Result                      |
    |----------------------------------------------|-----------------------------|
    | cognifold run timeline.json --fast            | Layer 1 only (<30s)         |
    | cognifold run timeline.json --fast --agent    | All 3 layers                |
    | cognifold run timeline.json --fast --layer 2  | Layer 2 only (graph loaded) |
    | cognifold run timeline.json --fast --no-embeddings --agent | Layer 1 + 2    |
    """
    from cognifold.pipeline.layered import LayeredPipeline

    logger.info("=== Fast Mode (Layered Pipeline) ===")
    print("Mode: FAST (layered pipeline)")

    # Build agent config if needed
    agent_config: Any = None
    prompt_profile: Any = None
    if args.agent:
        from cognifold.agent import AgentConfig
        from cognifold.agent.prompt_profile import load_prompt_profiles

        base_agent_config = AgentConfig(
            model_name=config.model.name,
            temperature=config.model.temperature,
            max_tokens=config.model.max_tokens,
            max_exploration_steps=config.model.max_exploration_steps,
        )

        if getattr(args, "prompt_profile", None):
            profiles_path = Path(args.prompt_profiles)
            if profiles_path.exists():
                profiles = load_prompt_profiles(profiles_path)
                prompt_profile = profiles.get(args.prompt_profile)
            if not prompt_profile:
                print(f"Error: Prompt profile not found: {args.prompt_profile}")
                return 1
            agent_config = prompt_profile.to_agent_config(base_agent_config)
        else:
            agent_config = base_agent_config

    batch_size = getattr(args, "batch_size", 10)

    pipeline = LayeredPipeline(
        agent_config=agent_config,
        prompt_profile=prompt_profile,
        batch_size=batch_size,
    )

    # Load timeline
    count = pipeline.load_timeline(timeline_path)
    print(f"Loaded {count} events from {timeline_path}")

    specific_layer = getattr(args, "layer", None)
    skip_embeddings = getattr(args, "no_embeddings", False)

    # Decide which layers to run
    if specific_layer:
        # Run only the specified layer
        if specific_layer == 1:
            pipeline.run_layer1()
        elif specific_layer == 2:
            # Layer 2 needs events in graph first
            pipeline.run_layer1()
            pipeline.run_layer2()
        elif specific_layer == 3:
            pipeline.run_layer1()
            pipeline.run_layer3()
    elif args.agent:
        # --fast --agent → all layers (or skip L3 with --no-embeddings)
        pipeline.run_layer1()
        pipeline.run_layer2()
        if not skip_embeddings:
            pipeline.run_layer3()
    else:
        # --fast only → Layer 1 (fast ingest, no LLM)
        pipeline.run_layer1()
        if not skip_embeddings:
            pipeline.run_layer3()

    stats = pipeline.stats

    # Print summary
    print("\nFast mode complete:")
    print(f"  Layer 1: {stats.layer1_events} events in {stats.layer1_time_ms:.0f}ms")
    if stats.layer2_batches > 0:
        print(
            f"  Layer 2: {stats.layer2_batches} batches, "
            f"{stats.layer2_plans} plans in {stats.layer2_time_ms:.0f}ms"
        )
    if stats.layer3_nodes_embedded > 0:
        print(
            f"  Layer 3: {stats.layer3_nodes_embedded} nodes embedded in {stats.layer3_time_ms:.0f}ms"
        )
    print(
        f"  Total: {stats.total_nodes} nodes, {stats.total_edges} edges, {stats.total_time_ms:.0f}ms"
    )
    if stats.errors:
        print(f"  Errors: {len(stats.errors)}")
        for err in stats.errors[:5]:
            print(f"    - {err}")

    logger.info(
        "Fast mode: %d events, %d nodes, %d edges, %.0fms",
        stats.layer1_events,
        stats.total_nodes,
        stats.total_edges,
        stats.total_time_ms,
    )

    # Generate visualization if output directory specified
    if args.output:
        from cognifold.scoring.ranker import ContextRanker, ScoringConfig
        from cognifold.simulator.visualizer import GraphVisualizer

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        viz_file = f"graph_{timestamp}.html"
        ranker = ContextRanker(ScoringConfig())
        context_ids = ranker.get_context_node_ids(pipeline.graph)
        node_scores = {s.node_id: s.composite_score for s in ranker.score_nodes(pipeline.graph)}

        visualizer = GraphVisualizer()
        output_path = visualizer.render(
            graph=pipeline.graph,
            output_path=output_dir / viz_file,
            context_node_ids=context_ids,
            node_scores=node_scores,
            title=f"Graph State [FAST] - {timeline_path.stem} ({timestamp})",
        )
        print(f"\nVisualization saved to: {output_path}")

        if getattr(args, "open", False):
            import webbrowser

            webbrowser.open(f"file://{output_path.absolute()}")

    # Save graph if requested
    if args.save_graph:
        from cognifold.graph.persistence import save_graph

        save_path = Path(args.save_graph)
        save_graph(pipeline.graph, save_path)
        print(f"Graph saved to: {save_path}")

    print(f"\nFull log saved to: {log_file}")
    return 0
