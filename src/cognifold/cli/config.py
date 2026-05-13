"""Config command for Cognifold CLI."""

from __future__ import annotations

import argparse


def add_config_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore
    """Add the config subcommand parser."""
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_parser.add_argument("--show", action="store_true", help="Show current configuration")
    config_parser.add_argument(
        "--generate", type=str, metavar="FILE", help="Generate example config file"
    )


def config_command(args: argparse.Namespace) -> int:
    """Handle config subcommand."""
    from cognifold.config import CognifoldConfig

    if args.generate:
        config = CognifoldConfig()
        config.to_yaml(args.generate)
        print(f"Configuration file generated: {args.generate}")
        return 0

    if args.show:
        config = CognifoldConfig.load()
        print("Current configuration:")
        print(f"  Model: {config.model.name}")
        print(f"    Temperature: {config.model.temperature}")
        print(f"    Max tokens: {config.model.max_tokens}")
        print(f"    Max exploration steps: {config.model.max_exploration_steps}")
        print("  Scoring weights:")
        print(f"    Alpha (PageRank): {config.scoring.alpha}")
        print(f"    Beta (Recency): {config.scoring.beta}")
        print(f"    Gamma (Access): {config.scoring.gamma}")
        print(f"    Decay rate: {config.scoring.decay_rate}")
        print("  Context window:")
        print(f"    Max nodes: {config.context.max_nodes}")
        print(f"    Min score threshold: {config.context.min_score_threshold}")
        print("  Logging:")
        print(f"    Level: {config.logging.level}")
        print(f"    File: {config.logging.file or '(none)'}")
        print("  Paths:")
        print(f"    Data dir: {config.data_dir}")
        print(f"    Output dir: {config.output_dir}")
        print(f"  API key: {'(set)' if config.api_key else '(not set)'}")
        return 0

    print("Use --show to display configuration or --generate FILE to create a config file")
    return 0
