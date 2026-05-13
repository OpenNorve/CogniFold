"""Command-line interface for Cognifold.

This module provides the main entry point for the Cognifold CLI tool.
Commands are organized into submodules for maintainability.
"""

from __future__ import annotations

import argparse
import sys

from cognifold.cli.build import add_build_parser, build_command
from cognifold.cli.client import add_client_parser, client_command
from cognifold.cli.config import add_config_parser, config_command
from cognifold.cli.generate import add_generate_parser, generate_command
from cognifold.cli.query import add_query_parser, query_command
from cognifold.cli.replay import add_replay_parser, replay_command
from cognifold.cli.run import add_run_parser, run_command
from cognifold.cli.serve import add_serve_parser, serve_command


def main() -> int:
    """Main entry point for the Cognifold CLI."""
    parser = argparse.ArgumentParser(
        description="Cognifold - Dynamic concept graph for event streams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simulation with default plans (no LLM)
  cognifold run data/mock_timeline.json

  # Run simulation with LLM agent
  cognifold run data/mock_timeline.json --agent

  # Run simulation and generate visualizations
  cognifold run data/mock_timeline.json --agent --output output/

  # Query the concept graph
  cognifold query --graph output/graph.json "What patterns exist?"

  # Get top concepts from the graph
  cognifold query --graph output/graph.json --top-concepts 10

  # Generate personal timeline events
  cognifold generate --domain personal-timeline --persona software_engineer --events 100

  # Generate computer activity events
  cognifold generate --domain computer-activity --profile software_developer --events 100

  # Generate service log events
  cognifold generate --domain service-logs --topology ecommerce --events 100

  # List available options for a domain
  cognifold generate --domain computer-activity --list

  # Build timeline from wiki/markdown files
  cognifold build-timeline --source wiki --input data/wiki/ -o data/timeline.json

  # Generate replay from logs
  cognifold replay logs/replay_*.jsonl -o output/replay.html --open

  # Show configuration
  cognifold config --show

  # Generate example config file
  cognifold config --generate config.yaml
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add subcommand parsers
    add_run_parser(subparsers)
    add_query_parser(subparsers)
    add_config_parser(subparsers)
    add_generate_parser(subparsers)
    add_replay_parser(subparsers)
    add_build_parser(subparsers)
    add_serve_parser(subparsers)
    add_client_parser(subparsers)

    # Version
    parser.add_argument("--version", action="version", version="cognifold 0.1.0")

    args = parser.parse_args()

    if args.command == "run":
        return run_command(args)
    elif args.command == "query":
        return query_command(args)
    elif args.command == "config":
        return config_command(args)
    elif args.command == "generate":
        return generate_command(args)
    elif args.command == "replay":
        return replay_command(args)
    elif args.command == "build-timeline":
        return build_command(args)
    elif args.command == "serve":
        return serve_command(args)
    elif args.command == "client":
        return client_command(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
