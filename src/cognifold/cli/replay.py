"""Replay command for Cognifold CLI."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_replay_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore
    """Add the replay subcommand parser."""
    replay_parser = subparsers.add_parser(
        "replay", help="Generate interactive replay from run logs"
    )
    replay_parser.add_argument("log", type=str, help="Path to replay log file (JSONL)")
    replay_parser.add_argument("--output", "-o", type=str, help="Output HTML file path")
    replay_parser.add_argument("--title", "-t", type=str, help="Title for the replay")
    replay_parser.add_argument("--start-step", type=int, help="Start from specific step")
    replay_parser.add_argument("--end-step", type=int, help="End at specific step")
    replay_parser.add_argument(
        "--open",
        action="store_true",
        help="Open replay in browser after generation",
    )


def replay_command(args: argparse.Namespace) -> int:
    """Handle replay subcommand."""
    from cognifold.replay.player import ReplayPlayer
    from cognifold.replay.renderer import ReplayRenderer

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return 1

    print(f"Loading replay log from {log_path}...")

    try:
        player = ReplayPlayer.from_log(log_path)
    except Exception as e:
        print(f"Error loading log: {e}")
        return 1

    print(f"Loaded {len(player.keyframes)} keyframes ({player.total_steps} steps)")

    # Determine output path
    output_path = Path(args.output) if args.output else log_path.with_suffix(".html")

    # Generate title
    title = args.title
    if not title:
        timeline_path = player.timeline_path
        if timeline_path:
            title = f"Graph Evolution Replay - {Path(timeline_path).stem}"
        else:
            title = "Graph Evolution Replay"

    # Render replay
    print("Generating replay visualization...")
    renderer = ReplayRenderer()
    renderer.render(player, output_path, title=title)

    print("\nReplay generated successfully:")
    print(f"  Output: {output_path}")
    print(f"  Steps: {player.total_steps}")
    print(f"  Keyframes: {len(player.keyframes)}")

    # Open in browser if requested
    if args.open:
        import webbrowser

        webbrowser.open(f"file://{output_path.absolute()}")
        print("Opened replay in browser")
    else:
        print("\nOpen the HTML file in a browser to view the replay.")
        print(
            "Keyboard shortcuts: Space (play/pause), Arrow keys (prev/next), Home/End (start/end)"
        )

    return 0
