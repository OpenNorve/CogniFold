"""Generate command for Cognifold CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def add_generate_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore
    """Add the generate subcommand parser."""
    gen_parser = subparsers.add_parser("generate", help="Generate event timeline using LLM")
    gen_parser.add_argument(
        "--domain",
        type=str,
        choices=["personal-timeline", "computer-activity", "service-logs", "claude-code"],
        default="personal-timeline",
        help="Event domain to generate (default: personal-timeline)",
    )
    gen_parser.add_argument(
        "--persona",
        "-p",
        type=str,
        help="Built-in persona name for personal-timeline domain",
    )
    gen_parser.add_argument("--persona-file", type=str, help="Path to custom persona JSON file")
    gen_parser.add_argument(
        "--profile",
        type=str,
        help="Work profile for computer-activity domain (software_developer, data_analyst, product_manager)",
    )
    gen_parser.add_argument(
        "--topology",
        type=str,
        help="Service topology for service-logs domain (ecommerce, saas_platform, microservices_demo)",
    )
    gen_parser.add_argument(
        "--session-profile",
        type=str,
        help="Session profile for claude-code domain (feature_development, bug_fix, refactoring)",
    )
    gen_parser.add_argument(
        "--events", "-n", type=int, default=100, help="Number of events to generate (default: 100)"
    )
    gen_parser.add_argument(
        "--days", "-d", type=int, default=3, help="Number of days to span (default: 3)"
    )
    gen_parser.add_argument(
        "--output", "-o", type=str, help="Output directory (default: data/generated/)"
    )
    gen_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    gen_parser.add_argument(
        "--list", action="store_true", help="List available options for the selected domain"
    )
    gen_parser.add_argument(
        "--list-personas",
        action="store_true",
        help="List available built-in personas (deprecated, use --list)",
    )


def generate_command(args: argparse.Namespace) -> int:
    """Handle generate subcommand."""
    import os

    # Handle list mode
    if args.list or args.list_personas:
        return _list_domain_options(args.domain)

    # Check API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable is required")
        print("Set it with: export GOOGLE_API_KEY='your-api-key'")
        return 1

    # Setup output directory
    output_dir = Path(args.output) if args.output else Path("data/generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dispatch to domain-specific generator
    if args.domain == "personal-timeline":
        return _generate_personal_timeline(args, output_dir)
    elif args.domain == "computer-activity":
        return _generate_computer_activity(args, output_dir)
    elif args.domain == "service-logs":
        return _generate_service_logs(args, output_dir)
    elif args.domain == "claude-code":
        return _generate_claude_code(args, output_dir)
    else:
        print(f"Error: Unknown domain: {args.domain}")
        return 1


def _list_domain_options(domain: str) -> int:
    """List available options for a domain."""
    if domain == "personal-timeline":
        from cognifold.generator.persona import SAMPLE_PERSONAS

        print("Available built-in personas (--persona):")
        for name, persona in SAMPLE_PERSONAS.items():
            print(f"  {name}: {persona.name} ({persona.occupation})")
    elif domain == "computer-activity":
        from cognifold.generator.computer_activity import SAMPLE_PROFILES

        print("Available work profiles (--profile):")
        for name, profile in SAMPLE_PROFILES.items():
            print(f"  {name}: {profile.name}")
            print(f"      Apps: {', '.join(profile.primary_apps[:3])}...")
    elif domain == "service-logs":
        from cognifold.generator.service_logs import SAMPLE_TOPOLOGIES

        print("Available service topologies (--topology):")
        for name, topology in SAMPLE_TOPOLOGIES.items():
            print(f"  {name}: {topology.description}")
            print(f"      Services: {', '.join(topology.services[:3])}...")
    elif domain == "claude-code":
        from cognifold.generator.claude_code import SAMPLE_SESSION_PROFILES

        print("Available session profiles (--session-profile):")
        for name, profile in SAMPLE_SESSION_PROFILES.items():
            print(f"  {name}: {profile.description}")
            print(f"      Tools: {', '.join(profile.common_tools[:4])}...")
    else:
        print(f"Unknown domain: {domain}")
        return 1
    return 0


def _generate_personal_timeline(args: argparse.Namespace, output_dir: Path) -> int:
    """Generate personal timeline events."""
    from cognifold.generator import EventGenerator, Persona
    from cognifold.generator.persona import get_sample_persona

    # Load persona
    persona = None
    if args.persona_file:
        try:
            persona = Persona.load(args.persona_file)
            print(f"Loaded persona from {args.persona_file}: {persona.name}")
        except FileNotFoundError:
            print(f"Error: Persona file not found: {args.persona_file}")
            return 1
        except Exception as e:
            print(f"Error loading persona: {e}")
            return 1
    elif args.persona:
        try:
            persona = get_sample_persona(args.persona)
            print(f"Using built-in persona: {persona.name}")
        except KeyError as e:
            print(f"Error: {e}")
            print("Use --list to see available options")
            return 1
    else:
        print("Error: Must specify either --persona or --persona-file")
        print("Use --list to see available built-in personas")
        return 1

    # Generate events
    print(f"Generating {args.events} personal timeline events over {args.days} days...")
    generator = EventGenerator()

    try:
        events = generator.generate(
            persona=persona,
            num_events=args.events,
            num_days=args.days,
        )
    except Exception as e:
        print(f"Error generating events: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    # Save timeline
    persona_slug = persona.name.lower().replace(" ", "_")
    output_path = output_dir / f"{persona_slug}_timeline.json"
    generator.save_timeline(
        events=events,
        path=output_path,
        persona=persona,
        description=f"Generated timeline for {persona.name} ({args.events} events, {args.days} days)",
    )

    _print_generation_summary(events, output_path, args.verbose)
    return 0


def _generate_computer_activity(args: argparse.Namespace, output_dir: Path) -> int:
    """Generate computer activity events."""
    from cognifold.generator.computer_activity import (
        ComputerActivityGenerator,
        get_work_profile,
    )

    # Load work profile
    if not args.profile:
        print("Error: Must specify --profile for computer-activity domain")
        print("Use --list to see available work profiles")
        return 1

    try:
        profile = get_work_profile(args.profile)
        print(f"Using work profile: {profile.name}")
    except KeyError as e:
        print(f"Error: {e}")
        print("Use --list to see available options")
        return 1

    # Generate events
    print(f"Generating {args.events} computer activity events over {args.days} days...")
    generator = ComputerActivityGenerator()

    try:
        events = generator.generate(
            work_profile=profile,
            num_events=args.events,
            num_days=args.days,
        )
    except Exception as e:
        print(f"Error generating events: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    # Save timeline
    profile_slug = profile.name.lower().replace(" ", "_")
    output_path = output_dir / f"computer_{profile_slug}_timeline.json"
    generator.save_timeline(
        events=events,
        path=output_path,
        work_profile=profile,
        description=f"Generated computer activity for {profile.name} ({args.events} events, {args.days} days)",
    )

    _print_generation_summary(events, output_path, args.verbose)
    return 0


def _generate_service_logs(args: argparse.Namespace, output_dir: Path) -> int:
    """Generate service log events."""
    from cognifold.generator.service_logs import (
        ServiceLogsGenerator,
        get_service_topology,
    )

    # Load topology
    if not args.topology:
        print("Error: Must specify --topology for service-logs domain")
        print("Use --list to see available service topologies")
        return 1

    try:
        topology = get_service_topology(args.topology)
        print(f"Using service topology: {topology.name}")
    except KeyError as e:
        print(f"Error: {e}")
        print("Use --list to see available options")
        return 1

    # Generate events
    print(f"Generating {args.events} service log events over {args.days} days...")
    generator = ServiceLogsGenerator()

    try:
        events = generator.generate(
            topology=topology,
            num_events=args.events,
            num_days=args.days,
        )
    except Exception as e:
        print(f"Error generating events: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    # Save timeline
    topology_slug = topology.name.lower().replace(" ", "_")
    output_path = output_dir / f"service_{topology_slug}_timeline.json"
    generator.save_timeline(
        events=events,
        path=output_path,
        topology=topology,
        description=f"Generated service logs for {topology.name} ({args.events} events, {args.days} days)",
    )

    _print_generation_summary(events, output_path, args.verbose)
    return 0


def _generate_claude_code(args: argparse.Namespace, output_dir: Path) -> int:
    """Generate Claude Code session events."""
    from cognifold.generator.claude_code import (
        ClaudeCodeGenerator,
        get_session_profile,
    )

    # Load session profile
    if not args.session_profile:
        print("Error: Must specify --session-profile for claude-code domain")
        print("Use --list to see available session profiles")
        return 1

    try:
        profile = get_session_profile(args.session_profile)
        print(f"Using session profile: {profile.name}")
    except KeyError as e:
        print(f"Error: {e}")
        print("Use --list to see available options")
        return 1

    # Generate events
    print(f"Generating {args.events} Claude Code session events over {args.days} day(s)...")
    generator = ClaudeCodeGenerator()

    try:
        events = generator.generate(
            session_profile=profile,
            num_events=args.events,
            num_days=args.days,
        )
    except Exception as e:
        print(f"Error generating events: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    # Save timeline
    profile_slug = profile.name.lower().replace(" ", "_")
    output_path = output_dir / f"claude_code_{profile_slug}_timeline.json"
    generator.save_timeline(
        events=events,
        path=output_path,
        session_profile=profile,
        description=f"Generated Claude Code session for {profile.name} ({args.events} events, {args.days} day(s))",
    )

    _print_generation_summary(events, output_path, args.verbose)
    return 0


def _print_generation_summary(
    events: list[dict[str, Any]], output_path: Path, verbose: bool
) -> None:
    """Print generation summary."""
    print("\nGeneration complete:")
    print(f"  Events generated: {len(events)}")
    print(f"  Output file: {output_path}")

    if verbose and events:
        print("\nSample events:")
        for event in events[:5]:
            ts = event.get("timestamp", "")
            title = event.get("title", "")
            etype = event.get("event_type", "")
            print(f"  [{ts}] {title} ({etype})")
        if len(events) > 5:
            print(f"  ... and {len(events) - 5} more")
