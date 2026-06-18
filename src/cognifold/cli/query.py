"""Query command for Cognifold CLI.

This module provides the 'query' subcommand for querying the concept graph.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognifold.query import MemoryQueryAgent, QueryResult
    from cognifold.query.models import QueryType


def add_query_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Add the query subcommand parser.

    Args:
        subparsers: The subparsers object from the main parser.
    """
    parser = subparsers.add_parser(
        "query",
        help="Query the concept graph for relevant context",
        description="""
Query the concept graph to retrieve relevant context.

The query command takes a natural language query and searches the graph
for relevant nodes, returning formatted context suitable for LLM consumption.

Query types:
  semantic   - Find nodes related to query meaning
  temporal   - Find nodes from recent time periods
  structural - Find highly connected/important nodes
  hybrid     - Combine all strategies (default)
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query with default hybrid strategy
  cognifold query --graph output/graph.json "What patterns exist around morning routines?"

  # Query with semantic focus
  cognifold query --graph output/graph.json --type semantic "Tell me about exercise habits"

  # Query with temporal focus (recent nodes)
  cognifold query --graph output/graph.json --type temporal "What happened recently?"

  # Query structural (important concepts)
  cognifold query --graph output/graph.json --type structural "What are the key concepts?"

  # Output as JSON
  cognifold query --graph output/graph.json --json "What actions are suggested?"

  # Get top concepts
  cognifold query --graph output/graph.json --top-concepts 10

  # Get recent intents
  cognifold query --graph output/graph.json --recent-intents 5
""",
    )

    # Required arguments
    parser.add_argument(
        "--graph",
        "-g",
        type=str,
        required=True,
        help="Path to the graph JSON file",
    )

    # Query string (positional, optional if using --top-concepts or --recent-actions)
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="Natural language query string",
    )

    # Query type
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        choices=["semantic", "temporal", "structural", "hybrid"],
        default="hybrid",
        help="Query strategy to use (default: hybrid)",
    )

    # Retrieval mode (new)
    parser.add_argument(
        "--retrieval",
        "-r",
        type=str,
        choices=["legacy", "bm25", "semantic", "hybrid"],
        default="legacy",
        help="Retrieval backend: legacy (keyword), bm25, semantic, hybrid (default: legacy)",
    )

    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.5,
        help="Weight for semantic scores in hybrid retrieval (default: 0.5)",
    )

    parser.add_argument(
        "--keyword-weight",
        type=float,
        default=0.5,
        help="Weight for keyword/BM25 scores in hybrid retrieval (default: 0.5)",
    )

    # Interactive mode
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode: query multiple times without reloading graph",
    )

    # Output options
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    parser.add_argument(
        "--max-nodes",
        type=int,
        default=20,
        help="Maximum number of nodes to return (default: 20)",
    )

    parser.add_argument(
        "--max-context",
        type=int,
        default=4000,
        help="Maximum context length in characters (default: 4000)",
    )

    # Convenience shortcuts
    parser.add_argument(
        "--top-concepts",
        type=int,
        metavar="N",
        help="Get top N most important concepts",
    )

    parser.add_argument(
        "--recent-intents",
        type=int,
        metavar="N",
        help="Get N most recent/relevant intents (goals/desires)",
    )

    # Backward compatibility alias
    parser.add_argument(
        "--recent-actions",
        type=int,
        metavar="N",
        dest="recent_intents",
        help="Alias for --recent-intents (deprecated)",
    )

    parser.add_argument(
        "--explain",
        type=str,
        metavar="NODE_ID",
        help="Get detailed explanation of a specific node",
    )

    # Prompt profile (validated for parity with `run`; see note below)
    parser.add_argument(
        "--prompt-profile",
        "--profile",
        dest="prompt_profile",
        type=str,
        help=(
            "Prompt profile ID (e.g., wiki-v1). Validated against the profiles "
            "file; surfaced in --verbose output. Note: query is read-only "
            "retrieval over a pre-built graph, so the profile does not change "
            "retrieval results -- profiles shape graph *building* via `run`."
        ),
    )
    parser.add_argument(
        "--prompt-profiles",
        type=str,
        default="configs/prompt_profiles.yaml",
        help="Path to prompt profiles YAML (default: configs/prompt_profiles.yaml)",
    )

    # Verbosity
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed query information",
    )


def _resolve_prompt_profile(name: str, profiles_path: str):
    """Load and return the named prompt profile, or None on error.

    On an unknown profile name (or missing file) prints the available profile
    names to stderr and returns None so the caller can exit non-zero. Uses the
    same loader the run command and benchmarks use
    (:func:`cognifold.agent.prompt_profile.load_prompt_profiles`).
    """
    from cognifold.agent.prompt_profile import load_prompt_profiles

    path = Path(profiles_path)
    if not path.exists():
        print(f"Error: Prompt profiles file not found: {path}", file=sys.stderr)
        return None

    profiles = load_prompt_profiles(path)
    profile = profiles.get(name)
    if profile is None:
        available = ", ".join(profiles) if profiles else "(none defined)"
        print(f"Error: Prompt profile not found: {name}", file=sys.stderr)
        print(f"Available profiles: {available}", file=sys.stderr)
        return None
    return profile


def _create_embedder(args: argparse.Namespace):
    """Create an embedder for semantic search if possible.

    Args:
        args: Command line arguments.

    Returns:
        NodeEmbedder or None if embedder cannot be created.
    """
    import os

    try:
        from cognifold.embeddings.config import EmbeddingConfig, EmbeddingProviderType
        from cognifold.embeddings.embedder import NodeEmbedder
    except ImportError:
        return None

    # Check for API keys
    if os.environ.get("GOOGLE_API_KEY"):
        config = EmbeddingConfig(
            provider=EmbeddingProviderType.GEMINI,
            dimensions=768,
        )
        return NodeEmbedder(config)
    elif os.environ.get("OPENAI_API_KEY"):
        config = EmbeddingConfig(
            provider=EmbeddingProviderType.OPENAI,
            dimensions=1536,
        )
        return NodeEmbedder(config)
    else:
        # Use mock embedder for testing without API keys
        config = EmbeddingConfig(
            provider=EmbeddingProviderType.MOCK,
            dimensions=128,
        )
        return NodeEmbedder(config)


def query_command(args: argparse.Namespace) -> int:
    """Execute the query command.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    from cognifold.graph.persistence import load_graph
    from cognifold.query import MemoryQueryAgent, QueryConfig, QueryType, RetrievalMode

    # Load the graph
    graph_path = Path(args.graph)
    if not graph_path.exists():
        print(f"Error: Graph file not found: {graph_path}", file=sys.stderr)
        return 1

    try:
        graph = load_graph(str(graph_path))
    except Exception as e:
        print(f"Error loading graph: {e}", file=sys.stderr)
        return 1

    # Validate prompt profile if provided (fails fast on typos, parity with `run`)
    prompt_profile = None
    if getattr(args, "prompt_profile", None):
        prompt_profile = _resolve_prompt_profile(args.prompt_profile, args.prompt_profiles)
        if prompt_profile is None:
            return 1

    # Map retrieval mode
    retrieval_mode_map = {
        "legacy": RetrievalMode.LEGACY,
        "bm25": RetrievalMode.BM25,
        "semantic": RetrievalMode.SEMANTIC,
        "hybrid": RetrievalMode.HYBRID,
    }
    retrieval_mode = retrieval_mode_map[args.retrieval]

    # Create embedder if needed for semantic/hybrid modes
    embedder = None
    if retrieval_mode in (RetrievalMode.SEMANTIC, RetrievalMode.HYBRID):
        embedder = _create_embedder(args)
        if embedder is None:
            print(
                "Warning: No embedder configured. Falling back to BM25 mode.",
                file=sys.stderr,
            )
            retrieval_mode = RetrievalMode.BM25

    # Create query config
    config = QueryConfig(
        max_nodes=args.max_nodes,
        max_context_chars=args.max_context,
        retrieval_mode=retrieval_mode,
        semantic_weight=args.semantic_weight,
        keyword_weight=args.keyword_weight,
    )

    # Create agent
    agent = MemoryQueryAgent(graph, config, embedder=embedder)

    # Handle convenience shortcuts
    if args.top_concepts:
        return _handle_top_concepts(agent, args)

    if args.recent_intents:
        return _handle_recent_intents(agent, args)

    if args.explain:
        return _handle_explain(agent, args)

    # Map query type
    query_type_map = {
        "semantic": QueryType.SEMANTIC,
        "temporal": QueryType.TEMPORAL,
        "structural": QueryType.STRUCTURAL,
        "hybrid": QueryType.HYBRID,
    }
    query_type = query_type_map[args.type]

    # Interactive mode
    if args.interactive:
        return _run_interactive(agent, args, query_type)

    # Regular query
    if not args.query:
        print("Error: Query string is required", file=sys.stderr)
        print(
            "Use --top-concepts, --recent-intents, --explain, or --interactive for other modes",
            file=sys.stderr,
        )
        return 1

    # Execute query
    if args.verbose:
        print(f"Querying graph with {graph.node_count} nodes...")
        print(f"Query: {args.query}")
        print(f"Type: {args.type}")
        print(f"Retrieval: {args.retrieval}")
        if prompt_profile is not None:
            print(
                f"Prompt profile: {prompt_profile.profile_id} "
                "(validated; does not affect retrieval)"
            )
        print()

    result = agent.query(
        query=args.query,
        query_type=query_type,
        max_nodes=args.max_nodes,
        max_context_chars=args.max_context,
    )

    # Output results
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        _print_result(result, args.verbose)

    return 0


def _run_interactive(
    agent: MemoryQueryAgent, args: argparse.Namespace, query_type: QueryType
) -> int:
    """Run interactive query mode.

    Args:
        agent: The query agent.
        args: Command line arguments.
        query_type: The query type to use.

    Returns:
        Exit code.
    """
    from cognifold.query import QueryType

    print("=" * 60)
    print("INTERACTIVE QUERY MODE")
    print("=" * 60)
    print(f"Graph: {args.graph}")
    print(f"Nodes: {agent.graph.node_count}")
    print(f"Retrieval: {args.retrieval}")
    print(f"Query type: {args.type}")
    print()
    print("Commands:")
    print("  Type a query and press Enter to search")
    print("  :top N      - Show top N concepts")
    print("  :recent N   - Show N recent intents")
    print("  :explain ID - Explain a node by ID")
    print("  :type TYPE  - Change query type (semantic/temporal/structural/hybrid)")
    print("  :quit       - Exit interactive mode")
    print()

    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive mode.")
            return 0

        if not query:
            continue

        # Handle commands
        if query.startswith(":"):
            parts = query[1:].split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("quit", "q", "exit"):
                print("Exiting interactive mode.")
                return 0

            elif cmd == "top":
                try:
                    n = int(arg) if arg else 5
                    concepts = agent.get_top_concepts(n=n)
                    print(f"\nTop {len(concepts)} Concepts:")
                    for i, c in enumerate(concepts, 1):
                        print(f"  {i}. {c.title} (score: {c.relevance_score:.2f})")
                    print()
                except ValueError:
                    print("Error: Invalid number")

            elif cmd == "recent":
                try:
                    n = int(arg) if arg else 5
                    intents = agent.get_recent_intents(n=n)
                    print(f"\nRecent {len(intents)} Intents:")
                    for i, intent in enumerate(intents, 1):
                        print(f"  {i}. {intent.title} (score: {intent.relevance_score:.2f})")
                    print()
                except ValueError:
                    print("Error: Invalid number")

            elif cmd == "explain":
                if not arg:
                    print("Error: Node ID required")
                    continue
                node = agent.explain_node(arg)
                if node:
                    print(f"\nNode: {node.node_id}")
                    print(f"Type: {node.node_type}")
                    print(f"Title: {node.title}")
                    if node.description:
                        print(f"Description: {node.description}")
                    if node.reasoning:
                        print(f"Reasoning: {node.reasoning}")
                    print()
                else:
                    print(f"Error: Node not found: {arg}")

            elif cmd == "type":
                type_map = {
                    "semantic": QueryType.SEMANTIC,
                    "temporal": QueryType.TEMPORAL,
                    "structural": QueryType.STRUCTURAL,
                    "hybrid": QueryType.HYBRID,
                }
                if arg.lower() in type_map:
                    query_type = type_map[arg.lower()]
                    print(f"Query type changed to: {arg.lower()}")
                else:
                    print(f"Error: Unknown type. Use: {', '.join(type_map.keys())}")

            else:
                print(f"Unknown command: {cmd}")

            continue

        # Execute query
        result = agent.query(
            query=query,
            query_type=query_type,
            max_nodes=args.max_nodes,
            max_context_chars=args.max_context,
        )

        print()
        print(f"Found {result.node_count} nodes in {result.query_time_ms:.1f}ms")
        print("-" * 40)

        # Show nodes
        for node in result.nodes[:10]:  # Show top 10
            print(f"  [{node.node_type}] {node.title} (score: {node.relevance_score:.3f})")

        if result.node_count > 10:
            print(f"  ... and {result.node_count - 10} more")

        # Show temporal references if any
        temporal_refs = result.query_metadata.get("temporal_references", [])
        if temporal_refs:
            print()
            print("Temporal references detected:")
            for ref in temporal_refs:
                print(f"  - '{ref['raw_text']}' ({ref['type']})")

        print()


def _handle_top_concepts(agent: MemoryQueryAgent, args: argparse.Namespace) -> int:
    """Handle --top-concepts shortcut.

    Args:
        agent: The query agent.
        args: Command line arguments.

    Returns:
        Exit code.
    """
    import json

    concepts = agent.get_top_concepts(n=args.top_concepts)

    if args.json:
        data = [
            {
                "node_id": c.node_id,
                "title": c.title,
                "description": c.description,
                "relevance_score": c.relevance_score,
                "reasoning": c.reasoning,
            }
            for c in concepts
        ]
        print(json.dumps(data, indent=2))
    else:
        print(f"Top {len(concepts)} Concepts:\n")
        for i, concept in enumerate(concepts, 1):
            print(f"{i}. {concept.title}")
            if concept.description:
                print(f"   {concept.description[:100]}...")
            if concept.reasoning:
                print(f"   Reasoning: {concept.reasoning}")
            print(f"   Relevance: {concept.relevance_score:.2f}")
            print()

    return 0


def _handle_recent_intents(agent: MemoryQueryAgent, args: argparse.Namespace) -> int:
    """Handle --recent-intents shortcut.

    Args:
        agent: The query agent.
        args: Command line arguments.

    Returns:
        Exit code.
    """
    import json

    # get_recent_intents falls back to get_recent_actions for backward compat
    intents = agent.get_recent_intents(n=args.recent_intents)

    if args.json:
        data = [
            {
                "node_id": i.node_id,
                "title": i.title,
                "description": i.description,
                "relevance_score": i.relevance_score,
                "reasoning": i.reasoning,
            }
            for i in intents
        ]
        print(json.dumps(data, indent=2))
    else:
        print(f"Recent {len(intents)} Intents:\n")
        for idx, intent in enumerate(intents, 1):
            print(f"{idx}. {intent.title}")
            if intent.description:
                print(f"   {intent.description[:100]}...")
            if intent.reasoning:
                print(f"   Reasoning: {intent.reasoning}")
            print(f"   Relevance: {intent.relevance_score:.2f}")
            print()

    return 0


# Backward compatibility alias
_handle_recent_actions = _handle_recent_intents


def _handle_explain(agent: MemoryQueryAgent, args: argparse.Namespace) -> int:
    """Handle --explain shortcut.

    Args:
        agent: The query agent.
        args: Command line arguments.

    Returns:
        Exit code.
    """
    import json

    node = agent.explain_node(args.explain)

    if node is None:
        print(f"Error: Node not found: {args.explain}", file=sys.stderr)
        return 1

    if args.json:
        data = {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "title": node.title,
            "description": node.description,
            "reasoning": node.reasoning,
            "grounded_in": node.grounded_in,
            "created_at": node.created_at.isoformat() if node.created_at else None,
            "data": node.data,
        }
        print(json.dumps(data, indent=2, default=str))
    else:
        print(f"Node: {node.node_id}")
        print(f"Type: {node.node_type}")
        print(f"Title: {node.title}")
        if node.description:
            print(f"Description: {node.description}")
        if node.reasoning:
            print(f"Reasoning: {node.reasoning}")
        if node.grounded_in:
            print(f"Grounded in: {', '.join(node.grounded_in)}")
        if node.created_at:
            print(f"Created: {node.created_at}")
        if node.data:
            print(f"Data: {json.dumps(node.data, indent=2)}")

    return 0


def _print_result(result: QueryResult, verbose: bool) -> int:
    """Print query result to stdout.

    Args:
        result: The query result.
        verbose: Whether to include detailed information.

    Returns:
        Exit code.
    """
    # Print context
    print("=" * 60)
    print("RETRIEVED CONTEXT")
    print("=" * 60)
    print()
    print(result.context)
    print()

    if verbose:
        print("=" * 60)
        print("QUERY DETAILS")
        print("=" * 60)
        print(f"Nodes returned: {result.node_count}")
        print(f"Nodes scanned: {result.total_nodes_scanned}")
        print(f"Query time: {result.query_time_ms:.1f}ms")
        print(f"Context length: {result.context_length} chars")
        print()

        if result.query_metadata:
            print("Query metadata:")
            for key, value in result.query_metadata.items():
                print(f"  {key}: {value}")
            print()

        print("Nodes by type:")
        for node_type in ["concept", "intent", "event", "time"]:
            nodes = result.get_nodes_by_type(node_type)
            # Also include legacy "action" nodes for backward compatibility
            if node_type == "intent":
                nodes = list(nodes) + list(result.get_nodes_by_type("action"))
            if nodes:
                print(f"  {node_type}: {len(nodes)}")

    return 0
