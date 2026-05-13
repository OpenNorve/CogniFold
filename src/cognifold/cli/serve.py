"""CLI command for running the Cognifold HTTP service."""

from __future__ import annotations

import argparse
from typing import Any


def add_serve_parser(subparsers: Any) -> None:
    """Add the 'serve' subcommand parser."""
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "serve",
        help="Start the Cognifold HTTP service",
        description="Run the FastAPI-based Cognifold service for event processing and graph queries.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes (default: 1)"
    )
    parser.add_argument(
        "--persist-dir",
        default="./sessions",
        help="Directory for session persistence (default: ./sessions)",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=100,
        help="Maximum concurrent sessions (default: 100)",
    )
    parser.add_argument(
        "--api-key",
        action="append",
        default=None,
        help="Valid API key (repeatable). Omit for no auth.",
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        default=False,
        help="Disable API key authentication",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)",
    )
    parser.add_argument(
        "--session-backend",
        default="file",
        choices=["file", "redis"],
        help="Session storage backend (default: file)",
    )
    parser.add_argument(
        "--redis-url",
        default="redis://localhost:6379/0",
        help="Redis URL (when --session-backend=redis)",
    )
    parser.add_argument(
        "--gunicorn",
        action="store_true",
        default=False,
        help="Run with Gunicorn instead of uvicorn (production)",
    )


def serve_command(args: argparse.Namespace) -> int:
    """Run the Cognifold HTTP service."""
    # Set up structured logging before anything else
    from cognifold.logging import setup_logging

    setup_logging()

    try:
        import uvicorn
    except ImportError:
        print(
            "Error: uvicorn is not installed. "
            "Install service dependencies: pip install cognifold[service]"
        )
        return 1

    from cognifold.service.app import AppSettings, create_app

    # Determine API keys
    api_keys: set[str] | None = None
    if not args.no_auth and args.api_key:
        api_keys = set(args.api_key)
    elif args.no_auth:
        api_keys = None
    else:
        # No keys provided and --no-auth not set: default to no auth with warning
        print("Warning: No API keys configured. Running without authentication.")
        print(
            "  Use --api-key <key> to require authentication, or --no-auth to silence this warning."
        )
        api_keys = None

    settings = AppSettings(
        persist_dir=args.persist_dir,
        max_sessions=args.max_sessions,
        api_keys=api_keys,
        session_backend=args.session_backend,
        redis_url=args.redis_url,
    )

    if args.gunicorn:
        try:
            import subprocess
            import sys

            cmd = [
                sys.executable,
                "-m",
                "gunicorn",
                "cognifold.service.wsgi:app",
                "--bind",
                f"{args.host}:{args.port}",
                "--workers",
                str(args.workers),
                "--worker-class",
                "uvicorn.workers.UvicornWorker",
                "--timeout",
                "120",
                "--access-logfile",
                "-",
            ]
            return subprocess.call(cmd)
        except ImportError:
            print(
                "Error: gunicorn is not installed. "
                "Install production dependencies: pip install cognifold[production]"
            )
            return 1

    app = create_app(settings)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
    )
    return 0
