"""Entry point for running Cognifold as a module.

Usage:
    python -m cognifold [command] [options]

Examples:
    python -m cognifold --help
    python -m cognifold generate --list
    python -m cognifold run data/mock_timeline.json --agent
"""

import sys

from cognifold.cli import main

if __name__ == "__main__":
    sys.exit(main())
