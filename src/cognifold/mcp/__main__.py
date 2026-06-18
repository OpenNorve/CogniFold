"""``python -m cognifold.mcp`` entry point."""

from __future__ import annotations

import sys

from cognifold.mcp.server import main

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        # Surface the "pip install 'cognifold[mcp]'" hint cleanly instead of a
        # traceback when the MCP SDK is not installed.
        print(str(exc), file=sys.stderr)
        sys.exit(1)
