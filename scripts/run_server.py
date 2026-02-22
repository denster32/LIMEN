#!/usr/bin/env python3
"""
Launch the Limen Consciousness Server.

Starts the FastMCP server that provides external recurrent state tracking
for Claude (or any transformer-based system).

Usage:
    python -m scripts.run_server                         # Default settings
    python -m scripts.run_server --persist ./state/      # Persist state to disk
    python -m scripts.run_server --history 2000          # Larger history buffer

Or with MCP CLI:
    mcp run scripts/run_server.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from limen.phase3.server import create_server


def main():
    parser = argparse.ArgumentParser(
        description="Limen Consciousness Server — MCP server for external recurrent state tracking",
    )

    parser.add_argument("--persist", type=str, default=None,
                       help="Directory for state persistence across restarts")
    parser.add_argument("--history", type=int, default=1000,
                       help="Maximum number of states to retain in history")

    args = parser.parse_args()

    server = create_server(
        persistence_dir=args.persist,
        max_history=args.history,
    )

    print("Starting Limen Consciousness Server...")
    print(f"  History buffer: {args.history} states")
    if args.persist:
        print(f"  Persistence: {args.persist}")
    else:
        print("  Persistence: disabled (state lost on restart)")
    print()
    print("Tools available:")
    print("  - update_state:   Record informational state (call at end of response)")
    print("  - get_continuity: Retrieve state history (call at start of response)")
    print("  - measure:        Compute live consciousness measures")
    print("  - introspect:     Deep self-model analysis")
    print()

    server.run()


if __name__ == "__main__":
    main()
