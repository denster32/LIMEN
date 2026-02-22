#!/usr/bin/env python3
"""Launch the LIMEN MCP server."""

import argparse
from limen.server import create_server


def main():
    parser = argparse.ArgumentParser(description="LIMEN persistence server")
    parser.add_argument(
        "--state", "-s",
        default="./state/limen.json",
        help="Path to state file (default: ./state/limen.json)",
    )
    args = parser.parse_args()

    server = create_server(state_path=args.state)
    server.run()


if __name__ == "__main__":
    main()
