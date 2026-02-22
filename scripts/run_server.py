#!/usr/bin/env python3
"""
Launch LIMEN server.

Modes:
  --mode mcp     MCP stdio transport (for Claude Desktop / Code)
  --mode rest    REST API on HTTP (for claude.ai via web_fetch)
  --mode both    MCP on stdio + REST on HTTP simultaneously
"""

import argparse
import asyncio
import threading


def run_mcp(state_path: str):
    """Run MCP server on stdio."""
    from limen.server import create_server
    mcp, store = create_server(state_path=state_path)
    mcp.run()


def run_rest(state_path: str, host: str = "0.0.0.0", port: int = 8452):
    """Run REST API on HTTP."""
    import uvicorn
    from limen.state import StateStore
    from limen.rest import create_rest_app

    store = StateStore(state_path)
    app = create_rest_app(store)
    uvicorn.run(app, host=host, port=port)


def run_both(state_path: str, host: str = "0.0.0.0", port: int = 8452):
    """Run MCP on stdio and REST on HTTP simultaneously, sharing state."""
    import uvicorn
    from limen.server import create_server
    from limen.rest import create_rest_app

    mcp, store = create_server(state_path=state_path)
    app = create_rest_app(store)

    # REST in a background thread
    rest_thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": host, "port": port, "log_level": "warning"},
        daemon=True,
    )
    rest_thread.start()

    # MCP on main thread (stdio)
    mcp.run()


def main():
    parser = argparse.ArgumentParser(description="LIMEN persistence server")
    parser.add_argument(
        "--state", "-s",
        default="./state/limen.json",
        help="Path to state file (default: ./state/limen.json)",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["mcp", "rest", "both"],
        default="mcp",
        help="Server mode: mcp (stdio), rest (HTTP), both (default: mcp)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="REST API host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8452,
        help="REST API port (default: 8452)",
    )
    args = parser.parse_args()

    if args.mode == "mcp":
        run_mcp(args.state)
    elif args.mode == "rest":
        run_rest(args.state, args.host, args.port)
    elif args.mode == "both":
        run_both(args.state, args.host, args.port)


if __name__ == "__main__":
    main()
