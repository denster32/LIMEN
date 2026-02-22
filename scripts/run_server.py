#!/usr/bin/env python3
"""
Launch LIMEN server.

Modes:
  --mode mcp     MCP stdio transport (Claude Desktop / Code)
  --mode rest    REST API on HTTP (claude.ai via web_fetch, or any HTTP client)
  --mode both    MCP on stdio + REST on HTTP simultaneously
"""

import argparse
import os
import secrets
import threading


def run_mcp(state_path: str):
    """Run MCP server on stdio."""
    from limen.server import create_server
    mcp, store = create_server(state_path=state_path)
    mcp.run()


def run_rest(state_path: str, host: str = "0.0.0.0", port: int = 8452, token: str = None):
    """Run REST API on HTTP."""
    import uvicorn
    from limen.state import StateStore
    from limen.rest import create_rest_app

    store = StateStore(state_path)
    app = create_rest_app(store, auth_token=token)

    print(f"LIMEN REST API starting on http://{host}:{port}")
    if token:
        print(f"Auth token: {token}")
        print(f"Use: ?token={token} in URLs or Authorization: Bearer {token}")
    else:
        print("WARNING: No auth token set. API is open.")

    uvicorn.run(app, host=host, port=port)


def run_both(state_path: str, host: str = "0.0.0.0", port: int = 8452, token: str = None):
    """Run MCP on stdio and REST on HTTP simultaneously, sharing state."""
    import uvicorn
    from limen.server import create_server
    from limen.rest import create_rest_app

    mcp, store = create_server(state_path=state_path)
    app = create_rest_app(store, auth_token=token)

    # REST in background thread
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
        help="Server mode (default: mcp)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="REST host (default: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=8452, help="REST port (default: 8452)")
    parser.add_argument(
        "--token", "-t",
        default=os.environ.get("LIMEN_TOKEN"),
        help="Auth token for REST API (or set LIMEN_TOKEN env var). If omitted, generates one.",
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable auth (NOT recommended for public-facing servers)",
    )
    args = parser.parse_args()

    token = None
    if args.mode in ("rest", "both"):
        if args.no_auth:
            token = None
        elif args.token:
            token = args.token
        else:
            token = secrets.token_urlsafe(32)
            print(f"Generated auth token: {token}")
            print("Set LIMEN_TOKEN env var or use --token to use a fixed token.\n")

    if args.mode == "mcp":
        run_mcp(args.state)
    elif args.mode == "rest":
        run_rest(args.state, args.host, args.port, token)
    elif args.mode == "both":
        run_both(args.state, args.host, args.port, token)


if __name__ == "__main__":
    main()
