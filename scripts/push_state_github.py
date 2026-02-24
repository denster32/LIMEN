#!/usr/bin/env python3
"""Conflict-safe state push via GitHub Contents API (SHA compare)."""

from __future__ import annotations

import argparse
import base64
import json
import urllib.request


def _request(url: str, token: str, method: str = "GET", body: dict | None = None) -> dict:
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        method=method,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--message", default="Update LIMEN state")
    parser.add_argument("--state-path", default="state/limen.json")
    parser.add_argument("--file", default="state/limen.json")
    args = parser.parse_args()

    with open(args.file, "rb") as fh:
        content = fh.read()

    api = f"https://api.github.com/repos/{args.owner}/{args.repo}/contents/{args.state_path}"
    remote = _request(f"{api}?ref={args.branch}", args.token)
    body = {
        "message": args.message,
        "content": base64.b64encode(content).decode("ascii"),
        "sha": remote["sha"],
        "branch": args.branch,
    }
    _request(api, args.token, method="PUT", body=body)
    print("push successful")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
