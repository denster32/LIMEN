#!/usr/bin/env python3
"""Conflict-safe sync to GitHub Contents API using sha optimistic concurrency."""

from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "state" / "limen.json"


def _request(url: str, token: str, method: str = "GET", data: dict | None = None) -> dict:
    body = None if data is None else json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, method=method, data=body)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--message", default="Update LIMEN state")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    args = parser.parse_args()

    if not args.token:
        raise ValueError("GitHub token required via --token or GITHUB_TOKEN")

    contents_url = f"https://api.github.com/repos/{args.repo}/contents/state/limen.json"
    remote = _request(f"{contents_url}?ref={args.branch}", args.token)
    sha = remote["sha"]

    content_b64 = base64.b64encode(STATE_PATH.read_bytes()).decode("ascii")
    payload = {
        "message": args.message,
        "content": content_b64,
        "sha": sha,
        "branch": args.branch,
    }

    try:
        response = _request(contents_url, args.token, method="PUT", data=payload)
    except urllib.error.HTTPError as exc:
        if exc.code == 409:
            raise RuntimeError("Conflict detected (sha mismatch). Pull latest and merge before retry.") from exc
        raise

    print(f"Updated state/limen.json on {args.repo}@{args.branch}: {response['commit']['sha']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
