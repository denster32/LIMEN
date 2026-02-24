#!/usr/bin/env python3
"""Push state/limen.json via GitHub Contents API with optimistic concurrency."""

from __future__ import annotations

import argparse
import base64
import json
import urllib.error
import urllib.request


API = "https://api.github.com/repos/{repo}/contents/state/limen.json"


def _request(url: str, token: str, method: str = "GET", data: dict | None = None) -> dict:
    payload = None if data is None else json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--token", required=True)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--message", default="chore: update limen state")
    parser.add_argument("--expected-sha", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    url = API.format(repo=args.repo)
    current = _request(f"{url}?ref={args.branch}", args.token)
    current_sha = current["sha"]

    if args.expected_sha and current_sha != args.expected_sha:
        raise SystemExit(f"SHA mismatch: expected {args.expected_sha}, got {current_sha}")

    with open("state/limen.json", "rb") as f:
        raw = f.read()

    body = {
        "message": args.message,
        "content": base64.b64encode(raw).decode("ascii"),
        "branch": args.branch,
        "sha": current_sha,
    }

    if not args.force:
        try:
            _request(url, args.token, method="PUT", data=body)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8")
            raise SystemExit(f"GitHub update failed (optimistic lock): {detail}") from exc
    else:
        _request(url, args.token, method="PUT", data=body)

    print(f"Updated state/limen.json on {args.repo}@{args.branch} using base sha {current_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
