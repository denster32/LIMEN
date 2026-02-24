#!/usr/bin/env python3
"""Sync state/limen.json to GitHub with optimistic concurrency."""

from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
from datetime import datetime
from typing import Any

from scripts.state_integrity import normalize_iso8601, with_integrity_fields


def _request(method: str, url: str, token: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload else None
    req = urllib.request.Request(url, method=method, data=data)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Authorization", f"Bearer {token}")
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def merge_states(remote: dict[str, Any], local: dict[str, Any]) -> dict[str, Any]:
    merged = dict(remote)
    merged["brief"] = local.get("brief", remote.get("brief", ""))
    merged["active"] = {**remote.get("active", {}), **local.get("active", {})}
    merged["pending"] = sorted(set(remote.get("pending", [])) | set(local.get("pending", [])))
    merged["avoid"] = sorted(set(remote.get("avoid", [])) | set(local.get("avoid", [])))
    merged["scratch"] = {**remote.get("scratch", {}), **local.get("scratch", {})}

    all_logs = remote.get("log", []) + local.get("log", [])
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for entry in sorted(all_logs, key=lambda item: normalize_iso8601(item["timestamp"])):
        key = (entry.get("timestamp", ""), entry.get("summary", ""))
        if key not in seen:
            deduped.append(entry)
            seen.add(key)
    merged["log"] = deduped

    merged_meta = {**remote.get("meta", {}), **local.get("meta", {})}
    merged_meta["total_conversations"] = max(merged_meta.get("total_conversations", 0), len(deduped))
    merged_meta["last_saved"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    merged["meta"] = merged_meta
    return with_integrity_fields(merged)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--path", default="state/limen.json")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--message", default="sync limen state")
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN"))
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("Provide --token or set GITHUB_TOKEN")

    with open("state/limen.json", "r", encoding="utf-8") as handle:
        local_state = json.load(handle)

    api_url = f"https://api.github.com/repos/{args.owner}/{args.repo}/contents/{args.path}?ref={args.branch}"
    remote_payload = _request("GET", api_url, args.token)
    remote_sha = remote_payload["sha"]
    remote_state = json.loads(base64.b64decode(remote_payload["content"]).decode("utf-8"))

    merged = merge_states(remote_state, local_state)
    content = base64.b64encode((json.dumps(merged, indent=2, ensure_ascii=False) + "\n").encode("utf-8")).decode("utf-8")
    put_payload = {"message": args.message, "content": content, "sha": remote_sha, "branch": args.branch}
    _request("PUT", api_url.split("?", 1)[0], args.token, put_payload)
    print("synced to GitHub")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
