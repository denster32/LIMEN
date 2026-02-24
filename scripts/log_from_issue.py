#!/usr/bin/env python3
"""Convert LIMEN issue payload into state update using reconcile flow."""

from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.reconcile_journal import reconcile
from scripts.state_ops import load_state, save_state, utc_now_iso


def parse_issue_event(title: str, body: str) -> dict:
    summary = title.replace("LIMEN:", "", 1).strip() if title.startswith("LIMEN:") else title.strip()
    event = {
        "timestamp": utc_now_iso(),
        "summary": summary,
        "projects": [],
        "decisions": [],
        "pending": [],
        "mistakes": [],
        "insights": [],
    }

    for line in body.splitlines():
        line = line.strip()
        for field in ("projects", "decisions", "pending", "mistakes", "insights"):
            if line.lower().startswith(f"{field}:"):
                event[field] = [x.strip() for x in line.split(":", 1)[1].split(",") if x.strip()]
    return event


def main() -> int:
    title = os.environ.get("ISSUE_TITLE", "")
    body = os.environ.get("ISSUE_BODY", "")
    event = parse_issue_event(title, body)

    state = load_state()
    reconcile(state, [event])
    save_state(state)

    Path("issue-entry.json").write_text(json.dumps(event, indent=2), encoding="utf-8")
    print(f"Logged: {event['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
