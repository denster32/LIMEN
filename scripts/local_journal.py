#!/usr/bin/env python3
"""Local-first journaling and reconciliation utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.state_integrity import utc_now_iso, with_integrity_fields

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "state" / "limen.json"
JOURNAL_PATH = ROOT / "state" / "journal.ndjson"


def queue_event(summary: str) -> None:
    event = {"timestamp": utc_now_iso(), "summary": summary}
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def reconcile_journal() -> int:
    if not JOURNAL_PATH.exists():
        return 0

    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    events: list[dict[str, Any]] = []
    for line in JOURNAL_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))

    if not events:
        return 0

    events.sort(key=lambda event: event["timestamp"])
    state.setdefault("log", []).extend(events)
    state.setdefault("meta", {})["total_conversations"] = max(
        state["meta"].get("total_conversations", 0), len(state["log"])
    )
    state["meta"]["last_saved"] = utc_now_iso()
    STATE_PATH.write_text(json.dumps(with_integrity_fields(state), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    JOURNAL_PATH.unlink()
    return len(events)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["queue", "reconcile"])
    parser.add_argument("--summary", help="Event summary for queue command")
    args = parser.parse_args()

    if args.command == "queue":
        if not args.summary:
            raise SystemExit("--summary is required for queue")
        queue_event(args.summary)
        print("queued")
        return 0

    applied = reconcile_journal()
    print(f"reconciled {applied} event(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
