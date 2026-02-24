#!/usr/bin/env python3
"""Apply local journal events into state/limen.json with deterministic merge rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from scripts.state_ops import compute_entry_hash, load_state, parse_iso8601, save_state, utc_now_iso

DEFAULT_JOURNAL = Path("state/local_journal.jsonl")


def _append_log(state: dict[str, Any], event: dict[str, Any]) -> None:
    previous_hash = state["log"][-1]["entry_hash"] if state["log"] else "GENESIS"
    entry = {
        "timestamp": event.get("timestamp", utc_now_iso()),
        "summary": event["summary"],
        "projects": event.get("projects", []),
        "decisions": event.get("decisions", []),
        "pending": event.get("pending", []),
        "mistakes": event.get("mistakes", []),
        "insights": event.get("insights", []),
        "entry_id": event.get("entry_id", str(uuid4())),
        "prev_hash": previous_hash,
    }
    entry["entry_hash"] = compute_entry_hash(entry)
    state["log"].append(entry)


def _apply_event(state: dict[str, Any], event: dict[str, Any]) -> None:
    if event.get("override"):
        for project, summary in event.get("active", {}).items():
            state["active"][project] = summary
        if "pending_replace" in event:
            state["pending"] = list(dict.fromkeys(event["pending_replace"]))

    for p in event.get("pending", []):
        if p not in state["pending"]:
            state["pending"].append(p)

    for a in event.get("avoid", []):
        if a not in state["avoid"]:
            state["avoid"].append(a)

    _append_log(state, event)


def reconcile(state: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_events = sorted(events, key=lambda item: parse_iso8601(item.get("timestamp", utc_now_iso())))
    for event in sorted_events:
        if "summary" not in event or not str(event["summary"]).strip():
            raise ValueError("journal event requires non-empty summary")
        _apply_event(state, event)
    state["meta"]["total_conversations"] = len(state["log"])
    state["meta"]["last_saved"] = utc_now_iso()
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", default=str(DEFAULT_JOURNAL))
    parser.add_argument("--keep", action="store_true", help="keep journal entries after successful merge")
    args = parser.parse_args()

    journal = Path(args.journal)
    if not journal.exists():
        print(f"No journal file at {journal}; nothing to do")
        return 0

    lines = [line.strip() for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        print("Journal is empty; nothing to reconcile")
        return 0
    events = [json.loads(line) for line in lines]

    state = load_state()
    reconcile(state, events)
    save_state(state)
    if not args.keep:
        journal.write_text("", encoding="utf-8")
    print(f"Reconciled {len(events)} journal events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
