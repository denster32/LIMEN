#!/usr/bin/env python3
"""Reconcile local journal events into state/limen.json deterministically."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.state_core import log_entry_hash, state_snapshot_checksum
from scripts.validate_state import validate_state

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "state" / "limen.json"
JOURNAL_PATH = ROOT / "state" / "journal.jsonl"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_log_entry(state: dict, event: dict) -> None:
    prev_hash = state["log"][-1]["entry_hash"] if state["log"] else "GENESIS"
    entry = {
        "timestamp": event["timestamp"],
        "summary": event["summary"],
        "source": event.get("source", "local-journal"),
        "pending": event.get("pending", []),
    }
    entry["entry_hash"] = log_entry_hash(prev_hash, entry)
    state["log"].append(entry)


def apply_event(state: dict, event: dict) -> None:
    kind = event.get("type")
    if kind == "set_brief":
        state["brief"] = event["brief"]
    elif kind == "set_active":
        state["active"][event["project"]] = event["status"]
    elif kind == "add_pending":
        state["pending"].append(event["item"])
    elif kind == "remove_pending":
        state["pending"] = [i for i in state["pending"] if i != event["item"]]
    elif kind == "add_avoid":
        state["avoid"].append(event["item"])
    elif kind == "log":
        _append_log_entry(state, event)
    else:
        raise ValueError(f"Unknown event type: {kind}")


def reconcile() -> int:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if not JOURNAL_PATH.exists() or JOURNAL_PATH.stat().st_size == 0:
        print("No journal events to reconcile")
        return 0

    events = [json.loads(line) for line in JOURNAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    events.sort(key=lambda e: e.get("timestamp", ""))

    for event in events:
        apply_event(state, event)

    state["meta"]["total_conversations"] = max(state["meta"]["total_conversations"], len(state["log"]))
    state["meta"]["last_saved"] = now_iso()
    state["meta"]["session_id"] = events[-1].get("session_id", state["meta"]["session_id"])
    state["meta"]["snapshot_checksum"] = state_snapshot_checksum(state)
    validate_state(state)

    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    JOURNAL_PATH.write_text("", encoding="utf-8")
    print(f"Reconciled {len(events)} event(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(reconcile())
