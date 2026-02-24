#!/usr/bin/env python3
"""Offline journal append/sync for intermittent connectivity workflows."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.migrate_state import migrate
from scripts.state_tools import utc_now_iso

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "state" / "limen.json"
JOURNAL_PATH = ROOT / "local" / "journal.ndjson"


def _load_state() -> dict:
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def append_event(summary: str, project: str | None, human_override: bool) -> None:
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": utc_now_iso(),
        "summary": summary,
        "projects": [project] if project else [],
        "human_override": human_override,
    }
    with JOURNAL_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def sync_journal() -> None:
    if not JOURNAL_PATH.exists() or JOURNAL_PATH.stat().st_size == 0:
        return

    state = _load_state()
    events = [json.loads(line) for line in JOURNAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    events.sort(key=lambda x: x["timestamp"])

    for event in events:
        state["log"].append(
            {
                "timestamp": event["timestamp"],
                "summary": event["summary"],
                "projects": event.get("projects", []),
            }
        )
        if event.get("human_override") and event.get("projects"):
            for project in event["projects"]:
                state["active"][project] = f"Updated by local override @ {event['timestamp']}"

    state["meta"]["total_conversations"] = max(state["meta"].get("total_conversations", 0), len(state["log"]))
    migrated = migrate(state)
    STATE_PATH.write_text(json.dumps(migrated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    JOURNAL_PATH.write_text("", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    append_p = sub.add_parser("append")
    append_p.add_argument("--summary", required=True)
    append_p.add_argument("--project")
    append_p.add_argument("--human-override", action="store_true")

    sub.add_parser("sync")
    args = parser.parse_args()

    if args.cmd == "append":
        append_event(args.summary, args.project, args.human_override)
    else:
        sync_journal()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
