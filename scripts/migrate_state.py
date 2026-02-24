#!/usr/bin/env python3
"""Backfill integrity and identity fields in state/limen.json."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.state_tools import compute_log_hash, compute_state_checksum, utc_now_iso

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "state" / "limen.json"


def migrate(state: dict) -> dict:
    meta = state.setdefault("meta", {})
    meta.setdefault("human_id", "human-default")
    meta.setdefault("agent_id", "agent-default")
    meta.setdefault("session_id", str(uuid.uuid4()))
    meta["last_saved"] = utc_now_iso()

    prev_hash = "GENESIS"
    for idx, entry in enumerate(state.get("log", []), start=1):
        entry.setdefault("entry_id", f"entry-{idx:06d}")
        entry["prev_hash"] = prev_hash
        entry["hash"] = compute_log_hash(entry, prev_hash)
        prev_hash = entry["hash"]

    meta["state_checksum"] = compute_state_checksum(state)
    return state


def main() -> int:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    migrated = migrate(state)
    STATE_PATH.write_text(json.dumps(migrated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("state migrated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
