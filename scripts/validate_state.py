#!/usr/bin/env python3
"""Validate LIMEN state file invariants."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from scripts.state_integrity import normalize_iso8601, verify_integrity

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "state" / "limen.json"

REQUIRED_TOP_LEVEL = {
    "brief": str,
    "active": dict,
    "pending": list,
    "avoid": list,
    "log": list,
    "scratch": dict,
    "meta": dict,
}

REQUIRED_META_FIELDS = ("version", "total_conversations", "human_id", "agent_id", "session_id")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_timestamp(value: str, label: str) -> None:
    try:
        datetime.fromisoformat(normalize_iso8601(value))
    except ValueError as exc:
        raise ValueError(f"{label} must be valid ISO-8601: {value}") from exc


def validate_state(state: dict) -> None:
    for field, expected_type in REQUIRED_TOP_LEVEL.items():
        _assert(field in state, f"Missing top-level field: {field}")
        _assert(isinstance(state[field], expected_type), f"{field} must be {expected_type.__name__}")

    _assert(state["brief"].strip() != "", "brief must not be empty")

    for key in ("pending", "avoid"):
        for i, item in enumerate(state[key]):
            _assert(isinstance(item, str), f"{key}[{i}] must be a string")

    for project, summary in state["active"].items():
        _assert(isinstance(project, str) and project.strip(), "active project names must be non-empty strings")
        _assert(isinstance(summary, str) and summary.strip(), f"active[{project}] must be non-empty string")

    log_entries = state["log"]
    for i, entry in enumerate(log_entries):
        _assert(isinstance(entry, dict), f"log[{i}] must be an object")
        for field in ("timestamp", "summary"):
            _assert(field in entry and isinstance(entry[field], str), f"log[{i}].{field} must be a string")
        _validate_timestamp(entry["timestamp"], f"log[{i}].timestamp")

    meta = state["meta"]
    for field in REQUIRED_META_FIELDS:
        _assert(field in meta, f"meta.{field} is required")
        if field == "total_conversations":
            _assert(isinstance(meta[field], int), "meta.total_conversations must be int")
        elif field == "version":
            _assert(isinstance(meta[field], int), "meta.version must be int")
        else:
            _assert(isinstance(meta[field], str) and meta[field].strip(), f"meta.{field} must be non-empty string")

    _assert(meta["total_conversations"] >= len(log_entries), "meta.total_conversations must be >= len(log)")
    if "last_saved" in meta:
        _assert(isinstance(meta["last_saved"], str), "meta.last_saved must be a string")
        _validate_timestamp(meta["last_saved"], "meta.last_saved")

    verify_integrity(state)


def main() -> int:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    validate_state(state)
    print("state/limen.json is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
