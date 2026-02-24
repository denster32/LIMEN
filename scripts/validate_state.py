#!/usr/bin/env python3
"""Validate LIMEN state file invariants."""

from __future__ import annotations

from scripts.state_ops import compute_entry_hash, load_state, parse_iso8601

REQUIRED_TOP_LEVEL = {
    "brief": str,
    "active": dict,
    "pending": list,
    "avoid": list,
    "log": list,
    "scratch": dict,
    "meta": dict,
}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_timestamp(value: str, label: str) -> None:
    try:
        parse_iso8601(value)
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
    last_hash = "GENESIS"
    for i, entry in enumerate(log_entries):
        _assert(isinstance(entry, dict), f"log[{i}] must be an object")
        for field in ("timestamp", "summary", "entry_id", "prev_hash", "entry_hash"):
            _assert(field in entry and isinstance(entry[field], str), f"log[{i}].{field} must be a string")
        _validate_timestamp(entry["timestamp"], f"log[{i}].timestamp")
        _assert(entry["prev_hash"] == last_hash, f"log[{i}].prev_hash must match previous entry hash")
        _assert(entry["entry_hash"] == compute_entry_hash(entry), f"log[{i}].entry_hash does not match payload")
        last_hash = entry["entry_hash"]

    meta = state["meta"]
    _assert("version" in meta and isinstance(meta["version"], int), "meta.version must be an integer")
    _assert("total_conversations" in meta and isinstance(meta["total_conversations"], int), "meta.total_conversations must be int")
    _assert(meta["total_conversations"] >= len(log_entries), "meta.total_conversations must be >= len(log)")
    for key in ("human_id", "agent_id", "session_id"):
        _assert(key in meta and isinstance(meta[key], str) and meta[key].strip(), f"meta.{key} must be a non-empty string")
    if "last_saved" in meta:
        _assert(isinstance(meta["last_saved"], str), "meta.last_saved must be a string")
        _validate_timestamp(meta["last_saved"], "meta.last_saved")


def main() -> int:
    state = load_state()
    validate_state(state)
    print("state/limen.json is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
