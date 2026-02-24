#!/usr/bin/env python3
"""Integrity helpers for LIMEN state."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


def normalize_iso8601(value: str) -> str:
    normalized = value
    if normalized.endswith("Z") and ("+" in normalized[10:] or "-" in normalized[10:]):
        normalized = normalized[:-1]
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    return normalized


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_log_entry_hash(entry: dict[str, Any], prev_hash: str) -> str:
    payload = {
        "timestamp": entry["timestamp"],
        "summary": entry["summary"],
        "projects": entry.get("projects", []),
        "decisions": entry.get("decisions", []),
        "pending": entry.get("pending", []),
        "mistakes": entry.get("mistakes", []),
        "insights": entry.get("insights", []),
        "prev_hash": prev_hash,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def with_integrity_fields(state: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(state)
    log = updated.get("log", [])
    prev_hash = "GENESIS"
    for entry in log:
        entry["prev_hash"] = prev_hash
        entry["entry_hash"] = compute_log_entry_hash(entry, prev_hash)
        prev_hash = entry["entry_hash"]

    meta = updated.setdefault("meta", {})
    if "last_saved" not in meta:
        meta["last_saved"] = utc_now_iso()

    meta_for_hash = deepcopy(meta)
    meta_for_hash.pop("snapshot_checksum", None)
    snapshot_payload = {
        "brief": updated.get("brief", ""),
        "active": updated.get("active", {}),
        "pending": updated.get("pending", []),
        "avoid": updated.get("avoid", []),
        "log_tip": prev_hash,
        "meta": meta_for_hash,
    }
    meta["snapshot_checksum"] = hashlib.sha256(canonical_json(snapshot_payload).encode("utf-8")).hexdigest()
    return updated


def verify_integrity(state: dict[str, Any]) -> None:
    prev_hash = "GENESIS"
    for idx, entry in enumerate(state.get("log", [])):
        if entry.get("prev_hash") != prev_hash:
            raise ValueError(f"log[{idx}].prev_hash mismatch")
        expected_hash = compute_log_entry_hash(entry, prev_hash)
        if entry.get("entry_hash") != expected_hash:
            raise ValueError(f"log[{idx}].entry_hash mismatch")
        prev_hash = expected_hash

    meta = state.get("meta", {})
    snapshot_checksum = meta.get("snapshot_checksum")
    if snapshot_checksum:
        computed = with_integrity_fields(state).get("meta", {}).get("snapshot_checksum")
        if snapshot_checksum != computed:
            raise ValueError("meta.snapshot_checksum mismatch")
