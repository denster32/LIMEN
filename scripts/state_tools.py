#!/usr/bin/env python3
"""Shared helpers for LIMEN state integrity and serialization."""

from __future__ import annotations

import hashlib
import json
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
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_log_hash(entry: dict[str, Any], prev_hash: str) -> str:
    payload = {
        "entry_id": entry.get("entry_id", ""),
        "timestamp": entry.get("timestamp", ""),
        "summary": entry.get("summary", ""),
        "projects": entry.get("projects", []),
        "decisions": entry.get("decisions", []),
        "pending": entry.get("pending", []),
        "mistakes": entry.get("mistakes", []),
        "insights": entry.get("insights", []),
    }
    blob = canonical_json({"prev_hash": prev_hash, "payload": payload})
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compute_state_checksum(state: dict[str, Any]) -> str:
    to_hash = {
        "brief": state.get("brief", ""),
        "active": state.get("active", {}),
        "pending": state.get("pending", []),
        "avoid": state.get("avoid", []),
        "log": state.get("log", []),
        "scratch": state.get("scratch", {}),
    }
    return hashlib.sha256(canonical_json(to_hash).encode("utf-8")).hexdigest()
