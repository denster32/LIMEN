#!/usr/bin/env python3
"""Shared state utilities for LIMEN workflows."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "state" / "limen.json"


def normalize_iso8601(value: str) -> str:
    normalized = value
    if normalized.endswith("Z") and ("+" in normalized[10:] or "-" in normalized[10:]):
        normalized = normalized[:-1]
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    return normalized


def parse_iso8601(value: str) -> datetime:
    return datetime.fromisoformat(normalize_iso8601(value))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_log_payload(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": entry.get("timestamp", ""),
        "summary": entry.get("summary", ""),
        "projects": entry.get("projects", []),
        "decisions": entry.get("decisions", []),
        "pending": entry.get("pending", []),
        "mistakes": entry.get("mistakes", []),
        "insights": entry.get("insights", []),
        "entry_id": entry.get("entry_id", ""),
        "prev_hash": entry.get("prev_hash", ""),
    }


def compute_entry_hash(entry: dict[str, Any]) -> str:
    body = json.dumps(canonical_log_payload(entry), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def load_state(path: Path = STATE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any], path: Path = STATE_PATH) -> None:
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
