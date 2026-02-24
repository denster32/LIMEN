#!/usr/bin/env python3
"""Core helpers for LIMEN state integrity and deterministic serialization."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def log_entry_hash(prev_hash: str, entry: dict) -> str:
    payload = {k: v for k, v in entry.items() if k != "entry_hash"}
    return sha256_text(f"{prev_hash}|{canonical_json(payload)}")


def state_snapshot_checksum(state: dict) -> str:
    payload = {k: v for k, v in state.items() if k != "meta"}
    return sha256_text(canonical_json(payload))
