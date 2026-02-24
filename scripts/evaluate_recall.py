#!/usr/bin/env python3
"""Simple recall quality evaluation from generated MEMORY.md."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.sync_memory import render_memory

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "state" / "limen.json"


def evaluate(state: dict) -> tuple[int, int]:
    memory = render_memory(state)
    expected = []
    expected.extend(state.get("active", {}).keys())
    expected.extend(state.get("pending", []))
    expected.extend(state.get("avoid", []))
    hits = sum(1 for item in expected if item in memory)
    return hits, len(expected)


def main() -> int:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    hits, total = evaluate(state)
    score = 100.0 if total == 0 else (hits / total) * 100
    print(f"Recall score: {hits}/{total} ({score:.1f}%)")
    if score < 90:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
