#!/usr/bin/env python3
"""Evaluate whether MEMORY.md preserves key recall targets from state."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sync_memory import render_memory
STATE_PATH = ROOT / "state" / "limen.json"


def recall_score(state: dict, memory_text: str) -> float:
    checks = []
    checks.append(state.get("brief", "") in memory_text)
    checks.extend(name in memory_text for name in state.get("active", {}).keys())
    checks.extend(item in memory_text for item in state.get("pending", []))
    checks.extend(item in memory_text for item in state.get("avoid", []))
    if not checks:
        return 1.0
    return sum(1 for c in checks if c) / len(checks)


def main() -> int:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    memory_text = render_memory(state)
    score = recall_score(state, memory_text)
    print(f"recall_score={score:.3f}")
    if score < 0.95:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
