#!/usr/bin/env python3
"""Evaluate whether MEMORY.md preserves key state context for cold starts."""

from __future__ import annotations

from scripts.state_ops import load_state
from scripts.sync_memory import render_memory


def evaluate() -> tuple[int, int, list[str]]:
    state = load_state()
    memory = render_memory(state)
    checks: list[tuple[str, bool]] = []

    for project in state.get("active", {}).keys():
        checks.append((f"active:{project}", project in memory))
    for item in state.get("pending", []):
        checks.append((f"pending:{item}", item in memory))
    for item in state.get("avoid", []):
        checks.append((f"avoid:{item}", item in memory))

    last_summary = state.get("log", [])[-1]["summary"] if state.get("log") else ""
    checks.append(("last_summary", last_summary in memory))

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    failures = [name for name, ok in checks if not ok]
    return passed, total, failures


def main() -> int:
    passed, total, failures = evaluate()
    print(f"Recall score: {passed}/{total}")
    if failures:
        print("Missing:", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
