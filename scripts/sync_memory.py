#!/usr/bin/env python3
"""Generate MEMORY.md from state/limen.json."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.state_tools import normalize_iso8601

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "state" / "limen.json"
MEMORY_PATH = ROOT / "MEMORY.md"


def _fmt_date(iso_timestamp: str | None) -> str:
    if not iso_timestamp:
        return datetime.now().strftime("%Y-%m-%d")
    try:
        dt = datetime.fromisoformat(normalize_iso8601(iso_timestamp))
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return iso_timestamp[:10]


def render_memory(state: dict) -> str:
    active = state.get("active", {})
    pending = state.get("pending", [])
    avoid = state.get("avoid", [])
    logs = state.get("log", [])
    meta = state.get("meta", {})

    lines: list[str] = ["# Memory", "", state.get("brief", "No brief set."), "", "## Working On"]

    if active:
        for name, description in active.items():
            lines.append(f"- **{name}** — {description}")
    else:
        lines.append("- (none)")

    lines += ["", "## Pending"]
    if pending:
        lines.extend(f"- {item}" for item in pending)
    else:
        lines.append("- (none)")

    lines += ["", "## Don't"]
    if avoid:
        lines.extend(f"- {item}" for item in avoid)
    else:
        lines.append("- (none)")

    lines += ["", "## Last Session"]
    if logs:
        lines.append(str(logs[-1].get("summary", "No summary available.")))
    else:
        lines.append("No sessions recorded.")

    lines += [
        "",
        "## Identity",
        f"- Human ID: `{meta.get('human_id', 'unset')}`",
        f"- Agent ID: `{meta.get('agent_id', 'unset')}`",
        f"- Session ID: `{meta.get('session_id', 'unset')}`",
        "",
        "---",
        f"*Updated {_fmt_date(meta.get('last_saved'))}. Integrity checksum: `{meta.get('state_checksum', 'unknown')}`. Full log: state/limen.json.*",
        "",
    ]

    return "\n".join(lines)


def main() -> int:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    MEMORY_PATH.write_text(render_memory(state), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
