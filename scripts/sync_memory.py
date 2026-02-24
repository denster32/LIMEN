#!/usr/bin/env python3
"""Generate MEMORY.md from state/limen.json."""

from __future__ import annotations

from datetime import datetime

from scripts.state_ops import load_state

ROOT_NOTE = "Full log: state/limen.json. Write back via GitHub API or issue titled `LIMEN: summary`."


def _fmt_date(iso_timestamp: str | None) -> str:
    if not iso_timestamp:
        return datetime.now().strftime("%Y-%m-%d")
    try:
        normalized = iso_timestamp
        if normalized.endswith("Z") and ("+" in normalized[10:] or "-" in normalized[10:]):
            normalized = normalized[:-1]
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        dt = datetime.fromisoformat(normalized)
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
    lines.extend(f"- {item}" for item in pending) if pending else lines.append("- (none)")

    lines += ["", "## Don't"]
    lines.extend(f"- {item}" for item in avoid) if avoid else lines.append("- (none)")

    lines += ["", "## Last Session"]
    lines.append(str(logs[-1].get("summary", "No summary available.")) if logs else "No sessions recorded.")

    lines += [
        "",
        "---",
        f"*Updated {_fmt_date(meta.get('last_saved'))}. {ROOT_NOTE}*",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    state = load_state()
    from pathlib import Path

    (Path(__file__).resolve().parents[1] / "MEMORY.md").write_text(render_memory(state), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
