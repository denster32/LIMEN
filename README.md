# LIMEN

Persistence for AI. No server. Just GitHub-managed files.

## Why LIMEN
LIMEN keeps long-term context in a repo so any model can load it from a URL and write updates back with standard GitHub primitives.

- **Read path:** `MEMORY.md` (short context for fast prompt priming)
- **Write path:** `state/limen.json` (source-of-truth data)
- **Automation:** scripts and CI keep both files consistent

## Quick start
1. Fork this repo.
2. Get a GitHub token with repo write access.
3. Point your AI to:
   - `https://raw.githubusercontent.com/<you>/LIMEN/main/MEMORY.md`
4. Update `state/limen.json` directly or through API/issue workflows.

## Data model
The canonical format is defined in `state/schema.json`.

```json
{
  "brief": "one-line memory",
  "active": {"Project": "status"},
  "pending": ["next action"],
  "avoid": ["anti-pattern"],
  "log": [{"timestamp": "ISO-8601", "summary": "session recap"}],
  "scratch": {},
  "meta": {"version": 2, "total_conversations": 1, "last_saved": "ISO-8601"}
}
```

## Local workflow
```bash
make validate      # validate state shape/invariants
make sync-memory   # regenerate MEMORY.md from state
make test          # run unit tests
make check         # full check + ensure MEMORY.md is up-to-date
```

## Write options
1. **GitHub API (recommended):** `PUT /repos/<you>/LIMEN/contents/state/limen.json`
2. **Manual commit:** edit `state/limen.json`, run `make sync-memory`, commit.
3. **Issue workflow (optional):** create issue titled `LIMEN: summary` and process it with automation.

## Design principles
- Keep infra minimal.
- Store structured truth (`state/limen.json`) and generated context (`MEMORY.md`) separately.
- Prefer deterministic scripts + CI over manual edits.

## License
LGPL-2.1
