# LIMEN

Persistence for AI. No app server. Git-managed files with integrity, portability, and offline-safe workflows.

## Why LIMEN
LIMEN keeps long-term context in a repo so any model can load it from a URL and write updates back with standard GitHub primitives.

- **Read path:** `MEMORY.md` (short context for fast prompt priming)
- **Write path:** `state/limen.json` (source-of-truth data)
- **Automation:** scripts and CI keep files consistent, validated, and deploy-ready

## Quick start
1. Fork this repo.
2. Create a GitHub token with repo write access.
3. Point your AI to:
   - `https://raw.githubusercontent.com/<you>/LIMEN/main/MEMORY.md`
4. Update `state/limen.json` directly or via journal/API workflows.

## Data model
The canonical format is defined in `state/schema.json`.

```json
{
  "brief": "one-line memory",
  "active": {"Project": "status"},
  "pending": ["next action"],
  "avoid": ["anti-pattern"],
  "log": [{"timestamp": "ISO-8601", "summary": "session recap", "entry_hash": "sha256"}],
  "scratch": {},
  "meta": {
    "version": 3,
    "total_conversations": 1,
    "human_id": "stable-human-id",
    "agent_id": "stable-agent-id",
    "session_id": "current-session-id",
    "snapshot_checksum": "sha256",
    "last_saved": "ISO-8601"
  }
}
```

## Local workflow
```bash
make validate          # validate state shape + invariants + tamper checks
make sync-memory       # regenerate MEMORY.md from state
make reconcile-journal # merge local queue into canonical state
make eval-recall       # verify cold-start recall quality
make test              # run unit tests
make check             # full check + ensure MEMORY.md is up-to-date
```

## Conflict-safe GitHub write
Optimistic concurrency is supported via SHA compare in the GitHub Contents API wrapper:

```bash
python3 scripts/github_sync.py --repo <owner>/<repo> --branch main --message "Update LIMEN state"
```

If the remote SHA changed, the script exits with a conflict error so you can merge first.

## Offline-first mode
1. Queue events to `state/journal.jsonl` (append-only).
2. Reconcile deterministically when online:
   ```bash
   make reconcile-journal
   ```
3. Regenerate read context:
   ```bash
   make sync-memory
   ```
4. Push conflict-safe with `scripts/github_sync.py`.

## Recovery and portability
Create encrypted exports and restore when needed:

```bash
python3 scripts/export_state.py --out backup.limen.enc.json --passphrase '<strong-passphrase>'
python3 scripts/import_state.py --in backup.limen.enc.json --passphrase '<strong-passphrase>'
```

## Design principles
- Keep infra minimal.
- Store structured truth (`state/limen.json`) and generated context (`MEMORY.md`) separately.
- Prefer deterministic scripts + CI over manual edits.
- Detect drift with hash-chained logs and snapshot checksums.

## Deployment readiness checklist
- [x] Deterministic generation (`scripts/sync_memory.py`)
- [x] Schema + invariant validation (`state/schema.json`, `scripts/validate_state.py`)
- [x] Conflict-safe write path (`scripts/github_sync.py`)
- [x] Offline reconciliation queue (`scripts/reconcile_journal.py`)
- [x] Encrypted backup/restore (`scripts/export_state.py`, `scripts/import_state.py`)
- [x] Recall quality gate (`scripts/evaluate_recall.py`, tests)

## License
LGPL-2.1
