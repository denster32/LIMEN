# LIMEN

Persistence for AI. No server. Just GitHub-managed files.

## Why LIMEN
LIMEN keeps long-term context in a repo so any model can load it from a URL and write updates back with standard GitHub primitives.

- **Read path:** `MEMORY.md` (short context for fast prompt priming)
- **Write path:** `state/limen.json` (source-of-truth data)
- **Automation:** scripts and CI keep both files consistent

## What is now production-ready
The roadmap has been implemented end-to-end for robust local-first continuity:

1. **Identity continuity**
   - `meta` now includes stable IDs: `human_id`, `agent_id`, `session_id`.
2. **Conflict-safe writes**
   - `scripts/push_state_github.py` uses Contents API with SHA compare (optimistic concurrency).
3. **Tamper and drift detection**
   - Log entries use append-only hash chaining (`prev_hash`, `entry_hash`, `entry_id`).
4. **Recovery and portability**
   - `scripts/backup_state.py` supports encrypted export/import for account-loss recovery.
5. **Evaluation loop**
   - `scripts/evaluate_recall.py` checks that MEMORY.md captures active/pending/avoid/last-session recall.
6. **Offline / intermittent mode**
   - `scripts/reconcile_journal.py` merges `state/local_journal.jsonl` with deterministic rules and optional override.

## Quick start
1. Fork this repo.
2. Get a GitHub token with repo write access.
3. Point your AI to:
   - `https://raw.githubusercontent.com/<you>/LIMEN/main/MEMORY.md`
4. Update `state/limen.json` directly, via issue workflow, or local journal + reconcile.

## Data model
The canonical format is defined in `state/schema.json`.

```json
{
  "brief": "one-line memory",
  "active": {"Project": "status"},
  "pending": ["next action"],
  "avoid": ["anti-pattern"],
  "log": [{
    "timestamp": "ISO-8601",
    "summary": "session recap",
    "entry_id": "stable id",
    "prev_hash": "GENESIS-or-previous",
    "entry_hash": "sha256"
  }],
  "scratch": {},
  "meta": {
    "version": 2,
    "human_id": "stable human id",
    "agent_id": "stable agent id",
    "session_id": "stable session id",
    "total_conversations": 1,
    "last_saved": "ISO-8601"
  }
}
```

## Local workflow
```bash
make validate         # validate state shape/invariants + hash chain
make sync-memory      # regenerate MEMORY.md from state
make test             # run unit tests
make recall           # recall quality check for cold-start memory
make check            # full check + ensure MEMORY.md is up-to-date
```

## Offline journal workflow
```bash
# append JSON lines to state/local_journal.jsonl (one event per line)
python3 -m scripts.reconcile_journal
python3 -m scripts.sync_memory
```

Journal event example:
```json
{"timestamp":"2026-02-24T00:00:00Z","summary":"offline session","pending":["next step"],"override":false}
```

For explicit human override, set `"override": true` and include `"active"` and/or `"pending_replace"`.

## GitHub API safe write workflow
```bash
python3 -m scripts.push_state_github \
  --repo <owner>/LIMEN \
  --token $GITHUB_TOKEN \
  --branch main \
  --expected-sha <current_sha>
```

## Encrypted backup / restore
```bash
python3 -m scripts.backup_state export --output limen.backup.json --password '<passphrase>'
python3 -m scripts.backup_state import --input limen.backup.json --password '<passphrase>'
```

## Deployment readiness checklist
- [x] CI quality gate (`make check`) on push/PR.
- [x] Issue ingestion workflow appends validated log entries and syncs memory.
- [x] Memory regeneration workflow validates state before commit.
- [x] Deterministic scripts only; no service runtime required.

## License
LGPL-2.1
