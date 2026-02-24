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
  "log": [
    {
      "timestamp": "ISO-8601",
      "summary": "session recap",
      "prev_hash": "GENESIS|<sha256>",
      "entry_hash": "<sha256>"
    }
  ],
  "scratch": {},
  "meta": {
    "version": 2,
    "human_id": "stable-human-id",
    "agent_id": "stable-agent-id",
    "session_id": "session-identifier",
    "total_conversations": 1,
    "last_saved": "ISO-8601",
    "snapshot_checksum": "<sha256>"
  }
}
```

## Local workflow
```bash
make validate         # validate state shape/invariants + integrity chain
make sync-memory      # regenerate MEMORY.md from state
make test             # run unit tests
make check            # full check + ensure MEMORY.md is up-to-date
make reconcile-journal  # apply offline queued events into state
```

## Deploy-ready write options
1. **GitHub API with optimistic concurrency (recommended):**
   ```bash
   python3 -m scripts.github_sync --owner <you> --repo LIMEN --branch main --token "$GITHUB_TOKEN"
   ```
   - Fetches remote `sha`
   - Merges deterministic fields
   - Writes with `sha` guard to prevent blind overwrite
2. **Manual commit:** edit `state/limen.json`, run `make sync-memory`, commit.
3. **Issue workflow (optional):** create issue titled `LIMEN: summary` and process it with automation.

## Completed robustness plan
The reliability gaps are now implemented:

1. **Identity continuity**
   - `meta.human_id`, `meta.agent_id`, `meta.session_id` are required and validated.
2. **Conflict-safe writes**
   - `scripts/github_sync.py` performs optimistic concurrency using GitHub Contents API `sha` checks with deterministic merge behavior.
3. **Tamper and drift detection**
   - `log` entries are hash chained (`prev_hash` + `entry_hash`).
   - `meta.snapshot_checksum` protects high-level state against drift.
4. **Recovery and portability**
   - `scripts/backup_state.py` provides encrypted export/import with OpenSSL AES-256 + PBKDF2.
5. **Evaluation loop**
   - Tests validate memory rendering and recall fields from a cold-start style state payload.
6. **Intermittent local/offline mode**
   - `scripts/local_journal.py queue` appends events locally.
   - `scripts/local_journal.py reconcile` deterministically applies queued events when online.

## Operational commands
```bash
# Queue local/offline event
python3 -m scripts.local_journal queue --summary "Spoke with model, decided X"

# Reconcile local queue back into canonical state
python3 -m scripts.local_journal reconcile

# Encrypted backup/export
python3 -m scripts.backup_state export --file backups/limen.enc --passphrase "<strong-passphrase>"

# Encrypted restore/import
python3 -m scripts.backup_state import --file backups/limen.enc --passphrase "<strong-passphrase>"
```

## License
LGPL-2.1
