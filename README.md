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
4. Validate and sync:
   - `make migrate-state`
   - `make check`

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
      "entry_id": "entry-000001",
      "timestamp": "ISO-8601",
      "summary": "session recap",
      "prev_hash": "GENESIS|<sha256>",
      "hash": "<sha256>"
    }
  ],
  "scratch": {},
  "meta": {
    "version": 2,
    "total_conversations": 1,
    "last_saved": "ISO-8601",
    "human_id": "stable-id",
    "agent_id": "stable-id",
    "session_id": "stable-id",
    "state_checksum": "sha256"
  }
}
```

## Local workflow
```bash
make migrate-state  # backfill IDs + integrity hashes/checksum
make validate       # validate state shape/invariants/integrity
make sync-memory    # regenerate MEMORY.md from state
make recall         # verify MEMORY.md covers recall targets
make test           # run unit tests
make check          # full check + ensure MEMORY.md is up-to-date
```

## Conflict-safe writes (GitHub API)
Use optimistic concurrency with SHA compare:

```bash
python3 scripts/push_state_github.py \
  --owner <you> --repo LIMEN --token $GITHUB_TOKEN --branch main
```

This reads the remote file SHA and submits a `PUT /contents` update with that SHA, preventing blind overwrite when concurrent writers update state first.

## Offline/intermittent mode
Queue local journal events while offline, then reconcile when online.

```bash
python3 scripts/local_journal.py append --summary "Did useful work" --project LIMEN
python3 scripts/local_journal.py sync
make sync-memory
```

Merge policy: deterministic timestamp ordering, with optional `--human-override` to explicitly update active project status.

## Recovery and portability
Create authenticated encrypted backups and restore later:

```bash
python3 scripts/export_import.py export --output backups/limen.enc.json --passphrase "$LIMEN_BACKUP_PASSPHRASE"
python3 scripts/export_import.py import --input backups/limen.enc.json --passphrase "$LIMEN_BACKUP_PASSPHRASE"
```

## Deployment readiness
- CI workflow (`.github/workflows/ci.yml`) runs `make check` on push/PR.
- `make check` enforces schema/invariants, memory sync consistency, recall quality threshold, and unit tests.
- Deterministic hash-chain and snapshot checksum provide tamper/drift detection.

## License
LGPL-2.1
