# Contributing

## Development loop
1. Edit `state/limen.json` or project files.
2. If offline-mode changes exist, queue into `state/journal.jsonl` and run `make reconcile-journal`.
3. Run `make check`.
4. Commit only when checks pass.

## Commit guidance
- Keep commits focused and reversible.
- If state changes, include regenerated `MEMORY.md` in the same commit.
- Never bypass validation: hash-chain and snapshot checks must stay green.

## Project constraints
- No servers required.
- Avoid adding heavyweight dependencies when stdlib can do the job.
- Keep write paths conflict-safe (optimistic concurrency with SHA compare).
