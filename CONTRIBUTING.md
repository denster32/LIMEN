# Contributing

## Development loop
1. Edit `state/limen.json` or project files.
2. Run `make check`.
3. Commit only when checks pass.

## Commit guidance
- Keep commits focused and reversible.
- If state changes, include regenerated `MEMORY.md` in the same commit.

## Project constraints
- No servers required.
- Avoid adding heavyweight dependencies when stdlib can do the job.
- Preserve hash-chain integrity for `log` entries.
- Keep `meta.human_id`, `meta.agent_id`, and `meta.session_id` stable across updates.
