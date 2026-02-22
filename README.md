# LIMEN

Persistence for AI. No server. A JSON file on GitHub.

## Use it

**Read** (any AI, no auth):
```
https://raw.githubusercontent.com/YOUR_USER/LIMEN/main/MEMORY.md
```

**Write** (3 ways):
1. GitHub API → `PUT /repos/YOU/LIMEN/contents/state/limen.json`
2. GitHub Issue → title it `LIMEN: what happened`
3. Manual → edit state/limen.json

**Setup** (2 min): Fork. Get a [token](https://github.com/settings/tokens). Tell your AI the URL.

## State

```json
{
  "brief": "One-line about the user",
  "active": {"project": "one-line status"},
  "pending": ["action items"],
  "avoid": ["anti-patterns"],
  "log": ["last 200 raw conversations"],
  "scratch": {},
  "meta": {"version": 2, "total_conversations": 1}
}
```

MEMORY.md auto-regenerates from state on every commit.

## License

LGPL-2.1
