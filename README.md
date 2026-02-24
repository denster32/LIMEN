# LIMEN

Persistence for AI. No server. A JSON file on GitHub.

## Use it

**Read** (any AI, no auth):
```
https://raw.githubusercontent.com/YOUR_USER/LIMEN/main/MEMORY.md
```

**Write** (log a session):
```
GitHub Issue → title: "LIMEN: what happened"
Body: projects, decisions, pending, mistakes, insights
```

**Update state directly** (brief, projects, avoid list):
```
GitHub Issue → title: "LIMEN-UPDATE: <field>"
```
Fields: `brief` | `avoid` | `active:<Project>` | `active-remove:<Project>` | `pending-done` | `scratch`

**Or via GitHub API:**
```
PUT /repos/YOU/LIMEN/contents/state/limen.json
```

**Setup** (2 min): Fork. Get a [token](https://github.com/settings/tokens). Tell your AI the URL.

## State

```json
{
  "brief": "One-line about the user",
  "active": {"project": "one-line status"},
  "pending": ["action items"],
  "avoid": ["anti-patterns"],
  "log": ["last 200 raw conversations"],
  "concepts": {"word": {"count": 5}},
  "distilled": {
    "top_concepts": ["recurring themes extracted from log"],
    "trajectory": "N sessions in last 7 days",
    "session_count": 42
  },
  "scratch": {"hypotheses": [], "questions": [], "threads": {}},
  "meta": {"version": 3, "total_conversations": 1}
}
```

MEMORY.md auto-regenerates from state on every commit. After each log entry,
concepts are extracted from decisions/insights/mistakes and surfaced in MEMORY.md
under "Recurring Themes" — the feedback loop that enables emergence.

## License

LGPL-2.1
