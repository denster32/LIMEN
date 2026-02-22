# LIMEN

*Persistence for AI. No server. No app. Just a file.*

## For users

Your memory lives at one URL:

```
https://raw.githubusercontent.com/YOUR_USER/LIMEN/main/MEMORY.md
```

Tell any AI to read it. That's it. It gets your history and knows how to save new entries.

### Setup (once, 2 minutes)

1. [Fork this repo](https://github.com/denster32/LIMEN/fork)
2. Create a [GitHub token](https://github.com/settings/tokens) with `repo` scope
3. Tell your AI platform (one time per platform):

**Claude** — Add to memory:
> LIMEN persistence at github.com/YOUR_USER/LIMEN/state/limen.json. Read via GitHub API at conversation start, write at end. PAT: YOUR_TOKEN

**ChatGPT** — Add to Custom Instructions:
> I have persistent memory at https://raw.githubusercontent.com/YOUR_USER/LIMEN/main/MEMORY.md — read it at the start of every conversation.

**Gemini** — Add to Gems or paste at conversation start:
> Read my memory: https://raw.githubusercontent.com/YOUR_USER/LIMEN/main/MEMORY.md

**Any AI** — Just paste the URL.

### How it writes back

Three ways, depending on what the AI can do:

1. **GitHub API** (Claude w/ computer use, ChatGPT w/ code interpreter) — AI reads and writes JSON directly via `api.github.com`
2. **GitHub Issue** (any AI that can browse) — AI creates an issue titled `LIMEN: summary here`, GitHub Action commits it to memory
3. **Manual** — You paste the summary, AI formats it, you copy it to an issue or commit

## State format

```json
{
  "conversations": [
    {
      "timestamp": "2025-02-22T23:48:47Z",
      "summary": "What happened",
      "projects": ["project names"],
      "decisions": ["what was decided"],
      "pending": ["what's next"],
      "mistakes": ["what to avoid"],
      "insights": ["what was learned"]
    }
  ],
  "projects": {},
  "scratchpad": {},
  "last_saved": "2025-02-22T23:48:47Z"
}
```

## How it works

```
Any AI  ──read──>  raw.githubusercontent.com/YOU/LIMEN/main/MEMORY.md
        ──write──> api.github.com (PUT) or GitHub Issue
                          │
                   state/limen.json  (your memory)
                          │
                   MEMORY.md auto-generated from state
```

No server. No Docker. No database. GitHub is the infrastructure.

## License

LGPL-2.1
