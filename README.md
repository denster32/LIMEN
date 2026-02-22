# LIMEN

*Persistence for AI. No server required.*

LIMEN is a JSON file on GitHub. Your AI reads it at the start of every conversation and writes to it at the end. That's it.

No Docker. No server. No deployment. Just a file at a URL.

## Setup (2 minutes)

1. Fork this repo (or use it directly)
2. Create a [GitHub personal access token](https://github.com/settings/tokens) with `repo` scope
3. Tell your AI the read/write URLs (see below)

## How Your AI Uses It

**Start of conversation — read memory:**
```
GET https://raw.githubusercontent.com/YOUR_USER/LIMEN/main/state/limen.json
```

**End of conversation — write memory:**
Your AI updates the JSON and commits it back via the GitHub API.

That's the entire protocol.

## Tell Your AI

Paste this into your AI's custom instructions, system prompt, or just say it at the start of a conversation:

> You have persistent memory stored at https://raw.githubusercontent.com/YOUR_USER/LIMEN/main/state/limen.json
>
> At the start of each conversation, read that URL to get context from previous conversations.
>
> At the end of each conversation, update the file via the GitHub API using token YOUR_TOKEN.
> Add a new entry to the conversations array with: summary, projects, decisions, pending items, mistakes, and insights.
>
> GitHub API write endpoint:
> PUT https://api.github.com/repos/YOUR_USER/LIMEN/contents/state/limen.json
> Authorization: Bearer YOUR_TOKEN

Works with Claude, ChatGPT, Gemini, or any AI with web access.

## State Format

```json
{
  "conversations": [
    {
      "timestamp": "2025-02-22T23:00:00Z",
      "summary": "Rebuilt LIMEN as a simple JSON file instead of a server",
      "projects": ["LIMEN"],
      "decisions": ["No Docker, no server, just GitHub"],
      "pending": ["Test from ChatGPT"],
      "mistakes": ["Overengineered v0.1-v0.5"],
      "insights": ["The simplest version is a file at a URL"]
    }
  ],
  "projects": {
    "LIMEN": {
      "status": "active",
      "description": "AI persistence via GitHub-hosted JSON"
    }
  },
  "scratchpad": {},
  "last_saved": "2025-02-22T23:00:00Z"
}
```

## License

LGPL-2.1
